package com.accretive.spark.recommendation

import java.util.Random

import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.ml.recommendation.ALS.Rating
import breeze.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.recommendation.ALSModel

class LatentMatrixFactorizationModel(
    val rank: Int,
    val userFeatures: RDD[LatentID], // bias and the user row
    val productFeatures: RDD[LatentID], // bias and the product row
    val globalBias: Float) {

  val log: Logger = LoggerFactory.getLogger(this.getClass)
  /** Predict the rating of one user for one product. */
  def predict(user: Long, product: Long): Float = {
    val uFeatures: Option[LatentID] = userFeatures.filter(u => u.id == user).collect().headOption
    val pFeatures: Option[LatentID] = productFeatures.filter(p => p.id == product).collect().headOption
    LatentMatrixFactorizationModel.predict(uFeatures, pFeatures, globalBias).rating
  }


  /**
   * Predict the rating of many users for many products.
   * The output RDD will return a prediction for all user - product pairs. For users or
   * products that were missing from the training data, the prediction will be made with the global
   * bias (global average) +- the user or product bias, if they exist.
   *
   * @param usersProducts  RDD of (user, product) pairs.
   * @return RDD of Ratings.
   */
  def predict(usersProducts: RDD[(Long, Long)]): RDD[Rating[Long]] = {
    val users = usersProducts.leftOuterJoin[LatentFactor](userFeatures.map(x => (x.id, x.latent))).
      map { case (user, (product, uFeatures)) =>
      (product, (user, uFeatures))
    }
    val sc = usersProducts.sparkContext
    val globalAvg = sc.broadcast(globalBias)
    users.leftOuterJoin[LatentFactor](productFeatures.map(x => (x.id, x.latent))).
      map { case (product, ((user, uFeatures), pFeatures)) =>
      LatentMatrixFactorizationModel.predict(Option(LatentID(uFeatures.head, user)),
        Option(LatentID(pFeatures.head, product)), globalAvg.value)
    }
  }
}

case class StreamingLatentMatrixFactorizationModel(
    override val rank: Int,
    override val userFeatures: RDD[LatentID], // bias and the user row
    override val productFeatures: RDD[LatentID], // bias and the product row
    override val globalBias: Float,
    observedExamples: Long)
  extends LatentMatrixFactorizationModel(rank, userFeatures, productFeatures, globalBias)

object LatentMatrixFactorizationModel extends Serializable {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  /**
   * Adds random factors for missing user - product entries and updates the global bias and
   * number of observed examples. Returns the initialized model, and number of examples in this rdd.
   */
  def initialize(
      ratings: RDD[Rating[Long]],
      params: LatentMatrixFactorizationParams,
      initialModel: ALSModel,
      initialLatentModel: Option[LatentMatrixFactorizationModel],
      isStreaming: Boolean = false): (LatentMatrixFactorizationModel, Long) = {
    val rank = params.getRank
    val seed = params.getSeed

    val randGenerator = new LatentFactorGenerator(rank)

    // Generate random entries for missing user-product factors
    val usersAndRatings: RDD[(Long, Rating[Long])] = ratings.map(r => (r.user, r))
    val productsAndRatings: RDD[(Long, Rating[Long])] = ratings.map(r => (r.item, r))
    val sc = ratings.sparkContext
    var userFeatures: RDD[LatentID] = initialLatentModel match {
      case Some(model) => model.userFeatures
      case None =>
        sc.parallelize(Seq.empty[(Long, LatentFactor)], ratings.partitions.length).map(x => LatentID(x._2, x._1))
    }

    var prodFeatures: RDD[LatentID] = initialLatentModel match {
      case Some(model) => model.productFeatures
      case None =>
        sc.parallelize(Seq.empty[(Long, LatentFactor)], ratings.partitions.length).map(x => LatentID(x._2, x._1))
    }

    val userFeaturesJoined = usersAndRatings.fullOuterJoin[LatentFactor](userFeatures.map(x => (x.id, x.latent)))
    val userFeaturesWithRandom = userFeaturesJoined.mapPartitionsWithIndex {
      case (partitionId, iterator) =>
        iterator.map { case (user, (rating, uFeatures)) =>
          (user, uFeatures.getOrElse(randGenerator.nextValue()))
        }
    }.map(x => LatentID(x._2, x._1))

    val prodFeaturesJoined = productsAndRatings.fullOuterJoin[LatentFactor](prodFeatures.map(x => (x.id, x.latent)))
    val prodFeaturesWithRandom = prodFeaturesJoined.mapPartitionsWithIndex { case (partitionId, iterator) =>
        iterator.map { case (user, (rating, pFeatures)) =>
          (user, pFeatures.getOrElse(randGenerator.nextValue()))
        }
    }.map(x => LatentID(x._2, x._1))

    val (ratingSum, numRatings) =
      ratings.map(r => (r.rating, 1L)).reduce((a, b) => (a._1 + b._1, a._2 + b._2))

    val (globalBias, numExamples) = initialModel.getOrElse(None) match {
      case streaming: StreamingLatentMatrixFactorizationModel =>
        val examples: Long = streaming.observedExamples + numRatings
        ((streaming.globalBias * streaming.observedExamples + ratingSum) / examples, examples)
      case _ => (ratingSum / numRatings, numRatings)
    }

    val initializedModel = initialModel.getOrElse(None) match {
      case streaming: StreamingLatentMatrixFactorizationModel =>
        StreamingLatentMatrixFactorizationModel(rank, userFeaturesWithRandom, prodFeaturesWithRandom,
          streaming.globalBias, streaming.observedExamples)
      case _ =>
        if (isStreaming) {
          StreamingLatentMatrixFactorizationModel(rank, userFeaturesWithRandom, prodFeaturesWithRandom, globalBias, numExamples)
        } else {
          new LatentMatrixFactorizationModel(rank, userFeatures, prodFeatures, globalBias)
        }
    }
    (initializedModel, numRatings)
  }

  def predict(
      uFeatures: Option[LatentID],
      pFeatures: Option[LatentID],
      globalBias: Float): Rating[Long] = {
    val user = uFeatures.head.id
    val product = pFeatures.head.id
    val finalRating =
      if (uFeatures.isDefined && pFeatures.isDefined) {
        Rating(user, product, LatentMatrixFactorizationModel.getRating(uFeatures.get, pFeatures.get,
          globalBias))
      } else if (uFeatures.isDefined) {
        log.warn(s"Product data missing for product id $product. Will use user factors.")
        val rating = globalBias + uFeatures.get.latent.bias
        Rating(user, product, 0f)
      } else if (pFeatures.isDefined) {
        log.warn(s"User data missing for user id $user. Will use product factors.")
        val rating = globalBias + pFeatures.get.latent.bias
        Rating(user, product, 0f)
      } else {
        log.warn(s"Both user and product factors missing for ($user, $product). " +
          "Returning global average.")
        val rating = globalBias
        Rating(user, product, 0f)
      }
    finalRating
  }

  def getRating(
      userFeatures: LatentID,
      prodFeatures: LatentID,
      bias: Float): Float = {
    val dot = userFeatures.latent.vector dot prodFeatures.latent.vector
    dot + userFeatures.latent.bias + prodFeatures.latent.bias + bias
  }
}

case class LatentFactor(var bias: Float, vector: breeze.linalg.DenseVector[Float]) extends Serializable {

  def +=(other: LatentFactor): this.type = {
    val resBias = bias + other.bias
    val resVector = vector + other.vector
    LatentFactor(resBias, resVector)
  }

  def add(other: LatentFactor): this.type = {
    this += other
  }

  override def toString: String = {
    s"bias: $bias, factors: " + vector.toString
  }
}

case class LatentID(var latent: LatentFactor, id: Long) extends Serializable {

  override def toString: String = {
    s"id: $id.toString, Factors: " + latent.vector.toString
  }
}

class LatentFactorGenerator(rank: Int) extends Serializable {

  private val random = new java.util.Random()

  def nextValue(): LatentFactor = {
    LatentFactor(random.nextFloat, randomDouble(rank).map(x => x.toFloat))
  }
}
