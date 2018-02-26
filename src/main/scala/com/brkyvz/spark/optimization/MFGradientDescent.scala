package com.brkyvz.spark.optimization

import com.brkyvz.spark.recommendation._
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.rdd.RDD

/**
 * A Gradient Descent Optimizer specialized for Matrix Factorization.
 *
 * @param params The parameters to use
 */
private[spark] class MFGradientDescent(params: LatentMatrixFactorizationParams) {

  def this() = this(new LatentMatrixFactorizationParams)

  def train(
      ratings: RDD[Rating[Long]],
      initialModel: LatentMatrixFactorizationModel,
      numExamples: Long): LatentMatrixFactorizationModel = {

    var userFeatures = initialModel.userFeatures
    var prodFeatures = initialModel.productFeatures
    val globalBias = initialModel.globalBias
    val lambda = params.getLambda
    val stepSize = params.getStepSize
    val stepDecay = params.getStepDecay
    val biasStepSize = params.getBiasStepSize
    val iter = params.getIter
    val intermediateStorageLevel = params.getIntermediateStorageLevel
    val rank = params.getRank

    for (i <- 0 until iter) {
      val currentStepSize = stepSize * math.pow(stepDecay, i)
      val currentBiasStepSize = biasStepSize * math.pow(stepDecay, i)
      val gradients: RDD[(Long, (Long, Float, LatentFactor))] = ratings.map(r => (r.user, r)).
        join[LatentFactor](userFeatures.map(x => (x.id, x.latent))).
        map { case (user, (rating, uFeatures)) =>
          (rating.item, (user, rating.rating, uFeatures))
        }
      val grad: RDD[((Long, LatentFactor), (Long, LatentFactor))] = gradients.join[LatentFactor](prodFeatures.map(x => (x.id, x.latent))).
        map { case (item, ((user, rating, uFeatures), pFeatures)) =>
          val step = MFGradientDescent.gradientStep(rating, LatentID(uFeatures, user), LatentID(pFeatures, item),
            globalBias, currentStepSize, currentBiasStepSize, lambda)
          ((user, step._1), (item, step._2))
        }.persist(intermediateStorageLevel)

      val userGradients: RDD[(Long, LatentFactor)] = grad.map(_._1)
        .aggregateByKey(LatentFactor(0f, new Array[Float](rank)))(
          seqOp = (base, example) => base += example,
          combOp = (a, b) => a += b
        )
      val prodGradients: RDD[(Long, LatentFactor)] = grad.map(_._2)
        .aggregateByKey(LatentFactor(0f, new Array[Float](rank)))(
          seqOp = (base, example) => base += example,
          combOp = (a, b) => a += b
        )
      userFeatures = userFeatures.map(x => (x.id, x.latent)).leftOuterJoin(userGradients) { case (id, base, gradient) =>
        gradient.foreach(g => base.divideAndAdd(g, numExamples))
        LatentID(base, id)
      }
      prodFeatures = prodFeatures.map(x => (x.id, x.latent)).leftOuterJoin(prodGradients) { case (id, base, gradient) =>
        gradient.foreach(g => base.divideAndAdd(g, numExamples))
        base
      }
    }
    initialModel match {
      case streaming: StreamingLatentMatrixFactorizationModel =>
        StreamingLatentMatrixFactorizationModel(rank, userFeatures, prodFeatures,
          globalBias, streaming.observedExamples)
      case _ =>
        new LatentMatrixFactorizationModel(rank, userFeatures, prodFeatures, globalBias)
    }
  }
}

private[spark] object MFGradientDescent extends Serializable {

  // Exposed for testing
  private[spark] def gradientStep(
      rating: Float,
      userFeatures: LatentID,
      prodFeatures: LatentID,
      bias: Float,
      stepSize: Double,
      biasStepSize: Double,
      lambda: Double): (LatentFactor, LatentFactor) = {
    val predicted = LatentMatrixFactorizationModel.getRating(userFeatures, prodFeatures, bias)
    val epsilon = rating - predicted
    val user = userFeatures.latent.vector
    val rank = user.length
    val prod = prodFeatures.latent.vector

    val featureGradients = Array.tabulate(rank) { i =>
      ((stepSize * (prod(i) * epsilon - lambda * user(i))).toFloat,
        (stepSize * (user(i) * epsilon - lambda * prod(i))).toFloat)
    }
    val userBiasGrad: Float = (biasStepSize * (epsilon - lambda * userFeatures.latent.bias)).toFloat
    val prodBiasGrad: Float = (biasStepSize * (epsilon - lambda * prodFeatures.latent.bias)).toFloat

    val uFeatures = featureGradients.map(_._1)
    val pFeatures = featureGradients.map(_._2)
    (LatentFactor(userBiasGrad, uFeatures), LatentFactor(prodBiasGrad, pFeatures))
  }
}
