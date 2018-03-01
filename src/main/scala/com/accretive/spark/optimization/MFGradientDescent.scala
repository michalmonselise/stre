package com.accretive.spark.optimization

import com.accretive.spark.recommendation._
import org.apache.spark.ml.recommendation.ALS.Rating
import breeze.linalg._
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
      val currentStepSize = (stepSize * math.pow(stepDecay, i)).toFloat
      val currentBiasStepSize = biasStepSize * math.pow(stepDecay, i)
      val gradients: RDD[(Long, (Long, Float, LatentFactor))] = ratings.map(r => (r.user, r)).
        join[LatentFactor](userFeatures.map(x => (x.id, x.latent))).
        map { case (user, (rating, uFeatures)) =>
          (rating.item, (user, rating.rating, uFeatures))
        }
      val grad: RDD[(Long, LatentFactor)] = gradients.join[LatentFactor](prodFeatures.map(x => (x.id, x.latent))).
        map { case (item, ((user, rating, uFeatures), pFeatures)) =>
          val step = MFGradientDescent.oneSidedGradientStep(rating, LatentID(uFeatures, user), LatentID(pFeatures, item),
            globalBias, currentStepSize, currentBiasStepSize, lambda)
          (user, step)
        }.persist(intermediateStorageLevel)

      val userGradients: RDD[(Long, LatentFactor)] = grad.aggregateByKey(LatentFactor(0f, DenseVector.zeros[Float](rank)))(
          seqOp = (base, example) => base += example,
          combOp = (a, b) => a += b
        )

      val uf = userFeatures.map(x => (x.id, x.latent)).leftOuterJoin[LatentFactor](userGradients)
        userFeatures = uf map {
        case (id, (base: LatentFactor, gradient: Option[LatentFactor])) =>
        val a = gradient.head.add(base)
        LatentID(a, id)
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
      stepSize: Float,
      biasStepSize: Double,
      lambda: Double): (LatentFactor, LatentFactor) = {
    val predicted = LatentMatrixFactorizationModel.getRating(userFeatures, prodFeatures, bias)
    val epsilon = rating - predicted
    val user = userFeatures.latent.vector
    val rank = user.length
    val prod = prodFeatures.latent.vector

    val uFeatures = stepSize * (user * epsilon - lambda * prod)
    val pFeatures = stepSize * (prod * epsilon - lambda * user)

    val userBiasGrad: Float = (biasStepSize * (epsilon - lambda * userFeatures.latent.bias)).toFloat
    val prodBiasGrad: Float = (biasStepSize * (epsilon - lambda * prodFeatures.latent.bias)).toFloat

    (LatentFactor(userBiasGrad, uFeatures), LatentFactor(prodBiasGrad, pFeatures))
  }

  private[spark] def oneSidedGradientStep(
                                   rating: Float,
                                   userFeatures: LatentID,
                                   prodFeatures: LatentID,
                                   bias: Float,
                                   stepSize: Float,
                                   biasStepSize: Double,
                                   lambda: Double): LatentFactor = {
    val predicted = LatentMatrixFactorizationModel.getRating(userFeatures, prodFeatures, bias)
    val epsilon = rating - predicted
    val user = userFeatures.latent.vector
    val rank = user.length
    val prod = prodFeatures.latent.vector

    val uFeatures = stepSize * (prod * epsilon - lambda * user)
    val userBiasGrad: Float = (biasStepSize * (epsilon - lambda * userFeatures.latent.bias)).toFloat

    LatentFactor(userBiasGrad, uFeatures)
  }
}
