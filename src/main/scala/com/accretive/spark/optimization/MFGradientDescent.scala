package com.accretive.spark.optimization

import breeze.linalg
import com.accretive.spark.recommendation._
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.recommendation.ALSModel
import breeze.linalg._
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

/**
 * A Gradient Descent Optimizer specialized for Matrix Factorization.
 *
 * @param params The parameters to use
 */
private[spark] class MFGradientDescent(params: LatentMatrixFactorizationParams) {

  def this() = this(new LatentMatrixFactorizationParams)

  def train(
           userFactors: DataFrame,
           itemFactors: DataFrame,
           ratings: DataFrame,
           globalBias: Float,
           rank: Int): DataFrame = {

    val lambda = params.getLambda.toFloat
    val stepSize = params.getStepSize
    val stepDecay = params.getStepDecay
    val biasStepSize = params.getBiasStepSize
    val iter = params.getIter
    val intermediateStorageLevel = params.getIntermediateStorageLevel
    val rank = params.getRank

    for (i <- 0 until iter) {
      val currentStepSize = (stepSize * math.pow(stepDecay, i)).toFloat
      val currentBiasStepSize = biasStepSize * math.pow(stepDecay, i)
      val userFactorsRenamed = userFactors.withColumnRenamed("features", "userFeatures")
      val itemFactorsRenamed = itemFactors.withColumnRenamed("features", "itemFeatures")
      val gradients = ratings.join(userFactorsRenamed, ratings.col("userid") === userFactors("id")).drop("id").drop("lastSpendDate")
      val grad = gradients.join(itemFactorsRenamed, gradients.col("performerid") === itemFactors.col("id")).drop("id")
      val step = grad.map(x =>
        MFGradientDescent.oneSidedGradientStep(x, globalBias, currentStepSize, currentBiasStepSize, lambda))
      val stepDF = step.toDF("userid", "performerid", "amount", "userFeatures", "userBiasGrad", "itemFeatures")
      val userGradients = stepDF.aggregateByKey(LatentFactor(0f, DenseVector.zeros[Float](rank)))(
          seqOp = (base, example) => base += example,
          combOp = (a, b) => a += b
        )

      val uf = userFactors.leftOuterJoin[LatentFactor](userGradients)
        userFeatures = uf map {
        case (id, (base: LatentFactor, gradient: Option[LatentFactor])) =>
        val a = gradient.head.add(base)
        LatentID(a, id)
      }
    }
  }
}



private[spark] object MFGradientDescent extends Serializable {

  // Exposed for testing
//  private[spark] def gradientStep(x: Row,
//      bias: Float,
//      stepSize: Float,
//      biasStepSize: Double,
//      lambda: Double): Row = {
//    val predicted = LatentMatrixFactorizationModel.getRating(userFeatures, prodFeatures, bias)
//    val epsilon = rating - predicted
//    val user = userFeatures.latent.vector
//    val rank = user.length
//    val prod = prodFeatures.latent.vector
//
//    val uFeatures = stepSize * (user * epsilon - lambda * prod)
//    val pFeatures = stepSize * (prod * epsilon - lambda * user)
//
//    val userBiasGrad: Float = (biasStepSize * (epsilon - lambda * userFeatures.latent.bias)).toFloat
//    val prodBiasGrad: Float = (biasStepSize * (epsilon - lambda * prodFeatures.latent.bias)).toFloat
//
//    (LatentFactor(userBiasGrad, uFeatures), LatentFactor(prodBiasGrad, pFeatures))
//  }

  def oneSidedGradientStep(x: Row,
                           bias: Float,
                           stepSize: Float,
                           biasStepSize: Double,
                           lambda: Float): Row = {
    val (userid, performerid, amount, userFeatures, bias, prodFeatures): (Long, Long, Float, Array[Float], Float, Array[Float]) =
      (x.getLong(0), x.getLong(1), x.getFloat(2), x.getList(3).toArray.map(_.toString.toFloat),
        x.getFloat(4), x.getList(5).toArray.map(_.toString.toFloat))
    val predicted: Float = getRating(x, bias)
    val epsilon: Float = amount - predicted
    val user: DenseVector[Float] = DenseVector(userFeatures)
    val rank = user.length
    val prod: DenseVector[Float] = DenseVector(prodFeatures)

    val uFeatures = stepSize * (prod * epsilon - lambda * user)
    val userBiasGrad: Float = (biasStepSize * (epsilon - lambda * bias)).toFloat

    Row(userid, performerid, amount, uFeatures, userBiasGrad, prodFeatures)
  }

  def getRating(x: Row,
                globalBias: Float): Float = {
    val (userid, performerid, amount, userFeatures, bias, prodFeatures): (Long, Long, Float, Array[Float], Float, Array[Float]) =
      (x.getLong(0), x.getLong(1), x.getFloat(2), x.getList(3).toArray.map(_.toString.toFloat),
        x.getFloat(4), x.getList(5).toArray.map(_.toString.toFloat))
    val uFeatures: DenseVector[Float] = DenseVector(userFeatures)
    val pFeatures: DenseVector[Float] = DenseVector(prodFeatures)
    val dotProd = uFeatures dot pFeatures
    dotProd + bias + globalBias
  }
}
