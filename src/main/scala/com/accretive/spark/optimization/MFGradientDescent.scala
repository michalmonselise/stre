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
class MFGradientDescent(params: LatentMatrixFactorizationParams) {

  def this() = this(new LatentMatrixFactorizationParams)

  def train(
           userFactors: DataFrame,
           itemFactors: DataFrame,
           ratings: DataFrame,
           globalBias: Double,
           rank: Int): DataFrame = {

    val lambda = params.getLambda.toDouble
    val stepSize = params.getStepSize
    val stepDecay = params.getStepDecay
    val biasStepSize = params.getBiasStepSize
    val iter = params.getIter
    val intermediateStorageLevel = params.getIntermediateStorageLevel
    val rank = params.getRank

    for (i <- 0 until iter) {
      val currentStepSize = stepSize * math.pow(stepDecay, i)
      val currentBiasStepSize = biasStepSize * math.pow(stepDecay, i)
      val userFactorsRenamed = userFactors.withColumnRenamed("features", "userFeatures")
      val itemFactorsRenamed = itemFactors.withColumnRenamed("features", "itemFeatures")
      val gradients = ratings.join(userFactorsRenamed, ratings.col("userid") === userFactors("id")).drop("id").drop("lastSpendDate")
      val grad = gradients.join(itemFactorsRenamed, gradients.col("performerid") === itemFactors.col("id")).drop("id")
      val step = grad.rdd.map(x => (x.getLong(0), x.getLong(1), x.getDouble(2), x.getList(3).toArray.map(_.toString.toDouble)
          ,x.getDouble(4), x.getList(5).toArray.map(_.toString.toDouble)))
      val step2: RDD[(Long, Long, Double, Array[Double], Double, Array[Double])] = step.map(x =>
        MFGradientDescent.oneSidedGradientStep(x._1, x._2, x._3, x._4, x._5, x._6, globalBias, currentStepSize, currentBiasStepSize, lambda))
      val stepDF = step2.toDF("userid", "performerid", "amount", "userFeatures", "userBiasGrad", "itemFeatures")
      val userGradients = step2.map(x => (x._1, x._4)).aggregateByKey((0, DenseVector.zeros[Double](rank)))(
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
//      bias: Double,
//      stepSize: Double,
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
//    val userBiasGrad: Double = (biasStepSize * (epsilon - lambda * userFeatures.latent.bias)).toDouble
//    val prodBiasGrad: Double = (biasStepSize * (epsilon - lambda * prodFeatures.latent.bias)).toDouble
//
//    (LatentFactor(userBiasGrad, uFeatures), LatentFactor(prodBiasGrad, pFeatures))
//  }

  def oneSidedGradientStep(userid: Long,
                           performerid: Long,
                           amount: Double,
                           userFeatures: Array[Double],
                           bias:Double,
                           prodFeatures: Array[Double],
                           globalBias: Double,
                           stepSize: Double,
                           biasStepSize: Double,
                           lambda: Double): (Long, Long, Double, Array[Double], Double, Array[Double]) = {
    val predicted: Double = getRating(userFeatures, prodFeatures, bias, globalBias)
    val epsilon: Double = amount - predicted
    val user: DenseVector[Double] = DenseVector(userFeatures)
    val rank = user.length
    val prod: DenseVector[Double] = DenseVector(prodFeatures)

    val uFeatures = stepSize * (prod * epsilon - lambda * user)
    val userBiasGrad: Double = biasStepSize * (epsilon - lambda * bias)

    (userid, performerid, amount, uFeatures.toArray, userBiasGrad, prodFeatures)
  }

  def getRating(userFeatures: Array[Double],
                prodFeatures: Array[Double],
                bias: Double,
                globalBias: Double): Double = {
    val uFeatures: DenseVector[Double] = DenseVector(userFeatures)
    val pFeatures: DenseVector[Double] = DenseVector(prodFeatures)
    val dotProd = uFeatures dot pFeatures
    dotProd + bias + globalBias
  }
}
