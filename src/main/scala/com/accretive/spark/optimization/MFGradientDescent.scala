package com.accretive.spark.optimization

import breeze.linalg
import com.accretive.spark.recommendation._
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.recommendation.ALSModel
import breeze.linalg._
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

/**
 * A Gradient Descent Optimizer specialized for Matrix Factorization.
 *
 * @param params The parameters to use
 */
class MFGradientDescent(params: LatentMatrixFactorizationParams) {

  def this() = this(new LatentMatrixFactorizationParams)
  val spark: SparkSession = SparkSession.builder().getOrCreate()
  def train(
           userFactors: org.apache.spark.sql.DataFrame,
           itemFactors: org.apache.spark.sql.DataFrame,
           ratings: org.apache.spark.sql.DataFrame,
           globalBias: Double,
           rank: Int): org.apache.spark.sql.DataFrame = {

    val lambda = params.getLambda
    val stepSize: Double = params.getStepSize
    val stepDecay = params.getStepDecay
    val biasStepSize = params.getBiasStepSize
    val iter = params.getIter
    //val intermediateStorageLevel = params.getIntermediateStorageLevel
    val rank = params.getRank

    def iteration(stepSize: Double,
                  biasStepSize: Double,
                  globalBias: Double,
                  stepDecay: Double,
                  rank: Int,
                  userFactors: org.apache.spark.sql.DataFrame,
                  itemFactors: org.apache.spark.sql.DataFrame,
                  ratings: org.apache.spark.sql.DataFrame,
                  lambda: Double,
                  iter: Int
                 ): org.apache.spark.sql.DataFrame = {
      val currentStepSize = stepSize * math.pow(stepDecay, iter)
      val currentBiasStepSize = biasStepSize * math.pow(stepDecay, iter)
      val userFactorsRenamed = userFactors.withColumnRenamed("features", "userFeatures")
      val itemFactorsRenamed = itemFactors.withColumnRenamed("features", "itemFeatures")
      val gradients = ratings.join(userFactorsRenamed, ratings.col("userid") === userFactors("id")).drop("id").drop("lastSpendDate")
      val grad = gradients.join(itemFactorsRenamed, gradients.col("performerid") === itemFactors.col("id")).drop("id")
      val step = grad.rdd.map(x => (x.getLong(0), x.getLong(1), x.getDouble(2), x.getList(3).toArray.map(_.toString.toDouble)
        ,x.getDouble(4), x.getList(5).toArray.map(_.toString.toDouble)))
      val schema = new StructType()
        .add(StructField("userid", LongType, nullable = true))
        .add(StructField("performerid", LongType, nullable = true))
        .add(StructField("amount", DoubleType, nullable = true))
        .add(StructField("userFeatures", ArrayType(DoubleType, containsNull = false), nullable = true))
        .add(StructField("userBiasGrad", DoubleType, nullable = true))
        .add(StructField("itemFeatures", ArrayType(DoubleType, containsNull = false), nullable = true))
      val step2: RDD[Row] = step.map(x =>
        MFGradientDescent.oneSidedGradientStep(x._1, x._2, x._3, x._4, x._5, x._6, globalBias, currentStepSize, currentBiasStepSize, lambda))
      val stepDF: org.apache.spark.sql.DataFrame = spark.createDataFrame(step2, schema)
      val output = stepDF.select("userid", "userFeatures").withColumnRenamed("userFeatures", "features")
      output
    }
    var curUserFactors = userFactors
    var prevUserFactors = userFactors
    for (i <- 0 until iter) {
      curUserFactors = iteration(stepSize, biasStepSize, globalBias, stepDecay, rank, prevUserFactors, itemFactors, ratings, lambda, i)
      prevUserFactors = curUserFactors
    }
  curUserFactors
  }
}



object MFGradientDescent extends Serializable {

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
                           lambda: Double): Row = {
    val predicted: Double = getRating(userFeatures, prodFeatures, bias, globalBias)
    val epsilon: Double = amount - predicted
    val user: DenseVector[Double] = DenseVector(userFeatures)
    val rank = user.length
    val prod: DenseVector[Double] = DenseVector(prodFeatures)

    val uFeatures = stepSize * (prod * epsilon - lambda * user)
    val userBiasGrad: Double = biasStepSize * (epsilon - lambda * bias)

    Row(userid, performerid, amount, uFeatures.toArray, userBiasGrad, prodFeatures)
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
