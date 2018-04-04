package com.accretivetg.spark.optimization

import com.accretivetg.spark.recommendation._
import breeze.linalg._
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable._

/**
 * A Gradient Descent Optimizer specialized for Matrix Factorization.
 *
 * @param params The parameters to use
 */
class MFGradientDescent(params: LatentMatrixFactorizationParams) {

  val log: Logger = LoggerFactory.getLogger(this.getClass)
  def this() = this(new LatentMatrixFactorizationParams)
  val spark = org.apache.spark.sql.SparkSession.builder().getOrCreate()
  def train(
           userFactors: org.apache.spark.sql.DataFrame,
           itemFactors: org.apache.spark.sql.DataFrame,
           ratings: org.apache.spark.sql.DataFrame,
           globalBias: Double,
           rank: Int,
           verbose: Boolean): org.apache.spark.sql.DataFrame = {

    val lambda = params.getLambda
    val stepSize: Double = params.getStepSize
    val stepDecay = params.getStepDecay
    val biasStepSize = params.getBiasStepSize
    val iter = params.getIter
    val intermediateStorageLevel = params.getIntermediateStorageLevel
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
                  k: Int
                 ): org.apache.spark.sql.DataFrame = {
      val currentStepSize = stepSize * math.pow(stepDecay, k)
      val currentBiasStepSize = biasStepSize * math.pow(stepDecay, k)
      var userFactorsBias = if (!userFactors.columns.contains("bias"))
        userFactors.withColumn("bias", org.apache.spark.sql.functions.rand()) else userFactors
      val userFactorsRenamed = userFactorsBias.withColumnRenamed("features", "userFeatures")
      val itemFactorsRenamed = itemFactors.withColumnRenamed("features", "itemFeatures")
      val gradients = ratings.join(userFactorsRenamed, ratings.col("userid") === userFactorsRenamed("id")).drop("id").drop("lastSpendDate")
      //print("gradient columns={}", gradients.columns.mkString)
      val grad = gradients.join(itemFactorsRenamed, gradients.col("performerid") === itemFactorsRenamed.col("id")).drop("id")
      //print("grad columns={}", grad.columns.mkString)
      val step = grad.rdd.map(x => (x.getLong(0), x.getLong(1), x.getDouble(2),
        x.getAs[scala.collection.mutable.WrappedArray[Float]](3), x.getDouble(4), x.getAs[scala.collection.mutable.WrappedArray[Float]](5)))
      val schema = new org.apache.spark.sql.types.StructType()
        .add(org.apache.spark.sql.types.StructField("id",
          org.apache.spark.sql.types.LongType, true))
        .add(org.apache.spark.sql.types.StructField("Features",
          org.apache.spark.sql.types.ArrayType(org.apache.spark.sql.types.FloatType, false), true))
      val step2: org.apache.spark.rdd.RDD[(Long, Array[Float])] = step.map(x =>
        MFGradientDescent.oneSidedGradientStep(x._1, x._2, x._3, x._4, x._5, x._6,
          globalBias, currentStepSize, currentBiasStepSize, lambda)).persist(intermediateStorageLevel)
      val userVectors = step2
        .map{ case (k: Long, v: Array[Float]) => (k, DenseVector(v)) }
        .foldByKey(DenseVector(Array.fill(params.getRank)(0f)))(_ += _)
        .mapValues(v => v.toArray).map({case (a,b) => Row(a,b.toArray) })
      val stepDF: org.apache.spark.sql.DataFrame = spark.createDataFrame(userVectors, schema)
      stepDF
    }

    var curUserFactors = userFactors
    var prevUserFactors = userFactors
    for (i <- 0 until iter) {
      if (verbose) {
        print("i={}", i)
        print("curUserFactors", curUserFactors.show.toString)
      }
      curUserFactors = iteration(stepSize, biasStepSize, globalBias, stepDecay, rank, prevUserFactors, itemFactors, ratings, lambda, i).cache
      prevUserFactors = curUserFactors
    }
  curUserFactors
  }
}



object MFGradientDescent extends Serializable {

  def oneSidedGradientStep(userid: Long,
                           performerid: Long,
                           amount: Double,
                           userFeatures: WrappedArray[Float],
                           bias:Double,
                           prodFeatures: WrappedArray[Float],
                           globalBias: Double,
                           stepSize: Double,
                           biasStepSize: Double,
                           lambda: Double): (Long, Array[Float]) = {
    val userF = userFeatures.toArray
    val prodF = prodFeatures.toArray
    val predicted: Double = getRating(userF, prodF, bias, globalBias)
    val epsilon: Double = amount - predicted
    val user: DenseVector[Float] = DenseVector(userF)
    val prod: DenseVector[Float] = DenseVector(prodF)

    val uFeatures: DenseVector[Float] = stepSize.toFloat * ((prod * epsilon.toFloat) - (user * lambda.toFloat))
    val scaledFeatures = scaleVector(uFeatures)
    val userBiasGrad: Double = biasStepSize * (epsilon - lambda * bias)

    (userid, scaledFeatures.toArray)
  }


  def getRating(userFeatures: Array[Float],
                prodFeatures: Array[Float],
                bias: Double,
                globalBias: Double): Double = {
    val uFeatures: DenseVector[Float] = DenseVector(userFeatures)
    val pFeatures: DenseVector[Float] = DenseVector(prodFeatures)
    val dotProd: Float = uFeatures dot pFeatures
    dotProd + bias + globalBias
  }

  def scaleVector(vec: DenseVector[Float]): DenseVector[Float] = {
    val magnitude = math.sqrt(vec dot vec).toFloat
    if (magnitude > 0) {vec / magnitude} else vec
  }
}




