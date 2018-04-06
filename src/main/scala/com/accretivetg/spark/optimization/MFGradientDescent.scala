package com.accretivetg.spark.optimization

import com.accretivetg.spark.recommendation._
import breeze.linalg._
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types._
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable
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
           userFactors: DataFrame,
           itemFactors: DataFrame,
           ratings: DataFrame,
           globalBias: Double,
           verbose: Boolean): DataFrame = {

    val lambda = params.lambda
    val stepSize: Double = params.stepSize
    val stepDecay = params.stepDecay
    val biasStepSize = params.biasStepSize
    val iter = params.iter
    val intermediateStorageLevel = params.intermediateStorageLevel
    val rank = params.rank

    def joiner(userFactors: DataFrame, itemFactors: DataFrame, ratings: DataFrame): DataFrame = {
      var userFactorsBias = if (!userFactors.columns.contains("bias"))
        userFactors.withColumn("bias", org.apache.spark.sql.functions.rand()) else userFactors
      val userFactorsRenamed = userFactorsBias.withColumnRenamed("features", "userFeatures")
      val itemFactorsRenamed = itemFactors.withColumnRenamed("features", "itemFeatures")
      val joinUsers = ratings.join(userFactorsRenamed, ratings.col("userid") === userFactorsRenamed("id")).drop("id").drop("lastSpendDate")
      val joinAll = joinUsers.join(itemFactorsRenamed, joinUsers.col("performerid") === itemFactorsRenamed.col("id")).drop("id")
      joinAll
    }

    def iteration(stepSize: Double,
                  biasStepSize: Double,
                  globalBias: Double,
                  stepDecay: Double,
                  rank: Int,
                  joinAll: DataFrame,
                  lambda: Double,
                  k: Int
                 ): DataFrame = {
      val currentStepSize = stepSize * math.pow(stepDecay, k)
      val currentBiasStepSize = biasStepSize * math.pow(stepDecay, k)
      val gradients = joinAll.withColumn("userFeatures", MFGradientDescent.oneSided(joinAll.col("amount"), joinAll.col("userFeatures"),
        joinAll.col("bias"), joinAll.col("itemFeatures"), lit(globalBias), lit(currentStepSize),
        lit(currentBiasStepSize), lit(lambda))).persist(intermediateStorageLevel)
      val schema = new StructType()
        .add(StructField("id", LongType, true))
        .add(StructField("Features", ArrayType(FloatType, false), true))
      val userVectors = gradients.select("userid", "userFeatures").rdd
        .map{ case Row(k: Long, v: WrappedArray[Float]) => (k, DenseVector(v.toArray)) }
        .foldByKey(DenseVector(Array.fill(params.rank)(0f)))(_ += _)
        .mapValues(v => v.toArray).map({case (a,b) => Row(a,b.toArray) })
      val stepDF: DataFrame = spark.createDataFrame(userVectors, schema)
      stepDF
    }

    val joinAll = joiner(userFactors, itemFactors, ratings)

    var curUserFactors = userFactors
    var prevUserFactors = userFactors
    for (i <- 0 until iter) {
      if (verbose) {
        print("i={}", i)
        print("curUserFactors", curUserFactors.show.toString)
      }
      curUserFactors = iteration(stepSize, biasStepSize, globalBias, stepDecay, rank, joinAll, lambda, i).cache
      prevUserFactors = curUserFactors
    }
  curUserFactors
  }


  def addNew(df: DataFrame, rank: Int): DataFrame = {
    val rand: java.util.Random = new java.util.Random
    val createRandomArray: UserDefinedFunction = udf((rank: Int) => {
      Array.fill(rank)(rand.nextFloat())
    })

    val dfArray = df.withColumn("userFeatures", createRandomArray(lit(rank)))
    val dfArrayBias = dfArray.withColumn("bias", org.apache.spark.sql.functions.rand())
    dfArrayBias
  }
}



object MFGradientDescent extends Serializable {

  val oneSided = udf((amount:Double,
    userFeatures: WrappedArray[Float],
    bias:Double,
    prodFeatures: WrappedArray[Float],
    globalBias: Double,
    stepSize: Double,
    biasStepSize: Double,
    lambda: Double) => {
    val userF = userFeatures.toArray
    val prodF = prodFeatures.toArray
    val predicted: Double = getRating(userF, prodF, bias, globalBias)
    val epsilon: Double = amount - predicted
    val user: DenseVector[Float] = DenseVector(userF)
    val prod: DenseVector[Float] = DenseVector(prodF)

    val uFeatures: DenseVector[Float] = stepSize.toFloat * ((prod * epsilon.toFloat) - (user * lambda.toFloat))
    val scaledFeatures = scaleVector(uFeatures)
    val userBiasGrad: Double = biasStepSize * (epsilon - lambda * bias)

    scaledFeatures.toArray
  })


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





