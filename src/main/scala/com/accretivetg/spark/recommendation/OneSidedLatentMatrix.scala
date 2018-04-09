package com.accretivetg.spark.recommendation

import com.accretivetg.spark.optimization._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.storage.StorageLevel
import java.util.Random._

class OneSidedLatentMatrix(params: LatentMatrixFactorizationParams) {
  protected val optimizer = new MFGradientDescent(params)

  def trainOn(userFactors: DataFrame,
              itemFactors: DataFrame,
              ratings: DataFrame,
              globalBias: Double, verbose: Boolean=false): DataFrame = {
    val userFactorsRenamed = userFactors.withColumnRenamed("features", "userFeatures")
    val usersDf: DataFrame = ratings.select("userid")
      .withColumnRenamed("userid", "id")
      .except(userFactorsRenamed.select("id"))
    var usersFactorsNew: DataFrame = makeNew(usersDf, params.rank)
    //userFactorsBias = userFactorsBias.union(usersFactorsNew)
    val users = optimizer.train(userFactorsRenamed, itemFactors, ratings, globalBias, verbose)
    users
  }

  def makeNew(df: DataFrame, rank: Int): DataFrame = {
    val rand: java.util.Random = new java.util.Random
    val createRandomArray: org.apache.spark.sql.expressions.UserDefinedFunction = udf((rank: Int) => {
      Array.fill(rank)(rand.nextFloat())
    })

    val dfArray = df.withColumn("userFeatures", createRandomArray(lit(rank)))
    val dfArrayBias = dfArray.withColumn("bias", org.apache.spark.sql.functions.rand())
    dfArrayBias
  }
}

