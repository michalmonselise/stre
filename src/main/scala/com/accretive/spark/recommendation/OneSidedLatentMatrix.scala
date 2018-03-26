package com.accretive.spark.recommendation

import java.util.Random

import breeze.linalg
import com.accretive.spark.optimization._
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.recommendation.ALSModel
import breeze.linalg._
import com.accretive.spark.recommendation.LatentMatrixFactorizationModel.log
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Column
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.storage.StorageLevel
import java.util.Random._

class OneSidedLatentMatrix(params: LatentMatrixFactorizationParams) {
  protected val optimizer = new MFGradientDescent(params)

  def trainOn(userFactors: org.apache.spark.sql.DataFrame,
              itemFactors: org.apache.spark.sql.DataFrame,
              ratings: org.apache.spark.sql.DataFrame,
              globalBias: Double, rank: Int, verbose: Boolean=false): Some[org.apache.spark.sql.DataFrame] = {
    val userFactorsRenamed = userFactors.withColumnRenamed("features", "userFeatures")
    val usersDf: org.apache.spark.sql.DataFrame = ratings.select("userid").withColumnRenamed("userid", "id").except(userFactorsRenamed.select("id"))
    var usersFactorsNew: org.apache.spark.sql.DataFrame = makeNew(usersDf, params.getRank)
    //userFactorsBias = userFactorsBias.union(usersFactorsNew)
    val users = Some(optimizer.train(userFactorsRenamed, itemFactors, ratings, globalBias, rank, verbose))
    users
  }

  def predict(userid: Long,
              performerid: Long,
              userFactors: Some[Array[Float]],
              itemFactors:Some[Array[Float]],
              ratings: org.apache.spark.sql.DataFrame,
              bias: Double,
              globalBias: Double): (Long, Long, Double) = {
    val finalRating =
      if (userFactors.isDefined && itemFactors.isDefined) {
        (userid, performerid, MFGradientDescent.getRating(userFactors.head, itemFactors.head, bias, globalBias))
      } else if (userFactors.isDefined) {
        //log.warn(s"Product data missing for product id $performerid. Will use user factors.")
        val rating = globalBias + bias
        (userid, performerid, 0.0)
      } else if (itemFactors.isDefined) {
        //log.warn(s"User data missing for user id $userid. Will use product factors.")
        val rating = globalBias + bias
        (userid, performerid, 0.0)
      } else {
        //log.warn(s"Both user and product factors missing for ($userid, $performerid). " +
         // "Returning global average.")
        val rating = globalBias
        (userid, performerid, 0.0)
      }
    finalRating
  }


  def makeNew(df: org.apache.spark.sql.DataFrame, rank: Int): org.apache.spark.sql.DataFrame = {
    val rand: java.util.Random = new java.util.Random
    val createRandomArray: org.apache.spark.sql.expressions.UserDefinedFunction = udf((rank: Int) => {
      Array.fill(rank)(rand.nextFloat())
    })

    val dfArray = df.withColumn("userFeatures", createRandomArray(lit(rank)))
    val dfArrayBias = dfArray.withColumn("bias", org.apache.spark.sql.functions.rand())
//    var df_dummy = df
//    var i: Int = 0
//    var inputCols: Array[String] = Array()
//    for (i <- 0 to rank) {
//      df_dummy = df_dummy.withColumn("feature".concat(i.toString), org.apache.spark.sql.functions.rand())
//      inputCols = inputCols :+ "feature".concat(i.toString)
//    }
//    val assembler = new org.apache.spark.ml.feature.VectorAssembler()
//      .setInputCols(inputCols)
//      .setOutputCol("userFeatures")
//    val output = assembler.transform(df_dummy)
//    output.withColumn("bias", org.apache.spark.sql.functions.rand()).select("id", "userFeatures", "bias")
    dfArrayBias
  }
}

