package com.accretive.spark.recommendation

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
import org.apache.spark.sql.functions.udf

class OneSidedLatentMatrix(params: LatentMatrixFactorizationParams) {
  protected val optimizer = new MFGradientDescent(params)

  def trainOn(userFactors: DataFrame, itemFactors: DataFrame, ratings: DataFrame,
              globalBias: Double, rank: Int): Some[DataFrame] = {
    var userFactorsBias: DataFrame = if (!userFactors.columns.contains("bias"))
      userFactors.withColumn("bias", rand()) else userFactors
    userFactorsBias = userFactorsBias.withColumnRenamed("features", "userFeatures")
    val usersDf: DataFrame = ratings.select("userid").withColumnRenamed("userid", "id").except(userFactorsBias.select("id"))
    var usersFactorsNew: DataFrame = makeNew(usersDf, params.getRank)
    val toArr: org.apache.spark.ml.linalg.Vector => Array[Double] = _.toArray
    val toArrUdf =udf(toArr)
    val toDoub: Array[Float] => Array[Double] = _.map(x => x.toDouble)
    val toDoubUdf = udf(toDoub)
    userFactorsBias = userFactorsBias.withColumn("userFeatures", toDoubUdf(userFactorsBias.col("userFeatures")))
    usersFactorsNew = usersFactorsNew.withColumn("userFeatures", toArrUdf(usersFactorsNew.col("userFeatures")))
    userFactorsBias = userFactorsBias.union(usersFactorsNew)
    val users = Some(optimizer.train(userFactorsBias, itemFactors, ratings, globalBias, rank))
    users
  }

  def predict(userid: Long,
              performerid: Long,
              userFactors: Some[Array[Double]],
              itemFactors:Some[Array[Double]],
              ratings: DataFrame,
              bias: Double,
              globalBias: Double): (Long, Long, Double) = {
    val finalRating =
      if (userFactors.isDefined && itemFactors.isDefined) {
        (userid, performerid, MFGradientDescent.getRating(userFactors.head, itemFactors.head, bias, globalBias))
      } else if (userFactors.isDefined) {
        log.warn(s"Product data missing for product id $performerid. Will use user factors.")
        val rating = globalBias + bias
        (userid, performerid, 0.0)
      } else if (itemFactors.isDefined) {
        log.warn(s"User data missing for user id $userid. Will use product factors.")
        val rating = globalBias + bias
        (userid, performerid, 0.0)
      } else {
        log.warn(s"Both user and product factors missing for ($userid, $performerid). " +
          "Returning global average.")
        val rating = globalBias
        (userid, performerid, 0.0)
      }
    finalRating
  }
  def makeNew(df: DataFrame, rank: Int): DataFrame = {
    var df_dummy = df
    var i: Int = 0
    var inputCols: Array[String] = Array()
    for (i <- 0 to rank) {
      df_dummy = df_dummy.withColumn("feature".concat(i.toString), rand())
      inputCols = inputCols :+ "feature".concat(i.toString)
    }
    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("userFeatures")
    val output = assembler.transform(df_dummy)
    output.withColumn("bias", rand()).select("id", "userFeatures", "bias")
  }
}

