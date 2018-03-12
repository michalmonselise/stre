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

class OneSidedLatentMatrix(params: LatentMatrixFactorizationParams) {
  protected val optimizer = new MFGradientDescent(params)

  def trainOn(userFactors: DataFrame, itemFactors: DataFrame, ratings: DataFrame,
              globalBias: Double, rank: Int): Some[DataFrame] = {
    val userFactorsBias: DataFrame = if (!userFactors.columns.contains("bias"))
      userFactors.withColumn("bias", rand()) else userFactors
    val usersDf: DataFrame = ratings.select("userid").except(userFactorsBias.select("id"))
    val usersFactorsNew: DataFrame = makeNew(userFactorsBias, params.getRank)
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
    df.withColumn("features", ).withColumn("bias", rand())
  }
}

