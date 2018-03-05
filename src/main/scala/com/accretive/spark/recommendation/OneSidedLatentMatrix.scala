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

class OneSidedLatentMatrix(params: LatentMatrixFactorizationParams) {
  protected val optimizer = new MFGradientDescent(params)

  def trainOn(userFactors: DataFrame, itemFactors: DataFrame, ratings: DataFrame,
              globalBias: Float, rank: Int): Some[DataFrame] = {
    val items = Some(optimizer.train(userFactors, itemFactors, ratings, globalBias, rank))
    items
  }

  def predict(x: Row,
               globalBias: Float): Row = {
    val (userid, performerid, amount, userFeatures, bias, prodFeatures): (Long, Long, Float, Array[Float], Float, Array[Float]) =
      (x.getLong(0), x.getLong(1), x.getFloat(2), x.getList(3).toArray.map(_.toString.toFloat),
        x.getFloat(4), x.getList(5).toArray.map(_.toString.toFloat))
    val finalRating =
      if (userFeatures.isDefined && prodFeatures.isDefined) {
        Rating(user, product, LatentMatrixFactorizationModel.getRating(uFeatures.get, pFeatures.get,
          globalBias))
      } else if (uFeatures.isDefined) {
        log.warn(s"Product data missing for product id $product. Will use user factors.")
        val rating = globalBias + uFeatures.get.latent.bias
        Rating(user, product, 0f)
      } else if (pFeatures.isDefined) {
        log.warn(s"User data missing for user id $user. Will use product factors.")
        val rating = globalBias + pFeatures.get.latent.bias
        Rating(user, product, 0f)
      } else {
        log.warn(s"Both user and product factors missing for ($user, $product). " +
          "Returning global average.")
        val rating = globalBias
        Rating(user, product, 0f)
      }
    finalRating
  }
}
