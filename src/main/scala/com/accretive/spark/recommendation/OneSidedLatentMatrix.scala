package com.accretive.spark.recommendation

import breeze.linalg
import com.accretive.spark.optimization._
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.recommendation.ALSModel
import breeze.linalg._
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

class OneSidedLatentMatrix(params: LatentMatrixFactorizationParams) {
  protected val optimizer = new MFGradientDescent(params)

  def trainOn(userFactors: DataFrame, itemFactors: DataFrame, ratings: RDD[Rating[Long]],
              globalBias: Float, rank: Int): Some[DataFrame] = {
    val items = Some(optimizer.train(userFactors, itemFactors, ratings, globalBias, rank))
    items
  }

}
