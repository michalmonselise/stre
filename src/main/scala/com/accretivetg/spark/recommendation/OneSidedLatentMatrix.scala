package com.accretivetg.spark.recommendation

import com.accretivetg.spark.optimization._
import breeze.linalg._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Column
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
    val usersDf: org.apache.spark.sql.DataFrame = ratings.select("userid")
      .withColumnRenamed("userid", "id")
      .except(userFactorsRenamed.select("id"))
    var usersFactorsNew: org.apache.spark.sql.DataFrame = makeNew(usersDf, params.getRank)
    //userFactorsBias = userFactorsBias.union(usersFactorsNew)
    val users = Some(optimizer.train(userFactorsRenamed, itemFactors, ratings, globalBias, rank, verbose))
    users
  }

//  def predict(userid: Long,
//              performerid: Long,
//              userFactors: Some[Array[Float]],
//              itemFactors:Some[Array[Float]],
//              ratings: org.apache.spark.sql.DataFrame,
//              bias: Double,
//              globalBias: Double): (Long, Long, Double) = {
//    val finalRating =
//      if (userFactors.isDefined && itemFactors.isDefined) {
//        (userid, performerid, MFGradientDescent.getRating(userFactors.head, itemFactors.head, bias, globalBias))
//      } else if (userFactors.isDefined) {
//        //log.warn(s"Product data missing for product id $performerid. Will use user factors.")
//        val rating = globalBias + bias
//        (userid, performerid, 0.0)
//      } else if (itemFactors.isDefined) {
//        //log.warn(s"User data missing for user id $userid. Will use product factors.")
//        val rating = globalBias + bias
//        (userid, performerid, 0.0)
//      } else {
//        //log.warn(s"Both user and product factors missing for ($userid, $performerid). " +
//         // "Returning global average.")
//        val rating = globalBias
//        (userid, performerid, 0.0)
//      }
//    finalRating
//  }


  def makeNew(df: org.apache.spark.sql.DataFrame, rank: Int): org.apache.spark.sql.DataFrame = {
    val rand: java.util.Random = new java.util.Random
    val createRandomArray: org.apache.spark.sql.expressions.UserDefinedFunction = udf((rank: Int) => {
      Array.fill(rank)(rand.nextFloat())
    })

    val dfArray = df.withColumn("userFeatures", createRandomArray(lit(rank)))
    val dfArrayBias = dfArray.withColumn("bias", org.apache.spark.sql.functions.rand())
    dfArrayBias
  }
}

