package com.accretivetg.spark.recommendation

/**
 * Parameters for training a Matrix Factorization Model
 */
case class LatentMatrixFactorizationParams() extends Serializable {
  var rank: Int = 20
  var stepSize: Double = 1.0
  var biasStepSize: Double = 1.0
  var stepDecay: Double = 0.9
  var lambda: Double = 10.0
  var iter: Int = 10
  var intermediateStorageLevel = org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER
  var seed: Long = System.currentTimeMillis()

}