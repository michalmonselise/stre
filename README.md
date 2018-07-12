Streaming Matrix Factorization for Spark
----------------------------------------

![Scheme](images/27qtcx.jpg)

This library contains methods to train a Matrix Factorization Recommendation System on Spark.
For user `u` and item `i`, the rating is calculated as:

`r = U(u) * P^T^(i) + bu(u) + mu`,

where `r` is the rating, `U` is the User Matrix, `P^T^` is the transpose of the product matrix,
`U(u)` corresponds to the `u`th row of `U`, `bu(u)` is the bias of the `u`th user and `mu` is the average global rating.

Gradient Descent is used to train the model.

The methodology in this library is described in Incremental Learning for Matrix Factorization in Recommender Systems by Mengshoel et al.


Installation
============

Include this package in your Spark Applications using:

### spark-shell or spark-submit

```
> $SPARK_HOME/bin/spark-shell --jars streamingMF-assembly-0.1.0.jar
```

### sbt

If you use the sbt-spark-package plugin, in your sbt build file, add:

```
spDependencies += "streamingMF:0.1.0"
```


```


Usage
=====

To train a streaming model, use the `StreamingLatentMatrixFactorization` class.
The following usage will train a Model that would predict ratings between 1.0, and 5.0 with rank 20:

```scala
import com.accretivetg.spark.optimization._
import com.accretivetg.spark.recommendation._
import org.apache.spark.streaming.dstream.ReceiverInputDStream
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.pubsub.SparkPubsubMessage
import org.apache.spark.sql.Row

val schema = new org.apache.spark.sql.types.StructType()
  .add(org.apache.spark.sql.types.StructField("user",
    org.apache.spark.sql.types.LongType, true))
  .add(org.apache.spark.sql.types.StructField("product",
    org.apache.spark.sql.types.LongType, true))
  .add(org.apache.spark.sql.types.StructField("amount",
    org.apache.spark.sql.types.DoubleType, true))

val one = new MFGradientDescent(params)

val stream1 = pubsubStream.map(message => new String(message.getData())).map(x => x.split(" "))
val stream2 = stream1.map(x =>
  Row(x.head.toLong, x.tail.head.toLong, x.tail.tail.head.toDouble))
stream2.print()
val stream3 = stream2.foreachRDD { x =>
    var m1 = one.train(prevUsers, itemFactors, currData, 0.001, params)
    }
m1.show
```

