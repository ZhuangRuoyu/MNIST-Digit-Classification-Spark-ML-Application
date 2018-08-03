#!/usr/bin/python

"""
k Nearest Neighbours Classifier Implementation
based on Spark Framework and DataFrame API.

In order to submit this to spark, please use spark_submit.sh
under the same folder.
"""

# import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, array, round
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, PCA
import argparse
import numpy as np
from scipy.stats import mode

# functions to be used

def prediction(test_X):
    """
    Make predictions.

    Input: test_X - Spark.sql DataFrame with a column named "features"
    Output: predicted class (integer from 0 to 9)
    """
    dist_array = np.linalg.norm(train_data_bc.value - test_X, axis=1)
    dist_idx = np.argpartition(dist_array, k, axis=0)[:k]

    klass = np.take(train_label_bc.value, dist_idx)
    result = mode(klass, axis=0)[0]
    return int(result)


# the main method begins here:

if __name__ == "__main__":

    # input arguments needed
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="the input path", default='hdfs://.../')
    parser.add_argument("--d", help="reduced dimension with PCA", default='100')
    parser.add_argument("--k", help="nearest neighbours number", default='10')
    args = parser.parse_args()
    input_path = args.input
    d = int(args.d)
    k = int(args.k)

    # start Spark session
    spark = SparkSession \
        .builder \
        .appName("k Nearest Neighbours Implementation with d = " \
                + args.d + " k = " + args.k) \
        .getOrCreate()

    # load the data
    test_df = spark.read.csv(input_path+"Test-label-28x28.csv", header=False, \
              inferSchema="true").withColumnRenamed("_c0", "label")
    train_df = spark.read.csv(input_path+"Train-label-28x28.csv",header=False, \
               inferSchema="true").withColumnRenamed("_c0", "label")

    ##################### Preprocessing #####################
    # assembler
    feature_list = test_df.columns[1:]
    assembler = VectorAssembler(inputCols=feature_list, \
                                outputCol="features_assembled")
    # PCA
    pca = PCA(k=d, inputCol="features_assembled", outputCol="features")
    # pipeline
    pipeline_preprocessing = Pipeline(stages=[assembler, pca])
    # build pipelined model with train data
    model_preprocessing = pipeline_preprocessing.fit(train_df)

    # transform train and test data with the model

    train_preprocessed = model_preprocessing.transform(train_df) \
                        .select(["label", "features"])
    # repartition test data to ensure certain parallelization
    test_preprocessed = model_preprocessing.transform(test_df) \
                        .select(["label", "features"]).repartition(20)

    # broadcast train data for all executors
    broadcast_data = train_preprocessed.collect()
    train_data_bc = spark.sparkContext \
                    .broadcast(np.array([i[1] for i in broadcast_data]) \
                    .reshape((60000, d)))
    train_label_bc = spark.sparkContext \
                    .broadcast(np.array([i[0] for i in broadcast_data]))

    ##################### Prediction #####################
    # predict on test data and persist the prediction for evaluation
    prediction_udf = udf(prediction, IntegerType())
    result_knn = test_preprocessed.withColumn("prediction", \
                 prediction_udf("features")).select("prediction", "label")

    # justify whether the results are correct
    bool_tran = udf(lambda x: 1 if x[0] == x[1] else 0, IntegerType())
    results = result_knn.withColumn("correctness", \
              bool_tran(array('label', 'prediction'))).collect()
    pred_prepared = spark.createDataFrame(results)

    ##################### Evaluation #####################
    # group for recall and precision separately
    grouped_t = pred_prepared.groupBy("label") \
                .agg({'correctness': 'sum', '*': 'count'}) \
                .withColumnRenamed("sum(correctness)", "tp") \
                .withColumnRenamed("count(1)", "t")

    grouped_p = pred_prepared.groupBy("prediction").count() \
                .withColumnRenamed("count", "p")

    joined_results = grouped_t.join(grouped_p, \
                     grouped_t.label == grouped_p.prediction, 'inner') \
                     .persist()

    evaluation_stats = joined_results \
                      .withColumn("recall/%", \
                      round(100*joined_results.tp/joined_results.t, 1)) \
                      .withColumn("precision/%", \
                      round(100*joined_results.tp/joined_results.p, 1)) \
                      .withColumn("f1_score/%", \
                      round(200 * joined_results.tp/joined_results.t * \
                       joined_results.tp/joined_results.p / \
                       (joined_results.tp/joined_results.t + \
                       joined_results.tp/joined_results.p), 1)) \
                      .select("label","recall/%","precision/%","f1_score/%") \
                      .sort("label")

    # show the results
    evaluation_stats.show()
