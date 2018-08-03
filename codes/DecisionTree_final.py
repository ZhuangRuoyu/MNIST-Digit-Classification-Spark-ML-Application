#!/usr/bin/python

"""
Decision Tree in Spark ML library exploration
based on Spark Framework and DataFrame API.

In order to submit this to spark, please use spark_submit.sh
under the same folder.
"""

# import libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import argparse


# main algorithm
if __name__ == "__main__":

    # input arguments needed
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="the input path", default='hdfs://.../')
    parser.add_argument("--d", help="reduced dimension with PCA", default='100')
    args = parser.parse_args()
    input_path = args.input
    d = int(args.d)

    # start Spark session
    spark = SparkSession \
        .builder \
        .appName("Decision Tree with d = " + args.d) \
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

    ##################### Decision Tree #####################
    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(maxDepth=12, minInstancesPerNode = 5, \
                                labelCol="label", featuresCol="features", \
                                seed=1234)

    ##################### Pipelined Model #####################
    pipeline_dt = Pipeline(stages=[assembler, pca, dt])


    # build pipelined model with train data
    model_dt = pipeline_dt.fit(train_df)

    ##################### Prediction #####################
    # make predictions
    result_dt = model_dt.transform(test_df)

    ##################### Evaluation #####################
    # compute accuracy
    evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(result_dt)
    print("\n+-------------------+")
    print("| Accuracy = %g%% |" % (100*accuracy))
    print("+-------------------+\n")
