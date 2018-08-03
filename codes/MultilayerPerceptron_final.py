#!/usr/bin/python

"""
Multilayer Perceptron in Spark ML library exploration
based on Spark Framework and DataFrame API.

In order to submit this to spark, please use spark_submit.sh
under the same folder.
"""


# import libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import argparse

# main algorithm
if __name__ == "__main__":

    # input arguments needed
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="the input path", default='hdfs://.../')
    parser.add_argument("--size", help="hidden layer size", default='100')
    args = parser.parse_args()
    input_path = args.input
    size = int(args.size)

    # start Spark session
    spark = SparkSession \
        .builder \
        .appName("Multilayer Perceptron with size = " + args.size) \
        .getOrCreate()

    # load the data
    test_df = spark.read.csv(input_path + "Test-label-28x28.csv", \
              header=False, inferSchema="true").withColumnRenamed("_c0", "label")
    train_df = spark.read.csv(input_path + "Train-label-28x28.csv", \
               header=False, inferSchema="true").withColumnRenamed("_c0", "label")

    ##################### Preprocessing #####################
    # assembler
    feature_list = test_df.columns[1:]
    assembler = VectorAssembler(inputCols=feature_list, outputCol="features")

    ##################### Multilayer Perceptron #####################
    # Train a MultilayerPerceptron model.
    layers = [784, size, 10]
    perceptron = MultilayerPerceptronClassifier(maxIter=100, layers=layers, \
                 blockSize=30, seed=1234)

    ##################### Pipelined Model #####################
    pipeline_per = Pipeline(stages=[assembler, perceptron])

    # train the model
    model_per = pipeline_per.fit(train_df)

    ##################### Prediction #####################
    # make predictions
    result_per = model_per.transform(test_df)

    ##################### Evaluation #####################
    # compute accuracy
    evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(result_per)
    print("\n+-------------------+")
    print("| Accuracy = %g%% |" % (100*accuracy))
    print("+-------------------+\n")
