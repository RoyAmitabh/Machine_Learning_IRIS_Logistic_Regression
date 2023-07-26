# Databricks notebook source
from pyspark.ml.feature import  VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
modelPath = "abfss://arc/iris_logisticregression_model"
inputPath = "abfss://arc/archive_data_iris_final.csv"
# Load the saved model from ADLS
model = LogisticRegressionModel.load(modelPath)
# Load new data for predictions
newData = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(inputPath)

# Prepare features by assembling them into a vector
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")
newDataWithFeatures = assembler.transform(newData)
# Apply the model to make predictions on the new data
predictions = model.transform(newDataWithFeatures)
# View the resulting predictions
predictions.show()
