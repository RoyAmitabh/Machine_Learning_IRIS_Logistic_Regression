# Databricks notebook source
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

# Load the Iris dataset from ADLS
data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("abfss://arc/archive_data_iris_final.csv")
data.show()
data.printSchema()


# Prepare features by assembling them into a vector
assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
assembledData = assembler.transform(data)

# Convert the target class labels into numeric indices
indexer = StringIndexer(inputCol="class", outputCol="labelIndex")
indexedData = indexer.fit(assembledData).transform(assembledData)

# Split the data into training and testing sets
(trainingData, testData) = indexedData.randomSplit([0.8, 0.2])

# Create a Logistic Regression classifier
lr = LogisticRegression(labelCol="labelIndex", featuresCol="features")

# Train the model
model = lr.fit(trainingData)

# Make predictions on the test set
predictions = model.transform(testData)
predictions.show()
# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))

# COMMAND ----------

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F

preds_and_labels = predictions.select(['prediction','labelIndex']).withColumn('labelIndex', F.col('labelIndex').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','labelIndex'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())

# COMMAND ----------

# Persist the trained model to ADLS
model.save("abfss://arc/iris_logisticregression_model")
