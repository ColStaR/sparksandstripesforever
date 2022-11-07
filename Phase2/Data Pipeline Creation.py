# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Data Pipeline Creation
# MAGIC 
# MAGIC Useful links? 
# MAGIC 
# MAGIC https://towardsdatascience.com/building-an-ml-application-with-mllib-in-pyspark-part-1-ac13f01606e2
# MAGIC 
# MAGIC https://docs.databricks.com/machine-learning/train-model/mllib/index.html
# MAGIC 
# MAGIC https://docs.databricks.com/_static/notebooks/binary-classification.html

# COMMAND ----------

from pyspark.sql.functions import col, floor
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import IntegerType

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics

import pandas as pd

# COMMAND ----------

# Azure Storage Container info
from pyspark.sql.functions import col, max

blob_container = "sasfcontainer" # The name of your container created in https://portal.azure.com
storage_account = "sasfstorage" # The name of your Storage account created in https://portal.azure.com
secret_scope = "sasfscope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "sasfkey" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# SAS Token login
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# Load Dataframes
print("**Loading Data")

# Inspect the Joined Data folders 
display(dbutils.fs.ls(f"{blob_url}"))

print("**Data Loaded")
print("**Loading Data Frames")

df_joined_data_3m = spark.read.parquet(f"{blob_url}/joined_data_3m")
display(df_joined_data_3m)

df_joined_data_all = spark.read.parquet(f"{blob_url}/joined_data_all")
display(df_joined_data_all)

print("**Data Frames Loaded")

# COMMAND ----------

# dataframe schema
print(df_joined_data_3m.count())
df_joined_data_3m.printSchema()

# COMMAND ----------

# Convert categorical features to One Hot Encoding
categoricalColumns = ["ORIGIN", "OP_UNIQUE_CARRIER", "TAIL_NUM", "ORIGIN_STATE_ABR", "DEST_STATE_ABR"]
# Features not included: DEP_DATETIME_LAG, FL_DATE, CRS_DEP_TIME, CANCELLATION_CODE, DEP_HOUR, DEP_DATETIME

stages = [] # stages in Pipeline

# NOTE: Had to cut out a bunch of features due to the sheer number of NULLS in them, which were causing the entire dataframe to be skipped. Will need to get the Null values either filled or dropped.

#for categoricalCol in categoricalColumns:
#    # Category Indexing with StringIndexer
#    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index").setHandleInvalid("skip")
#    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
#    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
#        
#    # Add stages.  These are not run here, but will run all at once later on.
#    stages += [stringIndexer, encoder]
#    
#print(stages)

#CRS_DEP_TIME_stringIdx = StringIndexer(inputCol="CRS_DEP_TIME", outputCol="CRS_DEP_TIME_INDEX").setHandleInvalid("skip")
#stages += [CRS_DEP_TIME_stringIdx]
#DEP_HOUR_stringIdx = StringIndexer(inputCol="DEP_HOUR", outputCol="DEP_HOUR_INDEX").setHandleInvalid("skip")
#stages += [DEP_HOUR_stringIdx]

print(stages)

# COMMAND ----------

# Create vectors for numeric and categorical variables
#numericCols = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_WAC", "DEP_TIME", "CANCELLED", "CRS_ELAPSED_TIME", "DISTANCE", "YEAR", "STATION", "DATE", "ELEVATION", "SOURCE", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlyVisibility", "HourlyWindSpeed", "DATE_HOUR", "distance_to_neighbor", "neighbor_call"]

# NOTE: Had to cut out a bunch of features due to the sheer number of NULLS in them, which were causing the entire dataframe to be skipped. Will need to get the Null values either filled or dropped.

#Works:
numericCols = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_WAC", "DEP_TIME", "CANCELLED", "CRS_ELAPSED_TIME", "DISTANCE", "YEAR", "STATION", "ELEVATION", "SOURCE", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlyVisibility", "HourlyWindSpeed", "distance_to_neighbor"]

#assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assemblerInputs = numericCols

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("skip")

stages += [assembler]

print(stages)

# COMMAND ----------

# Run the pipeline
partialPipeline = Pipeline().setStages(stages)

pipelineModel = partialPipeline.fit(df_joined_data_3m)

preppedDataDF = pipelineModel.transform(df_joined_data_3m)

# COMMAND ----------

#totalFeatures = [*categoricalColumns, *numericCols]
#print(categoricalColumns, "\n")
#print(numericCols, "\n")
#print(totalFeatures, "\n")

# Fit model to prepped data
lrModel = LogisticRegression(featuresCol = "features", labelCol = "DEP_DEL15").fit(preppedDataDF)

# ROC for training data
display(lrModel, preppedDataDF, "ROC")

# COMMAND ----------

display(lrModel, preppedDataDF)

# COMMAND ----------

selectedcols = ["DEP_DEL15", "features"]

dataset = preppedDataDF.select(selectedcols)

display(dataset)

# COMMAND ----------

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)

print(trainingData.count())

print(testData.count())

# COMMAND ----------

# Logistic Regression
# Create initial LogisticRegression model

lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", maxIter=10)

 

# Train model with Training Data

lrModel = lr.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the transform() method.

# LogisticRegression.transform() will only use the 'features' column.

predictions = lrModel.transform(testData)

# COMMAND ----------

# View model's predictions and probabilities of each prediction class

# You can select any columns in the above schema to view as well

selected = predictions.select("DEP_DEL15", "prediction", "probability")

display(selected)

# COMMAND ----------

# Evaluate model

evaluator = BinaryClassificationEvaluator(labelCol = "DEP_DEL15")

evaluator.evaluate(predictions)

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

print(lr.explainParams())

# COMMAND ----------

# Create ParamGrid for Cross Validation

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator

cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations

cvModel = cv.fit(trainingData)

# this will likely take a fair amount of time because of the amount of models that we're creating and testing

# COMMAND ----------

# Use the test set to measure the accuracy of the model on new data

predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation

# Evaluate best model

evaluator.evaluate(predictions)

# COMMAND ----------

print('Model Intercept: ', cvModel.bestModel.intercept)

# COMMAND ----------

weights = cvModel.bestModel.coefficients

weights = [(float(w),) for w in weights]  # convert numpy type to float, and to tuple

weightsDF = spark.createDataFrame(weights, ["Feature Weight"])

display(weightsDF)

# COMMAND ----------

# View best model's predictions and probabilities of each prediction class

#predictions = predictions.withColumn("label", predictions["DEP_DEL15"])

#selected = predictions.select("label", "prediction", "probability")
selected = predictions.select("DEP_DEL15", "prediction", "probability")

display(selected)

# COMMAND ----------

# Metric Evaluation
metrics = MulticlassMetrics(predictions.select("DEP_DEL15", "prediction").rdd)
#display(predictions.select("label", "prediction").rdd.collect())

TP = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 1) & (col("prediction") == 1)).count()
#print(TP)
TN = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 0) & (col("prediction") == 0)).count()
#print(TN)
FP = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 1) & (col("prediction") == 0)).count()
#print(FP)
FN = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 0) & (col("prediction") == 1)).count()
#print(FN)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TN + TP) / (TN + TP + FP + FN)

print("Manual Precision: ", precision)
print("Manual Recall: ", recall)
print("Manual Accuracy: ", accuracy)

# Computed using MulticlassMetrics
print("Summary Stats")
print("Precision = %s" % metrics.precision(1))
print("Recall = %s" % metrics.recall(1))
print("Accuracy = %s" % metrics.accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Workspace

# COMMAND ----------

display(df_joined_data_3m)

# COMMAND ----------

display(preppedDataDF)

# COMMAND ----------

numericCols = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_WAC", "DEP_TIME", "CANCELLED", "CRS_ELAPSED_TIME", "DISTANCE", "YEAR", "STATION", "DATE", "ELEVATION", "SOURCE", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlyVisibility", "HourlyWindSpeed", "DATE_HOUR", "distance_to_neighbor", "neighbor_call"]

display(preppedDataDF.select(numericCols))

# COMMAND ----------

df2 = df_joined_data_3m.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in numericCols])
display(df2)

# COMMAND ----------


