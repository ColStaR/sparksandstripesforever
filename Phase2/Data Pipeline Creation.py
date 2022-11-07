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

df_joined_data_3m = spark.read.parquet(f"{blob_url}/joined_data_all")
display(df_joined_data_3m)

print("**Data Frames Loaded")

# COMMAND ----------

# dataframe schema
df_joined_data_3m.printSchema()

# COMMAND ----------

# Convert categorical features to One Hot Encoding
categoricalColumns = ["ORIGIN", "OP_UNIQUE_CARRIER", "TAIL_NUM", "ORIGIN_STATE_ABR", "DEST_STATE_ABR"]
# Features not included: DEP_DATETIME_LAG, FL_DATE, CRS_DEP_TIME, CANCELLATION_CODE, DEP_HOUR, DEP_DATETIME

stages = [] # stages in Pipeline

for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
    
print(stages)

CRS_DEP_TIME_stringIdx = StringIndexer(inputCol="CRS_DEP_TIME", outputCol="CRS_DEP_TIME_INDEX")
stages += [CRS_DEP_TIME_stringIdx]
DEP_HOUR_stringIdx = StringIndexer(inputCol="DEP_HOUR", outputCol="DEP_HOUR_INDEX")
stages += [DEP_HOUR_stringIdx]

print(stages)

# COMMAND ----------

# Create vectors for numeric and categorical variables
numericCols = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_WAC", "DEP_TIME", "CANCELLED", "CRS_ELAPSED_TIME", "DISTANCE", "YEAR", "STATION", "DATE", "ELEVATION", "SOURCE", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlyVisibility", "HourlyWindSpeed", "DATE_HOUR", "distance_to_neighbor", "neighbor_call"]

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

stages += [assembler]

print(stages)

# COMMAND ----------

# Run the pipeline
partialPipeline = Pipeline().setStages(stages)

pipelineModel = partialPipeline.fit(df_joined_data_3m)

preppedDataDF = pipelineModel.transform(df_joined_data_3m)

# COMMAND ----------

# Fit model to prepped data
lrModel = LogisticRegression().fit(preppedDataDF)

# ROC for training data
display(lrModel, preppedDataDF, "ROC")

# COMMAND ----------

display(lrModel, preppedDataDF)

# COMMAND ----------

selectedcols = ["label", "features"] + cols

dataset = preppedDataDF.select(selectedcols)

display(dataset)

# COMMAND ----------

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)

print(trainingData.count())

print(testData.count())

# COMMAND ----------

# Logistic Regression
# Create initial LogisticRegression model

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

 

# Train model with Training Data

lrModel = lr.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the transform() method.

# LogisticRegression.transform() will only use the 'features' column.

predictions = lrModel.transform(testData)

# COMMAND ----------

# View model's predictions and probabilities of each prediction class

# You can select any columns in the above schema to view as well

selected = predictions.select("label", "prediction", "probability", "age", "occupation")

display(selected)

# COMMAND ----------



 

# Evaluate model

evaluator = BinaryClassificationEvaluator()

evaluator.evaluate(predictions)

# COMMAND ----------

display(df_joined_data_3m.select("HourlyWindSpeed", "DEP_DEL15").filter(col("HourlyWindSpeed") > 50))

# COMMAND ----------


