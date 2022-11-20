# Databricks notebook source
# MAGIC %md 
# MAGIC ### Notebook setup
# MAGIC 1. Import libraries
# MAGIC 2. Connect to Azure
# MAGIC 3. Connect data

# COMMAND ----------

# Import libraries
from pyspark.sql.functions import col, floor, countDistinct
from pyspark.sql.functions import isnan, when, count, col
import pyspark.sql.functions as F
from pyspark.sql.functions import mean
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
from pyspark.sql import SQLContext

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

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
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"

print("**Data Loaded")
print("**Loading Data Frames")

# full airflights dataset
df_flights = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")

# joined 3 months
df_3m = spark.read.parquet(f"{blob_url}/joined_data_3m")
# display(df_3m)

# joined full set
df_all = spark.read.parquet(f"{blob_url}/joined_data_all")
# display(df_all)

print("**Data Frames Loaded")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### EDA Highlights
# MAGIC 1. Flights data
# MAGIC 2. Joined dataset

# COMMAND ----------

# MAGIC %md
# MAGIC Highlights from flights data analysis

# COMMAND ----------

# Examine dataset
display(df_flights.limit(10))

# COMMAND ----------

# Examine the relationship between records in the dataset
# get the size of the full dataset = 74,177,433
print("size of the full dataset:", df_flights.count())

# get the size of the number of unique records in the dataset 
print("size of the unique records in the dataset full dataset:", df_flights.distinct().count())

# COMMAND ----------

# duplicate records are observed. Before conducting more analysis, we will remove duplicate
# from preliminary analysis, we also want to focus our analysis on key attributes
keep_columns = ['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN', 'ORIGIN_STATE_ABR', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST', 'DEST_STATE_ABR', 'DEST_WAC', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'CANCELLED', 'CANCELLATION_CODE', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'YEAR']

df_flights1 = df_flights.select(keep_columns).distinct()

# COMMAND ----------

# Continue to explore relationship between rows by analyzing ID fields
print("distinct ORIGIN: ", df_flights1.select("ORIGIN").distinct().count())
print("distinct ORIGIN_AIRPORT_ID: ", df_flights1.select("ORIGIN_AIRPORT_ID").distinct().count())
print("distinct ORIGIN_AIRPORT_SEQ_ID: ",df_flights1.select("ORIGIN_AIRPORT_SEQ_ID").distinct().count())
print("distinct DEST: ",df_flights1.select("DEST").distinct().count())
print("distinct DEST_AIRPORT_ID: ",df_flights1.select("DEST_AIRPORT_ID").distinct().count())
print("distinct DEST_AIRPORT_SEQ_ID: ",df_flights1.select("DEST_AIRPORT_SEQ_ID").distinct().count())
print("distinct OP_UNIQUE_CARRIER: ",df_flights1.select("OP_UNIQUE_CARRIER").distinct().count())
print("distinct TAIL_NUM: ", df_flights1.select("TAIL_NUM").distinct().count())
print("distinct OP_CARRIER_FL_NUM: ",df_flights1.select("OP_CARRIER_FL_NUM").distinct().count())
print("distinct ORIGIN_WAC: ",df_flights1.select("ORIGIN_WAC").distinct().count())

# COMMAND ----------

# quick examination of the target variable
df_flights1.groupBy('DEP_DEL15').count().show()

# COMMAND ----------

# From analyzing the aiport ID fields, we notice that there are airport IDs and airport seq IDs. 
# the following cells dive deep into these airport identifying fields

# do these columns have the same code for some airports?
print(df_flights1.filter(col("DEST_AIRPORT_ID") == col("DEST_AIRPORT_SEQ_ID")).count())

# COMMAND ----------

# check relationship between the Aiport_IDs and the SEQ_IDs
from pyspark.sql.functions import countDistinct
# Check if 1 seq_ID map to a max of 1 aiport_id - expected yes
df_flights1.groupBy("ORIGIN_AIRPORT_SEQ_ID") \
  .agg(countDistinct("ORIGIN_AIRPORT_ID").alias("ORIGIN_AIRPORT_ID")) \
  .sort(F.desc("ORIGIN_AIRPORT_ID")) \
  .show(3)

# COMMAND ----------

# Check if 1 aiport_id maps to 1 or more aiport_seq_id - expected yes
df_flights1.groupBy("ORIGIN_AIRPORT_ID") \
  .agg(countDistinct("ORIGIN_AIRPORT_SEQ_ID").alias("ORIGIN_AIRPORT_SEQ_ID")) \
  .sort(F.desc("ORIGIN_AIRPORT_SEQ_ID")) \
  .show(3)

# COMMAND ----------

print("Unique origin count (expect 388): ", df_flights1.select("ORIGIN_AIRPORT_ID","ORIGIN").distinct().count())
print("Unique dest count (expect 386): ", df_flights1.select("DEST_AIRPORT_ID","DEST").distinct().count())

# COMMAND ----------

# MAGIC %md From analyzing the aiport_IDs against the aiport_seq_IDs, we can see that:
# MAGIC - airport_seq_id uniquely identifies an airport (1 to 1 relationship with aiport_id) whereas a single aiport_id can have multiple airport_seq_id
# MAGIC - airport_id uniquely match to the aiport code, which contradicts the flights data documentation
# MAGIC  <br/> <br/> Conculsion: 
# MAGIC - the airport_seq_ID are more accurate tracker of airports. As time permit, we will further improve our model performance by improving our join algorithim to consider the movement of airport locations that is tracked by the airport_seq_id
# MAGIC - since an aiport_id can only have 1 active aiport_seq_id at a time, for the purpose of building the unique flight tracker we can continue to use the airport_id
# MAGIC - airport_id columns (ORIGIN_AIRPORT_ID and DEST_AIRPORT_ID) uniquely match to their airport code (ORIGIN and DEST). This contradicts the flight dictionary documentation. Furthermore airport_seq_ids (ORIGIN_AIRPORT_SEQ_ID and DEST_AIRPORT_SEQ_ID) uniquely identifie an airport (1 to 1 relationship with the aiport_ids) whereas a single aiport_id can have multiple airport_seq_id. As such the airport_seq_IDs are more accurate tracker of airports. As time permit, we will further improve our model performance by improving our join algorithm to consider the movement of airport locations that is tracked by the airport_seq_id as opposed to solely relying on the airport_ids

# COMMAND ----------

# Next we will focus on identifying the unique record Id to build the flight column
print("count of dataset: ", df_flights1.distinct().count())
print("count of row from the proposed record id: ", df_flights1.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "OP_CARRIER_FL_NUM", "OP_UNIQUE_CARRIER", "DEP_TIME").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC Highlights from the joined dataset analysis

# COMMAND ----------

# lets look at the joined dataset
df_all.printSchema()

# COMMAND ----------

# dataset record count
print(df_all.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Fix nulls - impute missing values with mean

# COMMAND ----------

# fix airline nulls
mean_val=df_all.select(mean(df_all.CRS_ELAPSED_TIME)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['CRS_ELAPSED_TIME'])

mean_val=df_all.select(mean(df_all.DISTANCE)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['DISTANCE'])

# COMMAND ----------

# fix weather nulls
mean_val=df_all.select(mean(df_all.ELEVATION)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['ELEVATION'])

mean_val=df_all.select(mean(df_all.HourlyAltimeterSetting)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyAltimeterSetting'])

mean_val=df_all.select(mean(df_all.HourlyDewPointTemperature)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyDewPointTemperature'])

mean_val=df_all.select(mean(df_all.HourlyWetBulbTemperature)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyWetBulbTemperature'])

mean_val=df_all.select(mean(df_all.HourlyDryBulbTemperature)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyDryBulbTemperature'])

mean_val=df_all.select(mean(df_all.HourlyPrecipitation)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyPrecipitation'])

mean_val=df_all.select(mean(df_all.HourlyStationPressure)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyStationPressure'])

mean_val=df_all.select(mean(df_all.HourlySeaLevelPressure)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlySeaLevelPressure'])

mean_val=df_all.select(mean(df_all.HourlyRelativeHumidity)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyRelativeHumidity'])

mean_val=df_all.select(mean(df_all.HourlyVisibility)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyVisibility'])

mean_val=df_all.select(mean(df_all.HourlyWindSpeed)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyWindSpeed'])

# COMMAND ----------

# Run correlation analysis
df_values2 = df_all.select('DEP_DEL15','QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_ELAPSED_TIME', 'DISTANCE','ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed')
df_values2.select([count(when(col('DEP_DEL15').isNull(),True))]).show()

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df_values2.columns, outputCol=vector_col)
df_vector = assembler.transform(df_values2).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector,vector_col).collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 

corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = df_values2.columns, index=df_values2.columns) 
corr_matrix_df .style.background_gradient(cmap='coolwarm').set_precision(2)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Feature Engineering

# COMMAND ----------

# Feature 1 (Analysis ID):  Create unique flight identifier
df_all = df_all.withColumn("flight_id",
                     F.concat(F.col("ORIGIN_AIRPORT_ID"),
                              F.lit("-"), F.col("DEST_AIRPORT_ID"),
                              F.lit("-"), F.col("FL_DATE"),
                              F.lit("-"), F.col("OP_CARRIER_FL_NUM"),
                              F.lit("-"), F.col("OP_UNIQUE_CARRIER"),
                              F.lit("-"), F.col("DEP_TIME")))

# COMMAND ----------

# next, sort the dataframe by carrier, flight date, tail number and departure time 
# so that we start to build lag features with the assumption that if a prior flight was delayed
# then the next flight riding on the same airplane will very likely be delayed
df_all = df_all.sort(df_all.OP_UNIQUE_CARRIER.asc(),df_all.FL_DATE.asc(),df_all.TAIL_NUM.asc(),df_all.DEP_TIME.asc())

# COMMAND ----------

# MAGIC %md
# MAGIC TAIL_NUM is a unique aircraft identifier. 
# MAGIC If an aircraft is delayed from its previous flight, then it is more likely that the delayed aircraft would delay its next flight. As such, we want to create a data feature that tracks if a flight's previous aircraft is delayed. 

# COMMAND ----------

# Helper Features
# Create partition key
w = Window.partitionBy("FL_DATE", "TAIL_NUM").orderBy("DEP_TIME")

# create columns over the partition key such that we obtain delays from the prior flight 
# also grab the 2 flight prior in case the difference between an aircraft's flights is within 2 hrs
is_1prev_delayed = F.lag("DEP_DEL15", 1).over(w)
is_2prev_delayed = F.lag("DEP_DEL15", 2).over(w)
df_all = df_all.withColumn("is_1prev_delayed", is_1prev_delayed)
df_all = df_all.withColumn("is_2prev_delayed", is_2prev_delayed)

# create temp helper data to see if there is a need to run 2 flights prior to avoid data leakage
prev_dep_time = F.lag("DEP_TIME", 1).over(w)
df_all = df_all.withColumn("prev_dep_time", prev_dep_time)
df_all = df_all.withColumn("dep_time_diff", (df_all["DEP_TIME"] - df_all["prev_dep_time"]))

# all features created in this cell are helper features that will not be selected for modeling

# COMMAND ----------

# check if there is a need to handle data leakage. 
# will need to handle leakage if the count is > 0
print(df_all.filter(col("prev_dep_time") < 200).count())

# COMMAND ----------

# Feature 2 (Machine learning feature):  flight tracker - is_prev_delayed
# since we have observed aircrafts with flights departing within a 2 window, we will create
# the is_prev_delayed column to avoid data leakage
# this feature will be selected for modeling
df_all = df_all.withColumn(
    "is_prev_delayed",
    F.when((df_all["dep_time_diff"]) >= 200, df_all['is_1prev_delayed'])
    .otherwise(df_all['is_2prev_delayed'])
)

# this would be a good point to checkpoint to blob storage

# COMMAND ----------

# MAGIC %md 
# MAGIC ### ML model Pipeline Creation

# COMMAND ----------

# Stages for working with categorical features

# Convert categorical features to One Hot Encoding
categoricalColumns = ["ORIGIN", "OP_UNIQUE_CARRIER", "TAIL_NUM", "ORIGIN_STATE_ABR", "DEST_STATE_ABR"]
# Features not included: DEP_DATETIME_LAG, FL_DATE, CRS_DEP_TIME, CANCELLATION_CODE, DEP_HOUR, DEP_DATETIME

stages = [] # stages in Pipeline

# NOTE: Had to cut out a bunch of features due to the sheer number of NULLS in them, which were causing the entire dataframe to be skipped. Will need to get the Null values either filled or dropped.

for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index").setHandleInvalid("skip")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
#        
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
#    
print(stages)

CRS_DEP_TIME_stringIdx = StringIndexer(inputCol="CRS_DEP_TIME", outputCol="CRS_DEP_TIME_INDEX").setHandleInvalid("skip")
stages += [CRS_DEP_TIME_stringIdx]
DEP_HOUR_stringIdx = StringIndexer(inputCol="DEP_HOUR", outputCol="DEP_HOUR_INDEX").setHandleInvalid("skip")
stages += [DEP_HOUR_stringIdx]

print(stages)

# COMMAND ----------

# Create vectors for numeric and categorical variables
#numericCols = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_WAC", "DEP_TIME", "CANCELLED", "CRS_ELAPSED_TIME", "DISTANCE", "YEAR", "STATION", "DATE", "ELEVATION", "SOURCE", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlyVisibility", "HourlyWindSpeed", "DATE_HOUR", "distance_to_neighbor", "neighbor_call"]

# NOTE: Had to cut out a bunch of features due to the sheer number of NULLS in them, which were causing the entire dataframe to be skipped. Will need to get the Null values either filled or dropped.

#Works:
# Removed: Date, date_hour, neighbor_call
numericCols = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_WAC", "DEP_TIME", "CANCELLED", "CRS_ELAPSED_TIME", "DISTANCE", "YEAR", "STATION", "ELEVATION", "SOURCE", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlyVisibility", "HourlyWindSpeed", "distance_to_neighbor"]

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
#assemblerInputs = numericCols

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("skip")

stages += [assembler]

print(stages)

# COMMAND ----------

# Run the pipeline
partialPipeline = Pipeline().setStages(stages)

# Run 3 Month
#pipelineModel = partialPipeline.fit(df_joined_data_3m)
#preppedDataDF = pipelineModel.transform(df_joined_data_3m)

# Run Full Time
pipelineModel = partialPipeline.fit(df_all)
preppedDataDF = pipelineModel.transform(df_all)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Modeling

# COMMAND ----------

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


