# Databricks notebook source
from pyspark.sql.functions import col, floor, countDistinct
from pyspark.sql.functions import isnan, when, count, col
import pyspark.sql.functions as F
from pyspark.sql import SQLContext
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
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
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
# display(dbutils.fs.ls(f"{data_BASE_DIR}"))

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

display(df_flights.limit(10))

# COMMAND ----------

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

display(df_flights1)

# COMMAND ----------

# Display summary stats for ID fields
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

# Examine the Aiport ID vs. Airport Seq ID in more detail

# do these columns have the same code for some airports?
print(df_flights1.filter(col("DEST_AIRPORT_ID") == col("DEST_AIRPORT_SEQ_ID")).count())

# COMMAND ----------

from pyspark.sql.functions import countDistinct
# Check if 1 seq_ID map to a max of 1 aiport_id - expected yes

df_flights1.groupBy("ORIGIN_AIRPORT_SEQ_ID") \
  .agg(countDistinct("ORIGIN_AIRPORT_ID").alias("ORIGIN_AIRPORT_ID")) \
  .sort(desc("ORIGIN_AIRPORT_ID")) \
  .show(3)

# COMMAND ----------

# Check if 1 aiport_id maps to 1 or more aiport_seq_id - expected yes
df_flights1.groupBy("ORIGIN_AIRPORT_ID") \
  .agg(countDistinct("ORIGIN_AIRPORT_SEQ_ID").alias("ORIGIN_AIRPORT_SEQ_ID")) \
  .sort(desc("ORIGIN_AIRPORT_SEQ_ID")) \
  .show(3)

# COMMAND ----------

print("Unique origin count (expect 388): ", df_flights1.select("ORIGIN_AIRPORT_ID","ORIGIN").distinct().count())
print("Unique origin count (expect 386): ", df_flights1.select("DEST_AIRPORT_ID","DEST").distinct().count())

# COMMAND ----------

# MAGIC %md From analyzing the aiport_IDs against the aiport_seq_IDs, we can see that:
# MAGIC - airport_seq_id uniquely identifies an airport (1 to 1 relationship with aiport_id) whereas a single aiport_id can have multiple airport_seq_id
# MAGIC - airport_id uniquely match to the aiport code, which contradicts the flights data documentation
# MAGIC  <br/> <br/> Conculsion: 
# MAGIC - the airport_seq_ID are more accurate tracker of airports. As time permit, we will further improve our model performance by improving our join algorithim to consider the movement of airport locations that is tracked by the airport_seq_id
# MAGIC - since an aiport_id can only have 1 active aiport_seq_id at a time, for the purpose of building the unique flight tracker we can continue to use the airport_id

# COMMAND ----------

# Next we will focus on identifying the unique record Id to build the flight column
print("count of dataset: ", df_flights1.distinct().count())
print("count of row from the proposed record id: ", df_flights1.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "OP_CARRIER_FL_NUM", "OP_UNIQUE_CARRIER", "DEP_TIME").distinct().count())

# COMMAND ----------

# MAGIC %md 
# MAGIC # Data Engineering - Flights Data 

# COMMAND ----------

# crreate flight_id, which uniquely identifies a flight
dff = df_flights1
dff = dff.withColumn("flight_id",
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
dff = dff.sort(dff.OP_UNIQUE_CARRIER.asc(),dff.FL_DATE.asc(),dff.TAIL_NUM.asc(),dff.DEP_TIME.asc())

# COMMAND ----------

# note .show() displays data in a way that becomes difficult to read as the number of columns increases
# for the remainder of the analysis, we will use display() instead
dff.show(20)

# COMMAND ----------

display(dff.limit(20))

# COMMAND ----------

# update again after pre-req on time conversion complete. 
# pending:
# 1. Avoid data leakage: convert dep time to local time and ensure that the difference between the dep times for flights with the same tail number is >2 hrs
# 2. Create helper column is_prev_flight_gt2hr
# 3. Data cleansing for delay 15
import pyspark.sql.window
w = Window.partitionBy("FL_DATE", "TAIL_NUM").orderBy("DEP_TIME")
is_prev_delayed = F.lag("DEP_DEL15", 1).over(w)
dff = dff.withColumn("is_prev_delayed", is_prev_delayed)

# COMMAND ----------

display(dff.limit(5))

# COMMAND ----------

# Next observe difference between joined dataset vs. dff

print(hi)
