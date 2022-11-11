# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Joined Data EDA

# COMMAND ----------

from pyspark.sql.functions import col, floor
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
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

# Initial Correlation of numerical features with DEL15
# Decent correlation (corr > abs(0.1)) for DEST_AIRPORT_ID, DEST_AIRPORT_SEQ_ID, DEP_TIME 

df_joined_data_all.printSchema()

features = ['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_WAC', 'DEP_TIME', 'DEP_DEL15', 'CANCELLED', 'CRS_ELAPSED_TIME', 'DISTANCE', 'YEAR', 'STATION', 'DATE', 'ELEVATION', 'SOURCE', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'DATE_HOUR', 'distance_to_neighbor', 'neighbor_call']
#print(features)

for feature in df_joined_data_all.select(features).columns:
    print(feature, df_joined_data_all.select(features).stat.corr(feature, "DEP_DEL15"))

# COMMAND ----------

# Total # of rows in dataset
totalRows = df_joined_data_all.count()
print(totalRows)

# COMMAND ----------

# Number and rate of of del15 flights
print(df_joined_data_all.filter(col("DEP_DEL15") == 1).count())
print(df_joined_data_all.filter(col("DEP_DEL15") == 1).count() / totalRows)

# COMMAND ----------



# COMMAND ----------

# Number and rate of of cancelled flights
print(df_joined_data_all.filter(col("CANCELLED") == 1).count())
print(df_joined_data_all.filter(col("CANCELLED") == 1).count() / totalRows)

# COMMAND ----------

# Rows where DEP_Del15 is null.
# Confirms that all of the rows where DEP_DEL15 is null are for cancelled flights.
display(df_joined_data_all.filter(col("DEP_DEL15").isNull()))
display(df_joined_data_all.filter(col("CANCELLED") == 1))
display(df_joined_data_all.filter(col("DEP_DEL15").isNull()).count())

# COMMAND ----------

# Rows where CANCELLATION_CODE is not null.
# Confirms that flights with null cancellation codes were non-cancelled flights
display(df_joined_data_all.filter(col("CANCELLATION_CODE").isNotNull()))
display(df_joined_data_all.filter(col("CANCELLATION_CODE").isNotNull()).count())

# COMMAND ----------

dfnum = df_joined_data_all[['DISTANCE', 'ELEVATION', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'distance_to_neighbor']]



# COMMAND ----------

from pyspark.sql.functions import col,isnan,when,count

df2 = dfnum.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in dfnum.columns])
df2.show()


# COMMAND ----------


dfnon = df_joined_data_all[['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_WAC', 'DEP_TIME', 'DEP_DEL15', 'CANCELLED', 'CRS_ELAPSED_TIME', 'DISTANCE', 'YEAR', 'STATION', 'DATE', 'ELEVATION', 'SOURCE', 'DATE_HOUR', 'neighbor_call']]

# COMMAND ----------

df3 = dfnon.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in dfnon.columns])
df3.show()


# COMMAND ----------

display(df_joined_data_all.filter(col("DATE_HOUR").isNotNull()))
display(df_joined_data_all.filter(col("DATE_HOUR").isNull()).count())

# COMMAND ----------


