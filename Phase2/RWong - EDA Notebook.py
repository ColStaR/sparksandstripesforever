# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC EDA Steps:
# MAGIC 
# MAGIC Numerical Values:
# MAGIC - Count # of values, NA's
# MAGIC - Describe distribution (mean, median, standard deviation, range)
# MAGIC - Plot distribution
# MAGIC - Unique Notes
# MAGIC 
# MAGIC Categorical Values:
# MAGIC - Count # of entries, NA's
# MAGIC - List unique values.
# MAGIC - Unique Notes

# COMMAND ----------

# MAGIC %md
# MAGIC - Column<'DEST_WAC'>,
# MAGIC - Column<'CRS_DEP_TIME'>,
# MAGIC -  Column<'DEP_TIME'>,
# MAGIC -  Column<'DEP_DELAY_NEW'>,
# MAGIC -  Column<'DEP_DEL15'>,
# MAGIC -  Column<'DEP_DELAY_GROUP'>,
# MAGIC -  Column<'CANCELLED'>,
# MAGIC -  Column<'CANCELLATION_CODE'>,
# MAGIC -  Column<'CRS_ELAPSED_TIME'>,
# MAGIC -  Column<'DISTANCE'>,
# MAGIC -  Column<'DISTANCE_GROUP'>,
# MAGIC -  Column<'CARRIER_DELAY'>,
# MAGIC -  Column<'WEATHER_DELAY'>,
# MAGIC -  Column<'NAS_DELAY'>,
# MAGIC -  Column<'SECURITY_DELAY'>,
# MAGIC -  Column<'LATE_AIRCRAFT_DELAY'>,
# MAGIC -  Column<'YEAR'>]

# COMMAND ----------

from pyspark.sql.functions import col, floor
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import IntegerType
import pandas as pd
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

print("**Loading Data")

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# Inspect the Mount's Final Project folder 
# Please IGNORE dbutils.fs.cp("/mnt/mids-w261/datasets_final_project/stations_data/", "/mnt/mids-w261/datasets_final_project_2022/stations_data/", recurse=True)
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

display(dbutils.fs.ls(f"{data_BASE_DIR}stations_data/"))

print("**Data Loaded")

#df_airlines_raw = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
#df_airlines = df_airlines_raw.distinct()
#display(df_airlines)

df_airlines_raw = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
df_airlines = df_airlines_raw.distinct()
display(df_airlines)

df_all = spark.read.parquet(f"{blob_url}/joined_data_all")

print("**Flight Data Loaded")

# COMMAND ----------

# User Defined Function that converts time values to hour integer.
# Input has string value of hours and minutes smashed together in one string. Need to return just the hour value as int.
def convertTimeToHours(inputTime):
    return floor(inputTime/ 100)

# Use convertTime with a withColumn function to apply convertTimeToHours to entire spark column efficiently.
convertTime = udf(lambda q : convertTimeToHours(q), IntegerType())

# COMMAND ----------


# DEST_WAC
display(df_airlines.select("DEST_WAC"))
df_airlines.select([count(when(col('DEST_WAC').isNull(),True))]).show()

#Notes: 
# Categorical data input as integers.

# COMMAND ----------

# CRS_DEP_TIME

#df_airlines.select("CRS_DEP_TIME").withColumn("Hour", convertTimeToHours(col("CRS_DEP_TIME"))).show()
display(df_airlines.select("CRS_DEP_TIME"))
df_airlines.select([count(when(col('CRS_DEP_TIME').isNull(),True))]).show()

#Notes: 
# Original scheduled departure time.
# Original data is in integer data but is meant to be converted to time. Will need to be converted to hour values. 
# Will also need to account for 12 AM values of 0, since integer drops leading zeroes.")

# COMMAND ----------

#DEP_TIME
display(df_airlines.select("DEP_TIME"))
df_airlines.select([count(when(col('DEP_TIME').isNull(),True))]).show()

#Notes: 
# Original data is in integer data but is meant to be converted to time. Will need to be converted to hour values. 
# Will also need to account for 12 AM values of 0, since integer drops leading zeroes.")
# Null values indicate flight was cancelled.

# COMMAND ----------

#DEP_DELAY_NEW
display(df_airlines.select("DEP_DELAY_NEW"))
df_airlines.select([count(when(col('DEP_DELAY_NEW').isNull(),True))]).show()

#Notes: 
# Original data is in integer data but is meant to be converted to time. Will need to be converted to hour values. 
# Will also need to account for 12 AM values of 0, since integer drops leading zeroes.")
# Null value indicates flight was cancelled.

# COMMAND ----------

#DEP_DEL15
display(df_airlines.select("DEP_DEL15"))
df_airlines.select([count(when(col('DEP_DEL15').isNull(),True))]).show()

print("% of flights delayed:", df_airlines.filter(df_airlines.DEP_DEL15 == 1).count() / (df_airlines.filter(df_airlines.DEP_DEL15 == 0).count() + df_airlines.filter(df_airlines.DEP_DEL15 == 1).count()))

#Notes: 
# Boolean feature. 1 = flight was delayed by at least 15 minutes.
# Null means flight was cancelled

# COMMAND ----------

#DEP_DELAY_GROUP
display(df_airlines.select("DEP_DELAY_GROUP"))
df_airlines.select([count(when(col('DEP_DELAY_GROUP').isNull(),True))]).show()
#Notes: 10.23k entries with negative delay group values.

# COMMAND ----------

#CANCELLED
display(df_airlines.select("CANCELLED"))
df_airlines.select([count(when(col('CANCELLED').isNull(),True))]).show()
print("% of flights cancelled:", df_airlines.filter(df_airlines.CANCELLED == 1).count() / (df_airlines.filter(df_airlines.CANCELLED == 0).count() + df_airlines.filter(df_airlines.CANCELLED == 1).count()))
print("# of flights delayed_15 and cancelled:", df_airlines.filter((df_airlines.DEP_DEL15 == 1) & (df_airlines.CANCELLED == 1)).count())

# Notes
# Boolean value. 1 = flight was cancelled.
# 3% of flights cancelled
# 1310 flights delayed and then cancelled. Statistically insignificant?

# COMMAND ----------

#CANCELLATION_CODE
display(df_airlines.select("CANCELLATION_CODE"))
df_airlines.select([count(when(col('CANCELLATION_CODE').isNull(),True))]).show()

# Notes
# Majority of cancellation code is B. Weather?
# Delay code: A = Carrier Code, B = Weather, C = National Aviation System, D = Security
# Nulls are flights that were not cancelled.
# Lots of null values due to flight not being cancelled.

# COMMAND ----------

#CRS_ELAPSED_TIME
display(df_airlines.select("CRS_ELAPSED_TIME"))
df_airlines.select([count(when(col('CRS_ELAPSED_TIME').isNull(),True))]).show()

# Notes
# Unknown null value meaning
# Correlation between elapsed time and delay? Longer flights more likely to get delayed?

# COMMAND ----------

#DISTANCE
display(df_airlines.select("DISTANCE"))
df_airlines.select([count(when(col('DISTANCE').isNull(),True))]).show()

# Notes
# Extreme value of 4,983
# Filter for low flight distance values. 
# Distance and elapsed time likely highly correlated. Feature engineer to go with just one?

# COMMAND ----------

#DISTANCE_GROUP
display(df_airlines.select("DISTANCE_GROUP"))
df_airlines.select([count(when(col('DISTANCE_GROUP').isNull(),True))]).show()

# Notes
# Derived from DISTANCE.

# COMMAND ----------

#CARRIER_DELAY
display(df_airlines.select("CARRIER_DELAY"))
df_airlines.select([count(when(col('CARRIER_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 1,971
# Loss of missing data. Assumed to be 0 delay value.

# COMMAND ----------

#WEATHER_DELAY
display(df_airlines.select("WEATHER_DELAY"))
df_airlines.select([count(when(col('WEATHER_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 1,152
# Lots of Null values. Assumed to be flights without delays, so values of 0.

# COMMAND ----------

#NAS_DELAY
display(df_airlines.select("NAS_DELAY"))
df_airlines.select([count(when(col('NAS_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 1,101
# Lots of Null values. Assumed to be flights without delays, so values of 0.

# COMMAND ----------

#SECURITY_DELAY
display(df_airlines.select("SECURITY_DELAY"))
df_airlines.select([count(when(col('SECURITY_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 241
# Lots of Null values. Assumed to be flights without delays, so values of 0.

# COMMAND ----------

#LATE_AIRCRAFT_DELAY
display(df_airlines.select("LATE_AIRCRAFT_DELAY"))
df_airlines.select([count(when(col('LATE_AIRCRAFT_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 1,313
# Lots of Null values. Assumed to be flights without delays, so values of 0.

# COMMAND ----------

#YEAR
display(df_airlines.select("YEAR"))
df_airlines.select([count(when(col('YEAR').isNull(),True))]).show()

# Notes
# Integer value for year. No surprises here!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Full Data Run

# COMMAND ----------

# Full Data Run

print("**Loading Full Flight Data")

df_full_airlines_raw = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
df_full_airlines_raw = df_full_airlines_raw.distinct()
display(df_full_airlines_raw)

print("**Full Flight Data Loaded")

# COMMAND ----------


# DEST_WAC
display(df_full_airlines_raw.select("DEST_WAC"))
df_full_airlines_raw.select([count(when(col('DEST_WAC').isNull(),True))]).show()

#Notes: 
# Categorical data input as integers.

# COMMAND ----------

# CRS_DEP_TIME

#df_full_airlines_raw.select("CRS_DEP_TIME").withColumn("Hour", convertTimeToHours(col("CRS_DEP_TIME"))).show()
display(df_full_airlines_raw.select("CRS_DEP_TIME"))
df_full_airlines_raw.select([count(when(col('CRS_DEP_TIME').isNull(),True))]).show()

#Notes: 
# Original scheduled departure time.
# Original data is in integer data but is meant to be converted to time. Will need to be converted to hour values. 
# Will also need to account for 12 AM values of 0, since integer drops leading zeroes.")

# COMMAND ----------

#DEP_TIME
display(df_full_airlines_raw.select("DEP_TIME"))
df_full_airlines_raw.select([count(when(col('DEP_TIME').isNull(),True))]).show()

#Notes: 
# Original data is in integer data but is meant to be converted to time. Will need to be converted to hour values. 
# Will also need to account for 12 AM values of 0, since integer drops leading zeroes.")
# Null values indicate flight was cancelled.

# COMMAND ----------

#DEP_DELAY_NEW
display(df_full_airlines_raw.select("DEP_DELAY_NEW"))
df_full_airlines_raw.select([count(when(col('DEP_DELAY_NEW').isNull(),True))]).show()

#Notes: 
# Original data is in integer data but is meant to be converted to time. Will need to be converted to hour values. 
# Will also need to account for 12 AM values of 0, since integer drops leading zeroes.")
# Null value indicates flight was cancelled.

# COMMAND ----------

#DEP_DEL15
display(df_full_airlines_raw.select("DEP_DEL15"))
df_full_airlines_raw.select([count(when(col('DEP_DEL15').isNull(),True))]).show()

print("% of flights delayed:", df_full_airlines_raw.filter(df_full_airlines_raw.DEP_DEL15 == 1).count() / (df_full_airlines_raw.filter(df_full_airlines_raw.DEP_DEL15 == 0).count() + df_full_airlines_raw.filter(df_full_airlines_raw.DEP_DEL15 == 1).count()))

#Notes: 
# Boolean feature. 1 = flight was delayed by at least 15 minutes.
# Null means flight was cancelled

# COMMAND ----------

#DEP_DELAY_GROUP
display(df_full_airlines_raw.select("DEP_DELAY_GROUP"))
df_full_airlines_raw.select([count(when(col('DEP_DELAY_GROUP').isNull(),True))]).show()
#Notes: 10.23k entries with negative delay group values.

# COMMAND ----------

#CANCELLED
display(df_full_airlines_raw.select("CANCELLED"))
df_full_airlines_raw.select([count(when(col('CANCELLED').isNull(),True))]).show()
print("% of flights cancelled:", df_full_airlines_raw.filter(df_full_airlines_raw.CANCELLED == 1).count() / (df_full_airlines_raw.filter(df_full_airlines_raw.CANCELLED == 0).count() + df_full_airlines_raw.filter(df_full_airlines_raw.CANCELLED == 1).count()))
print("# of flights delayed_15 and cancelled:", df_full_airlines_raw.filter((df_full_airlines_raw.DEP_DEL15 == 1) & (df_full_airlines_raw.CANCELLED == 1)).count())

# Notes
# Boolean value. 1 = flight was cancelled.
# 3% of flights cancelled
# 1310 flights delayed and then cancelled. Statistically insignificant?

# COMMAND ----------

#CANCELLATION_CODE
display(df_full_airlines_raw.select("CANCELLATION_CODE"))
df_full_airlines_raw.select([count(when(col('CANCELLATION_CODE').isNull(),True))]).show()

# Notes
# Majority of cancellation code is B. Weather?
# Nulls are flights that were not cancelled.
# Lots of null values due to flight not being cancelled.

# COMMAND ----------

#CRS_ELAPSED_TIME
display(df_full_airlines_raw.select("CRS_ELAPSED_TIME"))
df_full_airlines_raw.select([count(when(col('CRS_ELAPSED_TIME').isNull(),True))]).show()

# Notes
# Unknown null value meaning

# COMMAND ----------

#DISTANCE
display(df_full_airlines_raw.select("DISTANCE"))
df_full_airlines_raw.select([count(when(col('DISTANCE').isNull(),True))]).show()

# Notes
# Extreme value of 4,983

# COMMAND ----------

#DISTANCE_GROUP
display(df_full_airlines_raw.select("DISTANCE_GROUP"))
df_full_airlines_raw.select([count(when(col('DISTANCE_GROUP').isNull(),True))]).show()

# Notes
# Derived from DISTANCE.

# COMMAND ----------

#CARRIER_DELAY
display(df_full_airlines_raw.select("CARRIER_DELAY"))
df_full_airlines_raw.select([count(when(col('CARRIER_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 1,971
# Loss of missing data. Assumed to be 0 delay value.

# COMMAND ----------

#WEATHER_DELAY
display(df_full_airlines_raw.select("WEATHER_DELAY"))
df_full_airlines_raw.select([count(when(col('WEATHER_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 1,152
# Lots of Null values. Assumed to be flights without delays, so values of 0.

# COMMAND ----------

#NAS_DELAY
display(df_full_airlines_raw.select("NAS_DELAY"))
df_full_airlines_raw.select([count(when(col('NAS_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 1,101
# Lots of Null values. Assumed to be flights without delays, so values of 0.

# COMMAND ----------

#SECURITY_DELAY
display(df_full_airlines_raw.select("SECURITY_DELAY"))
df_full_airlines_raw.select([count(when(col('SECURITY_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 241
# Lots of Null values. Assumed to be flights without delays, so values of 0.

# COMMAND ----------

#LATE_AIRCRAFT_DELAY
display(df_full_airlines_raw.select("LATE_AIRCRAFT_DELAY"))
df_full_airlines_raw.select([count(when(col('LATE_AIRCRAFT_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 1,313
# Lots of Null values. Assumed to be flights without delays, so values of 0.

# COMMAND ----------

#YEAR
display(df_full_airlines_raw.select("YEAR"))
df_full_airlines_raw.select([count(when(col('YEAR').isNull(),True))]).show()

# Notes
# Integer value for year. No surprises here!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Additional EDA

# COMMAND ----------

# Correlation of features with DEL15
features = ["DEP_DEL15", "DEST_WAC", "CRS_DEP_TIME", "CRS_ELAPSED_TIME", "DISTANCE"]

for feature in df_airlines.select(features).columns:
    print(feature, df_airlines.select(features).stat.corr(feature, "DEP_DEL15"))

# COMMAND ----------

# Correlation of features with DEL15 on full data set
features = ["DEP_DEL15", "DEST_WAC", "CRS_DEP_TIME", "CRS_ELAPSED_TIME", "DISTANCE"]

for feature in df_full_airlines_raw.select(features).columns:
    print(feature, df_full_airlines_raw.select(features).stat.corr(feature, "DEP_DEL15"))

# COMMAND ----------

df_airlines_columns = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "FL_DATE", "OP_UNIQUE_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN", "ORIGIN_STATE_ABR", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_STATE_ABR", "DEST_WAC", "CRS_DEP_TIME", "DEP_TIME", "DEP_DEL15", "CANCELLED", "CANCELLATION_CODE", "CRS_ELAPSED_TIME", "DISTANCE", "YEAR"]

df_airlines_null_counts = df_airlines.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in df_airlines_columns])
display(df_airlines_null_counts)

# COMMAND ----------

df_airlines_null_percentage = df_airlines_null_counts.select("TAIL_NUM", "DEP_TIME", "DEP_DEL15", "CANCELLATION_CODE", "CRS_ELAPSED_TIME")
display(df_airlines_null_percentage)

# df_airlines.count() = 42430592
# TAIL_NUM = 242827 = 0.572%
# DEP_TIME = 852812 = 2.009%
# DEP_DEL15 = 857939 = 2.021%
# CANCELLATION_CODE = 41556551 = 97.940%
# CRS_ELAPSED_TIME = 170 0.00004%

# COMMAND ----------

print(df_airlines.count())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Joined Data EDA

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

all_cols_except_timestamp = ['ORIGIN', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_STATE_ABR', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_STATE_ABR', 'DEST_WAC', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DEL15', 'CANCELLED', 'CANCELLATION_CODE', 'CRS_ELAPSED_TIME', 'DISTANCE', 'YEAR', 'DEP_HOUR', 'DEP_DATETIME', 'STATION', 'DATE', 'ELEVATION', 'SOURCE', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'DATE_HOUR', 'distance_to_neighbor', 'neighbor_call']

df2 = df_joined_data_all.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in all_cols_except_timestamp])
display(df2)

# COMMAND ----------

features = ['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_WAC', 'DEP_TIME', 'DEP_DEL15', 'CANCELLED', 'CRS_ELAPSED_TIME', 'DISTANCE', 'YEAR', 'DATE', 'ELEVATION', 'SOURCE', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'DATE_HOUR']

df2 = df_joined_data_all.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in features])

display(df2)

# COMMAND ----------

df2.show()

# COMMAND ----------


