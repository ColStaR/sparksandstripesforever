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

df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
display(df_airlines)

print("**Flight Data Loaded")

# COMMAND ----------

# UDF that converts time values to hour integer.
# Input has string value of hours and minutes smashed together in one string. Need to return just the hour value as int.
def convertTimeToHours(inputTime):
    return floor(inputTime/ 100)
    
convertTime = udf(lambda q : convertTimeToHours(q), IntegerType())

# COMMAND ----------

# DEST_WAC
df_airlines.describe(["DEST_WAC"]).show()
df_airlines.select([count(when(col('DEST_WAC').isNull(),True))]).show()
print("Notes: ")

# COMMAND ----------

# CRS_DEP_TIME

#df_airlines.select("CRS_DEP_TIME").withColumn("Hour", convertTimeToHours(col("CRS_DEP_TIME"))).show()
df_airlines.describe(["CRS_DEP_TIME"]).show()
df_airlines.select([count(when(col('CRS_DEP_TIME').isNull(),True))]).show()
print("Notes: ")

# COMMAND ----------

#DEP_TIME
df_airlines.describe(["DEP_TIME"]).show()
df_airlines.select([count(when(col('DEP_TIME').isNull(),True))]).show()
print("Notes: ")

# COMMAND ----------

#DEP_DELAY_NEW
df_airlines.describe(["DEP_DELAY_NEW"]).show()
df_airlines.select([count(when(col('DEP_DELAY_NEW').isNull(),True))]).show()
print("Notes: ")

# COMMAND ----------

#DEP_DEL15
df_airlines.describe(["DEP_DEL15"]).show()
df_airlines.select([count(when(col('DEP_DEL15').isNull(),True))]).show()
print(df_airlines.filter(df_airlines.DEP_DEL15 == 1).count())
print(df_airlines.filter(df_airlines.DEP_DEL15 == 0).count())
print(df_airlines.filter(df_airlines.DEP_DEL15 == 1).count() / (df_airlines.filter(df_airlines.DEP_DEL15 == 0).count() + df_airlines.filter(df_airlines.DEP_DEL15 == 1).count()))
print("Notes: ")

# COMMAND ----------


