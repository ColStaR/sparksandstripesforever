# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Data Cleaning Notebook

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Info:

# COMMAND ----------

from pyspark.sql.functions import col, floor
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import IntegerType
import pandas as pd

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
print("**Loading Data Frames")

df_airlines_raw = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
df_airlines = df_airlines_raw.distinct()
display(df_airlines)

df_weather_raw = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/")
df_weather = df_weather_raw.distinct()
display(df_weather)

df_stations_raw = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
df_stations = df_stations_raw.distinct()
display(df_stations)

print("**Data Frames Loaded")

# COMMAND ----------

# Features of Interest
flights_features = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "FL_DATE", "OP_UNIQUE_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM", \
                    "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN", "ORIGIN_STATE_ABR", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", \
                    "DEST", "DEST_STATE_ABR", "DEST_WAC", "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY_NEW", "DEP_DEL15", "DEP_DELAY_GROUP", "CANCELLED", "CANCELLATION_CODE", \
                    "CRS_ELAPSED_TIME", "DISTANCE", "DISTANCE_GROUP", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY", "YEAR"]
weather_features = ["STATION", "DATE", "ELEVATION", "SOURCE", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyPressureChange", "HourlyRelativeHumidity", \
                    "HourlySkyConditions", "HourlySeaLevelPressure", "HourlyStationPressure", "HourlyVisibility", "HourlyWetBulbTemperature", "HourlyWindGustSpeed", \
                    "HourlyWindSpeed", "YEAR"]
stations_features = []

# COMMAND ----------

display(df_airlines.select(flights_features))
display(df_weather.select(weather_features))
display(df_stations)

# COMMAND ----------


