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
# MAGIC - Column<'STATION'>,
# MAGIC - Column<'DATE'>,
# MAGIC - Column<'ELEVATION'>,
# MAGIC - Column<'SOURCE'>,
# MAGIC - Column<'HourlyDewPointTemperature'>,
# MAGIC - Column<'HourlyDryBulbTemperature'>,
# MAGIC - Column<'HourlyPressureChange'>,
# MAGIC - Column<'HourlyRelativeHumidity'>,
# MAGIC - Column<'HourlySkyConditions'>,
# MAGIC - Column<'HourlySeaLevelPressure'>,
# MAGIC - Column<'HourlyStationPressure'>,
# MAGIC - Column<'HourlyVisibility'>,
# MAGIC - Column<'HourlyWetBulbTemperature'>,
# MAGIC - Column<'HourlyWindGustSpeed'>,
# MAGIC - Column<'HourlyWindSpeed'>,
# MAGIC - Column<'YEAR'>

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
df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")

print("**Data Loaded")

df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/")

display(df_weather)

print("**Flight Data Loaded")

# COMMAND ----------

# User Defined Function that converts time values to hour integer.
# Input has string value of hours and minutes smashed together in one string. Need to return just the hour value as int.
def convertTimeToHours(inputTime):
    return floor(inputTime/ 100)

# Use convertTime with a withColumn function to apply convertTimeToHours to entire spark column efficiently.
convertTime = udf(lambda q : convertTimeToHours(q), IntegerType())

# COMMAND ----------

#STATION
display(df_weather.select("STATION"))
df_weather.select([count(when(col('STATION').isNull(),True))]).show()

# Notes
# No Nulls in Station

# COMMAND ----------



# COMMAND ----------

#DATE
display(df_weather.select("DATE"))
df_weather.select([count(when(col('DATE').isNull(),True))]).show()

# Notes
# No Nulls in date
# format may be incorrect

# COMMAND ----------

#ELEVATION
display(df_weather.select(col("ELEVATION").cast('int')))
df_weather.select([count(when(col("ELEVATION").cast('int').isNull(),True))]).show()

# Notes
# Not many Nulls

# COMMAND ----------

#SOURCE
display(df_weather.select("SOURCE"))
df_weather.select([count(when(col('SOURCE').isNull(),True))]).show()

# Notes
# No nulls for source

# COMMAND ----------

#HourlyDewPointTemperature
display(df_weather.select(col("HourlyDewPointTemperature").cast('int')))
df_weather.select([count(when(col("HourlyDewPointTemperature").cast('int').isNull(),True))]).show()

# Notes
# 17% missing
# Normally Distributed

# COMMAND ----------

#HourlyDryBulbTemperature
display(df_weather.select(col("HourlyDryBulbTemperature").cast('int')))
df_weather.select([count(when(col("HourlyDryBulbTemperature").cast('int').isNull(),True))]).show()

# Notes
# mean and median seem high
# 2% missing
# Normally distributed

# COMMAND ----------

#HourlyPressureChange
display(df_weather.select(col("HourlyPressureChange").cast('int')))
df_weather.select([count(when(col("HourlyPressureChange").cast('int').isNull(),True))]).show()

# Notes
# Pressure in hectopascals does not seem to change often
# Data is 72% missing, may be worth dropping

# COMMAND ----------

#HourlyRelativeHumidity
display(df_weather.select(col("HourlyRelativeHumidity").cast('int')))
df_weather.select([count(when(col("HourlyRelativeHumidity").cast('int').isNull(),True))]).show()

# Notes
# Right Tailed dist
# 17% missing, perhaps replace with mean?

# COMMAND ----------

#HourlySkyConditions
display(df_weather.select("HourlySkyConditions"))
df_weather.select([count(when(col('HourlySkyConditions').isNull(),True))]).show()

# Notes
# Extreme value of 9 meters per second
# large quantity of nulls

# COMMAND ----------

#HourlySeaLevelPressure
display(df_weather.select(col("HourlySeaLevelPressure").cast('int')))
df_weather.select([count(when(col("HourlySeaLevelPressure").cast('int').isNull(),True))]).show()

# Notes
# very tight distribution, more than half missing, may be worth dropping

# COMMAND ----------

#HourlyStationPressure
display(df_weather.select(col("HourlyStationPressure").cast('int')))
df_weather.select([count(when(col("HourlyStationPressure").cast('int').isNull(),True))]).show()

# Notes
# half of data is missing
# right tailed

# COMMAND ----------

#HourlyVisibility
display(df_weather.select(col("HourlyVisibility").cast('int')))
df_weather.select([count(when(col("HourlyVisibility").cast('int').isNull(),True))]).show()

# Notes
# 1/3 missing
# outlier at 99?

# COMMAND ----------

#HourlyWetBulbTemperature
display(df_weather.select(col("HourlyWetBulbTemperature").cast('int')))
df_weather.select([count(when(col("HourlyWetBulbTemperature").cast('int').isNull(),True))]).show()
# Notes
# Normal Distribution
# Outliers at -90?
# Half missing

# COMMAND ----------

#HourlyWindSpeed
display(df_weather.select(col("HourlyWindSpeed").cast('int')))
df_weather.select([count(when(col("HourlyWindSpeed").cast('int').isNull(),True))]).show()

# Notes
# Data is tightly distributed otherwise
# Outliers present at max

# COMMAND ----------

#HourlyWindGustSpeed
display(df_weather.select(col("HourlyWindGustSpeed").cast('int')))
df_weather.select([count(when(col("HourlyWindGustSpeed").cast('int').isNull(),True))]).show()

# Notes
# Extreme value of 9 meters per second
# large quantity of nulls

# COMMAND ----------

#YEAR
display(df_weather.select("YEAR"))
df_weather.select([count(when(col('YEAR').isNull(),True))]).show()
# Notes
# No Nulls for year

# COMMAND ----------



df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
display(df_stations)

# COMMAND ----------



data_BASE_DIR = "dbfs:/mnt/mids-w261/"



data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))





df_weather_1y = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_1y/")
display(df_weather_1y)

# COMMAND ----------

#STATION
display(df_weather_1y.select("STATION"))
df_weather_1y.select([count(when(col('STATION').isNull(),True))]).show()

# Notes
# No Nulls in Station

# COMMAND ----------

#DATE
display(df_weather_1y.select("DATE"))
df_weather_1y.select([count(when(col('DATE').isNull(),True))]).show()

# Notes
# No Nulls in date
# format may be incorrect

# COMMAND ----------

#ELEVATION
display(df_weather_1y.select(col("ELEVATION").cast('int')))
df_weather_1y.select([count(when(col("ELEVATION").cast('int').isNull(),True))]).show()

# Notes
# Extreme value of 9 meters per second
# Not a lot missing

# COMMAND ----------

#SOURCE
display(df_weather_1y.select("SOURCE"))
df_weather_1y.select([count(when(col('SOURCE').isNull(),True))]).show()

# Notes
# No nulls for source

# COMMAND ----------

#HourlyDewPointTemperature
display(df_weather_1y.select(col("HourlyDewPointTemperature").cast('int')))
df_weather_1y.select([count(when(col("HourlyDewPointTemperature").cast('int').isNull(),True))]).show()

# Notes
# 17% missing
# Normally Distributed

# COMMAND ----------

#HourlyDryBulbTemperature
display(df_weather_1y.select(col("HourlyDryBulbTemperature").cast('int')))
df_weather_1y.select([count(when(col("HourlyDryBulbTemperature").cast('int').isNull(),True))]).show()

# Notes
# mean and median seem high
# 2% missing
# Normally distributed

# COMMAND ----------

#HourlyPressureChange
display(df_weather_1y.select(col("HourlyPressureChange").cast('int')))
df_weather_1y.select([count(when(col("HourlyPressureChange").cast('int').isNull(),True))]).show()

# Notes
# Pressure in hectopascals does not seem to change often
# Data is 72% missing, may be worth dropping

# COMMAND ----------

#HourlyRelativeHumidity
display(df_weather_1y.select(col("HourlyRelativeHumidity").cast('int')))
df_weather_1y.select([count(when(col("HourlyRelativeHumidity").cast('int').isNull(),True))]).show()

# Notes
# Right Tailed dist
# 17% missing, perhaps replace with mean?

# COMMAND ----------

#HourlySkyConditions
display(df_weather_1y.select("HourlySkyConditions"))
df_weather_1y.select([count(when(col('HourlySkyConditions').isNull(),True))]).show()

# Notes
# Extreme value of 9 meters per second
# Half data is missing quantity of nulls

# COMMAND ----------

#HourlySeaLevelPressure
display(df_weather_1y.select(col("HourlySeaLevelPressure").cast('int')))
df_weather_1y.select([count(when(col("HourlySeaLevelPressure").cast('int').isNull(),True))]).show()

# Notes
# Most of distrubtion is centered around two points
# Data is still relativeley tight

# COMMAND ----------

#HourlyStationPressure
display(df_weather_1y.select(col("HourlyStationPressure").cast('int')))
df_weather_1y.select([count(when(col("HourlyStationPressure").cast('int').isNull(),True))]).show()

# Notes
# half of data is missing
# right tailed

# COMMAND ----------

#HourlyVisibility
display(df_weather_1y.select(col("HourlyVisibility").cast('int')))
df_weather_1y.select([count(when(col("HourlyVisibility").cast('int').isNull(),True))]).show()

# Notes
# 1/3 missing
# outlier at 99?

# COMMAND ----------

#HourlyWetBulbTemperature
display(df_weather_1y.select(col("HourlyWetBulbTemperature").cast('int')))
df_weather_1y.select([count(when(col("HourlyWetBulbTemperature").cast('int').isNull(),True))]).show()
# Notes
# Normal Distribution
# Outliers at -90?
# Half missing
# Extremes on both ends

# COMMAND ----------

#HourlyWindSpeed
display(df_weather_1y.select(col("HourlyWindSpeed").cast('int')))
df_weather_1y.select([count(when(col("HourlyWindSpeed").cast('int').isNull(),True))]).show()

# Notes
# Data is tightly distributed otherwise
# Outliers present at max

# COMMAND ----------

#HourlyWindGustSpeed
display(df_weather_1y.select(col("HourlyWindGustSpeed").cast('int')))
df_weather_1y.select([count(when(col("HourlyWindGustSpeed").cast('int').isNull(),True))]).show()

# Notes
# Extreme value of 145 meters per second
# large quantity of nulls

# COMMAND ----------

#YEAR
display(df_weather_1y.select("YEAR"))
df_weather_1y.select([count(when(col('YEAR').isNull(),True))]).show()
# Notes
# No Nulls for year
