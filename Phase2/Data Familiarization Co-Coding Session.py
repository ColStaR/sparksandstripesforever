# Databricks notebook source


# COMMAND ----------

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

from pyspark.sql.functions import col
print("**Loading Data")

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# Inspect the Mount's Final Project folder 
# Please IGNORE dbutils.fs.cp("/mnt/mids-w261/datasets_final_project/stations_data/", "/mnt/mids-w261/datasets_final_project_2022/stations_data/", recurse=True)
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

display(dbutils.fs.ls(f"{data_BASE_DIR}stations_data/"))

print("**Data Loaded")

# COMMAND ----------

# Load 2015 Q1 for Flights        dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/
df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
display(df_airlines)

# Keep: CANCELLATION_CODE, 
# Drop: DIVERTED
# Dropping diverted data because it only has 3 flights in the 3 months. Very minimal gains in information for so much extra consideration/effort to implement.

# For development expediency, run 3 EDA on 3 month data set at first. Expand to larger data sets later by changing data source.
# Check number of diverted flights, see how frequent they are.
# Do EDA on joined data sets afterwards.

# COMMAND ----------

list(df_airlines)

# COMMAND ----------

#df_airlines.select("ORIGIN_AIRPORT_ID").distinct().count() # 315
df_airlines.select("ORIGIN_AIRPORT_SEQ_ID").distinct().count() # 315

# COMMAND ----------

# TODO: Check for NA's in CRS_DEP_TIME
# 
#from pyspark.sql.functions import col,isnan,when,count
#df_airlines.select("CRS_DEP_TIME").select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_Columns.columns]().count()

# COMMAND ----------

# Join on ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID, FL_DATE, DEP_TIME
# FIX DEP_TIME TO ONLY GET HOURS.

# COMMAND ----------

# Indicator that there is a lot of duplicate data in our flight data. Entire dataset duplicated twice?
#df_airlines.distinct().count() # 1403471
#df_airlines.count() # 2806942

# Unique ID will be:
# ["ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "OP_CARRIER_FL_NUM", "OP_UNIQUE_CARRIER"]
# "DEP_TIME" is optional for the unique identifiers, but will be needed for later joins.
#df_airlines.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "DEP_TIME", "OP_UNIQUE_CARRIER").distinct().count() # 1387996
#df_airlines.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "DEP_TIME", "TAIL_NUM").distinct().count() # 1399257
df_airlines.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "DEP_TIME", "OP_CARRIER_FL_NUM", "OP_UNIQUE_CARRIER").distinct().count() # 1403448


# COMMAND ----------

# Create deduplicated flight data
df_airlines_dedup = df_airlines.distinct()
df_airlines_dedup.count()

# COMMAND ----------

df_airlines_dedup

# COMMAND ----------

df_airlines.select("ORIGIN_STATE_ABR").distinct().count()

# COMMAND ----------

df_airlines.describe(["OP_UNIQUE_CARRIER", "OP_CARRIER"]).show()

# COMMAND ----------

# OP_CARRIER_AIRLINE_ID matches OP_UNIQUE_CARRIER
df_airlines.select("OP_UNIQUE_CARRIER", "OP_CARRIER_AIRLINE_ID").distinct().count()

# COMMAND ----------

# OP_Unique_Carrier has same entries as OP_CARRIER.
df_airlines.select("OP_UNIQUE_CARRIER", "OP_CARRIER").filter(df_airlines.OP_UNIQUE_CARRIER != df_airlines.OP_CARRIER).collect()

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/").filter(col('DATE') < "2015-04-01T00:00:00.000")
display(df_weather)

# Need to break up DATE into Hours, Day, Month, Year.
# 

# COMMAND ----------

list(df_weather)

# COMMAND ----------

# Load all stations data
df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
display(df_stations)

# Keep: station_id, neighbor_call, distance_to_neighbor
# Drop: All others

# COMMAND ----------

df_station.count()

# COMMAND ----------

df_weather.count()

# COMMAND ----------

# Joining airlines and Stations on Origin, Origin_state_abr, and a modified neighbor call field.

# COMMAND ----------

# Joining Airlines and Weather on derived features for date, hour number, and stationID

# COMMAND ----------

#35 flights
#17 weather
#3 Station
#55

# Nash takes Weather, Station tables
# Nina takes first 17 tables of Flights
# Ryan takes last 18 tables of Flights

