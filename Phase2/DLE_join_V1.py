# Databricks notebook source
# data_size = '_3m'
# data_size = '_6m'
# data_size = '_1y'
data_size = ''  # all

# COMMAND ----------

from pyspark.sql.functions import col, max

blob_container = "sasfcontainer" # The name of your container created in https://portal.azure.com
storage_account = "sasfstorage" # The name of your Storage account created in https://portal.azure.com
secret_scope = "sasfscope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "sasfkey" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import concat, col, lit, substring, avg
from pyspark.sql.types import StringType,BooleanType,DateType
from pyspark.sql.functions import substring, length, expr, to_timestamp

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

# MAGIC %md
# MAGIC 
# MAGIC ## Join Airlines and Weather datasets

# COMMAND ----------

# Load 2015 Q1 for Flights        dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/
df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data{data_size}/").distinct()\
                        .select('QUARTER','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','FL_DATE','OP_UNIQUE_CARRIER','TAIL_NUM','OP_CARRIER_FL_NUM',
                                'ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','ORIGIN','ORIGIN_STATE_ABR','ORIGIN_WAC',
                                'DEST', 'DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','DEST_STATE_ABR','DEST_WAC','CRS_DEP_TIME',
                                'DEP_TIME','DEP_DEL15', 'DEP_DELAY', 'DIVERTED', 'DIV_AIRPORT_LANDINGS','CANCELLED','CANCELLATION_CODE',
                                'CRS_ELAPSED_TIME','DISTANCE','YEAR')

df_airlines = df_airlines.withColumn("CRS_DEP_TIME", concat(lit('000'), col("CRS_DEP_TIME").cast(StringType())) ) \
                         .withColumn("DEP_HOUR", expr("substring(CRS_DEP_TIME, -4, 2)") ) \
                         .withColumn("DEP_DAY", expr("substring(FL_DATE, 0, 10)") ) \
                         .withColumn("DEP_DATETIME", concat( col('DEP_DAY'), lit('T'), col('DEP_HOUR'), lit(':00:00')) ) \
                         .withColumn("DEP_DATETIME_LAG", to_timestamp(col("DEP_DATETIME")) - expr("INTERVAL 2 HOURS") )\
                         .dropna(subset = ['TAIL_NUM'])

# add linen to drop nulls for tail number

# COMMAND ----------

# Load all stations data
df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*").distinct()
# display(df_stations)

# COMMAND ----------

print(df_airlines.select('ORIGIN', 'ORIGIN_STATE_ABR').distinct().count())
airports = df_airlines.select('ORIGIN', 'ORIGIN_STATE_ABR').distinct()
airports = airports.withColumn("neighbor_call", concat(lit("K"), col("ORIGIN")))

# COMMAND ----------

stations = df_stations.select('neighbor_call','neighbor_state', 'station_id', 'distance_to_neighbor','neighbor_lat','neighbor_lon').distinct()

stations = stations.withColumnRenamed('neighbor_state','ORIGIN_STATE_ABR').withColumnRenamed('station_id','STATION')

# COMMAND ----------

airport_stations = airports.join(stations, ['neighbor_call','ORIGIN_STATE_ABR']).distinct()

closest_stations = airport_stations.select("ORIGIN","distance_to_neighbor").groupBy("ORIGIN").min('distance_to_neighbor')\
                                        .withColumnRenamed('min(distance_to_neighbor)','distance_to_neighbor')

next_closest_stations = airport_stations.filter(col('distance_to_neighbor')!=0)\
                                        .select("ORIGIN","distance_to_neighbor").groupBy("ORIGIN").min('distance_to_neighbor')\
                                        .withColumnRenamed('min(distance_to_neighbor)','distance_to_neighbor')

furthest_stations = airport_stations.select("ORIGIN","distance_to_neighbor").groupBy("ORIGIN").max('distance_to_neighbor')\
                                        .withColumnRenamed('max(distance_to_neighbor)','distance_to_neighbor')

final_stations = closest_stations.join(next_closest_stations, ['ORIGIN','distance_to_neighbor'], how='outer')\
                                    .join(furthest_stations, ['ORIGIN','distance_to_neighbor'], how='outer')

airport_stations = airport_stations.join(final_stations, ['ORIGIN','distance_to_neighbor']).drop("ORIGIN_STATE_ABR").cache()

stations_list = airport_stations.select('STATION').distinct().toPandas()['STATION'].to_list()


# COMMAND ----------

# Load the 2015 Q1 for Weather .filter(col('DATE') < "2015-04-01T00:00:00.000")
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data{data_size}/") \
                       .select('STATION','DATE','ELEVATION','SOURCE', 'HourlyAltimeterSetting',
                               'HourlyDewPointTemperature', 'HourlyWetBulbTemperature','HourlyDryBulbTemperature', 
                               'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 
                               'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed')\
                       .filter(col("STATION").isin(stations_list) ) \
                       .withColumn("DATE_HOUR", expr("substring(DATE, 0, 13)") ) \
                       .withColumn("DEP_DATETIME_LAG", to_timestamp(col("DATE_HOUR")) + expr("INTERVAL 1 HOURS"))

# 'HourlyPressureChange','HourlyWindGustSpeed',
# 'MonthlyMeanTemperature', 'MonthlyMaximumTemperature', 'MonthlyGreatestSnowDepth',
# 'MonthlyGreatestSnowfall', 'MonthlyTotalSnowfall', 'MonthlyTotalLiquidPrecipitation',
# 'MonthlyMinimumTemperature'


# display(df_weather)

# COMMAND ----------

airport_weather = df_weather.join(airport_stations, 'STATION').drop("STATION",'neighbor_call').cache()
# display(airport_weather)

# COMMAND ----------

airport_weather = airport_weather.groupBy('ORIGIN','DEP_DATETIME_LAG') \
                                 .agg( *(avg(c).alias(c) for c in airport_weather.columns if c not in {'ORIGIN','DEP_DATETIME_LAG'}) )


# COMMAND ----------

df_final = df_airlines.join(airport_weather, ['DEP_DATETIME_LAG','ORIGIN']).cache()

# COMMAND ----------

# df_final.select('ORIGIN').count() # 40933735 rows for all data, 1353640 for 3m data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Convert time to UTC

# COMMAND ----------

# Datasource: https://github.com/lxndrblz/Airports
timezones = spark.read.csv(f"{blob_url}/airport_timezones.txt", header=True).select("code","time_zone_id").withColumnRenamed("code",'ORIGIN')

df_final = df_final.join(timezones, ['ORIGIN'])

# COMMAND ----------

df_final = df_final.withColumn("UTC_DEP_DATETIME_LAG", F.to_utc_timestamp(col("DEP_DATETIME_LAG"), col("time_zone_id")))
df_final = df_final.withColumn("UTC_DEP_DATETIME", F.to_utc_timestamp(col("DEP_DATETIME"), col("time_zone_id")))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Add unique identifier from Nina

# COMMAND ----------

df_final = df_final.withColumn('DEP_TIME_CLEANED', F.coalesce('DEP_TIME', 'CRS_DEP_TIME'))

df_final = df_final.withColumn('flight_id',
                     concat(F.col('ORIGIN_AIRPORT_ID'),
                              lit('-'), col('DEST_AIRPORT_ID'),
                              lit('-'), col('FL_DATE'),
                              lit('-'), col('OP_CARRIER_FL_NUM'),
                              lit('-'), col('OP_UNIQUE_CARRIER'),
                              lit('-'), col('DEP_TIME_CLEANED')))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Clean Nulls in Delay classification column

# COMMAND ----------

df_final = df_final.fillna(1, ['DEP_DEL15'])
df_final = df_final.fillna(0, ['HourlyPrecipitation'])

# COMMAND ----------

df_final.select('ORIGIN').count() # full data - 40933735, 3m data - 1353640

# COMMAND ----------

display(df_final)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Save to blob storage

# COMMAND ----------

if data_size == '':
    datalabel = '_all'
else:
    datalabel = data_size

df_final.write.mode('overwrite').parquet(f"{blob_url}/joined_data{datalabel}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 3m data ran in 3 mins from start to finish (including some unneccesary commands)
# MAGIC 
# MAGIC All data ran in 11 mins from start to finish

# COMMAND ----------


