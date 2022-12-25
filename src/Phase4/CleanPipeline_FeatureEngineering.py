# Databricks notebook source
# Import libraries
from pyspark.sql.functions import col, floor, countDistinct
from pyspark.sql.functions import isnan, when, count, col
import pyspark.sql.functions as F
from pyspark.sql.functions import mean
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
from pyspark.sql import SQLContext
from pyspark.sql import types

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

import numpy as np
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

# read injoined full set
df_all = spark.read.parquet(f"{blob_url}/joined_data_all")
display(df_all)
# read in seconedary dataset for special days
df_calendar = spark.read.option("header", True) \
                     .csv(f'{blob_url}/SpecialDaySchedule2.csv')

print("**Data Frames Loaded")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ### [Feature engineering] effect of special days:

# COMMAND ----------

# Data Engineering: effect of special days (e.g. holiday, Covid lockdown)
def get_date_effect(df_primary, df_sec):
    # clean up date format from the special day dataset
    df_sec = df_sec.withColumn('year', F.split(df_calendar['SpecialDate'], '/').getItem(2)) \
       .withColumn('month', F.split(df_calendar['SpecialDate'], '/').getItem(0)) \
       .withColumn('day', F.split(df_calendar['SpecialDate'], '/').getItem(1))
    
    df_sec = df_sec.withColumn('SpecialDate2', F.concat(F.lit("20"),F.col("year"),
                                                     F.lit("-"), F.col("month"),
                                                     F.lit("-"), F.col("day")) \
                  .cast(types.DateType()))
    
    # keep the secondary dataset lean by dropping unnecessary columns
    df_sec = df_sec.drop("year", "month", "day")
    
    # join to the primary dataset
    df_primary = df_primary.join(df_sec, df_primary.FL_DATE == df_sec.SpecialDate2, 'left')
    
    # any missing value from the join is because the given day is not a special day, so fill effect with 'None'
    df_primary = df_primary.na.fill(value = "NONE",subset=['AssumedEffect_Text'])
    
    df_primary = df_primary.drop('SpecialDate', 'SpecialDateType', 'SpecialDate2')
    
    return df_primary

# COMMAND ----------

# perform join to include special date effect to the primary dataframe 
df_all = get_date_effect(df_all, df_calendar)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### [Feature engineering] Airline effectiveness:

# COMMAND ----------

# function for getting the airline effectiveness of the starting year
def get_airline_type(df_1):
    '''For the starting year of the dataset, we will calculate the airline effectiveness using
    that year's data since there is no previous year data to reference from.
    This is not introducing data leakage since we already know what happened that year'''
    # Create temp table to store how many flights were delayed from a carrier
    tab = df_1.select(['OP_UNIQUE_CARRIER','DEP_DEL15']).\
       groupBy('OP_UNIQUE_CARRIER').\
       agg(F.sum('DEP_DEL15').alias('TotalDelay'))
    
    # Create temp table to store how many flights were scheduled from a carrier
    tab_flight = df_1.select(['OP_UNIQUE_CARRIER','flight_id']).\
        groupBy('OP_UNIQUE_CARRIER').\
        agg(F.count('flight_id').alias('TotalFlight'))
    
    tab_flight = tab_flight.withColumnRenamed("OP_UNIQUE_CARRIER","OP_UNIQUE_CARRIER2")
    tab_flight = tab_flight.drop("OP_UNIQUE_CARRIER")
    
    # bring the temp tables for airline together 
    tab_airline_effect = tab.join(tab_flight, tab.OP_UNIQUE_CARRIER==tab_flight.OP_UNIQUE_CARRIER2, 'inner')
    tab_airline_effect = tab_airline_effect.drop('OP_UNIQUE_CARRIER')
    
    # calculate percentage delay from airline
    tab_airline_effect = tab_airline_effect.withColumn("perc_delay", 
                                                   ((tab_airline_effect["TotalDelay"] /
                                                     tab_airline_effect["TotalFlight"])*100)
                                                  )
    
    # create perc_delay column based on quantiles
    tmp_airline = tab_airline_effect.toPandas()
    tmp_airline['airline_type'] = np.where(
    tmp_airline['perc_delay']>tmp_airline['perc_delay'].quantile(0.75), "BelowAverage",
        np.where(tmp_airline['perc_delay']<tmp_airline['perc_delay'].quantile(0.25), "AboveAverage",
                 "Average"))
    
    # get ready to join the temp table back to the spark dataframe
    tmp_airline=spark.createDataFrame(tmp_airline)
    
    output_df = df_1.join(tmp_airline, df_1.OP_UNIQUE_CARRIER==tmp_airline.OP_UNIQUE_CARRIER2, 'inner')
    output_df = output_df.drop('OP_UNIQUE_CARRIER2')
    return output_df

# COMMAND ----------

# function for getting the airline effectiveness of the subsequent years
def get_airline_type2(df_m1, df_1):
    '''this function creates the airline_type column with 3 possible values:
        BelowAverage - airlines who had more historical percentage delays than 75% of the airlines
        AboveAverage - airlines who had less historical percentage delays than 25% of the airlines
        Average - airlines with histoical percentage delays that fall within the 25%-75% quantiles
        
        Input of this function: 
             - df_m1 for df containing the current year minus 1
             - df_1 for df containing the current year
        Output of this function: input dataframe plus additional airlines columns'''
    
    # Create temp table to store how many flights were delayed from a carrier
    tab = df_m1.select(['OP_UNIQUE_CARRIER','DEP_DEL15']).\
       groupBy('OP_UNIQUE_CARRIER').\
       agg(F.sum('DEP_DEL15').alias('TotalDelay'))
    
    # Create temp table to store how many flights were scheduled from a carrier
    tab_flight = df_m1.select(['OP_UNIQUE_CARRIER','flight_id']).\
        groupBy('OP_UNIQUE_CARRIER').\
        agg(F.count('flight_id').alias('TotalFlight'))
    
    tab_flight = tab_flight.withColumnRenamed("OP_UNIQUE_CARRIER","OP_UNIQUE_CARRIER2")
    tab_flight = tab_flight.drop("OP_UNIQUE_CARRIER")
    
    # bring the temp tables for airline together 
    tab_airline_effect = tab.join(tab_flight, tab.OP_UNIQUE_CARRIER==tab_flight.OP_UNIQUE_CARRIER2, 'inner')
    
    # calculate percentage delay from airline
    tab_airline_effect = tab_airline_effect.withColumn("perc_delay", 
                                                   ((tab_airline_effect["TotalDelay"] /
                                                     tab_airline_effect["TotalFlight"])*100)
                                                  )
    
    # create perc_delay column based on quantiles
    tmp_airline = tab_airline_effect.toPandas()
    tmp_airline['airline_type'] = np.where(
    tmp_airline['perc_delay']>tmp_airline['perc_delay'].quantile(0.75), "BelowAverage",
        np.where(tmp_airline['perc_delay']<tmp_airline['perc_delay'].quantile(0.25), "AboveAverage",
                 "Average"))
    
    # get ready to join the temp table back to the spark dataframe
    tmp_airline=spark.createDataFrame(tmp_airline)
    tmp_airline = tmp_airline.select(['OP_UNIQUE_CARRIER2','perc_delay', 'airline_type'])
    
    output_df = df_1.join(tmp_airline, df_1.OP_UNIQUE_CARRIER==tmp_airline.OP_UNIQUE_CARRIER2, 'left')
    output_df = output_df.drop(col('OP_UNIQUE_CARRIER2'))
    return output_df

# COMMAND ----------

def get_airline_effect(df, year_start, year_end):
    # splite full data by year to prep the dataset for further engineering 
    # this approach guarantees recency while protect against data leakage
    for i in range(year_start, year_end + 1):
        globals()[f"df{i}"] = df.filter(col("YEAR") == i)
    
    # for the starting year of the dataset, get airline effect by using current year's data
    globals()[f"df_{year_start}"] = get_airline_type(globals()[f"df{year_start}"])
    globals()[f"df_{year_start}"] = globals()[f"df_{year_start}"].drop('TotalDelay', 'TotalFlight')
    df = globals()[f"df_{year_start}"]
    
    # for years onwards, get airline effect by using previous year's data
    for i in range(year_start + 1, year_end + 1):
        globals()[f"df_{i}"] = get_airline_type2(globals()[f"df{i-1}"], globals()[f"df{i}"])
        
        # fix nulls for new airlines introduced by assuming their effectiveness is the same as last year's overall average across all airlines
        mean_val=globals()[f"df_{i}"].select(mean(globals()[f"df_{i}"].perc_delay)).collect()
        avg = mean_val[0][0]
        globals()[f"df_{i}"] = globals()[f"df_{i}"].na.fill(avg,subset=['perc_delay'])
        globals()[f"df_{i}"] = globals()[f"df_{i}"].na.fill(value = "Average",subset=['airline_type'])
        
        df = df.union(globals()[f"df_{i}"])

    return df

# COMMAND ----------

df_all = get_airline_effect(df_all, 2015, 2021)

# COMMAND ----------

# df_all.count()

# COMMAND ----------

# Memory management - clean up temp database used to create the airline effectiveness col:
for i in range(2015, 2022):
    globals()[f"df_{i}"].unpersist()
    globals()[f"df{i}"].unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### [Feature engineering] Delayed flight tracker:

# COMMAND ----------

def get_flight_tracker(df):
    # prep the dataframe lag calculation
    df = df.sort(df.OP_UNIQUE_CARRIER.asc(),df.FL_DATE.asc(),df.TAIL_NUM.asc(),df.DEP_TIME.asc())
    
    # Helper Features
    # Create partition key
    w = Window.partitionBy("FL_DATE", "TAIL_NUM").orderBy("DEP_TIME_CLEANED")

    # create columns over the partition key such that we obtain delays from the prior flight 
    # also grab the 2 flight prior in case the difference between an aircraft's flights is within 2 hrs
    is_1prev_delayed = F.lag("DEP_DEL15", 1).over(w)
    is_2prev_delayed = F.lag("DEP_DEL15", 2).over(w)
    df = df.withColumn("is_1prev_delayed", is_1prev_delayed)
    df = df.withColumn("is_2prev_delayed", is_2prev_delayed)

    # create temp helper data to see if there is a need to run 2 flights prior to avoid data leakage
    prev_dep_time = F.lag("DEP_TIME_CLEANED", 1).over(w)
    df = df.withColumn("prev_dep_time", prev_dep_time)
    df = df.withColumn("dep_time_diff", (df["DEP_TIME_CLEANED"] - df["prev_dep_time"]))

    # all helper features will not be selected for modeling
    # now create our flight tracker column that will be used for modeling
    df = df.withColumn(
    "is_prev_delayed",
    F.when((df["dep_time_diff"]) >= 200, df['is_1prev_delayed'])
    .otherwise(df['is_2prev_delayed']))
    
    # replace null with 0, meaning first flight of the date (with null) is not delayed
    df = df.na.fill(value = 0,subset=['is_prev_delayed'])
    
    return df    

# COMMAND ----------

df_all = get_flight_tracker(df_all)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pagerank Join

# COMMAND ----------

PageranksByYear = spark.read.csv(f"{blob_url}/PageranksByYear.csv", header=True)


def join_pr(df,pr):
    pr = pr.withColumnRenamed('YEAR','YEAR2').distinct()
    df = df.withColumn('YEARLAG1',
                           df.YEAR - 1)



    df = df.join(pr, (df["YEARLAG1"] == pr["YEAR2"]) &
       ( df["DEST"] == pr["id"]),"inner")

    df = df.drop("YEAR2","id")
    
    df = df.withColumn("pagerank",df.pagerank.cast('double'))
    return df



# COMMAND ----------

df_all_PR = join_pr(df_all,PageranksByYear) 
#display(df_all_PR)
df_all = df_all_PR

# COMMAND ----------

# MAGIC %md
# MAGIC ### [Feature engineering] Freezing Rain and Blowing Snow:

# COMMAND ----------

from pyspark.sql.functions import when

def get_rough_weather(df):
    df = df.withColumn("Blowing_Snow", when(df.AggHourlyPresentWeatherType.contains('BLSN'),1) \
          .otherwise(0))

    df = df.withColumn("Freezing_Rain", when(df.AggHourlyPresentWeatherType.contains('FZRA'),1) \
          .otherwise(0))
    
    df = df.withColumn("Rain", when(df.AggHourlyPresentWeatherType.contains('RA'),1) \
          .otherwise(0))
    
    df = df.withColumn("Snow", when(df.AggHourlyPresentWeatherType.contains('SN'),1) \
          .otherwise(0))
    
    df = df.withColumn("Thunder", when(df.AggHourlyPresentWeatherType.contains('TS'),1) \
          .otherwise(0))
    
    
    #df = df.drop("HourlyPresentWeatherType")
    return df

# COMMAND ----------

df_all = get_rough_weather(df_all)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint to Blob Storage BEORE DOWNSAMPLE

# COMMAND ----------

#checkpoint to blob storage
df_all.write.mode("overwrite").parquet(f"{blob_url}/joined_all_with_efeatures")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Downsampling

# COMMAND ----------

# def downsampleYearly(dataFrameInput):
    
#     ###### TO DO: Might be able to extract distinct years without converting to rdd
#     listOfYears = dataFrameInput.select("YEAR").distinct().filter(col("YEAR") != 2021).rdd.flatMap(list).collect()
    
#     downsampledDF = None

#     for currentYear in listOfYears:

#         print(f"Processing Year: {currentYear}")
#         currentYearDF = dataFrameInput.filter(col("YEAR") == currentYear).cache()

#         # Downscale the data such that there are roughly equal amounts of rows where DEP_DEL15 == 0 and DEP_DEL15 == 1, which aids in training.

#         currentYearDF_downsampling_0 = currentYearDF.filter(col("DEP_DEL15") == 0)
#         print(f"@- currentYearDF_downsampling_0.count() = {currentYearDF_downsampling_0.count()}")
#         currentYearDF_downsampling_1 = currentYearDF.filter(col("DEP_DEL15") == 1)
#         print(f"@- currentYearDF_downsampling_1.count() = {currentYearDF_downsampling_1.count()}")

#         downsampling_ratio = (currentYearDF_downsampling_1.count() / currentYearDF_downsampling_0.count())

#         currentYearDF_downsampling_append = currentYearDF_downsampling_0.sample(fraction = downsampling_ratio, withReplacement = False, seed = 261)

#         currentYearDF_downsampled = currentYearDF_downsampling_1.union(currentYearDF_downsampling_append)
#         print(f"@- currentYearDF_downsampled.count() = {currentYearDF_downsampled.count()}")
                
#         if downsampledDF == None:
#             downsampledDF = currentYearDF_downsampled
#             print(f"@- downsampledDF.count() = {downsampledDF.count()}")
# #         else:
#             downsampledDF = downsampledDF.union(currentYearDF_downsampled).cache()
#             print(f"@- downsampledDF.count() = {downsampledDF.count()}")
            
        
#     downsampledDF = downsampledDF.union(dataFrameInput.filter(col("YEAR"))==2021)
            
#     return downsampledDF
    

# COMMAND ----------

# downsampledDF = downsampleYearly(df_all)

# COMMAND ----------

def downsample(df):
    downsampling_0 = df.filter((col("DEP_DEL15") == 0) & (col('YEAR')<2021))
    downsampling_1 = df.filter((col("DEP_DEL15") == 1) | (col('YEAR')==2021))

    downsampling_ratio = (downsampling_1.count() / downsampling_0.count())

    downsampling_append = downsampling_0.sample(fraction = downsampling_ratio, withReplacement = False, seed = 261)
    downsampledDF = downsampling_1.union(downsampling_append)
    
    return downsampledDF

# COMMAND ----------

downsampledDF = downsample(df_all)
downsampledDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint to Blob Storage AFTER DOWNSAMPLE

# COMMAND ----------

downsampledDF.write.mode("overwrite").parquet(f"{blob_url}/joined_all_with_efeatures_Downsampled")

# COMMAND ----------

display(df_all)

# COMMAND ----------

# Total # of rows in dataset
totalRows = df_all.count()
print(totalRows)

# Number and rate of of BLSN FLights
print("Blowing_Snow:")
print(df_all.filter(col("Blowing_Snow") == 1).count())
print(df_all.filter(col("Blowing_Snow") == 1).count() / totalRows)

# Number and rate of FZRN flights
# Number and rate of of del15 flights
print("Freezing_Rain:")
print(df_all.filter(col("Freezing_Rain") == 1).count())
print(df_all.filter(col("Freezing_Rain") == 1).count() / totalRows)

print("Rain:")
print(df_all.filter(col("Rain") == 1).count())
print(df_all.filter(col("Rain") == 1).count() / totalRows)

print("Snow:")
print(df_all.filter(col("Snow") == 1).count())
print(df_all.filter(col("Snow") == 1).count() / totalRows)

print("Thunder:")
print(df_all.filter(col("Thunder") == 1).count())
print(df_all.filter(col("Thunder") == 1).count() / totalRows)

print("Wind:")
print(df_all.filter(col("Wind") == 1).count())
print(df_all.filter(col("Wind") == 1).count() / totalRows)

print("Hail:")
print(df_all.filter(col("Hail") == 1).count())
print(df_all.filter(col("Hail") == 1).count() / totalRows)


# COMMAND ----------


print("Tornado:")
print(df_all.filter(col("Tornado") == 1).count())
print(df_all.filter(col("Tornado") == 1).count() / totalRows)


# COMMAND ----------

# Total # of rows in dataset
delRows = df_all.filter(col("DEP_DEL15") == 1).count()
print(delRows)

# Number and rate of of BLSN FLights
print("Blowing_Snow:")
print(df_all.filter((col("Blowing_Snow") == 1) & (col("DEP_DEL15") ==1)).count())
print(df_all.filter((col("Blowing_Snow") == 1) & (col("DEP_DEL15") ==1)).count()/ delRows)

# Number and rate of FZRN flights
# Number and rate of of del15 flights
print("Freezing_Rain:")
print(df_all.filter((col("Freezing_Rain") == 1) & (col("DEP_DEL15") ==1)).count())
print(df_all.filter((col("Freezing_Rain") == 1) & (col("DEP_DEL15") ==1)).count()/ delRows)

print("Rain:")
print(df_all.filter((col("Rain") == 1) & (col("DEP_DEL15") ==1)).count())
print(df_all.filter((col("Rain") == 1) & (col("DEP_DEL15") ==1)).count()/ delRows)

print("Snow:")
print(df_all.filter((col("Snow") == 1) & (col("DEP_DEL15") ==1)).count())
print(df_all.filter((col("Snow") == 1) & (col("DEP_DEL15") ==1)).count()/ delRows)

print("Thunder:")
print(df_all.filter((col("Thunder") == 1) & (col("DEP_DEL15") ==1)).count())
print(df_all.filter((col("Thunder") == 1) & (col("DEP_DEL15") ==1)).count()/ delRows)



# COMMAND ----------

df_all.printSchema()

# COMMAND ----------


