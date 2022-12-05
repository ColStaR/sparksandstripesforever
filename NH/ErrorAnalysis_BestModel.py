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
df_all = spark.read.parquet(f"{blob_url}/joined_all_with_efeatures")

# read in result dataset
df_ml = spark.read.parquet(f"{blob_url}/best_GBT_predictions_12042214")

print("**Data Frames Loaded")

# COMMAND ----------

df_ml.printSchema()

# COMMAND ----------

df_ml.select('DEP_DEL15')

# COMMAND ----------

# Create Result analysis features: r_tf, r_tn, r_fp, r_fn

def get_result(df):
    df = df.withColumn("r_tp", \
         when((df.DEP_DEL15 == 1) & (df.prediction == 1), 1) \
                             .otherwise(0))

    df = df.withColumn("r_tn", \
         when((df.DEP_DEL15 == 0) & (df.prediction == 0), 1) \
                             .otherwise(0))

    df = df.withColumn("r_fp", \
         when((df.DEP_DEL15 == 0) & (df.prediction == 1), 1) \
                             .otherwise(0))

    df = df.withColumn("r_fn", \
         when((df.DEP_DEL15 == 1) & (df.prediction == 0), 1) \
                             .otherwise(0))
    return df

# COMMAND ----------

df_ml = get_result(df_ml)

# COMMAND ----------

#  Create a column for result type
df_ml = df_ml.withColumn("r_type", \
     when(df_ml.r_tp == 1, 'tp') \
                   .when(df_ml.r_tn == 1, 'tn') \
                   .when(df_ml.r_fp == 1, 'fp') \
                   .when(df_ml.r_fn == 1, 'fn'))

# COMMAND ----------

# separate datasets into 4:
# r_tp, r_tn, r_fp, r_fn
df_tp = df_ml.filter(col("r_tp") == 1)
df_tn = df_ml.filter(col("r_tn") == 1)
df_fp = df_ml.filter(col("r_fp") == 1)
df_fn = df_ml.filter(col("r_fn") == 1)

# COMMAND ----------

# verify all results are covered by the 4 scenarios:
c_results = df_ml.count() 
c_tp = df_tp.count()
c_tn = df_tn.count()
c_fp = df_fp.count()
c_fn = df_fn.count()
print("The diff. between the total number of rows from the prediction dataset against the 4 scenarios is: ",
    c_results - c_tp - c_tn - c_fp - c_fn)

# COMMAND ----------

# get total number of rows predicted
c_results

# COMMAND ----------

c_total = df_all.filter(col('YEAR') == 2021).count()

# COMMAND ----------

# get number of records dropped
print('The total number of records dropped is: ', c_total - c_results)
print('The % of records dropped is: ', (c_total - c_results)/c_total*100)

# COMMAND ----------

# get the result percentage
print("The Percentage of true positives against all predictions are: ", c_tp / c_results * 100)
print("The Percentage of true negatives against all predictions are: ", c_tn / c_results * 100)
print("The Percentage of false positives against all predictions are: ", c_fp / c_results * 100)
print("The Percentage of false negatives against all predictions are: ", c_fn / c_results * 100)

# COMMAND ----------

# Deep dive into analysis by breaking df_ml by dep_del15
df_ml_delay = df_ml.filter(col("DEP_DEL15") == 1)
df_ml_nodelay = df_ml.filter(col("DEP_DEL15") == 0)

# COMMAND ----------

# Result Graphs： 
# hourly precipitation
# 1. is_prev_delayed
# 3. DEP HOUR
# 4. airline
# 5. Effect
display(df_ml.select('HourlyPrecipitation', 'is_prev_delayed', 'DEP_HOUR','AssumedEffect_Text', 'MONTH', 'OP_UNIQUE_CARRIER','DEP_DEL15','r_type'))

# COMMAND ----------

count_delay = df_ml_delay.count()
count_NoDelay = df_ml_nodelay.count()
count_fp = df_fp.count()
count_fn = df_fn.count()

# COMMAND ----------

df_ml_delay_hp = df_ml_delay.groupBy("HourlyPrecipitation") \
  .agg(count("HourlyPrecipitation").alias("HourlyPrecipitationCount"))

display(df_ml_delay_hp)

# COMMAND ----------

df_ml_nodelay_hp = df_ml_nodelay.groupBy("HourlyPrecipitation") \
  .agg(count("HourlyPrecipitation").alias("HourlyPrecipitationCount"))

display(df_ml_nodelay_hp)

# COMMAND ----------

# #Zoom in: is_prev_delayed
# df_ml_nodelay2 = df_ml_nodelay.groupBy("is_prev_delayed") \
#   .agg(count("is_prev_delayed").alias("Total_is_prev_delayed_0"))

# df_ml_nodelay2 = df_ml_nodelay2.withColumn("is_prev_delayed_percent_0", col('Total_is_prev_delayed_0') / count_delay * 100)
# display(df_ml_nodelay2)

# COMMAND ----------


