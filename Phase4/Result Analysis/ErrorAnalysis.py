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
df_ml = spark.read.parquet(f"{blob_url}/best_LR_predictions")

print("**Data Frames Loaded")

# COMMAND ----------

df_ml.printSchema()

# COMMAND ----------

# get the size of the number of records predicted
print("Number of records predicted: ", df_ml.count())

# COMMAND ----------

# broad brush analysis
display(df_ml)

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

display(df_ml.select('r_tf', 'r_tn', 'r_fp', 'r_fn'))

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

print("The Percentage of true positives against all predictions are: ", c_tp / c_results * 100)
print("The Percentage of true negatives against all predictions are: ", c_tn / c_results * 100)
print("The Percentage of false positives against all predictions are: ", c_fp / c_results * 100)
print("The Percentage of false negatives against all predictions are: ", c_fn / c_results * 100)

# COMMAND ----------

df_ml.printSchema()

# COMMAND ----------

df_ml.select('is_prev_delayed').distinct().show()

# COMMAND ----------

# Area of analysis of interest by feature importance: 
# 1. HourlyPrecipitation -> HourlyPrecipitation
# 2. is_prev_delayed -> is_prev_delayed
# 3. Freezing_rain -> Freezing_Rain, Thunder -> Thunder, BlowingSnow -> Blowing_Snow
# 4. DEP_HOUR -> DEP_HOUR
# 5. AssumedEffect_XmasP -> AssumedEffect_Text
# 6. MONTH -> MONTH
# 7. OP_UNIQUE_CARRIER ->  OP_UNIQUE_CARRIER

# COMMAND ----------

df_ml_delay = df_ml.filter(col("DEP_DEL15") == 1)
df_ml_nodelay = df_ml.filter(col("DEP_DEL15") == 0)

# COMMAND ----------

# Graph： Hourly precipitation for d
display(df_ml.select('HourlyPrecipitation', 'is_prev_delayed', 'DEP_HOUR','AssumedEffect_Text', 'MONTH', 'OP_UNIQUE_CARRIER','DEP_DEL15'))

# COMMAND ----------

# # Graph： Hourly precipitation across all data 
# display(df_ml.select('HourlyPrecipitation', 'is_prev_delayed', 'DEP_HOUR','AssumedEffect_Text', 'MONTH', 'OP_UNIQUE_CARRIER','DEP_DEL15','r_type').filter((col('r_type') == "fn") | (col('r_type')== "fp")))

# COMMAND ----------



# COMMAND ----------

# Result Graphs： 
# 1. Hourly precipitation across all data 
# 2. is_prev_delayed
# 3. DEP HOUR
# 4. airline
# 5. Effect
display(df_ml.select('HourlyPrecipitation', 'is_prev_delayed', 'DEP_HOUR','AssumedEffect_Text', 'MONTH', 'OP_UNIQUE_CARRIER','DEP_DEL15','r_type'))

# COMMAND ----------

display(df_ml_delay.select('HourlyPrecipitation', 'is_prev_delayed', 'DEP_HOUR','AssumedEffect_Text', 'MONTH', 'OP_UNIQUE_CARRIER'))

# COMMAND ----------

display(df_ml_nodelay.select('HourlyPrecipitation', 'is_prev_delayed', 'DEP_HOUR','AssumedEffect_Text', 'MONTH', 'OP_UNIQUE_CARRIER'))

# COMMAND ----------

display(df_ml.select('HourlyPrecipitation', 'is_prev_delayed', 'DEP_HOUR','AssumedEffect_Text', 'MONTH', 'OP_UNIQUE_CARRIER','DEP_DEL15'))

# COMMAND ----------

display(df_fn.select('HourlyPrecipitation', 'is_prev_delayed', 'DEP_HOUR','AssumedEffect_Text', 'MONTH', 'OP_UNIQUE_CARRIER'))

# COMMAND ----------

display(df_fp.select('HourlyPrecipitation', 'is_prev_delayed', 'DEP_HOUR','AssumedEffect_Text', 'MONTH', 'OP_UNIQUE_CARRIER'))

# COMMAND ----------

display(df_all.select('HourlyPrecipitation', 'is_prev_delayed', 'DEP_HOUR','AssumedEffect_Text', 'MONTH', 'OP_UNIQUE_CARRIER','DEP_DEL15'))

# COMMAND ----------


