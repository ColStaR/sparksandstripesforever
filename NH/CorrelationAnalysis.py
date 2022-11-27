# Databricks notebook source
# Import libraries
from pyspark.sql.functions import col, floor, countDistinct
from pyspark.sql.functions import isnan, when, count, col
import pyspark.sql.functions as F
from pyspark.sql.functions import mean
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
from pyspark.sql import SQLContext

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

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

df_all = spark.read.parquet(f"{blob_url}/joined_all_with_efeatures_v2_No2015")
df_all0 = spark.read.parquet(f"{blob_url}/joined_data_all")
print("**Data Frames Loaded")

# COMMAND ----------

df_all = df_all.withColumn("pagerank", df_all.pagerank.cast('double'))

# COMMAND ----------

print(df_all0.select([count(when(col('DEP_DELAY').isNull(),True))]).show())
print(df_all0.select([count(when(col('DEP_DEL15').isNull(),True))]).show())

# COMMAND ----------

df_all.printSchema()

# COMMAND ----------

# fix airline nulls
mean_val=df_all.select(mean(df_all.CRS_ELAPSED_TIME)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['CRS_ELAPSED_TIME'])

mean_val=df_all.select(mean(df_all.DISTANCE)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['DISTANCE'])

mean_val=df_all.select(mean(df_all.DEP_DELAY)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['DEP_DELAY'])


# COMMAND ----------

# fix weather nulls
mean_val=df_all.select(mean(df_all.ELEVATION)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['ELEVATION'])

mean_val=df_all.select(mean(df_all.HourlyAltimeterSetting)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyAltimeterSetting'])

mean_val=df_all.select(mean(df_all.HourlyDewPointTemperature)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyDewPointTemperature'])

mean_val=df_all.select(mean(df_all.HourlyWetBulbTemperature)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyWetBulbTemperature'])

mean_val=df_all.select(mean(df_all.HourlyDryBulbTemperature)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyDryBulbTemperature'])

mean_val=df_all.select(mean(df_all.HourlyPrecipitation)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyPrecipitation'])

mean_val=df_all.select(mean(df_all.HourlyStationPressure)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyStationPressure'])

mean_val=df_all.select(mean(df_all.HourlySeaLevelPressure)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlySeaLevelPressure'])

mean_val=df_all.select(mean(df_all.HourlyRelativeHumidity)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyRelativeHumidity'])

mean_val=df_all.select(mean(df_all.HourlyVisibility)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyVisibility'])

mean_val=df_all.select(mean(df_all.HourlyWindSpeed)).collect()
avg = mean_val[0][0]
df_all = df_all.na.fill(avg,subset=['HourlyWindSpeed'])

# COMMAND ----------

# Run Pearson correlation analysis

# Select features to run corelation against
df_values2 = df_all.select('DEP_DELAY', 'DEP_DEL15', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_ELAPSED_TIME', 'DISTANCE', 'is_prev_delayed', 'is_prev_diverted', 'perc_delay', 'pagerank', 'ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed').limit(1000)

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df_values2.columns, outputCol=vector_col)
df_vector = assembler.transform(df_values2).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector,vector_col).collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 

corr_matrix_pearson = pd.DataFrame(data=corr_matrix, columns = df_values2.columns, index=df_values2.columns)
corr_matrix_pearson.style.background_gradient(cmap='coolwarm').set_precision(2)
# corr_matrix_pearson[['DEP_DELAY', 'DEP_DEL15']].sort_values(by=['DEP_DEL15'], ascending=False).style.background_gradient(cmap='coolwarm').set_precision(2)

# COMMAND ----------

corr_matrix_pearson[['DEP_DEL15']].sort_values(by=['DEP_DEL15'], ascending=False).style.background_gradient(cmap='coolwarm').set_precision(2)

# COMMAND ----------

# Run Spearman correlation analysis

# Select features to run corelation against
df_values2 = df_all.select('DEP_DELAY', 'DEP_DEL15', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_ELAPSED_TIME', 'DISTANCE', 'is_prev_delayed', 'is_prev_diverted', 'perc_delay', 'pagerank', 'ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed')

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df_values2.columns, outputCol=vector_col)
df_vector = assembler.transform(df_values2).select(vector_col)

# get correlation matrix
matrix_spearman = Correlation.corr(df_vector,vector_col, method='spearman').collect()[0][0] 
corr_matrix_spearman = matrix_spearman.toArray().tolist() 

corr_matrix_spearman = pd.DataFrame(data=corr_matrix_spearman, columns = df_values2.columns, index=df_values2.columns) 
corr_matrix_spearman.style.background_gradient(cmap='coolwarm').set_precision(2)

# COMMAND ----------

corr_matrix_spearman[['DEP_DEL15']].sort_values(by=['DEP_DEL15'], ascending=False).style.background_gradient(cmap='coolwarm').set_precision(2)

# COMMAND ----------


