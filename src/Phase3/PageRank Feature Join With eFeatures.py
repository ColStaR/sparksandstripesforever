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

df_joined_all_with_efeatures = spark.read.parquet(f"{blob_url}/joined_all_with_efeatures")
display(df_joined_all_with_efeatures)

PageranksByYear = spark.read.csv(f"{blob_url}/PageranksByYear.csv", header=True)
PageranksByYear = PageranksByYear.withColumnRenamed('YEAR','YEAR2').distinct()

display(PageranksByYear)




df_join_PR = df_joined_all_with_efeatures.withColumn('YEARLAG1',
                       df_joined_all_with_efeatures.YEAR - 1)

    
    
df_join_PR = df_join_PR.join(PageranksByYear, (df_join_PR["YEARLAG1"] == PageranksByYear["YEAR2"]) &
   ( df_join_PR["DEST"] == PageranksByYear["id"]),"inner")

df_join_PR = df_join_PR.drop("YEAR2","id")


display(df_join_PR)

# COMMAND ----------

df_join_PR.select('YEAR').distinct().show()

# COMMAND ----------

df_join_PR.count()

# COMMAND ----------

#df_join_PR.write.mode("overwrite").parquet(f"{blob_url}/joined_all_with_efeatures_v2_No2015")


# COMMAND ----------

df_join_PR = df_join_PR.drop('DEP_DATETIME_LAG', 'UTC_DEP_DATETIME_LAG', 'UTC_DEP_DATETIME')

# COMMAND ----------

df2 = df_join_PR.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in df_join_PR.columns])
df2.show()


# COMMAND ----------

PageranksByYear = PageranksByYear.withColumn("pagerank",PageranksByYear.pagerank.cast('double'))
display(PageranksByYear)

#display(PageranksByYear.filter(PageranksByYear.YEAR2 == "2015"))
#display(PageranksByYear.filter(PageranksByYear.YEAR2 == "2016"))



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

PageranksByYear = spark.read.csv(f"{blob_url}/PageranksByYear.csv", header=True)
PageranksByYear = PageranksByYear.withColumnRenamed('YEAR','YEAR2').distinct()


PageranksByYear = PageranksByYear.withColumn("pagerank",PageranksByYear.pagerank.cast('double'))
display(PageranksByYear)

