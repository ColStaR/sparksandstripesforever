# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # GraphFrames Analysis

# COMMAND ----------

pip install graphframes

# COMMAND ----------

from pyspark.sql.functions import col, floor
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
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

# COMMAND ----------

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

print("**Data Loaded")
print("**Loading Data Frames")

df_joined_data_3m = spark.read.parquet(f"{blob_url}/joined_data_3m")
display(df_joined_data_3m)

df_joined_data_all = spark.read.parquet(f"{blob_url}/joined_data_all")
display(df_joined_data_all)

print("**Data Frames Loaded")

# COMMAND ----------

from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import * 

# COMMAND ----------

df_joined_data_all = spark.read.parquet(f"{blob_url}/joined_data_all")
display(df_joined_data_all)

# COMMAND ----------

vertices = df_joined_data_all[['ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN', 'flight_id', 'UTC_DEP_DATETIME', 'DISTANCE']]

# COMMAND ----------

vertices = vertices.withColumnRenamed("ORIGIN_AIRPORT_SEQ_ID","id")
#vertices.distinct()

# COMMAND ----------

vertices.display()

# COMMAND ----------

edges = df_joined_data_all[['ORIGIN_AIRPORT_SEQ_ID',  'DEST_AIRPORT_SEQ_ID', 'ORIGIN_STATE_ABR', 'DEST_STATE_ABR']]

# COMMAND ----------

edges = edges.withColumnRenamed("ORIGIN_AIRPORT_SEQ_ID","src")
edges = edges.withColumnRenamed("DEST_AIRPORT_SEQ_ID","dst")
edges.display()


# COMMAND ----------

edges.display()

# COMMAND ----------

from pyspark.sql.functions import countDistinct

edges.select(countDistinct("src")).show()
edges.select(countDistinct("dst")).show()


# COMMAND ----------

from graphframes import * 

g = GraphFrame(vertices, edges)
print(g)

# COMMAND ----------

results = g.pageRank(resetProbability=0.15, tol=0.01)
display(results.vertices)

# COMMAND ----------

display(results.edges)


# COMMAND ----------

results = g.pageRank(resetProbability=0.15, tol=0.05)
display(results.vertices)

# COMMAND ----------

display(results.edges)


# COMMAND ----------

results = g.pageRank(resetProbability=0.15, tol=0.10)
display(results.vertices)

# COMMAND ----------

display(results.edges)

# COMMAND ----------

vertices = df_joined_data_all[['DEST', 'ORIGIN_AIRPORT_SEQ_ID','flight_id', 'UTC_DEP_DATETIME', 'DISTANCE']]
vertices = vertices.withColumnRenamed("DEST","id")
vertices.display()

# COMMAND ----------

edges = df_joined_data_all[['DEST', 'ORIGIN', 'ORIGIN_STATE_ABR', 'DEST_STATE_ABR']]
edges = edges.withColumnRenamed("DEST","src")
edges = edges.withColumnRenamed("ORIGIN","dst")
edges.display()


# COMMAND ----------

from pyspark.sql.functions import countDistinct

edges.select(countDistinct("src")).show()
edges.select(countDistinct("dst")).show()

# COMMAND ----------

g = GraphFrame(vertices, edges)

results = g.pageRank(resetProbability=0.15, tol=0.01)
display(results.vertices)

# COMMAND ----------

from graphframes import * 

g = GraphFrame(vertices, edges)

results = g.pageRank(resetProbability=0.15, tol=0.01)
display(results.vertices)
#display(results.edges)

# COMMAND ----------

from graphframes import * 

g = GraphFrame(vertices, edges)

results1 = g.pageRank(resetProbability=0.15, tol=0.05)
display(results.vertices)
display(results.edges)

# COMMAND ----------

from graphframes import * 

g = GraphFrame(vertices, edges)

results2 = g.pageRank(resetProbability=0.15, tol=0.10)
display(results.vertices)
display(results.edges)
