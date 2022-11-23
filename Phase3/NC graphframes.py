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

df_joined_data_all_N21 = df_joined_data_all.filter(df_joined_data_all.YEAR != "2021")
display(df_joined_data_all_N21)
#display(df_joined_data_all_N21.filter(df_joined_data_all.YEAR == "2021"))


# COMMAND ----------

#from pyspark.sql.functions import countDistinct

#edges.select(countDistinct("src")).show()
#edges.select(countDistinct("dst")).show()


# COMMAND ----------

from pyspark.sql.functions import countDistinct

edges.select(countDistinct("src")).show()
edges.select(countDistinct("dst")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC - 621/662  Airport Sq IDs
# MAGIC - 334 unique airport DESTs

# COMMAND ----------

from graphframes import * 
#Non-21

vertices = df_joined_data_all_N21.select('DEST').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_N21.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

# COMMAND ----------

results = g.pageRank(resetProbability=0.15, tol=0.01)
display(results.vertices)

# COMMAND ----------

display(results.edges)

# COMMAND ----------

from graphframes import * 
#Non-21
#2015-Q1

df_joined_data_all_15Q1 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2015") & (df_joined_data_all.QUARTER == "1"))

vertices = df_joined_data_all_15Q1.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_15Q1.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results15Q1 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results15Q1.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2015-Q2

df_joined_data_all_15Q2 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2015") & (df_joined_data_all.QUARTER == "2"))

vertices = df_joined_data_all_15Q2.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_15Q2.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results15Q2 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results15Q2.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2015-Q3

df_joined_data_all_15Q3 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2015") & (df_joined_data_all.QUARTER == "3"))

vertices = df_joined_data_all_15Q3.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_15Q3.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results15Q3 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results15Q3.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2015-Q4

df_joined_data_all_15Q4 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2015") & (df_joined_data_all.QUARTER == "4"))

vertices = df_joined_data_all_15Q4.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_15Q4.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results15Q4 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results15Q4.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2016-Q1

df_joined_data_all_16Q1 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2016") & (df_joined_data_all.QUARTER == "1"))

vertices = df_joined_data_all_16Q1.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_16Q1.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results16Q1 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results16Q1.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2016-Q2

df_joined_data_all_16Q2 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2016") & (df_joined_data_all.QUARTER == "2"))

vertices = df_joined_data_all_16Q2.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_16Q2.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results16Q2 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results16Q2.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2016-Q3

df_joined_data_all_16Q3 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2016") & (df_joined_data_all.QUARTER == "3"))

vertices = df_joined_data_all_16Q3.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_16Q3.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results16Q3 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results16Q3.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2016-Q4

df_joined_data_all_16Q4 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2016") & (df_joined_data_all.QUARTER == "4"))

vertices = df_joined_data_all_16Q4.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_16Q4.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results16Q4 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results16Q4.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2017-Q1

df_joined_data_all_17Q1 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2017") & (df_joined_data_all.QUARTER == "1"))

vertices = df_joined_data_all_17Q1.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()
                                                        
edges = df_joined_data_all_17Q1.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results17Q1 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results17Q1.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2017-Q2

df_joined_data_all_17Q2 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2017") & (df_joined_data_all.QUARTER == "2"))

vertices = df_joined_data_all_17Q2.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()
                                                        
edges = df_joined_data_all_17Q2.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results17Q2 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results17Q2.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2017-Q3

df_joined_data_all_17Q3 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2017") & (df_joined_data_all.QUARTER == "3"))

vertices = df_joined_data_all_17Q3.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_17Q3.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results17Q3 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results17Q3.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2017-Q4

df_joined_data_all_17Q4 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2017") & (df_joined_data_all.QUARTER == "4"))

vertices = df_joined_data_all_17Q4.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_17Q4.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results17Q4 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results17Q4.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2018-Q1

df_joined_data_all_18Q1 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2018") & (df_joined_data_all.QUARTER == "1"))

vertices = df_joined_data_all_18Q1.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_18Q1.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results18Q1 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results18Q1.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2018-Q2

df_joined_data_all_18Q2 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2018") & (df_joined_data_all.QUARTER == "2"))

vertices = df_joined_data_all_18Q2.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_18Q2.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results18Q2 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results18Q2.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2018-Q3

df_joined_data_all_18Q3 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2018") & (df_joined_data_all.QUARTER == "3"))

vertices = df_joined_data_all_18Q3.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_18Q3.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results18Q3 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results18Q3.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2018-Q4

df_joined_data_all_18Q4 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2018") & (df_joined_data_all.QUARTER == "4"))

vertices = df_joined_data_all_18Q4.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_18Q4.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results18Q4 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results18Q4.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2019-Q1

df_joined_data_all_19Q1 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2019") & (df_joined_data_all.QUARTER == "1"))

vertices = df_joined_data_all_19Q1.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_19Q1.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results19Q1 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results19Q1.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2019-Q2

df_joined_data_all_19Q2 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2019") & (df_joined_data_all.QUARTER == "2"))

vertices = df_joined_data_all_19Q2.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_19Q2.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results19Q2 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results19Q2.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2019-Q3

df_joined_data_all_19Q3 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2019") & (df_joined_data_all.QUARTER == "3"))

vertices = df_joined_data_all_19Q3.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_19Q3.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results19Q3 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results19Q3.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2019-Q4

df_joined_data_all_19Q4 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2019") & (df_joined_data_all.QUARTER == "4"))

vertices = df_joined_data_all_19Q4.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_19Q4.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results19Q4 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results19Q4.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2020-Q1

df_joined_data_all_20Q1 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2020") & (df_joined_data_all.QUARTER == "1"))

vertices = df_joined_data_all_20Q1.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_20Q1.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results20Q1 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results20Q1.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2020-Q2

df_joined_data_all_20Q2 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2020") & (df_joined_data_all.QUARTER == "2"))

vertices = df_joined_data_all_20Q2.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_20Q2.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results20Q2 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results20Q2.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2020-Q3

df_joined_data_all_20Q3 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2020") & (df_joined_data_all.QUARTER == "3"))

vertices = df_joined_data_all_20Q3.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_20Q3.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results20Q3 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results20Q3.vertices)

# COMMAND ----------

from graphframes import * 
#Non-21
#2020-Q4

df_joined_data_all_20Q4 = df_joined_data_all_N21.filter((df_joined_data_all.YEAR == "2020") & (df_joined_data_all.QUARTER == "4"))

vertices = df_joined_data_all_20Q4.select('DEST','YEAR','QUARTER').withColumnRenamed('DEST','id').distinct()

edges = df_joined_data_all_20Q4.select('DEST','ORIGIN').withColumnRenamed("DEST","src").withColumnRenamed("ORIGIN","dst")

g = GraphFrame(vertices, edges)

results20Q4 = g.pageRank(resetProbability=0.15, tol=0.01)
display(results20Q4.vertices)

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame

resultsByQuarterYear = [results15Q1.vertices, results15Q2.vertices, results15Q3.vertices, results15Q4.vertices,results16Q1.vertices,results16Q2.vertices,results16Q3.vertices,results16Q4.vertices,results17Q1.vertices,results17Q2.vertices,results17Q3.vertices,results17Q4.vertices,results18Q1.vertices,results18Q2.vertices,results18Q3.vertices,results18Q4.vertices,results19Q1.vertices,results19Q2.vertices,results19Q3.vertices,results19Q4.vertices,results20Q1.vertices,results20Q2.vertices,results20Q3.vertices,results20Q4.vertices ]

#test = results15Q1.vertices.union(results15Q2.vertices, )
#display(test)
#resultsByQuarterYear = pd.concat(resultsByQuarterYear)
#type(results15Q1.vertices)

resultsByQuarterYearDF = reduce(DataFrame.union, resultsByQuarterYear)
display(resultsByQuarterYearDF)


# COMMAND ----------


