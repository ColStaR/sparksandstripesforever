# Databricks notebook source
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

df = spark.read.parquet(f"{blob_url}/joined_data_all")

print(df.count())

# Datasource: https://github.com/lxndrblz/Airports
timezones = spark.read.csv(f"{blob_url}/airport_timezones.txt", header=True).select("code","time_zone_id").withColumnRenamed("code",'ORIGIN')

df = df.join(timezones, ['ORIGIN'])

print(df.count())

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.select('DEP_DATETIME_LAG','time_zone_id').withColumn("UTC_DEP_DATETIME_LAG", F.to_utc_timestamp(col("DEP_DATETIME_LAG"), col("time_zone_id"))))

# COMMAND ----------


