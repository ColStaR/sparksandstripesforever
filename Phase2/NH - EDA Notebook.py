# Databricks notebook source
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

# MAGIC %md
# MAGIC - Column<'QUARTER'>,
# MAGIC - Column<'MONTH'>,
# MAGIC - Column<'DAY_OF_MONTH'>,
# MAGIC - Column<'DAY_OF_WEEK'>,
# MAGIC - Column<'FL_DATE'>,
# MAGIC - Column<'OP_UNIQUE_CARRIER'>,
# MAGIC - Column<'TAIL_NUM'>,
# MAGIC - Column<'OP_CARRIER_FL_NUM'>,
# MAGIC - Column<'ORIGIN_AIRPORT_ID'>,
# MAGIC - Column<'ORIGIN_AIRPORT_SEQ_ID'>,
# MAGIC - Column<'ORIGIN'>,
# MAGIC - Column<'ORIGIN_STATE_ABR'>,
# MAGIC - Column<'ORIGIN_WAC'>,
# MAGIC - Column<'DEST_AIRPORT_ID'>,
# MAGIC - Column<'DEST_AIRPORT_SEQ_ID'>,
# MAGIC - Column<'DEST'>,
# MAGIC - Column<'DEST_STATE_ABR'>,

# COMMAND ----------

from pyspark.sql.functions import col, floor
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import IntegerType
import pandas as pd

# COMMAND ----------

print("**Loading Data")

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# Inspect the Mount's Final Project folder 
# Please IGNORE dbutils.fs.cp("/mnt/mids-w261/datasets_final_project/stations_data/", "/mnt/mids-w261/datasets_final_project_2022/stations_data/", recurse=True)
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

display(dbutils.fs.ls(f"{data_BASE_DIR}stations_data/"))

print("**Data Loaded")

df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
display(df_airlines)

print("**Flight Data Loaded")

# COMMAND ----------

# User Defined Function that converts time values to hour integer.
# Input has string value of hours and minutes smashed together in one string. Need to return just the hour value as int.
def convertTimeToHours(inputTime):
    return floor(inputTime/ 100)

# Use convertTime with a withColumn function to apply convertTimeToHours to entire spark column efficiently.
convertTime = udf(lambda q : convertTimeToHours(q), IntegerType())

# COMMAND ----------

df_airlines_dedup = df_airlines.distinct()

# COMMAND ----------

# MAGIC %md
# MAGIC After removing duplicates, we end up with 1,403,471 records in the 3 month airlines data
# MAGIC 
# MAGIC Now we will drop irrelevant records for further analysis

# COMMAND ----------

df_nh = df_airlines_dedup.select(['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN', 'ORIGIN_STATE_ABR', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST', 'DEST_STATE_ABR', 'DEP_DEL15'])

# COMMAND ----------

# completing sanity check and confirm that there is no record lost in translation
df_nh.count()

# COMMAND ----------

display(df_nh)

# COMMAND ----------

# 316 unique airports (ORIGIN and DEST)
# 13 diff. carriers
# 4420 tail numbers
print("distinct ORIGIN and DEST: 316 from display, ", df_nh.select("ORIGIN").distinct().count(), " from count")
print("distinct ORIGIN_AIRPORT_SEQ_ID: ",df_nh.select("ORIGIN_AIRPORT_SEQ_ID").distinct().count())
print("distinct airports of DEST_AIRPORT_ID: ",df_nh.select("DEST_AIRPORT_ID").distinct().count())
print("distinct OP_UNIQUE_CARRIER: 13")
print("distinct tail number: 4420")
print("distinct OP_CARRIER_FL_NUM: ",df_nh.select("OP_CARRIER_FL_NUM").distinct().count())
print("distinct ORIGIN_WAC: ",df_nh.select("ORIGIN_WAC").distinct().count())

# COMMAND ----------

# investigate if the following has 1-1 mapping:
# 1. ORIGIN vs. ORIGIN_AIRPORT_SEQ_ID vs. ORIGIN_AIRPORT_ID has 1 to 1 m
# 2. DEST vs. DEST_AIRPORT_ID
print("ORIGIN check, expect 315: ", df_airlines.select("ORIGIN", "ORIGIN_AIRPORT_SEQ_ID","ORIGIN_AIRPORT_ID").distinct().count())
print("DEST check, expect 315: ", df_airlines.select("DEST", "DEST_AIRPORT_ID").distinct().count())

#from running 3 months of data, we conclude that these columns contain duplicate. 
# Hypothesis: Only 1 of these set of columns should be selected for joining
# Next step: run EDA on full dataset to verify hypothesis

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

df_values = df_nh.select('QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK','DEP_DEL15')

# COMMAND ----------

df_values.count()

# COMMAND ----------

df_values.columns

# COMMAND ----------

df_values2 = df_values.na.fill(value=0,subset=['DEP_DEL15'])

# COMMAND ----------

df_values2.columns

# COMMAND ----------

df_values2.select([count(when(col('DEP_DEL15').isNull(),True))]).show()

# COMMAND ----------

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df_values2.columns, outputCol=vector_col)
df_vector = assembler.transform(df_values2).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector,vector_col).collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 

# COMMAND ----------

corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = df_values2.columns, index=df_values2.columns) 
corr_matrix_df .style.background_gradient(cmap='coolwarm').set_precision(2)

# COMMAND ----------

df_nh.groupBy("DAY_OF_WEEK") \
       .count().sort("DAY_OF_WEEK").show()


# COMMAND ----------

from pyspark.sql.functions import sum, col

# COMMAND ----------

df_nh.select('DEP_DEL15').distinct().show()

# COMMAND ----------

# for code debuging only
simpleData = [("James","Sales","NY",90000,34,10000),
    ("Michael","Sales","NV",86000,56,20000),
    ("Robert","Sales","CA",81000,30,23000),
    ("Maria","Finance","CA",90000,24,23000),
    ("Raman","Finance","DE",99000,40,24000),
    ("Scott","Finance","NY",83000,36,19000),
    ("Jen","Finance","NY",79000,53,15000),
    ("Jeff","Marketing","NV",80000,25,18000),
    ("Kumar","Marketing","NJ",91000,50,21000)
  ]

schema = ["employee_name","department","state","salary","age","bonus"]
df = spark.createDataFrame(data=simpleData, schema = schema)
type(df)

# COMMAND ----------

display(df)

# COMMAND ----------

# for code debugging only
from pyspark.sql.functions import sum, col, desc
df.groupBy("state") \
  .agg(sum("salary").alias("sum_salary")) \
  .filter(col("sum_salary") > 100000)  \
  .sort(desc("sum_salary")) \
  .show()

# COMMAND ----------

df_nh.groupBy('DEP_DEL15').count().show()

# COMMAND ----------

delay_by_dow = df_nh.groupBy('DAY_OF_WEEK') \
  .agg(sum('DEP_DEL15').alias('Delayed Flights')) \
  .sort('DAY_OF_WEEK').toPandas()

# COMMAND ----------

type(delay_by_dow)

# COMMAND ----------

delay_by_dow.sort_values("DAY_OF_WEEK").set_index("DAY_OF_WEEK").plot()

# COMMAND ----------

df_nh.groupBy("DAY_OF_WEEK")

# COMMAND ----------

data_pandas = data_parquet.groupBy("year").agg(F.mean("mean_temp").alias("Mean␣
,→Temp")).toPandas()
data_pandas.sort_values("year").set_index("year").plot()

# COMMAND ----------

df_nh.groupBy("DAY_OF_WEEK").count() \
  .show()

# COMMAND ----------

display(df_airlines.select("LATE_AIRCRAFT_DELAY"))
df_airlines.select([count(when(col('LATE_AIRCRAFT_DELAY').isNull(),True))]).show()

# COMMAND ----------

#CANCELLED
display(df_airlines.select("CANCELLED"))
df_airlines.select([count(when(col('CANCELLED').isNull(),True))]).show()
print("% of flights cancelled:", df_airlines.filter(df_airlines.CANCELLED == 1).count() / (df_airlines.filter(df_airlines.CANCELLED == 0).count() + df_airlines.filter(df_airlines.CANCELLED == 1).count()))
print("# of flights delayed_15 and cancelled:", df_airlines.filter((df_airlines.DEP_DEL15 == 1) & (df_airlines.CANCELLED == 1)).count())

# Notes
# Boolean value. 1 = flight was cancelled.
# 3% of flights cancelled
# 1310 flights delayed and then cancelled. Statistically insignificant?

# COMMAND ----------

#LATE_AIRCRAFT_DELAY
display(df_airlines.select("LATE_AIRCRAFT_DELAY"))
df_airlines.select([count(when(col('LATE_AIRCRAFT_DELAY').isNull(),True))]).show()

# Notes
# Extreme value of 1,313
# Lots of Null values. Assumed to be flights without delays, so values of 0.

# COMMAND ----------

# MAGIC %md full set analysis

# COMMAND ----------

# Analyze duplicate columns and unique record ID on the full dataset
df_full = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")

# COMMAND ----------

# get the size of the full dataset = 74,177,433
print("size of the full dataset:", df_full.count())

# get the size of the number of unique records in the dataset 
print("size of the unique records in the dataset full dataset:", df_full.distinct().count())

# as expected, we are seeing duplicate records again in the full dataset. 
# as a part of the data pipeline, the first task will be removing duplicate

# COMMAND ----------

# verify record unique id (previously hypothsized)
# expected: 42,430,592
# df_full.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "OP_CARRIER_FL_NUM", "OP_UNIQUE_CARRIER", "TAIL_NUM") \
#     .distinct().count() # 42430587

# df_full.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "OP_CARRIER_FL_NUM", "OP_UNIQUE_CARRIER") \
#     .distinct().count() #42430577

df_full.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "OP_CARRIER_FL_NUM", "OP_UNIQUE_CARRIER", "DEP_TIME") \
    .distinct().count() #42430592 

# COMMAND ----------

# MAGIC %md 
# MAGIC Confirming the unique record key is:
# MAGIC 1. "ORIGIN_AIRPORT_ID", 
# MAGIC 2. "DEST_AIRPORT_ID", 
# MAGIC 3. "FL_DATE", 
# MAGIC 4. "OP_CARRIER_FL_NUM", 
# MAGIC 5. "OP_UNIQUE_CARRIER", 
# MAGIC 6. "DEP_TIME"

# COMMAND ----------

# 315 unique counts of ORIGIN and DEST exist in the 3 months data
# Now let's get a sense of how many we get from the full data
print("distinct size of ORIGIN: ", df_full.select("ORIGIN").distinct().count())
print("distinct size of ORIGIN_AIRPORT_SEQ_ID: ", df_full.select("ORIGIN_AIRPORT_SEQ_ID").distinct().count())
print("distinct size of ORIGIN_AIRPORT_ID: ", df_full.select("ORIGIN_AIRPORT_ID").distinct().count())
print("distinct size of DEST: ", df_full.select("DEST").distinct().count())
print("distinct size of DEST_AIRPORT_ID: ", df_full.select("DEST_AIRPORT_ID").distinct().count())

# COMMAND ----------

# MAGIC %md 
# MAGIC It's interesting to note that the number of origin vs. dest numbers no longer equal in the full dataset
# MAGIC There are more airports of Origin than airports of DEST
# MAGIC We also see that there are significantly more ORIGIN_AIRPORT_SEQ_ID than there is in ORIGIN
# MAGIC 
# MAGIC Next step:  
# MAGIC 1. Idenfity which airports are missing from DEST but are present in ORIGIN
# MAGIC 2. Explore more into the ORIGIN_AIRPORT_SEQ_ID

# COMMAND ----------

print("DEST check, expect 386: ", df_full.select("DEST", "DEST_AIRPORT_ID").distinct().count())
print("ORIGIN check, expect 388: ", df_full.select("ORIGIN", "ORIGIN_AIRPORT_ID").distinct().count())

# Result: confirmed that 1-1 mapping exist for columns within this set

# COMMAND ----------


