# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Data Pipeline Creation
# MAGIC 
# MAGIC Adapted From:
# MAGIC 
# MAGIC https://towardsdatascience.com/building-an-ml-application-with-mllib-in-pyspark-part-1-ac13f01606e2
# MAGIC 
# MAGIC https://docs.databricks.com/machine-learning/train-model/mllib/index.html
# MAGIC 
# MAGIC https://docs.databricks.com/_static/notebooks/binary-classification.html

# COMMAND ----------

from pyspark.sql.functions import col, floor
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import IntegerType

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import percent_rank

from sklearn.utils import parallel_backend
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
from joblibspark import register_spark

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

df_joined_data_2y = df_joined_data_all.filter(col("YEAR") <= 2016).cache()
display(df_joined_data_2y)

df_joined_data_all = spark.read.parquet(f"{blob_url}/joined_data_all")
display(df_joined_data_all)


#df_joined_data_2015_2020 = df_joined_data_all.filter(col("YEAR") < 2021)
#display(df_joined_data_all)

#df_joined_data_2021 = df_joined_data_all.filter(col("YEAR") == 2021)
#display(df_joined_data_2021)

print("**Data Frames Loaded")

# COMMAND ----------

# Data cleaning Tasks

#df_joined_data_3m = df_joined_data_3m.na.fill(value = 1,subset=["DEP_DEL15"])
df_joined_data_2y = df_joined_data_2y.na.fill(value = 1,subset=["DEP_DEL15"])
df_joined_data_all = df_joined_data_all.na.fill(value = 1,subset=["DEP_DEL15"])
#display(df_joined_data_3m)
#display(df_joined_data_all)

#df_joined_data_2015_2020 = df_joined_data_2015_2020.na.fill(value = 1,subset=["DEP_DEL15"])
#df_joined_data_2021 = df_joined_data_2021.na.fill(value = 1,subset=["DEP_DEL15"])

# COMMAND ----------

# dataframe schema
print(df_joined_data_all.count())
df_joined_data_all.printSchema()
#print(df_joined_data_all.columns)

# COMMAND ----------

# Convert categorical features to One Hot Encoding

# Join v1 columns:
#categoricalColumns = ["ORIGIN", "OP_UNIQUE_CARRIER", "TAIL_NUM", "ORIGIN_STATE_ABR", "DEST_STATE_ABR"]

categoricalColumns = ['ORIGIN', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_STATE_ABR',  'DEST_AIRPORT_SEQ_ID', 'DEST_STATE_ABR', 'CRS_DEP_TIME']
#categoricalColumns = ['ORIGIN', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_STATE_ABR',  'DEST_AIRPORT_SEQ_ID', 'DEST_STATE_ABR', 'CRS_DEP_TIME', 'YEAR']
# Features Not Included: DEP_DATETIME_LAG, 'CRS_ELAPSED_TIME', 'DISTANCE','DEP_DATETIME','DATE','ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyPressureChange', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'HourlyWindGustSpeed', 'MonthlyMeanTemperature', 'MonthlyMaximumTemperature', 'MonthlyGreatestSnowDepth', 'MonthlyGreatestSnowfall', 'MonthlyTotalSnowfall', 'MonthlyTotalLiquidPrecipitation', 'MonthlyMinimumTemperature', 'DATE_HOUR', 'distance_to_neighbor', 'neighbor_lat', 'neighbor_lon', 'time_zone_id', 'UTC_DEP_DATETIME_LAG', 'UTC_DEP_DATETIME', DEP_DEL15,  'flight_id', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID','ORIGIN_WAC', 'DEST_WAC', 'CANCELLED', 'CANCELLATION_CODE', 'SOURCE'

# Is including this data leakage? 'DEP_TIME', 'DEP_HOUR', 

stages = [] # stages in Pipeline

# NOTE: Had to cut out a bunch of features due to the sheer number of NULLS in them, which were causing the entire dataframe to be skipped. Will need to get the Null values either filled or dropped.

for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index").setHandleInvalid("skip")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
#        
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
#    
#print(stages)


# COMMAND ----------

# Create vectors for numeric and categorical variables

# Join v2 columns:
#numericCols = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_WAC", "DEP_TIME", "CANCELLED", "CRS_ELAPSED_TIME", "DISTANCE", "YEAR", "ELEVATION", "SOURCE", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlyVisibility", "HourlyWindSpeed", "distance_to_neighbor"]

numericCols = ['CRS_ELAPSED_TIME', 'DISTANCE','ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed']
# Features Not Included: 'DEP_DATETIME','DATE', 'HourlyWindGustSpeed', 'MonthlyMeanTemperature', 'MonthlyMaximumTemperature', 'MonthlyGreatestSnowDepth', 'MonthlyGreatestSnowfall', 'MonthlyTotalSnowfall', 'MonthlyTotalLiquidPrecipitation', 'MonthlyMinimumTemperature', 'DATE_HOUR', 'time_zone_id', 'UTC_DEP_DATETIME_LAG', 'UTC_DEP_DATETIME', 'HourlyPressureChange', 'distance_to_neighbor', 'neighbor_lat', 'neighbor_lon'

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("skip")

stages += [assembler]

#print(stages)

# COMMAND ----------

# Takes about 4 minutes for Full

# Run the pipeline
partialPipeline = Pipeline().setStages(stages)

# Run 3 Month
#pipelineModel = partialPipeline.fit(df_joined_data_3m)
#preppedDataDF = pipelineModel.transform(df_joined_data_3m)

# Run 2 Year
pipelineModel = partialPipeline.fit(df_joined_data_2y)
preppedDataDF = pipelineModel.transform(df_joined_data_2y)

# Run Full Time
#pipelineModel = partialPipeline.fit(df_joined_data_all)
#preppedDataDF = pipelineModel.transform(df_joined_data_all)

# Run Test Set of 2017 Data
df_joined_data_2017 = df_joined_data_all.filter(col("YEAR") == 2017)
pipelineModel_2017 = partialPipeline.fit(df_joined_data_2017)
preppedDataDF_2017 = pipelineModel_2017.transform(df_joined_data_2017)

display(preppedDataDF)

# COMMAND ----------

# Takes about 30 minutes for Full

#totalFeatures = [*categoricalColumns, *numericCols]
#print(categoricalColumns, "\n")
#print(numericCols, "\n")
#print(totalFeatures, "\n")

# Fit model to prepped data
lrModel = LogisticRegression(featuresCol = "features", labelCol = "DEP_DEL15").fit(preppedDataDF)

# ROC for training data
display(lrModel, preppedDataDF, "ROC")

# COMMAND ----------

display(lrModel, preppedDataDF)

# COMMAND ----------

def createLinearRegressionModel(num_iterations, trainingData_input):
    lr = LogisticRegression(num_iterations, labelCol="DEP_DEL15", featuresCol="features")
    lrModel = lr.fit(trainingData_input)
    return lrModel

def reportMetrics(inputMetrics):
    print("Precision = %s" % inputMetrics.precision(1))
    print("F1 = %s" % inputMetrics.fMeasure(1.0,1.0))
    print("Recall = %s" % inputMetrics.recall(1))
    print("Accuracy = %s" % inputMetrics.accuracy)

def runLogisticRegression(linearRegressionModel, testData):
    predictions = linearRegressionModel.transform(testData)
#    selected = predictions.select("DEP_DEL15", "prediction", "probability")
#    display(selected)
    metrics = MulticlassMetrics(predictions.select("DEP_DEL15", "prediction").rdd)
    
    return metrics
    
def runBlockingTimeSeriesCrossValidation(dataFrameInput):
    
    topMetrics = None
    topYear = None
    topModel = None
    
    listOfYears = dataFrameInput.select("YEAR").distinct().rdd.flatMap(list).collect()
    print("listOfYears:", listOfYears)

    for year in listOfYears:
        currentYear = year

        currentYearDF = dataFrameInput.filter(col("YEAR") == currentYear).cache()

        preppedDataDF = currentYearDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))

#        display(preppedDataDF)

        selectedcols = ["DEP_DEL15", "YEAR", "QUARTER", "DEP_DATETIME_LAG_percent", "features"]

        dataset = preppedDataDF.select(selectedcols).cache()

#        display(dataset)
        
        trainingData = dataset.filter(col("DEP_DATETIME_LAG_percent") <= .70)
        trainingTestData = dataset.filter(col("DEP_DATETIME_LAG_percent") > .70)
        display(trainingTestData)
        lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", maxIter=10)

        lrModel = lr.fit(trainingData)

        currentYearMetrics = runLogisticRegression(lrModel, trainingTestData)
        
        print(f"\n - {year}")
        print("trainingData Count:", trainingData.count())
        print("trainingTestData Count:", trainingTestData.count())
#        print("categoricalColumns =", categoricalColumns)
#        print("numericalCols =", numericCols)
        reportMetrics(currentYearMetrics)
        
        if topMetrics == None:
            topMetrics = currentYearMetrics
            topYear = year
            topModel = lrModel
        else:
            if currentYearMetrics.precision(1) > topMetrics.precision(1):
                topMetrics = currentYearMetrics
                topYear = year
                topModel = lrModel
    
    print("\n** Best Metrics **")
    print("topYear = %s" % topYear)
    print("categoricalColumns =", categoricalColumns)
    print("numericalCols =", numericCols)
    reportMetrics(topMetrics)
    print("Top Model:", topModel)
    
    return topModel

# COMMAND ----------

# 8 minutes for 2015, 2016
linearRegressionTopModel = runBlockingTimeSeriesCrossValidation(preppedDataDF)

# COMMAND ----------

preppedDataDF_2017_withPercent = preppedDataDF_2017.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))

#        display(preppedDataDF)

selectedcols = ["DEP_DEL15", "YEAR", "QUARTER", "DEP_DATETIME_LAG_percent", "features"]

testDataset_2017 = preppedDataDF_2017_withPercent.select(selectedcols).cache()

display(testDataset_2017)

#testDataMetrics = runLogisticRegression(linearRegressionTopModel, testDataset_2017)

# Keeps erroring out. Maybe removal of year category caused the vector sizes to differ?
#SparkException: [FAILED_EXECUTE_UDF] Failed to execute user defined function (ProbabilisticClassificationModel$$Lambda$8686/1058826222: (struct<type:tinyint,size:int,indices:array<int>,values:array<double>>) => struct<type:tinyint,size:int,indices:array<int>,values:array<double>>)
#Caused by: IllegalArgumentException: requirement failed: BLAS.dot(x: Vector, y:Vector) was given Vectors with non-matching sizes: x.size = 14918, y.size = 15596
#org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 1491.0 failed 4 times, most recent failure: Lost task 0.3 in stage 1491.0 (TID 5927) (10.139.64.49 executor 71): org.apache.spark.SparkException: [FAILED_EXECUTE_UDF] Failed to execute user defined function (ProbabilisticClassificationModel$$Lambda$8686/1058826222: (struct<type:tinyint,size:int,indices:array<int>,values:array<double>>) => struct<type:tinyint,size:int,indices:array<int>,values:array<double>>)

predictions = linearRegressionTopModel.transform(testDataset_2017)
#    selected = predictions.select("DEP_DEL15", "prediction", "probability")
display(predictions)
#testDataMetrics = MulticlassMetrics(predictions.select("DEP_DEL15", "prediction").rdd)

#reportMetrics(testDataMetrics)

# COMMAND ----------

listOfYears = df_joined_data_2y.select("YEAR").distinct().rdd.flatMap(list).collect()
print(listOfYears)

# COMMAND ----------

# 13 minutes for 2 years

currentYear = listOfYears[0]

currentYearDF = preppedDataDF.filter(col("YEAR") == currentYear).cache()

preppedDataDF = currentYearDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))

display(preppedDataDF)

selectedcols = ["DEP_DEL15", "YEAR", "QUARTER", "DEP_DATETIME_LAG_percent", "features"]

dataset = preppedDataDF.select(selectedcols)

display(dataset)

# COMMAND ----------

# Took 12 minutes for 1 year

# Training set of last thirty %.
trainingData = dataset.filter(col("DEP_DATETIME_LAG_percent") <= .70)
trainingTestData = dataset.filter(col("DEP_DATETIME_LAG_percent") > .70)

#trainingData = dataset.filter(col("QUARTER") != 4)
#testData = dataset.filter(col("QUARTER") == 4)

print("trainingData Count:", trainingData.count())
print("trainingTestData Count:", trainingTestData.count())

#display(trainingData)

# COMMAND ----------

# Takes about 32 minutes for Full

lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", maxIter=10)

# Train model with Training Data

lrModel = lr.fit(trainingData)

#linearRegressionModel_Training = createLinearRegressionModel(trainingData, maxIter=10)
runLogisticRegression(lrModel, trainingTestData)
#runLogisticRegression(lrModel, testData)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Other Code

# COMMAND ----------

# Takes about 19 Minutes for Full

preppedDataDF = preppedDataDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))

display(preppedDataDF)

selectedcols = ["DEP_DEL15", "YEAR", "QUARTER", "DEP_DATETIME_LAG_percent", "features"]

dataset = preppedDataDF.select(selectedcols)

display(dataset)

# COMMAND ----------

# Takes about 9 minutes for Full

#(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)

# Training set of last thirty %.
trainingData = dataset.filter(col("YEAR") < 2021).filter(col("DEP_DATETIME_LAG_percent") <= .70)
trainingTestData = dataset.filter(col("YEAR") < 2021).filter(col("DEP_DATETIME_LAG_percent") > .70)
testData = dataset.filter(col("YEAR") == 2021)

#trainingData = dataset.filter(col("QUARTER") != 4)
#testData = dataset.filter(col("QUARTER") == 4)

print(trainingData.count())

print(testData.count())

display(trainingData)

# COMMAND ----------

# Takes about 32 minutes for Full

lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", maxIter=10)

# Train model with Training Data

lrModel = lr.fit(trainingData)

#linearRegressionModel_Training = createLinearRegressionModel(trainingData, maxIter=10)
runLogisticRegression(lrModel, trainingTestData)
#runLogisticRegression(lrModel, testData)

# COMMAND ----------

#display(lrModel.transform(trainingTestData).select("DEP_DEL15", "prediction", "probability"))

# COMMAND ----------

# Takes about 13 minutes for Full

# Logistic Regression
# Create initial LogisticRegression model

lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", maxIter=10)

# Train model with Training Data

lrModel = lr.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the transform() method.

# LogisticRegression.transform() will only use the 'features' column.

predictions = lrModel.transform(testData)

# COMMAND ----------

# View model's predictions and probabilities of each prediction class

# You can select any columns in the above schema to view as well

selected = predictions.select("DEP_DEL15", "prediction", "probability")

display(selected)

# COMMAND ----------

metrics = MulticlassMetrics(predictions.select("DEP_DEL15", "prediction").rdd)

# Computed using MulticlassMetrics

print("categoricalColumns =", categoricalColumns)
print("numericalCols =", numericCols)
print("Precision = %s" % metrics.precision(1))
print("Recall = %s" % metrics.recall(1))
print("Accuracy = %s" % metrics.accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC v2 Data Set
# MAGIC 
# MAGIC Q1-4 Averages:
# MAGIC Precision = 0.04045
# MAGIC Recall = 0.4993
# MAGIC Accuracy = 0.815275
# MAGIC 
# MAGIC Standard random 70-30 split
# MAGIC Precision = 0.03550097100701333
# MAGIC Recall = 0.5323700456294176
# MAGIC Accuracy = 0.8167548729588924
# MAGIC 
# MAGIC Percentage 70-30 split
# MAGIC Precision = 0.03555337944803945
# MAGIC Recall = 0.5082754984010343
# MAGIC Accuracy = 0.8156185988403839
# MAGIC 
# MAGIC v2 Data Set
# MAGIC 
# MAGIC 70-30 percentage split
# MAGIC categoricalColumns = ['ORIGIN', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN_STATE_ABR', 'DEST_STATE_ABR']
# MAGIC numericalCols = ['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_WAC', 'DEP_TIME', 'CANCELLED', 'CRS_ELAPSED_TIME', 'DISTANCE', 'YEAR', 'ELEVATION', 'SOURCE', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'distance_to_neighbor']
# MAGIC Precision = 0.03523330919916401
# MAGIC Recall = 0.5101780622485635
# MAGIC Accuracy = 0.8156644749442552
# MAGIC 
# MAGIC 70-30 % split with complete set of new variables
# MAGIC categoricalColumns = ['ORIGIN', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_STATE_ABR', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_STATE_ABR', 'DEST_WAC', 'CRS_DEP_TIME', 'DEP_TIME', 'CANCELLED', 'CANCELLATION_CODE', 'YEAR', 'DEP_HOUR', 'SOURCE', 'flight_id']
# MAGIC numericalCols = ['CRS_ELAPSED_TIME', 'DISTANCE', 'ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyPressureChange', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'distance_to_neighbor', 'neighbor_lat', 'neighbor_lon']
# MAGIC Precision = 0.6197107659346546
# MAGIC Recall = 0.5882053889171327
# MAGIC Accuracy = 0.5623380362798733
# MAGIC 
# MAGIC 70-30 % split with refined variables
# MAGIC categoricalColumns = ['ORIGIN', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_STATE_ABR', 'DEST_AIRPORT_SEQ_ID', 'DEST_STATE_ABR', 'CRS_DEP_TIME', 'YEAR']
# MAGIC numericalCols = ['CRS_ELAPSED_TIME', 'DISTANCE', 'ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed']
# MAGIC Precision = 0.01718794406241851
# MAGIC Recall = 0.5598992135603573
# MAGIC Accuracy = 0.8082395453515033

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Old Code Below

# COMMAND ----------

# Evaluate model

evaluator = BinaryClassificationEvaluator(labelCol = "DEP_DEL15")

evaluator.evaluate(predictions)

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

print(lr.explainParams())

# COMMAND ----------

# Create ParamGrid for Cross Validation

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
# This one takes a few minutes...

cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=2)

# Run cross validations

cvModel = cv.fit(trainingData)

# this will likely take a fair amount of time because of the amount of models that we're creating and testing

# COMMAND ----------

# Use the test set to measure the accuracy of the model on new data

predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation

# Evaluate best model

evaluator.evaluate(predictions)

# COMMAND ----------

print('Model Intercept: ', cvModel.bestModel.intercept)

# COMMAND ----------

weights = cvModel.bestModel.coefficients

weights = [(float(w),) for w in weights]  # convert numpy type to float, and to tuple

weightsDF = spark.createDataFrame(weights, ["Feature Weight"])

display(weightsDF)

# COMMAND ----------

# View best model's predictions and probabilities of each prediction class

#predictions = predictions.withColumn("label", predictions["DEP_DEL15"])

#selected = predictions.select("label", "prediction", "probability")
selected = predictions.select("DEP_DEL15", "prediction", "probability")

display(selected)

# COMMAND ----------

# Metric Evaluation
metrics = MulticlassMetrics(predictions.select("DEP_DEL15", "prediction").rdd)

# Computed using MulticlassMetrics

print("categoricalColumns =", categoricalColumns)
print("numericalCols =", numericCols)
print("Precision = %s" % metrics.precision(1))
print("Recall = %s" % metrics.recall(1))
print("Accuracy = %s" % metrics.accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Saved Results
# MAGIC #### 3 Months
# MAGIC 
# MAGIC 11/6/22: V1 Numerical values only:
# MAGIC numericCols = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_WAC", "DEP_TIME", "CANCELLED", "CRS_ELAPSED_TIME", "DISTANCE", "YEAR", "STATION", "ELEVATION", "SOURCE", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlyVisibility", "HourlyWindSpeed", "distance_to_neighbor"]
# MAGIC Precision = 0.04726737410292584
# MAGIC Recall = 0.6066761139977956
# MAGIC Accuracy = 0.7976539367365639
# MAGIC 
# MAGIC 11/7/22: V1 Categorical + V1 Numeric. CRS_DEP_TIME_stringIdx and DEP_TIME were not indexed.
# MAGIC categoricalColumns = ['ORIGIN', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN_STATE_ABR', 'DEST_STATE_ABR']
# MAGIC numericalCols = ['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_WAC', 'DEP_TIME', 'CANCELLED', 'CRS_ELAPSED_TIME', 'DISTANCE', 'YEAR', 'STATION', 'DATE', 'ELEVATION', 'SOURCE', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'DATE_HOUR', 'distance_to_neighbor', 'neighbor_call']
# MAGIC Precision = 0.10488814633726859
# MAGIC Recall = 0.5929578442947427
# MAGIC Accuracy = 0.800846964537787
# MAGIC 
# MAGIC 11/7/2022: V1 Categorical + V1 Numeric with string indexed categoricals.
# MAGIC categoricalColumns = ['ORIGIN', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN_STATE_ABR', 'DEST_STATE_ABR']
# MAGIC numericalCols = ['QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_WAC', 'DEP_TIME', 'CANCELLED', 'CRS_ELAPSED_TIME', 'DISTANCE', 'YEAR', 'STATION', 'ELEVATION', 'SOURCE', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'distance_to_neighbor']
# MAGIC Precision = 0.10630601910320389
# MAGIC Recall = 0.5898360432682496
# MAGIC Accuracy = 0.8004823858378799
# MAGIC 
# MAGIC #### Total Time

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Workspace

# COMMAND ----------

display(df_joined_data_3m)

# COMMAND ----------

display(preppedDataDF)

# COMMAND ----------

#numericCols = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_WAC", "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_WAC", "DEP_TIME", "CANCELLED", "CRS_ELAPSED_TIME", "DISTANCE", "YEAR", "STATION", "DATE", "ELEVATION", "SOURCE", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlyVisibility", "HourlyWindSpeed", "DATE_HOUR", "distance_to_neighbor", "neighbor_call"]

display(preppedDataDF.select(numericCols))

# COMMAND ----------

df2 = df_joined_data_3m.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in numericCols])
display(df2)

# COMMAND ----------

#["ORIGIN", "OP_UNIQUE_CARRIER", "TAIL_NUM", "ORIGIN_STATE_ABR", "DEST_STATE_ABR"]

df2 = df_joined_data_3m.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in categoricalColumns])
display(df2)


# COMMAND ----------

display(df_joined_data_3m.filter(col("TAIL_NUM").isNull()))

# COMMAND ----------

register_spark() # register spark backend

# COMMAND ----------

with parallel_backend('spark', n_jobs=3):
  scores = cross_val_score(clf, iris.data, iris.target, cv=5)

print(scores)

# COMMAND ----------

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from joblibspark import register_spark
from sklearn.utils import parallel_backend

register_spark() # register spark backend

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC(gamma='auto')

clf = GridSearchCV(svr, parameters, cv=5)

with parallel_backend('spark', n_jobs=3):
  scores = cross_val_score(clf, iris.data, iris.target, cv=5)


# COMMAND ----------

df_basket1 = df_joined_data_3m.select("HourlyDewPointTemperature", "HourlyDryBulbTemperature", F.percent_rank().over(Window.partitionBy().orderBy(df_joined_data_3m['HourlyDryBulbTemperature'])).alias("percent_rank"))
display(df_basket1)

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}datasets_final_project_2022/"))

# COMMAND ----------

df_test = spark.read.parquet(f"{data_BASE_DIR}datasets_final_project_2022/parquet_airlines_data/")
display(df_test.filter(col("YEAR") == 2021))

# COMMAND ----------

display(df_joined_data_all.filter(col("YEAR") == 2020))

# COMMAND ----------

features = ['ORIGIN', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_STATE_ABR',  'DEST_AIRPORT_SEQ_ID', 'DEST_STATE_ABR', 'CRS_DEP_TIME', 'YEAR', 'CRS_ELAPSED_TIME', 'DISTANCE','ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed']

df2 = df_joined_data_all.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in features])

df2.show()

# COMMAND ----------


