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
from pyspark.ml.feature import StandardScaler

from sklearn.utils import parallel_backend
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
from joblibspark import register_spark

from pyspark.ml.classification import RandomForestClassifier

import pandas as pd
import numpy as np

from datetime import datetime

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

def getCurrentDateTimeFormatted():
    return str(datetime.utcnow()).replace(" ", "-").replace(":", "-").replace(".", "-")

def resetMetricsToAzure_LR():
    backup_metrics = spark.read.parquet(f"{blob_url}/logistic_regression_metrics")
    backup_date_string = getCurrentDateTimeFormatted()
    backup_metrics.write.parquet(f"{blob_url}/metrics_backups/logistic_regression_metrics-{backup_date_string}")
    
    columns = ["date_time","precision", "f0.5", "recall", "accuracy", "regParam", "elasticNetParam", "maxIter", "threshold"]
    data = [(datetime.utcnow(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)]
    rdd = spark.sparkContext.parallelize(data)
    dfFromRDD = rdd.toDF(columns)
    
    dfFromRDD.write.mode('overwrite').parquet(f"{blob_url}/logistic_regression_metrics")
    print("LR Metrics Reset")

def saveMetricsToAzure_LR(input_model, input_metrics):
    columns = ["date_time","precision", "f0.5", "recall", "accuracy", "regParam", "elasticNetParam", "maxIter", "threshold"]
    data = [(datetime.utcnow(), input_metrics.precision(1), input_metrics.fMeasure(label = 1.0, beta = 0.5), \
             input_metrics.recall(1), input_metrics.accuracy, input_model.getRegParam(), \
             input_model.getElasticNetParam(), input_model.getMaxIter(), input_model.getThreshold())]
    rdd = spark.sparkContext.parallelize(data)
    dfFromRDD = rdd.toDF(columns)
    
    dfFromRDD.write.mode('append').parquet(f"{blob_url}/logistic_regression_metrics")
    print("LR Metrics Saved Successfully!")
    
def resetMetricsToAzure_RF():
    backup_metrics = spark.read.parquet(f"{blob_url}/random_forest_metrics")
    backup_date_string = getCurrentDateTimeFormatted()
    backup_metrics.write.parquet(f"{blob_url}/metrics_backups/random_forest_metrics-{backup_date_string}")
    
    columns = ["date_time","precision", "f0.5", "recall", "accuracy", "numTrees", "maxDepth", "maxBins"]
    data = [(datetime.utcnow(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)]
    rdd = spark.sparkContext.parallelize(data)
    dfFromRDD = rdd.toDF(columns)
    
    dfFromRDD.write.mode('overwrite').parquet(f"{blob_url}/random_forest_metrics")
    print("RF Metrics Reset")

def saveMetricsToAzure_RF(input_model, input_precision, input_fPointFive, input_recall, input_accuracy):
    columns = ["date_time","precision", "f0.5", "recall", "accuracy", "numTrees", "maxDepth", "maxBins"]
    data = [(datetime.utcnow(), input_precision, input_fPointFive, \
             input_recall, input_accuracy, input_model.numTrees(), \
             input_model.maxDepth(), input_model.maxBins())]
    rdd = spark.sparkContext.parallelize(data)
    dfFromRDD = rdd.toDF(columns)
    
    dfFromRDD.write.mode('append').parquet(f"{blob_url}/random_forest_metrics")
    print("RF Metrics Saved Successfully!")

    
# WARNING: Will Delete Current Metrics for Logistic Regression
#resetMetricsToAzure_LR()
# WARNING: Will Delete Current Metrics for Logistic Regression


# WARNING: Will Delete Current Metrics for Random Forest
#resetMetricsToAzure_RF()
# WARNING: Will Delete Current Metricsfor Random Forest

# COMMAND ----------

# Load Dataframes
print("**Loading Data")

# Inspect the Joined Data folders 
display(dbutils.fs.ls(f"{blob_url}"))

print("**Data Loaded")
print("**Loading Data Frames")

#df_joined_data_3m = spark.read.parquet(f"{blob_url}/joined_data_3m")
#display(df_joined_data_3m)

df_joined_data_all = spark.read.parquet(f"{blob_url}/joined_data_all")
display(df_joined_data_all)

#df_joined_data_2y = df_joined_data_all.filter(col("YEAR") <= 2016).cache()
#display(df_joined_data_2y)

df_joined_data_pre2021 = df_joined_data_all.filter(col("YEAR") < 2021).cache()
display(df_joined_data_pre2021)

df_joined_data_2021 = df_joined_data_all.filter(col("YEAR") == 2021)
display(df_joined_data_2021)

print("**Data Frames Loaded")

# COMMAND ----------

# Data cleaning Tasks

# Fills in NA values for cancelled flights.

#df_joined_data_3m = df_joined_data_3m.na.fill(value = 1,subset=["DEP_DEL15"])
#df_joined_data_2y = df_joined_data_2y.na.fill(value = 1,subset=["DEP_DEL15"])
df_joined_data_all = df_joined_data_all.na.fill(value = 1,subset=["DEP_DEL15"])
df_joined_data_pre2021 = df_joined_data_pre2021.na.fill(value = 1,subset=["DEP_DEL15"])
df_joined_data_2021 = df_joined_data_2021.na.fill(value = 1,subset=["DEP_DEL15"])
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

categoricalColumns = ['ORIGIN', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_STATE_ABR',  'DEST_AIRPORT_SEQ_ID', 'DEST_STATE_ABR', 'CRS_DEP_TIME', 'YEAR']
# Features Not Included: DEP_DATETIME_LAG, 'CRS_ELAPSED_TIME', 'DISTANCE','DEP_DATETIME','DATE','ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyPressureChange', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'HourlyWindGustSpeed', 'MonthlyMeanTemperature', 'MonthlyMaximumTemperature', 'MonthlyGreatestSnowDepth', 'MonthlyGreatestSnowfall', 'MonthlyTotalSnowfall', 'MonthlyTotalLiquidPrecipitation', 'MonthlyMinimumTemperature', 'DATE_HOUR', 'distance_to_neighbor', 'neighbor_lat', 'neighbor_lon', 'time_zone_id', 'UTC_DEP_DATETIME_LAG', 'UTC_DEP_DATETIME', DEP_DEL15, 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID','ORIGIN_WAC', 'DEST_WAC', 'CANCELLED', 'CANCELLATION_CODE', 'SOURCE'

# Could not use , 'flight_id'. Leads to buffer overflow error.
# org.apache.spark.SparkException: Job aborted due to stage failure: Task 2 in stage 57.0 failed 4 times, most recent failure: Lost task 2.3 in stage 57.0 (TID 280) (10.139.64.6 executor 0): org.apache.spark.SparkException: Kryo serialization failed: Buffer overflow. Available: 0, required: 31

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
numericCols = ['CRS_ELAPSED_TIME', 'DISTANCE','ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed']
# Features Not Included: 'DEP_DATETIME','DATE', 'HourlyWindGustSpeed', 'MonthlyMeanTemperature', 'MonthlyMaximumTemperature', 'MonthlyGreatestSnowDepth', 'MonthlyGreatestSnowfall', 'MonthlyTotalSnowfall', 'MonthlyTotalLiquidPrecipitation', 'MonthlyMinimumTemperature', 'DATE_HOUR', 'time_zone_id', 'UTC_DEP_DATETIME_LAG', 'UTC_DEP_DATETIME', 'HourlyPressureChange', 'distance_to_neighbor', 'neighbor_lat', 'neighbor_lon'

# scaler = StandardScaler(inputCol=numericCols, outputCol="scaledFeatures", withStd=True, withMean=False)

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

# Adds Features vector to data frames as part of pipeline.
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("skip")

stages += [assembler]

#print(stages)

# COMMAND ----------

# Takes about 4 minutes for Full

# Create the pipeline to be applied to the dataframes
partialPipeline = Pipeline().setStages(stages)

# Apply pipeline to 3 Month
#pipelineModel = partialPipeline.fit(df_joined_data_3m)
#preppedDataDF = pipelineModel.transform(df_joined_data_3m)

# Apply pipeline to 2 Year
#pipelineModel = partialPipeline.fit(df_joined_data_2y)
#preppedDataDF = pipelineModel.transform(df_joined_data_2y)

# Apply pipeline to Full Time
pipelineModel = partialPipeline.fit(df_joined_data_all)
preppedDataDF = pipelineModel.transform(df_joined_data_all).cache()

# Apply pipeline to Pre-2021
#pipelineModel_pre2021 = partialPipeline.fit(df_joined_data_pre2021)
#preppedDataDF_pre2021 = pipelineModel_pre2021.transform(df_joined_data_pre2021)

# Apply pipeline to 2021
#pipelineModel_2021 = partialPipeline.fit(df_joined_data_2021)
#preppedDataDF_2021 = pipelineModel_2021.transform(df_joined_data_2021)

#display(preppedDataDF)

# COMMAND ----------

#display(preppedDataDF)

# COMMAND ----------

# Takes about 30 minutes for Full

#Displays ROC graph

#totalFeatures = [*categoricalColumns, *numericCols]
#print(categoricalColumns, "\n")
#print(numericCols, "\n")
#print(totalFeatures, "\n")

# Fit model to prepped data
#lrModel = LogisticRegression(featuresCol = "features", labelCol = "DEP_DEL15").fit(preppedDataDF)

# ROC for training data
#display(lrModel, preppedDataDF, "ROC")

# COMMAND ----------

#display(lrModel, preppedDataDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Logistic Regression

# COMMAND ----------

def createLogisticRegressionModel(trainingData_input, maxIter_input):
    """
    Creates Logistic Regression model trained on trainingData_input.
    """
    lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", maxIter = maxIter_input)
    lrModel = lr.fit(trainingData_input)
    return lrModel

def reportMetrics(inputMetrics):
    """Outputs metrics (currently only for Linear Regression?)"""
    print("Precision = %s" % inputMetrics.precision(1))
    print("F0.5 = %s" % inputMetrics.fMeasure(label = 1.0, beta = 0.5))
    print("Recall = %s" % inputMetrics.recall(1))
    print("Accuracy = %s" % inputMetrics.accuracy)

def runLogisticRegression(linearRegressionModel, testData):
    """
    Applies a logistic regression model to the test data provided, and return the metrics from the test evaluation.
    Realize now that the model input can be any model, and does not necessarily need to be logistic regression.
    Maybe try using with other models?
    """
    predictions = linearRegressionModel.transform(testData)
#    selected = predictions.select("DEP_DEL15", "prediction", "probability")
#    display(selected)
    metrics = MulticlassMetrics(predictions.select("DEP_DEL15", "prediction").rdd)
    
    return metrics
    
def runBlockingTimeSeriesCrossValidation(dataFrameInput, regParam_input, elasticNetParam_input, maxIter_input, threshold_input):
    """
    Conducts the Blocking Time Series Cross Validation.
    Accepts the full dataFrame of all years. 
    Is hard coded to use pre-2021 data as training data, which it will cross validate against.
    After all cross validations, will select best model from each year, and then apply the test 2021 data against it for final evaluation.
    Prints metrics from final test evaluation at the end.
    """
    print(f"\n@ Starting runBlockingTimeSeriesCrossValidation")
    print(f"@ {regParam_input}, {elasticNetParam_input}, {maxIter_input}, {threshold_input}")
    print(f"@ {getCurrentDateTimeFormatted()}")
    topMetrics = None
    topYear = None
    topModel = None
    
    # list all of the years that the data will be trained against.
    listOfYears = dataFrameInput.select("YEAR").distinct().filter(col("YEAR") != 2021).rdd.flatMap(list).collect()
    print("listOfYears:", listOfYears)

    # Iterate through each of the individual years in the training data set.
    for year in listOfYears:
        
        currentYear = year
        print(f"Processing Year: {currentYear}")
        print(f"@ {getCurrentDateTimeFormatted()}")
        currentYearDF = dataFrameInput.filter(col("YEAR") == currentYear).cache()
        
        # Upscale the data such that there are roughly equal amounts of rows where DEP_DEL15 == 0 and DEP_DEL15 == 1, which aids in training.
        
        currentYearDF_upsampling_0 = currentYearDF.filter(col("DEP_DEL15") == 0)
        print(f"@- df_testData0.count() = {currentYearDF_upsampling_0.count()}")
        currentYearDF_upsampling_1 = currentYearDF.filter(col("DEP_DEL15") == 1)
        print(f"@- df_testData1.count() = {currentYearDF_upsampling_1.count()}")

        upsampling_ratio = (currentYearDF_upsampling_0.count() / currentYearDF_upsampling_1.count()) - 1

        currentYearDF_upsampling_append = currentYearDF.filter(col("DEP_DEL15") == 1).sample(fraction = upsampling_ratio, withReplacement = True, seed = 261)
        print(f"@- currentYearDF_upsampling_append.count() = {currentYearDF_upsampling_append.count()}")

        currentYearDF_upsampled = currentYearDF.unionAll(currentYearDF_upsampling_append)
        print(f"@- currentYearDF_upsampled.count() = {currentYearDF_upsampled.count()}")
        
        # Adds a percentage column to each year's data frame, with the percentage corresponding to percentage of the year's time. 
        # 0% = earliest time that year. 100% = latest time that year.
        preppedDataDF = currentYearDF_upsampled.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))

#        display(preppedDataDF)

        # remove unneeded columns. All feature values are captured in "features". All the other retained features are for row tracking.
        selectedcols = ["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]
        dataset = preppedDataDF.select(selectedcols).cache()

#        display(dataset)
        
        # The training set is the data from the 70% earliest data.
        # Test set is the latter 30% of the data.
        trainingData = dataset.filter(col("DEP_DATETIME_LAG_percent") <= .70)
        trainingTestData = dataset.filter(col("DEP_DATETIME_LAG_percent") > .70)
#        display(trainingTestData)
        
        # Create and train a logistic regression model for the year based on training data.
        # Note: createLinearRegressionModel() function would not work here for some reason.
        lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", regParam = regParam_input, elasticNetParam = elasticNetParam_input, maxIter = maxIter_input, threshold = threshold_input)
        lrModel = lr.fit(trainingData)

        currentYearMetrics = runLogisticRegression(lrModel, trainingTestData)
        
        # Print the year's data and evaluation metrics
#        print(f"\n - {year}")
#        print("trainingData Count:", trainingData.count())
#        print("trainingTestData Count:", trainingTestData.count())
#        print("categoricalColumns =", categoricalColumns)
#        print("numericalCols =", numericCols)
#        reportMetrics(currentYearMetrics)
        
        # Compare and store top models and metrics.
        if topMetrics == None:
            topMetrics = currentYearMetrics
            topYear = year
            topModel = lrModel
        else:
            if currentYearMetrics.precision(1) > topMetrics.precision(1):
                topMetrics = currentYearMetrics
                topYear = year
                topModel = lrModel
    
    # TODO: Ensemble models across all years?
    
    # Print the metrics of the best model from the cross validated years.
#    print("\n** Best Metrics **")
#    print("topYear = %s" % topYear)
#    print("categoricalColumns =", categoricalColumns)
#    print("numericalCols =", numericCols)
#    reportMetrics(topMetrics)
#    print("Top Model:", topModel)
    

    print(f"@ Starting Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    # Prepare 2021 Test Data
    currentYearDF = dataFrameInput.filter(col("YEAR") == 2021).cache()
    preppedDataDF = currentYearDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))
#        display(preppedDataDF)
    selectedcols = ["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]
    dataset = preppedDataDF.select(selectedcols).cache()
#        display(dataset)

    # Evaluate best model from cross validation against the test data frame of 2021 data, then print evaluation metrics.
    testDataSet = dataset
#    display(testDataSet)
    lr = topModel
    testMetrics = runLogisticRegression(topModel, testDataSet)

    print(f"\n - 2021")
    print("Model Parameters:")
    print("regParam:", lr.getRegParam())
    print("elasticNetParam:", lr.getElasticNetParam())
    print("maxIter:", lr.getMaxIter())
    print("threshold:", lr.getThreshold())
#    print("Weights Col:", lr.getWeightCol())
    print("testDataSet Count:", testDataSet.count())
    reportMetrics(testMetrics)
    saveMetricsToAzure_LR(lr, testMetrics)
    print(f"@ Finised Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    

# COMMAND ----------

# Hyperparameter Tuning Parameter Grid
# Each CV takes one hour. Do the math.

#regParamGrid = [0.0, 0.01, 0.5, 2.0]
#elasticNetParamGrid = [0.0, 0.5, 1.0]
#maxIterGrid = [1, 5, 10]

regParamGrid = [0.0]
elasticNetParamGrid = [0]
maxIterGrid = [10]
thresholdGrid = [0.5]

for regParam in regParamGrid:
    print(f"! regParam = {regParam}")
    for elasticNetParam in elasticNetParamGrid:
        print(f"! elasticNetParam = {elasticNetParam}")
        for maxIter in maxIterGrid:
            print(f"! maxIter = {maxIter}")
            for threshold in thresholdGrid:
                print(f"! threshold = {threshold}")
                runBlockingTimeSeriesCrossValidation(preppedDataDF, regParam, elasticNetParam, maxIter, threshold)
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")


# COMMAND ----------

current_metrics = spark.read.parquet(f"{blob_url}/logistic_regression_metrics")
display(current_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Logistic Regression Results
# MAGIC 
# MAGIC 11/16/22
# MAGIC 55 minutes
# MAGIC regParam = 0.0
# MAGIC elasticNetParam = 0.0
# MAGIC maxIter = 10
# MAGIC threshold = .5
# MAGIC testDataSet Count: 5771706
# MAGIC Precision = 0.029235971681133347
# MAGIC F1 = 0.05415838783534103
# MAGIC Recall = 0.36706913430609295
# MAGIC Accuracy = 0.80939517709322
# MAGIC 
# MAGIC 11/20/22
# MAGIC regParam = .01
# MAGIC elasticNetParam = 0.0
# MAGIC maxIter = 1
# MAGIC threshold = .5
# MAGIC testDataSet Count: 5771706
# MAGIC Precision = 0.013125369557125526
# MAGIC F1 = 0.025499071285593207
# MAGIC Recall = 0.4452561639953396
# MAGIC Accuracy = 0.8127451398252094
# MAGIC 
# MAGIC 11/21/22
# MAGIC Model Parameters:
# MAGIC regParam: 0.0
# MAGIC elasticNetParam: 0.0
# MAGIC maxIter: 10
# MAGIC threshold: 0.3
# MAGIC testDataSet Count: 5771706
# MAGIC Precision = 0.2542970733396268
# MAGIC F1 = 0.27699824117668764
# MAGIC Recall = 0.3041497728493967
# MAGIC Accuracy = 0.7522193611386304

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Random Forest

# COMMAND ----------

def reportMetrics_rf(precision, recall, accuracy):
    """Outputs metrics (currently only for Linear Regression?)"""
    print("Precision = %s" % precision)
#    print("F0.5 = %s" % inputMetrics.fMeasure(label = 1.0, beta = 0.5))
    print("Recall = %s" % recall)
    print("Accuracy = %s" % accuracy)

def runRandomForest(randomForestModel, testData):
    """
    Applies a logistic regression model to the test data provided, and return the metrics from the test evaluation.
    Realize now that the model input can be any model, and does not necessarily need to be logistic regression.
    Maybe try using with other models?
    """
    predictions = randomForestModel.transform(testData)
#    selected = predictions.select("DEP_DEL15", "prediction", "probability")
#    display(selected)

    TP = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 1) & (col("prediction") == 1)).count()
    print(f"TP = {TP}")
    TN = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 0) & (col("prediction") == 0)).count()
    print(f"TN = {TN}")
    FP = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 0) & (col("prediction") == 1)).count()
    print(f"FP = {FP}")
    FN = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 1) & (col("prediction") == 0)).count()
    print(f"FN = {FN}")

    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    accuracy = (TN + TP) / (TN + TP + FP + FN)
    
    fPointFive = ((1 + (0.5^2)) * precision * recall) / ((0.5^2) * precision + recall)
    
    return precision, fPointFive, recall, accuracy
    
def runBlockingTimeSeriesCrossValidation_rf(dataFrameInput, input_numTrees, input_maxDepth, input_maxBins):
    """
    Conducts the Blocking Time Series Cross Validation for RandomForest.
    Accepts the full dataFrame of all years. 
    Is hard coded to use pre-2021 data as training data, which it will cross validate against.
    After all cross validations, will select best model from each year, and then apply the test 2021 data against it for final evaluation.
    Prints metrics from final test evaluation at the end.
    """
    print(f"\n@ Starting runBlockingTimeSeriesCrossValidation_rf")
    print(f"@ {input_numTrees}, {input_maxDepth}, {input_maxBins}")
    print(f"@ {getCurrentDateTimeFormatted()}")
    
    topPrecision = None
    topYear = None
    topModel = None
    
    # list all of the years that the data will be trained against.
    listOfYears = dataFrameInput.select("YEAR").distinct().filter(col("YEAR") != 2021).rdd.flatMap(list).collect()
    print("listOfYears:", listOfYears)

    # Iterate through each of the individual years in the training data set.
    for year in listOfYears:
        
        currentYear = year
        print(f"Processing Year: {currentYear}")
        print(f"@ {getCurrentDateTimeFormatted()}")
        currentYearDF = dataFrameInput.filter(col("YEAR") == currentYear).cache()
        
        # Upscale the data such that there are roughly equal amounts of rows where DEP_DEL15 == 0 and DEP_DEL15 == 1, which aids in training.
        
        currentYearDF_upsampling_0 = currentYearDF.filter(col("DEP_DEL15") == 0)
        print(f"@- df_testData0.count() = {currentYearDF_upsampling_0.count()}")
        currentYearDF_upsampling_1 = currentYearDF.filter(col("DEP_DEL15") == 1)
        print(f"@- df_testData1.count() = {currentYearDF_upsampling_1.count()}")

        upsampling_ratio = (currentYearDF_upsampling_0.count() / currentYearDF_upsampling_1.count()) - 1

        currentYearDF_upsampling_append = currentYearDF.filter(col("DEP_DEL15") == 1).sample(fraction = upsampling_ratio, withReplacement = True, seed = 261)
        print(f"@- currentYearDF_upsampling_append.count() = {currentYearDF_upsampling_append.count()}")

        currentYearDF_upsampled = currentYearDF.unionAll(currentYearDF_upsampling_append)
        print(f"@- currentYearDF_upsampled.count() = {currentYearDF_upsampled.count()}")
        
        # Adds a percentage column to each year's data frame, with the percentage corresponding to percentage of the year's time. 
        # 0% = earliest time that year. 100% = latest time that year.
        preppedDataDF = currentYearDF_upsampled.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))

#        display(preppedDataDF)

        # remove unneeded columns. All feature values are captured in "features". All the other retained features are for row tracking.
        selectedcols = ["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]
        dataset = preppedDataDF.select(selectedcols).cache()

#        display(dataset)
        
        # The training set is the data from the 70% earliest data.
        # Test set is the latter 30% of the data.
        trainingData = dataset.filter(col("DEP_DATETIME_LAG_percent") <= .70)
        trainingTestData = dataset.filter(col("DEP_DATETIME_LAG_percent") > .70)
#        display(trainingTestData)
        
        # Create and train a logistic regression model for the year based on training data.
        # Note: createLinearRegressionModel() function would not work here for some reason.
        rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", numTrees = input_numTrees, maxDepth = input_maxDepth, maxBins = input_maxBins)
        rfModel = rf.fit(trainingData)

        precision, fPointFive, recall, accuracy = runRandomForest(rfModel, trainingTestData)
            
        
        # Print the year's data and evaluation metrics
#        print(f"\n - {year}")
#        print("trainingData Count:", trainingData.count())
#        print("trainingTestData Count:", trainingTestData.count())
#        print("categoricalColumns =", categoricalColumns)
#        print("numericalCols =", numericCols)
#        reportMetrics(currentYearMetrics)
        
        # Compare and store top models and metrics.
        if topMetrics == None:
            topPrecision = precision
            topYear = year
            topModel = rfModel
        else:
            if precision > topPrecision:
                topPrecision = precision
                topYear = year
                topModel = rfModel
    
    # TODO: Ensemble models across all years?
    
    # Print the metrics of the best model from the cross validated years.
#    print("\n** Best Metrics **")
#    print("topYear = %s" % topYear)
#    print("categoricalColumns =", categoricalColumns)
#    print("numericalCols =", numericCols)
#    reportMetrics(topMetrics)
#    print("Top Model:", topModel)
    

    print(f"@ Starting Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    # Prepare 2021 Test Data
    currentYearDF = dataFrameInput.filter(col("YEAR") == 2021).cache()
    preppedDataDF = currentYearDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))
#        display(preppedDataDF)
    selectedcols = ["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]
    dataset = preppedDataDF.select(selectedcols).cache()
#        display(dataset)

    # Evaluate best model from cross validation against the test data frame of 2021 data, then print evaluation metrics.
    testDataSet = dataset
#    display(testDataSet)
    precision, fPointFive, recall, accuracy = runRandomForest(topModel, testDataSet)

    print(f"\n - 2021")
    print("Model Parameters:")
    print("numTrees:", topModel.numTrees())
    print("maxDepth:", topModel.maxDepth())
    print("maxBins:", topModel.maxBins())
    print("testDataSet Count:", testDataSet.count())
    reportMetrics_rf(precision, fPointFive, recall, accuracy)
#    saveMetricsToAzure_LR(lr, testMetrics)
    print(f"@ Finised Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    

# COMMAND ----------

#numTrees = [10, 25, 50]
#maxDepth = [4, 8, 16]
#maxBins = [32, 64, 128]

numTreesGrid = [10]
maxDepthGrid = [4]
maxBinsGrid = [32]

for numTrees in numTreesGrid:
    print(f"! numTrees = {numTrees}")
    for maxDepth in maxDepthGrid:
        print(f"! maxDepth = {maxDepth}")
        for maxBins in maxBinsGrid:
            print(f"! maxBins = {maxBins}")
            runBlockingTimeSeriesCrossValidation_rf(preppedDataDF, numTrees, maxDepth, maxBins)
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

# COMMAND ----------

# Test Grid

testPreppedData_2017 = preppedDataDF.filter(col("YEAR") == 2017)
testPreppedData_2021 = preppedDataDF.filter(col("YEAR") == 2021)

testPreppedData = testPreppedData_2017.unionAll(testPreppedData_2021)


numTreesGrid = [10]
maxDepthGrid = [4]
maxBinsGrid = [32]

for numTrees in numTreesGrid:
    print(f"! numTrees = {numTrees}")
    for maxDepth in maxDepthGrid:
        print(f"! maxDepth = {maxDepth}")
        for maxBins in maxBinsGrid:
            print(f"! maxBins = {maxBins}")
            runBlockingTimeSeriesCrossValidation_rf(testPreppedData, numTrees, maxDepth, maxBins)
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

# COMMAND ----------

# # Logistic Regression
# # Create initial LogisticRegression model

# lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", maxIter=10)
rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", numTrees=10)
 
    
testPreppedData_2017 = preppedDataDF.filter(col("YEAR") == 2017)
testPreppedData_2021 = preppedDataDF.filter(col("YEAR") == 2021)
print("Data Loaded")

# # Train model with Training Data

rfModel = rf.fit(testPreppedData_2017)

predictions = rfModel.transform(testPreppedData_2021)

# View model's predictions and probabilities of each prediction class

# You can select any columns in the above schema to view as well

selected = predictions.select("DEP_DEL15", "prediction", "probability")

display(selected)

# Metric Evaluation
metrics = MulticlassMetrics(predictions.select("DEP_DEL15", "prediction").rdd)
#display(predictions.select("label", "prediction").rdd.collect())

TP = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 1) & (col("prediction") == 1)).count()
print(TP)
TN = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 0) & (col("prediction") == 0)).count()
print(TN)
FP = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 0) & (col("prediction") == 1)).count()
print(FP)
FN = predictions.select("DEP_DEL15", "prediction").filter((col("DEP_DEL15") == 1) & (col("prediction") == 0)).count()
print(FN)

if (TP + FP) == 0:
    precision = 0
else:
    precision = TP / (TP + FP)
    
if (TP + FN) == 0:
    recall = 0
else:
    recall = TP / (TP + FN)
    
accuracy = (TN + TP) / (TN + TP + FP + FN)

print("Manual Precision: ", precision)
print("Manual Recall: ", recall)
print("Manual Accuracy: ", accuracy)

#print("categoricalColumns =", categoricalColumns)
#print("numericalCols =", numericCols)
#print("Precision = %s" % metrics.precision(1))
#print("Recall = %s" % metrics.recall(1))
#print("Accuracy = %s" % metrics.accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Other Code
# MAGIC 
# MAGIC Holding onto because it will be useful later on, or might provide useful features.
# MAGIC 
# MAGIC BinaryClassificationEvaluator
# MAGIC 
# MAGIC GridSearch
# MAGIC 
# MAGIC Model Weights

# COMMAND ----------

# 22 minutes
testPreppedData_2017 = preppedDataDF.filter(col("YEAR") == 2017)
testPreppedData_2021 = preppedDataDF.filter(col("YEAR") == 2021)
print("Data Loaded")

# Testing Upsampling

currentYearDF_upsampling_0 = testPreppedData_2017.filter(col("DEP_DEL15") == 0)
print(f"df_testData0.count() = {currentYearDF_upsampling_0.count()}")

currentYearDF_upsampling_1 = testPreppedData_2017.filter(col("DEP_DEL15") == 1)
print(f"df_testData0.count() = {currentYearDF_upsampling_1.count()}")

upsampling_ratio = (currentYearDF_upsampling_0.count() / currentYearDF_upsampling_1.count()) - 1

currentYearDF_upsampling_append = testPreppedData_2017.filter(col("DEP_DEL15") == 1).sample(fraction = upsampling_ratio, withReplacement = True, seed = 261)
print(currentYearDF_upsampling_append.count())

testPreppedData_2017 = testPreppedData_2017.unionAll(currentYearDF_upsampling_append)
print(testPreppedData_2017.count())

testModel = createLogisticRegressionModel(testPreppedData_2017, 10)
print(testModel)
testMetrics = runLogisticRegression(testModel, testPreppedData_2021)
print(testMetrics)
reportMetrics(testMetrics)
#saveMetricsToAzure_LR(testModel, testMetrics)

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

# MAGIC %md
# MAGIC 
# MAGIC # Workspace

# COMMAND ----------

# Experimenting with upsampling to solve data imbalance

df_testData0 = df_joined_data_all.select("DEP_DEL15", "YEAR").filter(col("DEP_DEL15") == 0)
print(df_testData0.count())

df_testData1 = df_joined_data_all.select("DEP_DEL15", "YEAR").filter(col("DEP_DEL15") == 1)
print(df_testData1.count())

ratio = (df_testData0.count() / df_testData1.count()) - 1

df_upsampled = df_joined_data_all.select("DEP_DEL15", "YEAR").filter(col("DEP_DEL15") == 1).sample(fraction = ratio, withReplacement = True, seed=3)
print(df_upsampled.count())

df = df_testData1.unionAll(df_upsampled)
print(df.count())

# COMMAND ----------

pageRanks_PriorTo21 = spark.read.csv(f"{blob_url}/PageRanks_PriorTo21.csv", header=True)
display(pageRanks_PriorTo21)

pageRanks_Quarter_Year = spark.read.csv(f"{blob_url}/PageRanks_Quarter_Year.csv", header=True)
display(pageRanks_Quarter_Year)

# COMMAND ----------


