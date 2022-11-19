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

#df_joined_data_3m = spark.read.parquet(f"{blob_url}/joined_data_3m")
#display(df_joined_data_3m)

df_joined_data_all = spark.read.parquet(f"{blob_url}/joined_data_all")
display(df_joined_data_all)

#df_joined_data_2y = df_joined_data_all.filter(col("YEAR") <= 2016).cache()
#display(df_joined_data_2y)

#df_joined_data_pre2021 = df_joined_data_all.filter(col("YEAR") < 2021).cache()
#display(df_joined_data_2y)

#df_joined_data_2021 = df_joined_data_all.filter(col("YEAR") == 2021).cache()
#display(df_joined_data_2y)

#df_joined_data_2015_2020 = df_joined_data_all.filter(col("YEAR") < 2021)
#display(df_joined_data_all)

#df_joined_data_2021 = df_joined_data_all.filter(col("YEAR") == 2021)
#display(df_joined_data_2021)

print("**Data Frames Loaded")

# COMMAND ----------

# Data cleaning Tasks

# Fills in NA values for cancelled flights.

#df_joined_data_3m = df_joined_data_3m.na.fill(value = 1,subset=["DEP_DEL15"])
df_joined_data_2y = df_joined_data_2y.na.fill(value = 1,subset=["DEP_DEL15"])
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
numericCols = ['CRS_ELAPSED_TIME', 'DISTANCE','ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed']
# Features Not Included: 'DEP_DATETIME','DATE', 'HourlyWindGustSpeed', 'MonthlyMeanTemperature', 'MonthlyMaximumTemperature', 'MonthlyGreatestSnowDepth', 'MonthlyGreatestSnowfall', 'MonthlyTotalSnowfall', 'MonthlyTotalLiquidPrecipitation', 'MonthlyMinimumTemperature', 'DATE_HOUR', 'time_zone_id', 'UTC_DEP_DATETIME_LAG', 'UTC_DEP_DATETIME', 'HourlyPressureChange', 'distance_to_neighbor', 'neighbor_lat', 'neighbor_lon'

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
preppedDataDF = pipelineModel.transform(df_joined_data_all)

# Apply pipeline to Pre-2021
#pipelineModel = partialPipeline.fit(df_joined_data_pre2021)
#preppedDataDF = pipelineModel.transform(df_joined_data_pre2021)

# Apply pipeline to 2021
#pipelineModel_2021 = partialPipeline.fit(df_joined_data_2021)
#preppedDataDF_2021 = pipelineModel_2021.transform(df_joined_data_2021)

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

def createLinearRegressionModel(num_iterations, trainingData_input):
    """
    Creates Logistic Regression model trained on trainingData_input.
    I realize now that this function is misnamed. Whoops!
    """
    lr = LogisticRegression(num_iterations, labelCol="DEP_DEL15", featuresCol="features")
    lrModel = lr.fit(trainingData_input)
    return lrModel

def reportMetrics(inputMetrics):
    """Outputs metrics (currently only for Linear Regression?)"""
    print("Precision = %s" % inputMetrics.precision(1))
    print("F1 = %s" % inputMetrics.fMeasure(1.0,1.0))
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
    
def runBlockingTimeSeriesCrossValidation(dataFrameInput):
    """
    Conducts the Blocking Time Series Cross Validation.
    Accepts the full dataFrame of all years. 
    Is hard coded to use pre-2021 data as training data, which it will cross validate against.
    After all cross validations, will select best model from each year, and then apply the test 2021 data against it for final evaluation.
    Prints metrics from final test evaluation at the end.
    """
    
    topMetrics = None
    topYear = None
    topModel = None
    
    # list all of the years that the data will be trained against.
    listOfYears = dataFrameInput.select("YEAR").distinct().filter(col("YEAR") != 2021).rdd.flatMap(list).collect()
    print("listOfYears:", listOfYears)

    # Iterate through each of the individual years in the training data set.
    for year in listOfYears:
        currentYear = year

        currentYearDF = dataFrameInput.filter(col("YEAR") == currentYear).cache()

        # Adds a percentage column to each year's data frame, with the percentage corresponding to percentage of the year's time. 
        # 0% = earliest time that year. 100% = latest time that year.
        preppedDataDF = currentYearDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))

#        display(preppedDataDF)

        # remove unneeded columns. All feature values are captured in "features". All the other retained features are for row tracking.
        selectedcols = ["DEP_DEL15", "YEAR", "QUARTER", "DEP_DATETIME_LAG_percent", "features"]
        dataset = preppedDataDF.select(selectedcols).cache()

#        display(dataset)
        
        # The training set is the data from the 70% earliest data.
        # Test set is the latter 30% of the data.
        trainingData = dataset.filter(col("DEP_DATETIME_LAG_percent") <= .70)
        trainingTestData = dataset.filter(col("DEP_DATETIME_LAG_percent") > .70)
        display(trainingTestData)
        
        # Create and train a logistic regression model for the year based on training data.
        # Note: createLinearRegressionModel() function would not work here for some reason.
        lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", maxIter=10)
        lrModel = lr.fit(trainingData)

        currentYearMetrics = runLogisticRegression(lrModel, trainingTestData)
        
        # Print the year's data and evaluation metrics
        print(f"\n - {year}")
        print("trainingData Count:", trainingData.count())
        print("trainingTestData Count:", trainingTestData.count())
#        print("categoricalColumns =", categoricalColumns)
#        print("numericalCols =", numericCols)
        reportMetrics(currentYearMetrics)
        
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
    print("\n** Best Metrics **")
    print("topYear = %s" % topYear)
    print("categoricalColumns =", categoricalColumns)
    print("numericalCols =", numericCols)
    reportMetrics(topMetrics)
    print("Top Model:", topModel)
    
    
    
    # Prepare 2021 Test Data
    currentYearDF = dataFrameInput.filter(col("YEAR") == 2021).cache()
    preppedDataDF = currentYearDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))
#        display(preppedDataDF)
    selectedcols = ["DEP_DEL15", "YEAR", "QUARTER", "DEP_DATETIME_LAG_percent", "features"]
    dataset = preppedDataDF.select(selectedcols).cache()
#        display(dataset)

    # Evaluate best model from cross validation against the test data frame of 2021 data, then print evaluation metrics.
    testDataSet = dataset
    display(testDataSet)
    lr = topModel
    testMetrics = runLogisticRegression(topModel, testDataSet)

    print(f"\n - 2021")
    print("testDataSet Count:", testDataSet.count())
#        print("categoricalColumns =", categoricalColumns)
#        print("numericalCols =", numericCols)
    reportMetrics(testMetrics)
    
# TODO: Way to save results from evaluation to a file or RDD? Would allow for easier automation for gridsearch.
    

# COMMAND ----------

# 55 Minutes for Full Data
runBlockingTimeSeriesCrossValidation(preppedDataDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Logistic Regression Results
# MAGIC 
# MAGIC 11/16/22
# MAGIC 55 minutes
# MAGIC testDataSet Count: 5771706
# MAGIC Precision = 0.029235971681133347
# MAGIC F1 = 0.05415838783534103
# MAGIC Recall = 0.36706913430609295
# MAGIC Accuracy = 0.80939517709322

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


