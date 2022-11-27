# Databricks notebook source
grid_spark_DF = spark.createDataFrame(grid_search.drop(columns=['trained_model']))
grid_spark_DF.write.mode('overwrite').parquet(f"{blob_url}/logistic_regression_grid_CV")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Data Pipeline Creation 
# MAGIC ## Version 2

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
from pyspark.ml.classification import LinearSVC

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

# COMMAND ----------

# Load Dataframes
print("**Loading Data")

# Inspect the Joined Data folders 
display(dbutils.fs.ls(f"{blob_url}"))

print("**Data Loaded")
print("**Loading Data Frame")

df_joined_data_all_with_efeatures = spark.read.parquet(f"{blob_url}/joined_all_with_efeatures_v2_No2015")
display(df_joined_data_all_with_efeatures)

print("**Data Frame Loaded")

# COMMAND ----------

# Cast pagerank feature from string to double
df_joined_data_all_with_efeatures = df_joined_data_all_with_efeatures.withColumn("pagerank",df_joined_data_all_with_efeatures.pagerank.cast('double'))

# COMMAND ----------

# dataframe schema
print("df_joined_data_all_with_efeatures.count()", df_joined_data_all_with_efeatures.count())
df_joined_data_all_with_efeatures.printSchema()

# COMMAND ----------

# Convert categorical features to One Hot Encoding

categoricalColumns = ['ORIGIN', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_STATE_ABR',  'DEST_AIRPORT_SEQ_ID', 'DEST_STATE_ABR', 'CRS_DEP_TIME', 'YEAR', 'AssumedEffect', 'is_prev_delayed', 'is_prev_diverted']

stages = [] # stages in Pipeline

# NOTE: rows with null values in any of their features will be dropped.

for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index").setHandleInvalid("skip")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])

    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
    
#print(stages)


# COMMAND ----------

# Create vectors for numeric and categorical variables

numericCols = ['CRS_ELAPSED_TIME', 'DISTANCE','ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyStationPressure', 'HourlySeaLevelPressure', 'HourlyRelativeHumidity', 'HourlyVisibility', 'HourlyWindSpeed', 'perc_delay', 'pagerank']

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

# Adds Features vector to data frames as part of pipeline.
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("skip")

stages += [assembler]
stages_NoEncoding += [assembler]

#print(stages)

# COMMAND ----------

# Takes about 4 minutes for Full

# Create the pipeline to be applied to the dataframes
partialPipeline = Pipeline().setStages(stages)

# Apply pipeline stages to data frame
pipelineModel = partialPipeline.fit(df_joined_data_all_with_efeatures)
preppedDataDF = pipelineModel.transform(df_joined_data_all_with_efeatures).cache()

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
# MAGIC # Generic Functions

# COMMAND ----------

def runModel(inputModel, testData):
    """
    Applies the input model to the test data provided, and return a data frame with the 
    model's predictions.
    """
    predictions = inputModel.transform(testData)
    
    return predictions

def extract_prob(v):
    """
    Extracts the predicted probability from the logistic regression model
    """
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None
#extract_prob_udf = F.udf(extract_prob, DoubleType())

def testModelPerformance(predictions):
    """
    Calculates and returns model evaluation metrics based on input predictions data frame.
    Had to do it this way because MulticlassMetrics class would not return necessary metrics
    for random forest for some reason.
    """
    def FScore(beta, precision, recall):
        if precision + recall == 0:
            F = 0
        else:
            F = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        return F
    
    metrics = MulticlassMetrics(predictions.select("DEP_DEL15", "prediction").rdd)
    
    TP = predictions.filter((col("DEP_DEL15")==1) & (col("prediction")==1)).count()
    TN = predictions.filter((col("DEP_DEL15")==0) & (col("prediction")==0)).count()
    FP = predictions.filter((col("DEP_DEL15")==0) & (col("prediction")==1)).count()
    FN = predictions.filter((col("DEP_DEL15")==1) & (col("prediction")==0)).count()

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
        
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    
    F1 = FScore(1, precision, recall)
    F05 = FScore(0.5, precision, recall)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return precision, recall, F05, F1, accuracy


def getDownsampledDataFrame(inputDF):
    """
    Rebalances dataframe based on DEP_DEL15 feature by returning a downsampled dataframe
    such that the distributions for DEP_DEL15's bool values are equivalent.
    """
    inputDF_downsampling_0 = inputDF.filter(col("DEP_DEL15") == 0)
    print(f"@- inputDF_downsampling_0.count() = {inputDF_downsampling_0.count()}")
    inputDF_downsampling_1 = inputDF.filter(col("DEP_DEL15") == 1)
    print(f"@- inputDF_downsampling_1.count() = {inputDF_downsampling_1.count()}")

    downsampling_ratio = (inputDF_downsampling_1.count() / inputDF_downsampling_0.count())

    inputDF_downsampling_append = inputDF_downsampling_0.sample(fraction = downsampling_ratio, withReplacement = False, seed = 261)

    inputDF_downsampled = inputDF_downsampling_1.unionAll(inputDF_downsampling_append)
    print(f"@- inputDF_downsampled.count() = {inputDF_downsampled.count()}")
    return inputDF_downsampled

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Logistic Regression

# COMMAND ----------

def runBlockingTimeSeriesCrossValidation_LR(dataFrameInput, regParam_input = 0.0, elasticNetParam_input = 0, maxIter_input = 10, thresholds_list = [0.5, 0.7]):
    """
    Conducts the Blocking Time Series Cross Validation.
    Accepts the full dataFrame of all years. 
    Is hard coded to use pre-2021 data as training data, which it will cross-validate against.
    After all cross validations, will select best model from each year, and then apply the test 2021 data against it for final evaluation.
    Prints metrics from final test evaluation at the end.
    """
    print(f"\n@ Starting runBlockingTimeSeriesCrossValidation")
    print(f"@ {regParam_input}, {elasticNetParam_input}, {maxIter_input}, {thresholds_list}")
    print(f"@ {getCurrentDateTimeFormatted()}")

    # list all of the years that the data will be trained against.
    listOfYears = dataFrameInput.select("YEAR").distinct().filter(col("YEAR") != 2021).rdd.flatMap(list).collect()
    print("listOfYears:", listOfYears)

    cv_stats = pd.DataFrame()

    # Iterate through each of the individual years in the training data set.
    for currentYear in listOfYears:

        print(f"Processing Year: {currentYear}")
        print(f"@ {getCurrentDateTimeFormatted()}")
        currentYearDF = dataFrameInput.filter(col("YEAR") == currentYear).cache()

        # Downscale the data such that there are roughly equal amounts of rows where DEP_DEL15 == 0 and DEP_DEL15 == 1, which aids in training.
        currentYearDF_downsampled = getDownsampledDataFrame(currentYearDF).cache()

        # Adds a percentage column to each year's data frame, with the percentage corresponding to percentage of the year's time. 
        # 0% = earliest time that year. 100% = latest time that year.
        preppedDF = currentYearDF_downsampled.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))

        # remove unneeded columns. All feature values are captured in "features" vector. All the other retained features are for row tracking.
        selectedcols = ["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]
        dataset = preppedDF.select(selectedcols).cache()

        # The training set is the data from the 70% earliest data, i.e. the earliest 70% of the year's data.
        # Test set is the latter 30% of the data, i.e. the last 30% of the year's data.
        trainingData = dataset.filter(col("DEP_DATETIME_LAG_percent") <= .70)
        trainingTestData = dataset.filter(col("DEP_DATETIME_LAG_percent") > .70)

        # Create and train a logistic regression model for the year based on training data.
        lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", regParam = regParam_input, elasticNetParam = elasticNetParam_input, 
                                maxIter = maxIter_input, threshold = 0.5, standardization = True)
        lrModel = lr.fit(trainingData)

        currentYearPredictions = runModelPredictions(lrModel, trainingTestData
                                                      ).withColumn("predicted_probability", extract_prob_udf(col("probability"))).cache()

        # After creating predictions percentages for each row in the trainingTestData from the model, evaluate predictions at different threshold values.
        for threshold in thresholds_list:
            thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > threshold).cast('double') )

            currentYearMetrics = testModelPerformance(thresholdPredictions)
            stats = pd.DataFrame([currentYearMetrics], columns=['val_Precision','val_Recall','val_F0.5','val_F1','val_Accuracy'])
            stats['year'] = year
            stats['regParam'] = regParam_input
            stats['elasticNetParam'] = elasticNetParam_input
            stats['maxIter'] = maxIter_input
            stats['threshold'] = threshold
            stats['trained_model'] = lrModel

            # Stores each of the validation metrics and fitted model for every CV in a Pandas DF, which is returned at the end.
            cv_stats = pd.concat([cv_stats,stats],axis=0)
            
    return cv_stats


def predictTestData_LR(cv_stats, dataFrameInput):
    """
    Evaluates trained models against the test data set, then returns a dataframe with all of the evaluation metrics.
    Takes in a pandas dataframe with CV evaluation metrics and the fitted models along with the data frame with the
    test data to be evaluated against from runBlockingTimeSeriesCrossValidation().
    """
    print(f"@ Starting Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    
    # Prepare 2021 Test Data
    currentYearDF = dataFrameInput.filter(col("YEAR") == 2021).cache()
    
    # NOTE: Might not need DEP_DATETIME_LAG_percent feature here. Remove?
    preppedDF = currentYearDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))
    selectedcols = ["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]
    dataset = preppedDF.select(selectedcols).cache()
    
    best_model = cv_stats.sort_values("val_F0.5", ascending=False).iloc[0]
    best_model_stats = cv_stats.sort_values("val_F0.5", ascending=False).iloc[[0]]
    
    currentYearPredictions = runModelPredictions(best_model['trained_model'], dataset
                                                  ).withColumn("predicted_probability", extract_prob_udf(col("probability")))
    thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > best_model['threshold']).cast('double') )
    
    currentYearMetrics = testModelPerformance(thresholdPredictions)
    stats = pd.DataFrame([currentYearMetrics], columns=['test_Precision','test_Recall','test_F0.5','test_F1','test_Accuracy'])
    stats = pd.concat([stats, best_model_stats], axis=1)
    
    return stats
    
# TODO: REWRITE TO FIT WITH NEW COLUMNS
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

# TODO: REWRITE TO FIT WITH NEW COLUMNS
def saveMetricsToAzure_LR(input_model, input_metrics):
    columns = ["date_time","precision", "f0.5", "recall", "accuracy", "regParam", "elasticNetParam", "maxIter", "threshold"]
    data = [(datetime.utcnow(), input_metrics.precision(1), input_metrics.fMeasure(label = 1.0, beta = 0.5), \
             input_metrics.recall(1), input_metrics.accuracy, input_model.getRegParam(), \
             input_model.getElasticNetParam(), input_model.getMaxIter(), input_model.getThreshold())]
    rdd = spark.sparkContext.parallelize(data)
    dfFromRDD = rdd.toDF(columns)
    
    dfFromRDD.write.mode('append').parquet(f"{blob_url}/logistic_regression_metrics")
    print("LR Metrics Saved Successfully!")
    
# WARNING: Will Delete Current Metrics for Logistic Regression
#resetMetricsToAzure_LR()
# WARNING: Will Delete Current Metrics for Logistic Regression
    

# COMMAND ----------

# Hyperparameter Tuning Parameter Grid for Logistic Regression
# Each CV takes about 20 minutes.

regParamGrid = [0.0, 0.01, 0.5, 2.0]
elasticNetParamGrid = [0.0, 0.5, 1.0]
maxIterGrid = [10, 50, 100]
thresholds = [0.5, 0.6, 0.7, 0.8]

grid_search = pd.DataFrame()

for maxIter in maxIterGrid:
    print(f"! maxIter = {maxIter}")
    for elasticNetParam in elasticNetParamGrid:
        print(f"! elasticNetParam = {elasticNetParam}")
        for regParam in regParamGrid:
            print(f"! regParam = {regParam}")
            try:
                cv_stats = runBlockingTimeSeriesCrossValidation_LR(preppedDataDF, regParam, elasticNetParam, maxIter, thresholds_list = thresholds)
                test_results = predictTestData_LR(cv_stats, preppedDataDF)

                grid_search = pd.concat([grid_search,test_results],axis=0)
            except:
                pass
            
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

grid_search

# COMMAND ----------

grid_spark_DF = spark.createDataFrame(grid_search.drop(columns=['trained_model']))
grid_spark_DF.write.mode('overwrite').parquet(f"{blob_url}/logistic_regression_grid_CV")

# COMMAND ----------

current_metrics = spark.read.parquet(f"{blob_url}/logistic_regression_grid_CV")
display(current_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Random Forest

# COMMAND ----------

def runBlockingTimeSeriesCrossValidation_RF(dataFrameInput, input_numTrees = 10, input_maxDepth = 4, input_maxBins = 32, thresholds_list = [0.5, 0.7]):
    """
    Conducts the Blocking Time Series Cross Validation.
    Accepts the full dataFrame of all years. 
    Is hard coded to use pre-2021 data as training data, which it will cross-validate against.
    After all cross validations, will select best model from each year, and then apply the test 2021 data against it for final evaluation.
    Prints metrics from final test evaluation at the end.
    """
    print(f"\n@ Starting runBlockingTimeSeriesCrossValidation")
    print(f"@ {regParam_input}, {elasticNetParam_input}, {maxIter_input}, {thresholds_list}")
    print(f"@ {getCurrentDateTimeFormatted()}")

    # list all of the years that the data will be trained against.
    listOfYears = dataFrameInput.select("YEAR").distinct().filter(col("YEAR") != 2021).rdd.flatMap(list).collect()
    print("listOfYears:", listOfYears)

    cv_stats = pd.DataFrame()

    # Iterate through each of the individual years in the training data set.
    for currentYear in listOfYears:

        print(f"Processing Year: {currentYear}")
        print(f"@ {getCurrentDateTimeFormatted()}")
        currentYearDF = dataFrameInput.filter(col("YEAR") == currentYear).cache()

        # Downscale the data such that there are roughly equal amounts of rows where DEP_DEL15 == 0 and DEP_DEL15 == 1, which aids in training.
        currentYearDF_downsampled = getDownsampledDataFrame(currentYearDF).cache()

        # Adds a percentage column to each year's data frame, with the percentage corresponding to percentage of the year's time. 
        # 0% = earliest time that year. 100% = latest time that year.
        preppedDF = currentYearDF_downsampled.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))

        # remove unneeded columns. All feature values are captured in "features" vector. All the other retained features are for row tracking.
        selectedcols = ["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]
        dataset = preppedDF.select(selectedcols).cache()

        # The training set is the data from the 70% earliest data, i.e. the earliest 70% of the year's data.
        # Test set is the latter 30% of the data, i.e. the last 30% of the year's data.
        trainingData = dataset.filter(col("DEP_DATETIME_LAG_percent") <= .70)
        trainingTestData = dataset.filter(col("DEP_DATETIME_LAG_percent") > .70)

        # Create and train a logistic regression model for the year based on training data.
        
        rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", numTrees = input_numTrees, maxDepth = input_maxDepth, maxBins = input_maxBins, threshold = 0.5, standardization = True)
        rfModel = rf.fit(trainingData)

        currentYearPredictions = runModelPredictions(rfModel, trainingTestData
                                                      ).withColumn("predicted_probability", extract_prob_udf(col("probability"))).cache()

        # After creating predictions percentages for each row in the trainingTestData from the model, evaluate predictions at different threshold values.
        for threshold in thresholds_list:
            thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > threshold).cast('double') )

            currentYearMetrics = testModelPerformance(thresholdPredictions)
            stats = pd.DataFrame([currentYearMetrics], columns=['val_Precision','val_Recall','val_F0.5','val_F1','val_Accuracy'])
            stats['year'] = year
            stats['regParam'] = regParam_input
            stats['elasticNetParam'] = elasticNetParam_input
            stats['maxIter'] = maxIter_input
            stats['threshold'] = threshold
            stats['trained_model'] = lrModel

            # Stores each of the validation metrics and fitted model for every CV in a Pandas DF, which is returned at the end.
            cv_stats = pd.concat([cv_stats,stats],axis=0)
            
    return cv_stats


def predictTestData_RF(cv_stats, dataFrameInput):
    """
    Evaluates trained models against the test data set, then returns a dataframe with all of the evaluation metrics.
    Takes in a pandas dataframe with CV evaluation metrics and the fitted models along with the data frame with the
    test data to be evaluated against from runBlockingTimeSeriesCrossValidation().
    """
    print(f"@ Starting Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    
    # Prepare 2021 Test Data
    currentYearDF = dataFrameInput.filter(col("YEAR") == 2021).cache()
    
    # NOTE: Might not need DEP_DATETIME_LAG_percent feature here. Remove?
    preppedDF = currentYearDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))
    selectedcols = ["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]
    dataset = preppedDF.select(selectedcols).cache()
    
    best_model = cv_stats.sort_values("val_F0.5", ascending=False).iloc[0]
    best_model_stats = cv_stats.sort_values("val_F0.5", ascending=False).iloc[[0]]
    
    currentYearPredictions = runModelPredictions(best_model['trained_model'], dataset
                                                  ).withColumn("predicted_probability", extract_prob_udf(col("probability")))
    thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > best_model['threshold']).cast('double') )
    
    currentYearMetrics = testModelPerformance(thresholdPredictions)
    stats = pd.DataFrame([currentYearMetrics], columns=['test_Precision','test_Recall','test_F0.5','test_F1','test_Accuracy'])
    stats = pd.concat([stats, best_model_stats], axis=1)
    
    return stats
    
# TODO: REWRITE TO FIT WITH NEW COLUMNS
def resetMetricsToAzure_RF():
    backup_metrics = spark.read.parquet(f"{blob_url}/logistic_regression_metrics")
    backup_date_string = getCurrentDateTimeFormatted()
    backup_metrics.write.parquet(f"{blob_url}/metrics_backups/logistic_regression_metrics-{backup_date_string}")
    
    columns = ["date_time","precision", "f0.5", "recall", "accuracy", "regParam", "elasticNetParam", "maxIter", "threshold"]
    data = [(datetime.utcnow(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)]
    rdd = spark.sparkContext.parallelize(data)
    dfFromRDD = rdd.toDF(columns)
    
    dfFromRDD.write.mode('overwrite').parquet(f"{blob_url}/logistic_regression_metrics")
    print("LR Metrics Reset")

# TODO: REWRITE TO FIT WITH NEW COLUMNS
def saveMetricsToAzure_RF(input_model, input_metrics):
    columns = ["date_time","precision", "f0.5", "recall", "accuracy", "regParam", "elasticNetParam", "maxIter", "threshold"]
    data = [(datetime.utcnow(), input_metrics.precision(1), input_metrics.fMeasure(label = 1.0, beta = 0.5), \
             input_metrics.recall(1), input_metrics.accuracy, input_model.getRegParam(), \
             input_model.getElasticNetParam(), input_model.getMaxIter(), input_model.getThreshold())]
    rdd = spark.sparkContext.parallelize(data)
    dfFromRDD = rdd.toDF(columns)
    
    dfFromRDD.write.mode('append').parquet(f"{blob_url}/logistic_regression_metrics")
    print("LR Metrics Saved Successfully!")
    
# WARNING: Will Delete Current Metrics for Logistic Regression
#resetMetricsToAzure_LR()
# WARNING: Will Delete Current Metrics for Logistic Regression
    

# COMMAND ----------

# Hyperparameter Tuning Parameter Grid for Random Forest
# Each CV takes about XX? minutes.

#numTrees = [10, 25, 50]
#maxDepth = [4, 8, 16]
#maxBins = [32, 64, 128]
#thresholds = [0.5, 0.6, 0.7, 0.8]

numTreesGrid = [10]
maxDepthGrid = [4]
maxBinsGrid = [32]
thresholds = [0.5, 0.6, 0.7, 0.8]

grid_search = pd.DataFrame()

for numTrees in numTreesGrid:
    print(f"! numTrees = {numTrees}")
    for maxDepth in maxDepthGrid:
        print(f"! maxDepth = {maxDepth}")
        for maxBins in maxBinsGrid:
            print(f"! maxBins = {maxBins}")
            try:
                cv_stats = runBlockingTimeSeriesCrossValidation_rf(preppedDataDF, numTrees, maxDepth, maxBins, thresholds_list = thresholds)
                test_results = predictTestData_RF(cv_stats, preppedDataDF)

                grid_search = pd.concat([grid_search,test_results],axis=0)
            except:
                pass
            
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

grid_search

# COMMAND ----------

grid_spark_DF = spark.createDataFrame(grid_search.drop(columns=['trained_model']))
grid_spark_DF.write.mode('overwrite').parquet(f"{blob_url}/random_forest_grid_CV")

# COMMAND ----------

current_metrics = spark.read.parquet(f"{blob_url}/random_forest_grid_CV")
display(current_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Linear Support Vector Machines

# COMMAND ----------

def runSupportVectorMachine(svmModel, testData):
    """
    Applies a Support Vector Machine model to the test data provided, and return the metrics from the test evaluation.
    """
    predictions = svmModel.transform(testData)
#    selected = predictions.select("DEP_DEL15", "prediction", "probability")
#    display(selected)
    metrics = MulticlassMetrics(predictions.select("DEP_DEL15", "prediction").rdd)
    
    return metrics
    
def runBlockingTimeSeriesCrossValidation_SVM(dataFrameInput, regParam_input, maxIter_input, threshold_input):
    """
    Conducts the Blocking Time Series Cross Validation.
    Accepts the full dataFrame of all years. 
    Is hard coded to use pre-2021 data as training data, which it will cross validate against.
    After all cross validations, will select best model from each year, and then apply the test 2021 data against it for final evaluation.
    Prints metrics from final test evaluation at the end.
    """
    print(f"\n@ Starting runBlockingTimeSeriesCrossValidation")
    print(f"@ {regParam_input}, {maxIter_input}, {threshold_input}")
    print(f"@ {getCurrentDateTimeFormatted()}")
    topMetrics = None
    topYear = None
    topModel = None
    trainingRows = 0
    
    # list all of the years that the data will be trained against.
    listOfYears = dataFrameInput.select("YEAR").distinct().filter(col("YEAR") != 2021).rdd.flatMap(list).collect()
    print("listOfYears:", listOfYears)

    # Iterate through each of the individual years in the training data set.
    for year in listOfYears:
        
        currentYear = year
        print(f"Processing Year: {currentYear}")
        print(f"@ {getCurrentDateTimeFormatted()}")
        currentYearDF = dataFrameInput.filter(col("YEAR") == currentYear).cache()
        
        # Downscale the data such that there are roughly equal amounts of rows where DEP_DEL15 == 0 and DEP_DEL15 == 1, which aids in training.
        
        currentYearDF_downsampling_0 = currentYearDF.filter(col("DEP_DEL15") == 0)
        print(f"@- currentYearDF_downsampling_0.count() = {currentYearDF_downsampling_0.count()}")
        currentYearDF_downsampling_1 = currentYearDF.filter(col("DEP_DEL15") == 1)
        print(f"@- currentYearDF_downsampling_1.count() = {currentYearDF_downsampling_1.count()}")

        downsampling_ratio = (currentYearDF_downsampling_1.count() / currentYearDF_downsampling_0.count())

        currentYearDF_downsampling_append = currentYearDF_downsampling_0.sample(fraction = downsampling_ratio, withReplacement = False, seed = 261)
        
        currentYearDF_downsampled = currentYearDF_downsampling_1.unionAll(currentYearDF_downsampling_append)
        trainingRows += currentYearDF_downsampled.count()
        print(f"@- currentYearDF_downsampled.count() = {currentYearDF_downsampled.count()}")
        
        # Adds a percentage column to each year's data frame, with the percentage corresponding to percentage of the year's time. 
        # 0% = earliest time that year. 100% = latest time that year.
        preppedDataDF = currentYearDF_downsampled.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))

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
        lsvc = LinearSVC(labelCol="DEP_DEL15", featuresCol="features", regParam = regParam_input, maxIter = maxIter_input, threshold = threshold_input)
        lsvc_model = lsvc.fit(trainingData)

        currentYearMetrics = runSupportVectorMachine(lsvc_model, trainingTestData)
        
        # Compare and store top models and metrics.
        if topMetrics == None:
            topMetrics = currentYearMetrics
            topYear = year
            topModel = lsvc_model
        else:
            if currentYearMetrics.precision(1.0) > topMetrics.precision(1.0):
                topMetrics = currentYearMetrics
                topYear = year
                topModel = lsvc_model
    
    # TODO: Ensemble models across all years?
    
    print(f"@ Training Test Metrics")
    reportMetrics(topMetrics)

    print(f"@ Starting Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    # Prepare 2021 Test Data
    currentYearDF = dataFrameInput.filter(col("YEAR") == 2021).cache()
    preppedDataDF = currentYearDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))
#        display(preppedDataDF)
    selectedcols = ["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]
    testDataSet = preppedDataDF.select(selectedcols).cache()
#        display(dataset)

    # Evaluate best model from cross validation against the test data frame of 2021 data, then print evaluation metrics.
    testMetrics = runLogisticRegression(topModel, testDataSet)

    print(f"\n - 2021")
    print("Model Parameters:")
    print("regParam:", topModel.getRegParam())
    print("maxIter:", topModel.getMaxIter())
    print("threshold:", topModel.getThreshold())
#    print("Weights Col:", lr.getWeightCol())
    print("testDataSet Count:", testDataSet.count())
    reportMetrics(testMetrics)
    saveMetricsToAzure_SVM(topModel, testMetrics)
    print(f"@ Finised Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    
    
def resetMetricsToAzure_SVM():
    backup_metrics = spark.read.parquet(f"{blob_url}/support_vector_machines_metrics")
    backup_date_string = getCurrentDateTimeFormatted()
    backup_metrics.write.parquet(f"{blob_url}/metrics_backups/support_vector_machines_metrics-{backup_date_string}")
    
    columns = ["date_time","precision", "f0.5", "recall", "accuracy", "regParam", "maxIter", "threshold"]
    data = [(datetime.utcnow(), 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)]
    rdd = spark.sparkContext.parallelize(data)
    dfFromRDD = rdd.toDF(columns)
    
    dfFromRDD.write.mode('overwrite').parquet(f"{blob_url}/support_vector_machines_metrics")
    print("SVM Metrics Reset")

def saveMetricsToAzure_SVM(input_model, input_metrics):
    columns = ["date_time","precision", "f0.5", "recall", "accuracy", "regParam", "maxIter", "threshold"]
    data = [(datetime.utcnow(), input_metrics.precision(1), input_metrics.fMeasure(label = 1.0, beta = 0.5), \
             input_metrics.recall(1), input_metrics.accuracy, input_model.getRegParam(), \
             input_model.getMaxIter(), input_model.getThreshold())]
    rdd = spark.sparkContext.parallelize(data)
    dfFromRDD = rdd.toDF(columns)
    
    dfFromRDD.write.mode('append').parquet(f"{blob_url}/support_vector_machines_metrics")
    print("SVM Metrics Saved Successfully")
    
# WARNING: Will Delete Current Metrics for Logistic Regression
#resetMetricsToAzure_SVM()
# WARNING: Will Delete Current Metrics for Logistic Regression

# COMMAND ----------

# Hyperparameter Tuning Parameter Grid
# Each CV takes one hour. Do the math.

#regParamGrid = [0.0, 0.01, 0.5, 2.0]
#maxIterGrid = [1, 5, 10]

regParamGrid = [0.0]
maxIterGrid = [10]
thresholdGrid = [0.5]

for regParam in regParamGrid:
    print(f"! regParam = {regParam}")
    for maxIter in maxIterGrid:
        print(f"! maxIter = {maxIter}")
        for threshold in thresholdGrid:
            print(f"! threshold = {threshold}")
            runBlockingTimeSeriesCrossValidation_SVM(preppedDataDF, regParam, maxIter, threshold)
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")


# COMMAND ----------

current_metrics = spark.read.parquet(f"{blob_url}/support_vector_machines_metrics")
display(current_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Gradient Boosted Tree

# COMMAND ----------

# GBT Test

from pyspark.ml.classification import GBTClassifier

def runGradientBoostedTree(gbtModel, testData):
    """
    Applies a logistic regression model to the test data provided, and return the metrics from the test evaluation.
    Realize now that the model input can be any model, and does not necessarily need to be logistic regression.
    Maybe try using with other models?
    """
    predictions = gbtModel.transform(testData)
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
    
    # Beta = 0.5, so (1 + (0.5 ^ 2)) = 1.25. Python didn't like (0.5 ^ 2) for some reason.
    fPointFive = (1.25 * precision * recall) / ((0.25 * precision) + recall)
    
    return precision, fPointFive, recall, accuracy

print(f"@ {getCurrentDateTimeFormatted()}")
testPreppedData_2017 = preppedDataDF.filter(col("YEAR") == 2017)
testPreppedData_2021 = preppedDataDF.filter(col("YEAR") == 2021)
print("Data Loaded")
print(f"@ {getCurrentDateTimeFormatted()}")

testPreppedData_2017_downsampling_0 = testPreppedData_2017.filter(col("DEP_DEL15") == 0).cache()
#        print(f"@- currentYearDF_downsampling_0.count() = {currentYearDF_downsampling_0.count()}")
testPreppedData_2017_downsampling_1 = testPreppedData_2017.filter(col("DEP_DEL15") == 1).cache()
#        print(f"@- currentYearDF_downsampling_1.count() = {currentYearDF_downsampling_1.count()}")

downsampling_ratio = (testPreppedData_2017_downsampling_1.count() / testPreppedData_2017_downsampling_0.count())

testPreppedData_2017_downsampling_append = testPreppedData_2017_downsampling_0.sample(fraction = downsampling_ratio, withReplacement = False, seed = 261)

testPreppedData_2017_downsampled = testPreppedData_2017_downsampling_1.unionAll(testPreppedData_2017_downsampling_append)
#        print(f"@- currentYearDF_downsampled.count() = {currentYearDF_downsampled.count()}")
print(f"@ testPreppedData_2017_downsampled(): {testPreppedData_2017_downsampled.count()}")    
print(f"@ {getCurrentDateTimeFormatted()}")
# Adds a percentage column to each year's data frame, with the percentage corresponding to percentage of the year's time. 
# 0% = earliest time that year. 100% = latest time that year.
preppedDataDF = testPreppedData_2017_downsampled.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG"))).cache()

gbt = GBTClassifier(featuresCol = 'features', labelCol = 'DEP_DEL15', maxIter=10)
gbt_model = gbt.fit(testPreppedData_2017)

precision, fPointFive, recall, accuracy = runGradientBoostedTree(gbt_model, preppedDataDF)

reportMetrics_rf(precision, fPointFive, recall, accuracy)

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

# MAGIC %md
# MAGIC 
# MAGIC # Workspace

# COMMAND ----------


