# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Data Pipeline Creation
# MAGIC 
# MAGIC Adapted From:
# MAGIC 
# MAGIC https://towardsdatascience.com/building-an-ml-application-with-mllib-in-pyspark-part-1-ac13f01606e2
# MAGIC 
# MAGIC https://spark.apache.org/docs/2.2.0/ml-classification-regression.html
# MAGIC 
# MAGIC https://docs.databricks.com/machine-learning/train-model/mllib/index.html
# MAGIC 
# MAGIC https://docs.databricks.com/_static/notebooks/binary-classification.html
# MAGIC 
# MAGIC https://pages.databricks.com/rs/094-YMS-629/images/02-Delta%20Lake%20Workshop%20-%20Including%20ML.html

# COMMAND ----------

from pyspark.sql.functions import col, floor
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import IntegerType, DoubleType

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
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import GBTClassifier

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

# Load Dataframes
print("**Loading Data")

# Inspect the Joined Data folders 
display(dbutils.fs.ls(f"{blob_url}"))

print("**Data Loaded")
print("**Loading Data Frames")

#df_joined_data_3m = spark.read.parquet(f"{blob_url}/joined_data_3m")
#display(df_joined_data_3m)

# df_joined_data_all = spark.read.parquet(f"{blob_url}/joined_data_all")
# display(df_joined_data_all)

df_joined_data_all_with_efeatures = spark.read.parquet(f"{blob_url}/joined_all_with_efeatures_Downsampled")
# df_joined_data_all_with_efeatures = df_joined_data_all_with_efeatures.withColumn("pagerank",df_joined_data_all_with_efeatures.pagerank.cast('double'))
display(df_joined_data_all_with_efeatures)

print("**Data Frames Loaded")

# COMMAND ----------

def getCurrentDateTimeFormatted():
    return str(datetime.utcnow()).replace(" ", "-").replace(":", "-").replace(".", "-")

def buildPipeline(trainDF, categoricals, numerics, Y="DEP_DEL15", oneHot=True, imputer=False, scaler=False):
    ## Current possible ways to handle categoricals in string indexer is 'error', 'keep', and 'skip'
#     indexers = map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'skip'), categoricals)
    
#     stages = list(indexers)
    
    stages = [] # stages in Pipeline

    # NOTE: Had to cut out a bunch of features due to the sheer number of NULLS in them, which were causing the entire dataframe to be skipped. Will need to get the Null values either filled or dropped.
    cat_ending = '_idx'
    if oneHot==True:
        cat_ending = '_class'
        
    
    for categoricalCol in categoricals:
        # Category Indexing with StringIndexer
        indexers = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "_idx").setHandleInvalid("skip")
        stages += [indexers]
        
        if oneHot==True:
            # Use OneHotEncoder to convert categorical variables into binary SparseVectors
            encoder = OneHotEncoder(inputCols=[indexers.getOutputCol()], outputCols=[categoricalCol + "_class"])
            # Add stages.  These are not run here, but will run all at once later on.
            stages += [encoder]
            
    featureCols = list(map(lambda c: c + cat_ending, categoricals)) + numerics
        
    ##############################################
    
#     if oneHot == True:
#         ohes = map(lambda c: OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class"),categoricals)
#         # Establish features columns
#         featureCols = list(map(lambda c: c+"_class", categoricals)) + numerics
#         stages += list(ohes)
#     else:
#         featureCols = list(map(lambda c: c+"_idx", categoricals)) + numerics
        
    if imputer == True:
        imputers = Imputer(inputCols = numerics, outputCols = numerics)
        stages += list(imputers)
    
    # Build the stage for the ML pipeline
    stages += [VectorAssembler(inputCols=featureCols, outputCol="features").setHandleInvalid("skip")]

    if scaler == True:
        # Apply StandardScaler to create scaledFeatures
        scalers = StandardScaler(inputCol="features",
                                outputCol="scaledFeatures",
                                withStd=True,
                                withMean=True)
        stages += [scalers]

    
    pipeline = Pipeline().setStages(stages)
    
    pipelineModel = pipeline.fit(trainDF)
    
    return pipelineModel


def getFeatureNames(preppedPipelineModel, featureCol='features'):
    # Get feature names for use later 
    meta = [f.metadata 
        for f in preppedPipelineModel.schema.fields 
        if f.name == featureCol][0]
    
    try:
        # access feature name and index
        feature_names = meta['ml_attr']['attrs']['binary'] + meta['ml_attr']['attrs']['numeric']
        # feature_names = pd.DataFrame(feature_names)
    except:
        feature_names = meta['ml_attr']['attrs']['nominal'] + meta['ml_attr']['attrs']['numeric']
        
    feature_names = [feature['name'] for feature in feature_names]
    
    return feature_names


# COMMAND ----------

Y = "DEP_DEL15"

categoricals = ['QUARTER','MONTH','DAY_OF_WEEK','OP_UNIQUE_CARRIER',
                'DEP_HOUR','AssumedEffect_Text','airline_type',
                'is_prev_delayed','Blowing_Snow','Freezing_Rain','Rain','Snow','Thunder']

numerics = ['DISTANCE','ELEVATION','HourlyAltimeterSetting','HourlyDewPointTemperature',
            'HourlyWetBulbTemperature','HourlyDryBulbTemperature','HourlyPrecipitation',
            'HourlyStationPressure','HourlySeaLevelPressure','HourlyRelativeHumidity',
            'HourlyVisibility','HourlyWindSpeed','perc_delay',
            'pagerank']

# COMMAND ----------

pipelineModel = buildPipeline(df_joined_data_all_with_efeatures.filter( col('YEAR') != 2021), 
                              categoricals, numerics, Y="DEP_DEL15", oneHot=True, imputer=False, scaler=True)

preppedData = pipelineModel.transform(df_joined_data_all_with_efeatures)
# preppedValid = pipelineModel.transform(valid).cache()
# preppedTest = pipelineModel.transform(test).cache()

preppedTrain = preppedData.filter( col('YEAR') != 2021)\
                   .withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))\
                   .cache()

# valid = df_joined_data_all_with_efeatures.filter((col('FL_DATE') >= '2020-07-01') & (col('FL_DATE') < '2021-01-01')).cache()
preppedTest = preppedData.filter(col('YEAR') == 2021).cache()

# COMMAND ----------

feature_names = getFeatureNames(preppedTrain)
print(len(feature_names))

# COMMAND ----------

def extract_prob(v):
    """
    Extracts the predicted probability from the logistic regression model
    """
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None
extract_prob_udf = F.udf(extract_prob, DoubleType())


    
def testModelPerformance(predictions, y='DEP_DEL15'):
    
    def FScore(beta, precision, recall):
        if precision + recall == 0:
            F = 0
        else:
            F = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        return F
    
#     metrics = MulticlassMetrics(predictions.select(y, "prediction").rdd)
    
    TP = predictions.filter((col(y)==1) & (col("prediction")==1)).count()
    TN = predictions.filter((col(y)==0) & (col("prediction")==0)).count()
    FP = predictions.filter((col(y)==0) & (col("prediction")==1)).count()
    FN = predictions.filter((col(y)==1) & (col("prediction")==0)).count()

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


###### THIS WILL NOT RUN - WORK IN PROGRESS
def predictTestData(cv_stats, dataFrameInput):
    
    print(f"@ Starting Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    # Prepare 2021 Test Data
    selectedcols = ["DEP_DEL15", "YEAR", "features"]
    dataset = dataFrameInput.select(selectedcols).cache()
    
    test_stats = pd.DataFrame()
    
    for row in range(len(cv_stats)):
        
        print(f'Testing model {row + 1} of {len(cv_stats)}')
        
        best_model = cv_stats.iloc[row]
        best_model_stats = cv_stats.iloc[[row]]

        currentYearPredictions = best_model['trained_model'].transform(dataset).withColumn("predicted_probability", extract_prob_udf(col("probability")))
        thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                             .withColumn("prediction", (col('predicted_probability') > best_model['threshold']).cast('double') )

#         thresholdPredictions = thresholdPredictions.withColumn("row_id", F.monotonically_increasing_id()).cache()

    #         if ensemble_predictions == None:
    #             ensemble_predictions = thresholdPredictions
    #         else:
    #             ensemble_predictions = ensemble_predictions.join(thresholdPredictions, ("row_id"))

        currentYearMetrics = testModelPerformance(thresholdPredictions)
        stats = pd.DataFrame([currentYearMetrics], columns=['test_Precision','test_Recall','test_F0.5','test_F1','test_Accuracy'])
        test_stats = pd.concat([test_stats, stats], axis=0)
        
    final_stats = pd.concat([stats.reset_index(drop=True), cv_stats.reset_index(drop=True)], axis=1)

    return final_stats
  
    
def getFeatureImportance(featureNames, coefficients):
    
    featureImportances = pd.DataFrame(zip(featureNames,coefficients), columns=['featureName','coefficient'])
    featureImportances['importance'] = featureImportances['coefficient'].abs()
    
    return featureImportances


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Logistic Regression

# COMMAND ----------

def runBlockingTimeSeriesCrossValidation_LR(preppedTrain, cv_folds=4, regParam_input=0, elasticNetParam_input=0,
                                         maxIter_input=10, thresholds_list = [0.5]):
    """
    Function which performs blocking time series cross validation
    Takes the pipeline-prepped DF as an input, with options for number of desired folds and logistic regression parameters
    Returns a pandas dataframe of validation performance metrics and the corresponding models
    """
    
    cutoff = 1/cv_folds
    
    cv_stats = pd.DataFrame()


    for i in range(cv_folds):
        print(f"! Running fold {i+1} of {cv_folds}")
        print(f"@ {getCurrentDateTimeFormatted()}")
        min_perc = i*cutoff
        max_perc = min_perc + cutoff
        train_cutoff = min_perc + (0.7 * cutoff)

        cv_train = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= min_perc) & (col("DEP_DATETIME_LAG_percent") < train_cutoff))\
                                .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]).cache()
        
        cv_val = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= train_cutoff) & (col("DEP_DATETIME_LAG_percent") < max_perc))\
                              .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]).cache()
        
        lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", regParam = regParam_input, elasticNetParam = elasticNetParam_input, 
                                maxIter = maxIter_input, threshold = 0.5, standardization = True)
        
        lrModel = lr.fit(cv_train)

        currentYearPredictions = lrModel.transform(cv_val).withColumn("predicted_probability", extract_prob_udf(col("probability"))).cache()

        for threshold in thresholds_list:
#            print(f"! Testing threshold {threshold}")

            thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > threshold).cast('double') )

            currentYearMetrics = testModelPerformance(thresholdPredictions)
            stats = pd.DataFrame([currentYearMetrics], columns=['val_Precision','val_Recall','val_F0.5','val_F1','val_Accuracy'])
            stats['cv_fold'] = i
            stats['regParam'] = regParam_input
            stats['elasticNetParam'] = elasticNetParam_input
            stats['maxIter'] = maxIter_input
            stats['threshold'] = threshold
            stats['trained_model'] = lrModel

            cv_stats = pd.concat([cv_stats,stats],axis=0)
            
    return cv_stats


# COMMAND ----------

# cv_stats = runBlockingTimeSeriesCrossValidation(preppedTrain, cv_folds=4, regParam_input=0, elasticNetParam_input=0,
#                                          maxIter_input=10, thresholds_list = [0.5])

# cv_stats

# COMMAND ----------

# feature_importances = getFeatureImportance(feature_names, list(cv_stats.iloc[0]['trained_model'].coefficients))
# feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# test_results = predictTestData(cv_stats, preppedTest)
# test_results

# COMMAND ----------

regParamGrid = [0.0, 0.01, 0.5, 1, 2.0]
elasticNetParamGrid = [0.0, 0.5, 1.0]
maxIterGrid = [5, 10, 50]
thresholds = [0.5, 0.6, 0.7, 0.8]

# regParamGrid = [0.0, 0.5]
# elasticNetParamGrid = [0.0, 1.0]
# maxIterGrid = [10]
# thresholds = [0.5, 0.6, 0.7, 0.8]

grid_search = pd.DataFrame()

for maxIter in maxIterGrid:
    print(f"~ maxIter = {maxIter}")
    for elasticNetParam in elasticNetParamGrid:
        print(f"~ elasticNetParam = {elasticNetParam}")
        for regParam in regParamGrid:
            print(f"~ regParam = {regParam}")
            try:
                cv_stats = runBlockingTimeSeriesCrossValidation(preppedTrain, cv_folds=4, regParam_input=regParam, 
                                                                elasticNetParam_input=elasticNetParam, maxIter_input=maxIter, 
                                                                thresholds_list = thresholds)
#                 test_results = predictTestData(cv_stats, preppedTest)

                grid_search = pd.concat([grid_search,cv_stats],axis=0)
            except:
                print('Error, continuing to next iteration')
                continue
            
test_results = predictTestData(grid_search, preppedTest)

print("~ Job Finished!")
print(f"~ {getCurrentDateTimeFormatted()}\n")

test_results

# COMMAND ----------

grid_spark_DF = spark.createDataFrame(test_results.drop(columns=['trained_model']))
grid_spark_DF.write.mode('overwrite').parquet(f"{blob_url}/logistic_regression_grid_CV_120122")

# COMMAND ----------

import joblib

joblib.dump(test_results, 'logistic_regression_grid_models_120122.pkl')

# COMMAND ----------

test_results.loc[test_results['val_F0.5'] == test_results['val_F0.5'].max(), 'trained_model']

# COMMAND ----------

feature_importances = getFeatureImportance(feature_names,)
feature_importances

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Random Forest

# COMMAND ----------

def runBlockingTimeSeriesCrossValidation_RF(preppedTrain, cv_folds=4, input_numTrees = 10, input_maxDepth = 4, input_maxBins = 32, thresholds_list = [0.5]):
    """
    Function which performs blocking time series cross validation
    Takes the pipeline-prepped DF as an input, with options for number of desired folds and logistic regression parameters
    Returns a pandas dataframe of validation performance metrics and the corresponding models
    """
    
    cutoff = 1/cv_folds
    
    cv_stats = pd.DataFrame()


    for i in range(cv_folds):
        min_perc = i*cutoff
        max_perc = min_perc + cutoff
        train_cutoff = min_perc + (0.7 * cutoff)

        cv_train = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= min_perc) & (col("DEP_DATETIME_LAG_percent") < train_cutoff))\
                                .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]).cache()
        
        cv_val = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= train_cutoff) & (col("DEP_DATETIME_LAG_percent") < max_perc))\
                              .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]).cache()
        
        rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", numTrees = input_numTrees, maxDepth = input_maxDepth, maxBins = input_maxBins)
        
        rfModel = rf.fit(cv_train)

        currentYearPredictions = rfModel.transform(cv_val).withColumn("predicted_probability", extract_prob_udf(col("probability"))).cache()

        for threshold in thresholds_list:

            thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > threshold).cast('double') )

            currentYearMetrics = testModelPerformance(thresholdPredictions)
            stats = pd.DataFrame([currentYearMetrics], columns=['val_Precision','val_Recall','val_F0.5','val_F1','val_Accuracy'])
            stats['cv_fold'] = i
            stats['numTrees'] = input_numTrees
            stats['maxDepth'] = input_maxDepth
            stats['maxBins'] = input_maxBins
            stats['threshold'] = threshold
            stats['trained_model'] = rfModel

            cv_stats = pd.concat([cv_stats,stats],axis=0)
            
    return cv_stats


# COMMAND ----------

cv_stats = runBlockingTimeSeriesCrossValidation_RF(preppedTrain, cv_folds=4, input_numTrees = 10, input_maxDepth = 4, input_maxBins = 32, thresholds_list = [0.5])

cv_stats

# COMMAND ----------

feature_importances = getFeatureImportance(feature_names, list(cv_stats.iloc[0]['trained_model'].coefficients))
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

test_results = predictTestData(cv_stats, preppedTest)
test_results

# COMMAND ----------

# Hyperparameter Tuning Parameter Grid for Random Forest
# Each CV takes about 30 minutes.

numTreesGrid = [10, 25, 50]
maxDepthGrid = [4, 8, 16]
maxBinsGrid = [32, 64, 128]
thresholds = [0.5, 0.6, 0.7, 0.8]

#numTreesGrid = [10, 25]
#maxDepthGrid = [4, 8]
#maxBinsGrid = [32, 64]
#thresholds = [0.5]

grid_search = pd.DataFrame()

for numTrees in numTreesGrid:
    print(f"! numTrees = {numTrees}")
    for maxDepth in maxDepthGrid:
        print(f"! maxDepth = {maxDepth}")
        for maxBins in maxBinsGrid:
            print(f"! maxBins = {maxBins}")
            for numTrees in numTreesGrid:
                print(f"! numTrees = {numTrees}")
            try:
                cv_stats = runBlockingTimeSeriesCrossValidation_RF(preppedTrain, 4, numTrees, maxDepth, maxBins, thresholds)
                test_results = predictTestData(cv_stats, preppedTest)

                grid_search = pd.concat([grid_search,test_results],axis=0)
            except:
                pass
            
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

grid_search

# COMMAND ----------

grid_search

# COMMAND ----------

grid_spark_DF = spark.createDataFrame(grid_search.drop(columns=['trained_model']))
display(grid_spark_DF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Support Vector Machines

# COMMAND ----------

def runBlockingTimeSeriesCrossValidation_SVM(preppedTrain, cv_folds=4, regParam_input=0, maxIter_input=10, thresholds_list = [0.5]):
    """
    Function which performs blocking time series cross validation
    Takes the pipeline-prepped DF as an input, with options for number of desired folds and logistic regression parameters
    Returns a pandas dataframe of validation performance metrics and the corresponding models
    """
    
    cutoff = 1/cv_folds
    
    cv_stats = pd.DataFrame()


    for i in range(cv_folds):
        min_perc = i*cutoff
        max_perc = min_perc + cutoff
        train_cutoff = min_perc + (0.7 * cutoff)

        cv_train = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= min_perc) & (col("DEP_DATETIME_LAG_percent") < train_cutoff))\
                                .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]).cache()
        
        cv_val = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= train_cutoff) & (col("DEP_DATETIME_LAG_percent") < max_perc))\
                              .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]).cache()
        
        lsvc = LinearSVC(labelCol="DEP_DEL15", featuresCol="features", regParam = regParam_input, maxIter = maxIter_input, threshold = 0.5, standardization = True)
        lsvc_model = lsvc.fit(cv_train)
        
        currentYearPredictions = lsvc_model.transform(cv_val).withColumn("predicted_probability", extract_prob_udf(col("probability"))).cache()

        for threshold in thresholds_list:

            thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > threshold).cast('double') )

            currentYearMetrics = testModelPerformance(thresholdPredictions)
            stats = pd.DataFrame([currentYearMetrics], columns=['val_Precision','val_Recall','val_F0.5','val_F1','val_Accuracy'])
            stats['cv_fold'] = i
            stats['regParam'] = regParam_input
            stats['maxIter'] = maxIter_input
            stats['threshold'] = threshold
            stats['trained_model'] = lsvc_model

            cv_stats = pd.concat([cv_stats,stats],axis=0)
            
    return cv_stats


# COMMAND ----------

cv_stats = runBlockingTimeSeriesCrossValidation_SVM(preppedTrain, cv_folds=4, regParam_input=0, maxIter_input=10, thresholds_list = [0.5])

cv_stats

# COMMAND ----------

feature_importances = getFeatureImportance(feature_names, list(cv_stats.iloc[0]['trained_model'].coefficients))
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

test_results = predictTestData(cv_stats, preppedTest)
test_results

# COMMAND ----------

#regParamGrid = [0.0, 0.01, 0.5, 2.0]
#elasticNetParamGrid = [0.0, 0.5, 1.0]
#maxIterGrid = [5, 10, 50]
#thresholds = [v0.5, 0.6, 0.7, 0.8]

regParamGrid = [0.0, 0.5]
maxIterGrid = [10]
thresholds = [0.5, 0.6, 0.7, 0.8]

grid_search = pd.DataFrame()

for maxIter in maxIterGrid:
    print(f"! maxIter = {maxIter}")
    for elasticNetParam in elasticNetParamGrid:
        print(f"! elasticNetParam = {elasticNetParam}")
        for regParam in regParamGrid:
            print(f"! regParam = {regParam}")
            try:
                cv_stats = runBlockingTimeSeriesCrossValidation_SVM(preppedTrain, cv_folds=4, regParam, maxIter, thresholds)
                test_results = predictTestData(cv_stats, preppedTest)

                grid_search = pd.concat([grid_search,test_results],axis=0)
            except:
                continue
            
                        
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

grid_search

# COMMAND ----------

display(preppedTest)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Multilayer Perceptron Classifier

# COMMAND ----------

def runBlockingTimeSeriesCrossValidation_MLP(preppedTrain, input_layers, cv_folds=4, maxIter = 100, blockSize = 128, stepSize = 0.03, thresholds_list = [0.5]):
    """
    Function which performs blocking time series cross validation
    Takes the pipeline-prepped DF as an input, with options for number of desired folds and logistic regression parameters
    Returns a pandas dataframe of validation performance metrics and the corresponding models
    """
    
    print(f"@ runBlockingTimeSeriesCrossValidation_MLP")
    print(f"@ {getCurrentDateTimeFormatted()}")
    
    cutoff = 1/cv_folds
    
    cv_stats = pd.DataFrame()

    for i in range(cv_folds):
        print(f"@ Running cv_fold {i}")
        print(f"@ {getCurrentDateTimeFormatted()}")
        
        min_perc = i*cutoff
        max_perc = min_perc + cutoff
        train_cutoff = min_perc + (0.7 * cutoff)

        cv_train = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= min_perc) & (col("DEP_DATETIME_LAG_percent") < train_cutoff))\
                                .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]).cache()
        
        cv_val = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= train_cutoff) & (col("DEP_DATETIME_LAG_percent") < max_perc))\
                              .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]).cache()
        
        # create the trainer and set its parameters
        multilayer_perceptrion_classifier = MultilayerPerceptronClassifier(
                    labelCol = "DEP_DEL15",
                    featuresCol = "features",
                    maxIter = maxIter,
                    layers = input_layers,
                    blockSize = blockSize,
                    stepSize = stepSize
                )

        # train the model
        mlpModel = multilayer_perceptrion_classifier.fit(cv_train)
        
        currentYearPredictions = mlpModel.transform(cv_val).withColumn("predicted_probability", extract_prob_udf(col("probability"))).cache()

        for threshold in thresholds_list:

            thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > threshold).cast('double') )

            currentYearMetrics = testModelPerformance(thresholdPredictions)
            stats = pd.DataFrame([currentYearMetrics], columns=['val_Precision','val_Recall','val_F0.5','val_F1','val_Accuracy'])
            stats['cv_fold'] = i
            stats['maxIter'] = maxIter
            stats['blockSize'] = blockSize
            stats['stepSize'] = stepSize
            stats['layers'] = input_layers
            stats['threshold'] = threshold
            stats['trained_model'] = mlpModel

            cv_stats = pd.concat([cv_stats,stats],axis=0)
            
    return cv_stats

def getFeatures(inputDF): 
    meta = [f.metadata 
        for f in inputDF.schema.fields 
        if f.name == 'features'][0]

    # access feature name and index
    feature_names = meta['ml_attr']['attrs']['binary'] + meta['ml_attr']['attrs']['numeric']
    # feature_names = pd.DataFrame(feature_names)
    feature_names = [feature['name'] for feature in feature_names]

    return feature_names

# COMMAND ----------

# Each CV takes about 90 minutes for 2 hidden layers.
# Takes about 45 for 1 hidden layer.

#regParamGrid = [0.0, 0.01, 0.5, 2.0]
#elasticNetParamGrid = [0.0, 0.5, 1.0]
#maxIterGrid = [5, 10, 50]
#thresholds = [v0.5, 0.6, 0.7, 0.8]

numFeatures = len(getFeatures(preppedTrain))

layers = [[numFeatures, 30, 15, 2], [numFeatures, 15, 2]]
cvfold = 4

# maxIterGrid = [100, 200]
# blockSizeGrid = [128, 256]
# stepSizeGrid = [0.03, 0.1]
# thresholds = [0.5, 0.6, 0.7, 0.8]

maxIterGrid = [100, 200]
blockSizeGrid = [128, 256]
stepSizeGrid = [0.03, 0.1]
thresholds = [0.5, 0.6, 0.7, 0.8]

grid_search = pd.DataFrame()

for maxIter in maxIterGrid:
    print(f"! maxIter = {maxIter}")
    for blockSize in blockSizeGrid:
        print(f"! blockSize = {blockSize}")
        for stepSize in stepSizeGrid:
            print(f"! stepSize = {stepSize}")
            for layer in layers:
                print(f"! layer = {layer}")
                try:
                    cv_stats = runBlockingTimeSeriesCrossValidation_MLP(preppedTrain, layer, cvfold, maxIter, blockSize, stepSize, thresholds)

                    grid_search = pd.concat([grid_search,cv_stats],axis=0)
                except:
                    continue
            
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

grid_search



# COMMAND ----------

test_results = predictTestData(grid_search, preppedTest)

# COMMAND ----------

grid_spark_DF = spark.createDataFrame(test_results.drop(columns=['trained_model']))
grid_spark_DF.write.mode('overwrite').parquet(f"{blob_url}/neural_network_grid_CV_120122")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Gradient Boosted Trees

# COMMAND ----------

def runBlockingTimeSeriesCrossValidation_GBT(preppedTrain, cv_folds=4, input_maxIter = 20, input_maxDepth = 5, input_maxBins = 32, input_stepSize = 0.1, thresholds_list = [0.5]):
    """
    Function which performs blocking time series cross validation
    Takes the pipeline-prepped DF as an input, with options for number of desired folds and logistic regression parameters
    Returns a pandas dataframe of validation performance metrics and the corresponding models
    """
    
    cutoff = 1/cv_folds
    
    cv_stats = pd.DataFrame()


    for i in range(cv_folds):
        min_perc = i*cutoff
        max_perc = min_perc + cutoff
        train_cutoff = min_perc + (0.7 * cutoff)

        cv_train = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= min_perc) & (col("DEP_DATETIME_LAG_percent") < train_cutoff))\
                                .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]).cache()
        
        cv_val = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= train_cutoff) & (col("DEP_DATETIME_LAG_percent") < max_perc))\
                              .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]).cache()
        
        rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", numTrees = input_numTrees, maxDepth = input_maxDepth, maxBins = input_maxBins)
        
        rfModel = rf.fit(cv_train)
        
        gbt = GBTClassifier(labelCol = 'DEP_DEL15', featuresCol = 'features', input_maxIter, input_maxDepth, input_maxBins, input_stepSize)
        gbt_model = gbt.fit(cv_train)

        currentYearPredictions = gbt_model.transform(cv_val).withColumn("predicted_probability", extract_prob_udf(col("probability"))).cache()

        for threshold in thresholds_list:

            thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > threshold).cast('double') )

            currentYearMetrics = testModelPerformance(thresholdPredictions)
            stats = pd.DataFrame([currentYearMetrics], columns=['val_Precision','val_Recall','val_F0.5','val_F1','val_Accuracy'])
            stats['cv_fold'] = i
            stats['maxIter'] = input_maxIter
            stats['maxDepth'] = input_maxDepth
            stats['maxBins'] = input_maxBins
            stats['stepSize'] = input_stepSize
            stats['threshold'] = threshold
            stats['trained_model'] = gbt_model

            cv_stats = pd.concat([cv_stats,stats],axis=0)
            
    return cv_stats


# COMMAND ----------

cv_stats = runBlockingTimeSeriesCrossValidation_GBT(preppedTrain, cv_folds=4, input_numTrees = 20, input_maxDepth = 5, input_maxBins = 32, input_stepSize = 0.1, thresholds_list = [0.5])

cv_stats

# COMMAND ----------

feature_importances = getFeatureImportance(feature_names, list(cv_stats.iloc[0]['trained_model'].coefficients))
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

test_results = predictTestData(cv_stats, preppedTest)
test_results

# COMMAND ----------

# Hyperparameter Tuning Parameter Grid for Random Forest
# Each CV takes about 30 minutes.

numTreesGrid = [10, 25, 50]
maxDepthGrid = [4, 8, 16]
maxBinsGrid = [32, 64, 128]
stepSizeGrid = [0.1]
thresholds = [0.5, 0.6, 0.7, 0.8]

#numTreesGrid = [10, 25]
#maxDepthGrid = [4, 8]
#maxBinsGrid = [32, 64]
#thresholds = [0.5]

grid_search = pd.DataFrame()

for numTrees in numTreesGrid:
    print(f"! numTrees = {numTrees}")
    for maxDepth in maxDepthGrid:
        print(f"! maxDepth = {maxDepth}")
        for maxBins in maxBinsGrid:
            print(f"! maxBins = {maxBins}")
            for stepSize in stepSizeGrid:
                print(f"! stepSize = {stepSize}")
            try:
                cv_stats = runBlockingTimeSeriesCrossValidation_GBT(preppedTrain, cv_folds, numTrees, maxDepth, maxBins, stepSize, thresholds_list = [0.5])
                test_results = predictTestData(cv_stats, preppedTest)

                grid_search = pd.concat([grid_search,test_results],axis=0)
            except:
                pass
            
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

grid_search

# COMMAND ----------

grid_search

# COMMAND ----------

grid_spark_DF = spark.createDataFrame(grid_search.drop(columns=['trained_model']))
display(grid_spark_DF)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Workspace

# COMMAND ----------


categoricals = ['QUARTER','MONTH','DAY_OF_WEEK','OP_UNIQUE_CARRIER',
                'DEP_HOUR','AssumedEffect_Text','airline_type',
                'is_prev_delayed','Blowing_Snow','Freezing_Rain','Rain','Snow','Thunder']

numerics = ['DISTANCE','ELEVATION','HourlyAltimeterSetting','HourlyDewPointTemperature',
            'HourlyWetBulbTemperature','HourlyDryBulbTemperature','HourlyPrecipitation',
            'HourlyStationPressure','HourlySeaLevelPressure','HourlyRelativeHumidity',
            'HourlyVisibility','HourlyWindSpeed','perc_delay',
            'pagerank']
            
for feature in categoricals:
    print(preppedTrain.filter(col(feature).isNull()).count())

# COMMAND ----------

for feature in numerics:
    print(preppedTrain.filter(col(feature).isNull()).count())

# COMMAND ----------


