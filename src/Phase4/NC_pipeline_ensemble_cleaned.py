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
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
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

Y = "DEP_DEL15"

categoricals = ['ORIGIN','QUARTER','MONTH','DAY_OF_WEEK','OP_UNIQUE_CARRIER','ORIGIN_STATE_ABR',
                'DEST','DEST_STATE_ABR','DEP_HOUR','AssumedEffect_Text','airline_type',
                'is_prev_delayed','Blowing_Snow','Freezing_Rain','Rain','Snow','Thunder']

numerics = ['DISTANCE','ELEVATION','HourlyAltimeterSetting','HourlyDewPointTemperature',
            'HourlyWetBulbTemperature','HourlyDryBulbTemperature','HourlyPrecipitation',
            'HourlyStationPressure','HourlySeaLevelPressure','HourlyRelativeHumidity',
            'HourlyVisibility','HourlyWindSpeed','perc_delay',
            'pagerank']


train = df_joined_data_all_with_efeatures.filter( col('YEAR') != 2021)\
                                         .withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))\
                                         .cache()

# valid = df_joined_data_all_with_efeatures.filter((col('FL_DATE') >= '2020-07-01') & (col('FL_DATE') < '2021-01-01')).cache()
test = df_joined_data_all_with_efeatures.filter(col('YEAR') == 2021).cache()

# COMMAND ----------

def getCurrentDateTimeFormatted():
    return str(datetime.utcnow()).replace(" ", "-").replace(":", "-").replace(".", "-")

def buildPipeline(trainDF, categoricals, numerics, Y="DEP_DEL15", oneHot=True, imputer=False, scaler=False):
    ## Current possible ways to handle categoricals in string indexer is 'error', 'keep', and 'skip'
    indexers = map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'skip'), categoricals)
    
    stages = list(indexers)
    
    if oneHot == True:
        ohes = map(lambda c: OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class"),categoricals)
        # Establish features columns
        featureCols = list(map(lambda c: c+"_class", categoricals)) + numerics
        stages += list(ohes)
    else:
        featureCols = list(map(lambda c: c+"_idx", categoricals)) + numerics
        
    if imputer==True:
        imputers = Imputer(inputCols = numerics, outputCols = numerics)
        stages += list(imputer)
    
    # Build the stage for the ML pipeline
    model_matrix_stages = stages + \
                         [VectorAssembler(inputCols=featureCols, outputCol="features").setHandleInvalid("skip")]

    if scaler == True:
        # Apply StandardScaler to create scaledFeatures
        scaler = StandardScaler(inputCol="features",
                                outputCol="scaledFeatures",
                                withStd=True,
                                withMean=True)

        pipeline = Pipeline(stages=model_matrix_stages+[scaler])
    else:
        pipeline = Pipeline(stages=model_matrix_stages)
    
    
    pipelineModel = pipeline.fit(trainDF)
    
    return pipelineModel


def getFeatureNames(preppedPipelineModel):
    # Get feature names for use later 
    meta = [f.metadata 
        for f in preppedPipelineModel.schema.fields 
        if f.name == 'features'][0]

    # access feature name and index
    feature_names = meta['ml_attr']['attrs']['binary'] + meta['ml_attr']['attrs']['numeric']
    # feature_names = pd.DataFrame(feature_names)
    feature_names = [feature['name'] for feature in feature_names]
    
    return feature_names


# COMMAND ----------

pipelineModel = buildPipeline(train, categoricals, numerics, Y="DEP_DEL15", oneHot=True, imputer=False, scaler=False)

preppedTrain = pipelineModel.transform(train).cache()
# preppedValid = pipelineModel.transform(valid).cache()
preppedTest = pipelineModel.transform(test).cache()

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
    
    metrics = MulticlassMetrics(predictions.select(y, "prediction").rdd)
    
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
        best_model = cv_stats.sort_values("val_F0.5", ascending=False).iloc[row]
        best_model_stats = cv_stats.sort_values("val_F0.5", ascending=False).iloc[[row]]

        currentYearPredictions = best_model['trained_model'].transform(dataset).withColumn("predicted_probability", extract_prob_udf(col("probability")))
        thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                             .withColumn("prediction", (col('predicted_probability') > best_model['threshold']).cast('double') )
        
        thresholdPredictions = thresholdPredictions.withColumn("row_id", F.monotonically_increasing_id())
        
#         if ensemble_predictions == None:
#             ensemble_predictions = thresholdPredictions
#         else:
#             ensemble_predictions = ensemble_predictions.join(thresholdPredictions, ("row_id"))
    
        currentYearMetrics = testModelPerformance(thresholdPredictions)
        stats = pd.DataFrame([currentYearMetrics], columns=['test_Precision','test_Recall','test_F0.5','test_F1','test_Accuracy'])
        test_stats = pd.concat([test_stats, stats], axis=1)
        
    final_stats = pd.concat([stats, cv_stats], axis=1)
    
    return final_stats
  
    
def getFeatureImportance(featureNames, coefficients):
    
    featureImportances = pd.DataFrame(zip(featureNames,coefficients), columns=['featureName','coefficient'])
    featureImportances['importance'] = featureImportances['coefficient'].abs()
    
    return featureImportances


# COMMAND ----------

def runBlockingTimeSeriesCrossValidation(preppedTrain, cv_folds=4, regParam_input=0, elasticNetParam_input=0,
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
            print(f"! Testing threshold {threshold}")

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

cv_stats = runBlockingTimeSeriesCrossValidation(preppedTrain, cv_folds=4, regParam_input=0, elasticNetParam_input=0,
                                         maxIter_input=10, thresholds_list = [0.5])

cv_stats

# COMMAND ----------

feature_importances = getFeatureImportance(feature_names, list(cv_stats.iloc[0]['trained_model'].coefficients))
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

test_results = predictTestData(cv_stats, preppedTest)
test_results

# COMMAND ----------

regParamGrid = [0.0, 0.01, 0.5, 2.0]
elasticNetParamGrid = [0.0, 0.5, 1.0]
maxIterGrid = [5, 10, 50]
thresholds = [0.5, 0.6, 0.7, 0.8]

# regParamGrid = [0.0, 0.5]
# elasticNetParamGrid = [0.0, 1.0]
# maxIterGrid = [10]
# thresholds = [0.5, 0.6, 0.7, 0.8]

grid_search = pd.DataFrame()

for maxIter in maxIterGrid:
    print(f"! maxIter = {maxIter}")
    for elasticNetParam in elasticNetParamGrid:
        print(f"! elasticNetParam = {elasticNetParam}")
        for regParam in regParamGrid:
            print(f"! regParam = {regParam}")
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

print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

test_results

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Support Vector Machines

# COMMAND ----------

def runBlockingTimeSeriesCrossValidation_SVM(preppedTrain, cv_folds=4, regParam_input=0,
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

        lsvc = LinearSVC(labelCol="DEP_DEL15", featuresCol="features", regParam = regParam_input, maxIter = maxIter_input)
        lsvc_model = lsvc.fit(cv_train)
        
        currentYearPredictions = lsvc_model.transform(cv_val).withColumn("predicted_probability", extract_prob_udf(col("probability"))).cache()
        
        for threshold in thresholds_list:
            print(f"! Testing threshold {threshold}")

            thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > threshold).cast('double') )

            currentYearMetrics = testModelPerformance(thresholdPredictions)
            stats = pd.DataFrame([currentYearMetrics], columns=['val_Precision','val_Recall','val_F0.5','val_F1','val_Accuracy'])
            stats['cv_fold'] = i
            stats['regParam'] = regParam_input
            stats['elasticNetParam'] = elasticNetParam_input
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

regParamGrid = [0.0, 0.01, 0.5, 2.0]
maxIterGrid = [5, 10, 50]
thresholds = [0.5, 0.6, 0.7, 0.8]

# regParamGrid = [0.0, 0.5]
# maxIterGrid = [10]
# thresholds = [0.5, 0.6, 0.7, 0.8]

grid_search = pd.DataFrame()

for maxIter in maxIterGrid:
    print(f"! maxIter = {maxIter}")
    for regParam in regParamGrid:
        print(f"! regParam = {regParam}")
        try:
            cv_stats = runBlockingTimeSeriesCrossValidation_SVM(preppedTrain, cv_folds=4, regParam_input=regParam, 
                                                            maxIter_input=maxIter, 
                                                            thresholds_list = thresholds)
#                 test_results = predictTestData(cv_stats, preppedTest)

            grid_search = pd.concat([grid_search,cv_stats],axis=0)
        except:
            print('Error, continuing to next iteration')
            continue
            
test_results = predictTestData(grid_search, preppedTest)

print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

test_results
