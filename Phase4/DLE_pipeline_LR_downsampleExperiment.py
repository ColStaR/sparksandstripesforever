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

df_joined_data_all_with_efeatures = spark.read.parquet(f"{blob_url}/joined_all_with_efeatures")
display(df_joined_data_all_with_efeatures)

print("**Data Frames Loaded")

# COMMAND ----------

def getCurrentDateTimeFormatted():
    return str(datetime.utcnow()).replace(" ", "-").replace(":", "-").replace(".", "-")
  
def downsample(df):
    downsampling_0 = df.filter((col("DEP_DEL15") == 0) & (col('YEAR')<2021))
    downsampling_1 = df.filter((col("DEP_DEL15") == 1) | (col('YEAR')==2021))

    downsampling_ratio = (downsampling_1.filter(col('YEAR')!=2021).count() / downsampling_0.count())

    downsampling_append = downsampling_0.sample(fraction = downsampling_ratio, withReplacement = False, seed = 261)
    downsampledDF = downsampling_1.union(downsampling_append)
    
    return downsampledDF

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

df_predownsampled = spark.read.parquet(f"{blob_url}/joined_all_with_efeatures_Downsampled")

print(df_predownsampled.count())
print(df_predownsampled.filter(col('YEAR')==2021).count())
print(df_predownsampled.filter(col('YEAR')!=2021).count())
print(df_predownsampled.filter((col('DEP_DEL15')==1) & (col('DEP_DEL15')!=2021)).count())
print(df_predownsampled.filter((col('DEP_DEL15')==0) & (col('DEP_DEL15')!=2021)).count())

# COMMAND ----------

print(df_joined_data_all_with_efeatures.count())
print(df_joined_data_all_with_efeatures.filter(col('YEAR')==2021).count())
print(df_joined_data_all_with_efeatures.filter(col('YEAR')!=2021).count())
print(df_joined_data_all_with_efeatures.filter((col('DEP_DEL15')==1) & (col('DEP_DEL15')!=2021)).count())
print(df_joined_data_all_with_efeatures.filter((col('DEP_DEL15')==0) & (col('DEP_DEL15')!=2021)).count())

# COMMAND ----------

df_joined_data_all_with_efeatures = downsample(df_joined_data_all_with_efeatures)

print(df_joined_data_all_with_efeatures.count())
print(df_joined_data_all_with_efeatures.filter(col('YEAR')==2021).count())
print(df_joined_data_all_with_efeatures.filter(col('YEAR')!=2021).count())
print(df_joined_data_all_with_efeatures.filter((col('DEP_DEL15')==1) & (col('DEP_DEL15')!=2021)).count())
print(df_joined_data_all_with_efeatures.filter((col('DEP_DEL15')==0) & (col('DEP_DEL15')!=2021)).count())

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
                              categoricals, numerics, Y="DEP_DEL15", oneHot=True, imputer=False, scaler=False)

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

def FScore(beta, precision, recall):
    if precision + recall == 0:
        F = 0
    else:
        F = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return F
FScore_udf = F.udf(FScore, DoubleType())
    
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
def predictTestData(cv_stats, dataFrameInput, featureCol='features'):
    
    print(f"@ Starting Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    # Prepare 2021 Test Data
    selectedcols = ["DEP_DEL15", "YEAR", featureCol]
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

    final_stats = pd.concat([test_stats.reset_index(drop=True), cv_stats.reset_index(drop=True)], axis=1)
    
    #clean up cachced DFs
    dataset.unpersist()
    
    return final_stats
  
    
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


def getFeatureImportance(featureNames, coefficients):
    
    featureImportances = pd.DataFrame(zip(featureNames,coefficients), columns=['featureName','coefficient'])
    featureImportances['importance'] = featureImportances['coefficient'].abs()
    
    return featureImportances.sort_values('importance', ascending=False)

# COMMAND ----------

def runBlockingTimeSeriesCrossValidation(preppedTrain, featureCol='features', cv_folds=4, regParam_input=0, elasticNetParam_input=0,
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
                                .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", featureCol]).cache()
        
        cv_val = preppedTrain.filter((col("DEP_DATETIME_LAG_percent") >= train_cutoff) & (col("DEP_DATETIME_LAG_percent") < max_perc))\
                              .select(["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", featureCol]).cache()
        
        lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol=featureCol, regParam = regParam_input, elasticNetParam = elasticNetParam_input, 
                                maxIter = maxIter_input, threshold = 0.5, standardization = True)

        lrModel = lr.fit(cv_train)
        
        currentYearPredictions = lrModel.transform(cv_val).withColumn("predicted_probability", extract_prob_udf(col("probability"))).cache()
        
        print(f"!! Starting threshold search")
        for threshold in thresholds_list:
#             print(f"! Testing threshold {threshold}")

            thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                         .withColumn("prediction", (col('predicted_probability') > threshold).cast('double')).cache()

            currentYearMetrics = testModelPerformance(thresholdPredictions)
            stats = pd.DataFrame([currentYearMetrics], columns=['val_Precision','val_Recall','val_F0.5','val_F1','val_Accuracy'])
            stats['cv_fold'] = i
            stats['regParam'] = regParam_input
            stats['elasticNetParam'] = elasticNetParam_input
            stats['maxIter'] = maxIter_input
            stats['threshold'] = threshold
            stats['trained_model'] = lrModel

            cv_stats = pd.concat([cv_stats,stats],axis=0)

    #clean up cachced DFs
    cv_train.unpersist()
    cv_val.unpersist()
    currentYearPredictions.unpersist()
    thresholdPredictions.unpersist()
    
    return cv_stats


# COMMAND ----------

regParamGrid = [0.5, 2]
elasticNetParamGrid = [0.5, 1]
maxIterGrid = [10]
thresholds = [0.5, 0.6]


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

                grid_search = pd.concat([grid_search,cv_stats],axis=0)
            except:
                print('Error, continuing to next iteration')
                continue


grid_search

# COMMAND ----------

grid_search[grid_search['val_F0.5']>0]

# COMMAND ----------

test_results = predictTestData(grid_search[grid_search['val_F0.5']>0], preppedTest)
test_results

# COMMAND ----------

test_results['test_F0.5'].max()

# COMMAND ----------

grid_spark_DF = spark.createDataFrame(test_results.drop(columns=['trained_model']))
grid_spark_DF.write.mode('overwrite').parquet(f"{blob_url}/logistic_regression_grid_CV_downsampleExperiment_2")

# COMMAND ----------

agg_results = test_results.drop(columns=['trained_model']).groupby(['regParam','elasticNetParam','maxIter','threshold']).mean()

rP, eNP, mI, thresh = agg_results[agg_results['val_F0.5'] == agg_results['val_F0.5'].max()].index[0]

best_model = test_results[(test_results['regParam']==rP) & 
                               (test_results['elasticNetParam']==eNP) & 
                               (test_results['maxIter']==mI) & 
                               (test_results['threshold']==thresh)]

best_model_save = best_model[best_model['val_F0.5']==best_model['val_F0.5'].max()].iloc[0]['trained_model']

best_model

# COMMAND ----------

preds = best_model_save.transform(preppedTest).withColumn("predicted_probability", extract_prob_udf(col("probability")))

preds.write.mode('overwrite').parquet(f"{blob_url}/best_LR_DownsampleExperiment_predictions_2")

# COMMAND ----------

feature_importances = getFeatureImportance(feature_names, best_model_save.coefficients)
feature_importances

# COMMAND ----------

featureImportanceDF = spark.createDataFrame(feature_importances)
featureImportanceDF.write.mode('overwrite').parquet(f"{blob_url}/best_LR_DownsampleExperiment_feature_importance_2")

feature_importances.head(50)

# COMMAND ----------

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(best_model_save.summary.roc.select('FPR').collect(),
         best_model_save.summary.roc.select('TPR').collect())
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'ROC, with AUC={best_model_save.summary.areaUnderROC}')
plt.show()


# COMMAND ----------

# maxFMeasure = fScoreThresholds.groupBy().max('F05').select('max(F05)').head()
# bestThreshold = fScoreThresholds.where(col('F05') == maxFMeasure['max(F05)']) \
#     .select('threshold').head()['threshold']

# bestThreshold

fScoreThresholds.agg({"F05": "max"}).show()

# COMMAND ----------

fScoreThresholds = best_model_save.summary.precisionByThreshold.join(best_model_save.summary.recallByThreshold, on='threshold')\
                                                               .withColumn('F05', FScore_udf(F.lit(0.5), col("precision"), col('recall')))

thresh = fScoreThresholds.select('threshold').collect()

# plt.figure(figsize=(15,10))
plt.plot(thresh, fScoreThresholds.select('F05').collect(),color='red')
plt.plot(thresh, fScoreThresholds.select('precision').collect(), color='green')
plt.plot(thresh, fScoreThresholds.select('recall').collect(), color='blue')
plt.legend(['F0.5 Score', 'Precision','Recall'], bbox_to_anchor=(1.05, 1))
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title(f'Scores By Threshold')
plt.show()

# COMMAND ----------

best_model_save.summaryplt.figure(figsize=(8,5))
plt.plot(best_model_save.summary.objectiveHistory)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### GBT

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

# Hyperparameter Tuning Parameter Grid for Random Forest
# Each CV takes about 30 minutes.

maxIterGrid = [5, 10, 50]
maxDepthGrid = [4, 8, 16]
maxBinsGrid = [32, 64, 128]
stepSizeGrid = [0.1, 0.5]
thresholds = [0.5, 0.6, 0.7, 0.8]


grid_search = pd.DataFrame()

for maxIter in maxIterGrid:
    print(f"! maxIter = {maxIter}")
    for stepSize in stepSizeGrid:
        print(f"! stepSize = {stepSize}")
        for maxBins in maxBinsGrid:
            print(f"! maxBins = {maxBins}")
            for maxDepth in maxDepthGrid:
                print(f"! maxDepth = {maxDepth}")
            try:
                cv_stats = runBlockingTimeSeriesCrossValidation_GBT(preppedTrain, cv_folds=4, input_maxIter = maxIter, 
                                                                    input_maxDepth = maxDepth, input_maxBins = maxBins, 
                                                                    input_stepSize = stepSize, thresholds_list = thresholds)

                grid_search = pd.concat([grid_search,cv_stats],axis=0)
            except:
                continue

test_results = predictTestData(cv_stats, preppedTest)

print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

test_results

# COMMAND ----------

grid_spark_DF = spark.createDataFrame(test_results.drop(columns=['trained_model']))
grid_spark_DF.write.mode('overwrite').parquet(f"{blob_url}/GBT_grid_CV_120122")

# COMMAND ----------

# grid_search['trained_model'][0].coefficients

len(list(grid_search['trained_model'].iloc[0].featureImportances))

# COMMAND ----------

# stages[0].getInputCol()
# stages[0].getOutputCol()

# stages[-1].getOutputCol()

# stages[-1].getInputCols()

display(preppedDataDF.select('ORIGIN'))

# COMMAND ----------

print(f"@ Starting Test Evaluation")
print(f"@ {getCurrentDateTimeFormatted()}")
# Prepare 2021 Test Data
currentYearDF = dataFrameInput.filter(col("YEAR") == 2021).cache()
preppedDF = currentYearDF.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))
selectedcols = ["DEP_DEL15", "YEAR", "DEP_DATETIME_LAG_percent", "features"]
dataset = preppedDF.select(selectedcols).cache()
#        display(dataset)

# Evaluate best model from cross validation against the test data frame of 2021 data, then print evaluation metrics.
testDataSet = dataset.limit(10).cache()


# COMMAND ----------

test_predictions = testDataSet.select("DEP_DEL15")

for i in range(len(cv_yearly_best)):
    currentYearPredictions = runLogisticRegression(cv_yearly_best.iloc[i]['trained_model'], testDataSet)\
                            .withColumn("predicted_probability", extract_prob_udf(col("probability"))).cache()
    
    thresholdPredictions = currentYearPredictions.select('DEP_DEL15','predicted_probability')\
                                                 .withColumn(f"{cv_yearly_best.iloc[i]['year']}_prediction", (col('predicted_probability') > cv_yearly_best.iloc[i]['threshold']).cast('double') )
        
    test_predictions = test_predictions.withColumn(f"{cv_yearly_best.iloc[i]['year']}", thresholdPredictions.select('prediction'))
    
test_predictions

# COMMAND ----------

test_eval = pd.DataFrame()
for year in cv_yearly_best['year'].unique():
    stats = pd.DataFrame([testModelPerformance(testMetrics)], columns=['test_Precision','test_Recall','test_F0.5','test_F1','test_Accuracy'])
    stats = pd.concat([stats, cv_yearly_best.iloc[[i]]], axis=1)
        
    test_eval = pd.concat([test_eval,stats],axis=0)
    
    
test_eval

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

# Downsampling Test

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
                runBlockingTimeSeriesCrossValidation_downsampling(preppedDataDF, regParam, elasticNetParam, maxIter, threshold)
print("! Job Finished!")
print(f"! {getCurrentDateTimeFormatted()}\n")

# COMMAND ----------

current_metrics = spark.read.parquet(f"{blob_url}/logistic_regression_metrics")
display(current_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Random Forest

# COMMAND ----------

def reportMetrics_rf(precision, fPointFive, recall, accuracy):
    """Outputs metrics (currently only for Linear Regression?)"""
    print("Precision = %s" % precision)
    print("F0.5 = %s" % fPointFive)
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
    
    # Beta = 0.5, so (1 + (0.5 ^ 2)) = 1.25. Python didn't like (0.5 ^ 2) for some reason.
    fPointFive = (1.25 * precision * recall) / ((0.25 * precision) + recall)
    
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
    
    topPrecision = 0
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
        
        # Downsample the data such that there are roughly equal amounts of rows where DEP_DEL15 == 0 and DEP_DEL15 == 1, which aids in training.
        
        currentYearDF_downsampling_0 = currentYearDF.filter(col("DEP_DEL15") == 0).cache()
#        print(f"@- currentYearDF_downsampling_0.count() = {currentYearDF_downsampling_0.count()}")
        currentYearDF_downsampling_1 = currentYearDF.filter(col("DEP_DEL15") == 1).cache()
#        print(f"@- currentYearDF_downsampling_1.count() = {currentYearDF_downsampling_1.count()}")

        downsampling_ratio = (currentYearDF_downsampling_1.count() / currentYearDF_downsampling_0.count())

        currentYearDF_downsampling_append = currentYearDF_downsampling_0.sample(fraction = downsampling_ratio, withReplacement = False, seed = 261)
        
        currentYearDF_downsampled = currentYearDF_downsampling_1.unionAll(currentYearDF_downsampling_append)
        trainingRows += currentYearDF_downsampled.count()
#        print(f"@- currentYearDF_downsampled.count() = {currentYearDF_downsampled.count()}")
        print(f"Finished Downsampling for {currentYear}")
        print(f"@ {getCurrentDateTimeFormatted()}")    
    
        # Adds a percentage column to each year's data frame, with the percentage corresponding to percentage of the year's time. 
        # 0% = earliest time that year. 100% = latest time that year.
        preppedDataDF = currentYearDF_downsampled.withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG"))).cache()

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
        
        print(f"Training model for {currentYear}")
        print(f"@ {getCurrentDateTimeFormatted()}")    
        rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", numTrees = input_numTrees, maxDepth = input_maxDepth, maxBins = input_maxBins)
        rfModel = rf.fit(trainingData)
        print(f"Finished training model for {currentYear}")
        print(f"@ {getCurrentDateTimeFormatted()}")    

        
        print(f"Evaluating test training data for {currentYear}")
        print(f"@ {getCurrentDateTimeFormatted()}")    
        precision, fPointFive, recall, accuracy = runRandomForest(rfModel, trainingTestData)
        print(f"Finished evaluating test training data for {currentYear}")
        print(f"@ {getCurrentDateTimeFormatted()}")    
        
        # Compare and store top models and metrics.
        if topPrecision == 0:
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
    testDataSet = preppedDataDF.select(selectedcols).cache()
#    display(testDataSet)

    # Evaluate best model from cross validation against the test data frame of 2021 data, then print evaluation metrics.
    print(f"Evaluating test data.")
    print(f"@ {getCurrentDateTimeFormatted()}")
    precision, fPointFive, recall, accuracy = runRandomForest(topModel, testDataSet)

    print(f"\n - 2021")
    print("Model Parameters:")
    print("numTrees:", input_numTrees)
    print("maxDepth:", input_maxDepth)
    print("maxBins:", input_maxBins)
    print("testDataSet Count:", testDataSet.count())
    print("Training Rows Count:", trainingRows)
    print("topYear:", topYear)
    reportMetrics_rf(precision, fPointFive, recall, accuracy)
    saveMetricsToAzure_RF(precision, fPointFive, recall, accuracy, input_numTrees, input_maxDepth, input_maxBins)
    print(f"@ Finised Test Evaluation")
    print(f"@ {getCurrentDateTimeFormatted()}")
    

# COMMAND ----------

# ~ 2.5 hours per CV. Evaluating each year takes about 20 minutes. Be mindful of how many parameters you are passing in.

#numTrees = [10, 25, 50]
#maxDepth = [4, 8, 16]
#maxBins = [32, 64, 128]

numTreesGrid = [10, 25]
maxDepthGrid = [4, 8]
maxBinsGrid = [32, 64]

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

# Show Saved Metrics
current_metrics = spark.read.parquet(f"{blob_url}/random_forest_metrics")
display(current_metrics)

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

pageRanks_PriorTo21 = spark.read.csv(f"{blob_url}/PageRanks_PriorTo21.csv", header=True)
display(pageRanks_PriorTo21)

pageRanks_Quarter_Year = spark.read.csv(f"{blob_url}/PageRanks_Quarter_Year.csv", header=True)
display(pageRanks_Quarter_Year)

# COMMAND ----------

display(df_joined_data_all_with_efeatures.filter(col("YEAR") == 2021))

# COMMAND ----------

display(df_joined_data_all_with_efeatures.filter(col("AssumedEffect").isNull()))
print(df_joined_data_all_with_efeatures.filter(col("AssumedEffect").isNull()).count())

display(df_joined_data_all_with_efeatures.filter(col("AssumedEffect").isNotNull()))
print(df_joined_data_all_with_efeatures.filter(col("AssumedEffect").isNotNull()).count())

# COMMAND ----------

display(df_joined_data_all_with_efeatures.filter(col("is_prev_delayed").isNull()))
print(df_joined_data_all_with_efeatures.filter(col("is_prev_delayed").isNull()).count())

display(df_joined_data_all_with_efeatures.filter(col("is_prev_delayed").isNotNull()))
print(df_joined_data_all_with_efeatures.filter(col("is_prev_delayed").isNotNull()).count())

# COMMAND ----------

display(df_joined_data_all_with_efeatures.filter(col("perc_delay").isNull()))
print(df_joined_data_all_with_efeatures.filter(col("perc_delay").isNull()).count())

display(df_joined_data_all_with_efeatures.filter(col("perc_delay").isNotNull()))
print(df_joined_data_all_with_efeatures.filter(col("perc_delay").isNotNull()).count())

# COMMAND ----------


