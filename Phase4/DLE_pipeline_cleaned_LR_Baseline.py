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

categoricals = ['QUARTER','MONTH','DAY_OF_WEEK','OP_UNIQUE_CARRIER']

numerics = ['DISTANCE','ELEVATION','HourlyAltimeterSetting','HourlyDewPointTemperature',
            'HourlyWetBulbTemperature','HourlyDryBulbTemperature','HourlyPrecipitation',
            'HourlyStationPressure','HourlySeaLevelPressure','HourlyRelativeHumidity',
            'HourlyVisibility','HourlyWindSpeed']

# COMMAND ----------

print(df_joined_data_all_with_efeatures.count())
print(df_joined_data_all_with_efeatures.filter(col('DEP_DEL15')==1).count())
print(df_joined_data_all_with_efeatures.filter(col('DEP_DEL15')==0).count())

pipelineModel = buildPipeline(df_joined_data_all_with_efeatures.filter( col('YEAR') != 2021), 
                              categoricals, numerics, Y="DEP_DEL15", oneHot=True, imputer=False, scaler=False)

preppedData = pipelineModel.transform(df_joined_data_all_with_efeatures)
print(preppedData.count())
print(preppedData.filter(col('DEP_DEL15')==1).count())
print(preppedData.filter(col('DEP_DEL15')==0).count())
# preppedValid = pipelineModel.transform(valid).cache()
# preppedTest = pipelineModel.transform(test).cache()

preppedTrain = preppedData.filter( col('YEAR') != 2021)\
                   .withColumn("DEP_DATETIME_LAG_percent", percent_rank().over(Window.partitionBy().orderBy("DEP_DATETIME_LAG")))\
                   .cache()

# valid = df_joined_data_all_with_efeatures.filter((col('FL_DATE') >= '2020-07-01') & (col('FL_DATE') < '2021-01-01')).cache()
preppedTest = preppedData.filter(col('YEAR') == 2021).cache()

# COMMAND ----------

(22310611 - 22185570) #/22310611

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

regParamGrid = [0.0, 0.01, 0.5, 2.0]
elasticNetParamGrid = [0.0, 0.5, 1.0]
maxIterGrid = [5, 10]
thresholds = [0.5, 0.6, 0.7]

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

timestamp = pd.to_datetime('today').strftime('%m%d%y%H')
grid_spark_DF = spark.createDataFrame(grid_search.drop(columns=['trained_model']))
grid_spark_DF.write.mode('overwrite').parquet(f"{blob_url}/BaselineLR_grid_CV_valResults_{timestamp}")

# COMMAND ----------

agg_results = grid_search.drop(columns=['trained_model']).groupby(['regParam','elasticNetParam','maxIter','threshold']).mean()

rP, eNP, mI, thresh = agg_results[agg_results['val_F0.5'] == agg_results['val_F0.5'].max()].index[0]

best_model = grid_search[(grid_search['regParam']==rP) & 
                               (grid_search['elasticNetParam']==eNP) & 
                               (grid_search['maxIter']==mI) & 
                               (grid_search['threshold']==thresh)]

best_model_save = best_model[best_model['val_F0.5']==best_model['val_F0.5'].max()].iloc[0]['trained_model']

best_model

# COMMAND ----------

timestamp

# COMMAND ----------

preds = best_model_save.transform(preppedTest).withColumn("predicted_probability", extract_prob_udf(col("probability")))

preds.write.mode('overwrite').parquet(f"{blob_url}/best_BaselineLR_predictions_{timestamp}")

# COMMAND ----------

test_results = predictTestData(grid_search[grid_search['val_F0.5']>0], preppedTest)
test_results

# COMMAND ----------

display(test_results.drop(columns=['trained_model']))

# COMMAND ----------

grid_spark_DF = spark.createDataFrame(test_results.drop(columns=['trained_model']))
grid_spark_DF.write.mode('overwrite').parquet(f"{blob_url}/baseline_logistic_regression_grid_CV_120322")

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

preds.write.mode('overwrite').parquet(f"{blob_url}/best_BaselineLR_predictions")

# COMMAND ----------

feature_importances = getFeatureImportance(feature_names, best_model_save.coefficients)
feature_importances

# COMMAND ----------

featureImportanceDF = spark.createDataFrame(feature_importances)
featureImportanceDF.write.mode('overwrite').parquet(f"{blob_url}/best_BaselineLR_feature_importance")

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

fScoreThresholds = best_model_save.summary.precisionByThreshold.join(best_model_save.summary.recallByThreshold, on='threshold')\
                                                               .withColumn('F05', FScore_udf(F.lit(0.5), col("precision"), col('recall')))


# COMMAND ----------

thresh = fScoreThresholds.select('threshold').collect()
scores = fScoreThresholds.select('F05', 'precision','recall').collect()
plt.plot(thresh, scores)
# plt.xlabel('Threshold')

# COMMAND ----------

# plt.figure(figsize=(15,8))
# plt.gca().set_prop_cycle(['red', 'green', 'blue'])
# fig, ax = plt.subplots()
# ax.set_color_cycle(['red', 'green', 'blue'])

plt.plot(thresh, scores)
plt.legend(['F0.5 Score', 'Precision','Recall'], bbox_to_anchor=(1.3, 1.0))
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title(f'Scores By Threshold')
plt.show()

# COMMAND ----------


# thresh = fScoreThresholds.select('threshold').collect()

# plt.figure(figsize=(15,10))
# plt.plot(thresh, fScoreThresholds.select('F05').collect(),color='red')
# plt.plot(thresh, fScoreThresholds.select('precision').collect(), color='green')
# plt.plot(thresh, fScoreThresholds.select('recall').collect(), color='blue')

plt.plot(fScoreThresholds.select('threshold').collect(),fScoreThresholds.select('F05', 'precision','recall').collect())
plt.legend(['F0.5 Score', 'Precision','Recall'], bbox_to_anchor=(1.3, 1.05))
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title(f'Scores By Threshold')
plt.show()

# COMMAND ----------

# best_model_save.summary

plt.figure(figsize=(8,5))
plt.plot(best_model_save.summary.objectiveHistory)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# COMMAND ----------


