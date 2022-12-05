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

# Load Dataframes
print("**Loading Data")

# Inspect the Joined Data folders 
display(dbutils.fs.ls(f"{blob_url}"))

print("**Data Loaded")
print("**Loading Data Frames")

LR_pred = spark.read.parquet(f"{blob_url}/best_LR_predictions").select('flight_id','DEP_DEL15','prediction'
                                                                      ).withColumnRenamed("prediction",'LR_prediction')

GBT_pred = spark.read.parquet(f"{blob_url}/best_GBT_predictions_12042202").select('flight_id','DEP_DEL15','prediction'
                                                                                 ).withColumnRenamed("prediction",'GBT_prediction')

# MLP1_pred = spark.read.parquet(f"{blob_url}/best_MLPNN_predictions").select('flight_id','DEP_DEL15','prediction'
#                                                                            ).withColumnRenamed("prediction",'MLP1_prediction')

MLP_pred = spark.read.parquet(f"{blob_url}/best_MLPNN_predictions_120322").select('flight_id','DEP_DEL15','prediction'
                                                                                  ).withColumnRenamed("prediction",'MLP_prediction')


print("**Data Frames Loaded")

# COMMAND ----------

full_preds = LR_pred.join(GBT_pred, ['flight_id','DEP_DEL15']).join(MLP_pred, ['flight_id','DEP_DEL15']) #.join(MLP2_pred, ['flight_id','DEP_DEL15'])
display(full_preds)

# COMMAND ----------


pred_columns = [column for column in full_preds.columns if 'pred' in column]

ensemble_predictions = full_preds.withColumn('total_pred', F.expr( '+'.join(pred_columns) ))\
                                 .withColumn('avg_pred', col("total_pred") / F.lit(len(pred_columns)))

display(ensemble_predictions)

# COMMAND ----------

metrics_LR = testModelPerformance(ensemble_predictions.withColumnRenamed("LR_prediction", 'prediction'))
metrics_GBT = testModelPerformance(ensemble_predictions.withColumnRenamed("GBT_prediction", 'prediction'))
metrics_MLP = testModelPerformance(ensemble_predictions.withColumnRenamed("MLP_prediction", 'prediction'))

# metrics_smallMajority = testModelPerformance(ensemble_predictions.withColumn("prediction", col('avg_pred')>=0.5))
metrics_bigMajority = testModelPerformance(ensemble_predictions.withColumn("prediction", col('avg_pred')>0.5))
metrics_any = testModelPerformance(ensemble_predictions.withColumn("prediction", col('avg_pred')>0))
metrics_all = testModelPerformance(ensemble_predictions.withColumn("prediction", col('avg_pred')==1))

metrics = [metrics_LR, metrics_GBT, metrics_MLP, metrics_bigMajority, metrics_any, metrics_all]
metric_type = ['Logistic Regression', 'Gradient Boosted Tree', 'Multilayer Perceptron', 'Ensemble Majority', 'Ensemble At Least One', 'Ensemble Unanimous']

stats = pd.DataFrame()
for metric, model_type in zip(metrics, metric_type):
    results = pd.DataFrame([metric], columns=['test_Precision','test_Recall','test_F0.5','test_F1','test_Accuracy'])
    results['Model'] = model_type
    
    stats = pd.concat([stats, results], axis=0)

stats

# COMMAND ----------

grid_spark_DF = spark.createDataFrame(grid_search.drop(columns=['trained_model']))
grid_spark_DF.write.mode('overwrite').parquet(f"{blob_url}/logistic_regression_grid_CV_112822")

# COMMAND ----------


