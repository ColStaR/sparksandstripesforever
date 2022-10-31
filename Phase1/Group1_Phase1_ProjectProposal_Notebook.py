# Databricks notebook source
# MAGIC %md
# MAGIC # Team 13 Phase 1 Project Proposal
# MAGIC ## Section 4, Group 1
# MAGIC ## Sparks and Stripes Forever
# MAGIC Nashat Cabral, Deanna Emery, Nina Huang, Ryan S. Wong

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Phase Leader Plan
# MAGIC 
# MAGIC Per the instructions, each weekly phase will be led by a different member of the team. Below is a table showing the planned leader for each of the upcoming phases.
# MAGIC 
# MAGIC https://docs.google.com/spreadsheets/d/1Va1bwlEmrIrOc1eFo1ySYlPQpt4kZjDqahQABgw0la4/edit#gid=0
# MAGIC 
# MAGIC 
# MAGIC | Phase Number | Phase Leader    |
# MAGIC | ------------ | --------------- |
# MAGIC | Phase 1      | Ryan S. Wong    |
# MAGIC | Phase 2      | Nashat Cabral   |
# MAGIC | Phase 3      | Deanna Emery    |
# MAGIC | Phase 4      | Nina Huang      |
# MAGIC | Phase 5      | Team Submission |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Credit Assignment Plan
# MAGIC ### Phase 1 Contributions
# MAGIC 
# MAGIC | Task                                                   | Estimated Time (hours) | Nashat Cabral | Deanna Emery | Nina Huang | Ryan S. Wong |
# MAGIC | ------------------------------------------------------ | ---------------------- | ------------- | ------------ | ---------- | ------------ |
# MAGIC | Project Abstract Section                               | 1                      | X             |              | X          |              |
# MAGIC | Data Overview Section                                  | 4                      | X             |              |            |              |
# MAGIC | The Desired Outcomes and Metrics Section               | 2                      |               |              |            | X            |
# MAGIC | Data Ingesting and Pipeline Section                    | 4                      |               |              |            | X            |
# MAGIC | Joining Datasets Section                               | 4                      |               | X            |            |              |
# MAGIC | Machine Learning Algorithms to be Used Section         | 2                      |               | X            |            |              |
# MAGIC | Resource Management & Performance Optimization Section | 4                      |               |              | X          |              |
# MAGIC | Train/Test Data Splits Section                         | 2                      |               |              | X          |              |
# MAGIC | Conclusions and Next Steps Section                     | 2                      | X             |              | X          |              |
# MAGIC | Open Issues or Problems Section                        | 2                      | X             |              | X          |              |
# MAGIC | Set up Databricks instance                             | 2                      |               |              |            | X            |
# MAGIC | Set up GitHub and Integrate with Databricks            | 1                      |               |              |            | X            |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Flight Delay Prediction Model Project
# MAGIC 
# MAGIC ## Project Team: Sparks and Stripes Forever
# MAGIC 
# MAGIC Nashat Cabral, Deanna Emery, Nina Huang, Ryan S. Wong

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Project Abstract
# MAGIC 
# MAGIC In the flight industry, delays are a key issue for airline companies, airports, and customers. These events account for significant financial and time losses across these groups. This project seeks to predict the occurrence, or lack thereof, a delayed departing flight by using airport, flight, and local weather data to create a machine learning model that can effectively predict flight delays. Any analyses and methods applied will come from the perspective of benefiting the customer, and thus the model would need to place greater emphasis on correctly identifying non-delayed flights, as incorrectly identifying these events would be detrimental to the customer experience. Such a model would be capable of minimizing costs while improving customer satisfaction in their business. This document outlines the high level plan for approaching this problem.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## The Data
# MAGIC 
# MAGIC For this project, we will be examining the data from two tables of considerable size, along with an additional smaller table:
# MAGIC 
# MAGIC - Flight Data: The first of the larger tables will consist of a collection of the “on-time” performance data for passenger flights from the U.S. Department of Transportation (DOT). These flights will be limited to domestic flights within the United States and its territories. 
# MAGIC - Weather Data: The second of the larger tables is composed of weather data that can be used to determine the effect of weather conditions on flight performance for airports within the region. 
# MAGIC - Airport Data: The final, smallest, table houses metadata on airports, such as their location and ID. 
# MAGIC 
# MAGIC By exploring these datasets in conjunction with one another, we hope to develop a better understanding of the fields within and their possible relationships to one another. The information on the datasets below is obtained from the dataset documentation provided, as well as preliminary analyses on fields of interest.
# MAGIC 
# MAGIC The Airline On-Time Performance Data table contains the scheduled and actual departure/arrival times for U.S. Domestic flights for qualifying airline carriers. These carriers must account for at least one percentage of U.S Domestic scheduled passenger revenues in order to qualify. Our data ranges from 2015 to 2021, for the purposes of this preliminary analysis of the data, we will be examining the data from “/parquet_airlines_data_3m/”  which consists of flight data from the first quarter of 2015. In this study, canceled flights will be considered with the same regard for customers as delayed flights. Variables of interest within this dataset include: 
# MAGIC 
# MAGIC - ORIGIN_AIRPORT_ID- Identifier for the airport of departure
# MAGIC - DEST_AIRPORT_ID- Identifier for the airport of arrival
# MAGIC - FL_DATE- scheduled flight date 
# MAGIC - DEP_DELAY_NEW- numerical variable, difference in minutes between scheduled and actual departure time with early departures are set to 0
# MAGIC - DEP_DEL15- binary categorical variable that indicates if a flight departure was delayed by more than 15 minutes
# MAGIC - ARR_DELAY_NEW-  numerical variable, difference in minutes between scheduled and actual arrival time, early arrivals are set to 0
# MAGIC - ARR_DEL15- binary categorical variable that indicates if a flight arrival was delayed by more than 15 minutes
# MAGIC - CANCELLED- binary categorical variable indicating whether flight was canceled 
# MAGIC - DIVERTED- binary categorical variable indicating whether flight was diverted
# MAGIC - CARRIER_DELAY - numerical variable, indicates time spent delayed due to carrier
# MAGIC - WEATHER_DELAY - numerical variable, indicates time spent delayed due to weather
# MAGIC - NAS_DELAY - numerical variable, indicates time spent delayed due to National Air System
# MAGIC - SECURITY_DELAY - numerical variable, indicates time spent delayed due to security
# MAGIC - LATE AIRCRAFT DELAY - numerical variable, indicates time spent delayed due to a late aircraft
# MAGIC - ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID, and FL_DATE will likely be combined to create a composite key as a unique identifer for each scheduled flight.
# MAGIC 
# MAGIC The below two figures display summary statistics for our numeric variables, as well as null value counts for our chosen variables.
# MAGIC ![img1](files/tables/airlinestats.PNG)
# MAGIC ![img2](files/tables/airlinenull.PNG)
# MAGIC 
# MAGIC Null values shown above may indicate flights without delays, but will likely need to be removed/modified moving forward.
# MAGIC 
# MAGIC The Quality Controlled Local Climatological Data contains summary data from weather stations housed at airports. These stations log daily temperature highs/lows, precipitation, wind speed, visibility, and storm characteristics. The available data ranges from 2015 to 2021, for the purposes of this preliminary analysis of the data, we will be examining the data from “/parquet_weather_data_3m/”  which consists of weather data from the first quarter of 2015. (expand on variables). Variables of interest within this dataset are any that may have a relationship with flight delays, such as: 
# MAGIC 
# MAGIC - Station - identifier for each station
# MAGIC - Date - Year-Month-Day-Hour-Minute-Second identifier for the date of a record, the field providing data to the hour allows for the field to identify hourly data.
# MAGIC - HourlyWindSpeed - numerical variable, indicates wind speed in meters per second, 9999’s are considered missing values.
# MAGIC - HourlySkyConditions  - Height in meters of the lowest cloud or obscuring item (max of 22,000)
# MAGIC - HourlyVisibility - numerical variable, distance in meters an object can be seen (max of 16000), 999999 is considered missing
# MAGIC - HourlyDryBulbTemperature - numerical variable, temperature of air in celsius, +9999 is considered missing
# MAGIC - HourlySeaLevelPressure - numerical variable, air pressure relative to Mean Sea Level in hectopascals, 99999 is considered missing
# MAGIC - Station and Date will likely be combined into a composite key as a unique identifier for each weather record.
# MAGIC 
# MAGIC The below two figures display summary statistics for our numeric variables, as well as null value counts for our chosen variables.
# MAGIC 
# MAGIC ![img3](files/tables/weatherstats.PNG)
# MAGIC 
# MAGIC ![img4](files/tables/weather_null.PNG)
# MAGIC 
# MAGIC The statistics table shows the max being 9s, which is likely representative of missing data.
# MAGIC The figure for null values above indicates a large portion of data is missing from our dataset, these values may negatively affect any attempted analyses and will likely need to be filtered out.
# MAGIC 
# MAGIC 
# MAGIC The final table, stations_data, houses valuable information on airport location including fields such as: 
# MAGIC - lat - latitude
# MAGIC - lon - longitude 
# MAGIC - station_id - identifier for each station
# MAGIC - Distance_to_neighbor - numeric variable, distance to neighboring station in meters  
# MAGIC - station_id will likely be used as the unique identifier for records within this dataset.
# MAGIC 
# MAGIC The below figure displays summary statistics for our numeric variables of Distance_to_neighbor
# MAGIC 
# MAGIC ![img5](files/tables/stationstats.PNG)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## The Desired Outcomes and Metrics
# MAGIC 
# MAGIC Our desired outcome is to create a machine learning model that will effectively predict what flights will be categorized as delayed. The ability to predict what flights will not arrive on time is highly valuable to individual air travelers. To this intended audience, the ability to accurately predict flight delays will allow them to better economy their time and improve travel scheduling. Recall that within the data set in question, the features “ArrDel15” and “DepDel15” having a value of 1 indicates that the flight had been delayed.
# MAGIC 
# MAGIC For evaluating the effectiveness of our model, it is important that we kept the audience’s intended usage for the model in mind. When predicting flight delays, a false positive prediction is far worse than a false negative outcome: a flight that is predicted to be on-time but arrives late means the flier will have to wait, but a flight that is predicted to be late but arrives on-time means the flier will miss their flight entirely. For that reason, our primary metrics for success will be the F1 and precision scores of the model, with the model’s accuracy being a secondary metric. F1 and precision were both chosen as our primary metrics due to their emphasis on minimizing false positive occurrences while still tracking the positive predictive capabilities of the model. Accuracy is considered to be a secondary metric, as a high accuracy score is not as useful for the business case as F1 and precision , but accuracy does provide a good baseline measure of success for overall predictive ability without overfitting. 
# MAGIC 
# MAGIC 
# MAGIC F1, precision, and accuracy are described further with their equations in the bullet points below.
# MAGIC 
# MAGIC - F1 is the harmonic mean of precision and recall, and is computed using the formula below.
# MAGIC 
# MAGIC \\( F_1 = 2 * \frac{precision \cdot recall}{precision + recall} = \frac{2 * TP}{2 * TP + FP + FN} \\)
# MAGIC 
# MAGIC - Precision is the rate of true positivity within the model. It is computed using the formula below, where TP = true positives and FP = false positives.
# MAGIC 
# MAGIC \\( Precision  = \frac{TP}{TP + FP} \\)
# MAGIC 
# MAGIC - Accuracy is the rate of all correctly classified observations. It is computed using the formula below, where TP = true positives, TN = true negatives, FP = false positives, and FN = false negatives.
# MAGIC 
# MAGIC \\( Accuracy  = \frac{TP + TN}{TP + TN + FP + FN} \\)
# MAGIC 
# MAGIC For creating a proper comparison and target for our model, we will compare our model against a simple baseline model. This baseline will be a model that predicts that every flight will not be delayed. Any improvements in our model over this baseline model will represent an overall improvement in the ability to correctly predict what flights will be delayed. Therefore, our desired model would be a model that has a high F1 and precision score while matching or exceeding the accuracy score of the baseline evaluation.
# MAGIC 
# MAGIC This desired outcome may lead to some interesting alternatives that would be worthwhile to explore as well. One possibility is to change the predicted value from being a categorical one to a numerical one, and focus our efforts towards getting accurate predictions of flight delays. This alternative would require measuring different metrics for success and incorporating different models, but it would be able to be done with the same data. The intended audience and use case would be the same as well.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## The Data Ingesting and Pipeline
# MAGIC 
# MAGIC The raw data in question comes from multiple data sets on third party servers while being multiple gigabytes in size. In order for the data to be made usable, we will establish a data pipeline that conducts the necessary tasks to ingest, clean, transform, join, and make available the data we are interested in. To create this data pipeline, we will be leveraging the ML Pipelines API library, since that library is designed for machine learning pipelines and integrates natively with Apache Spark on our DataBricks instance.
# MAGIC 
# MAGIC The general process for data ingestion and processing through the pipeline is currently planned to take the following steps:
# MAGIC 1. Import data into HDFS
# MAGIC     - The data in question is currently only available on a set of third party servers. We will first copy the raw data files from their current location into our Microsoft Azure  HDFS instance, uncompressing them so that they are fully accessible. 
# MAGIC 2. Convert CSV into Parque
# MAGIC     - Once stored within our cloud storage, we will convert the raw files from their native CSV format into the Parquet file format, as Parquet provides superior storage and computation performance compared to CSV. A copy of this converted raw data in Parque format will then be stored back onto our HDFS instance, which will then be transferred to our DataBricks instance for additional transformations.
# MAGIC 3. Clean data
# MAGIC     - As with all raw data, we are expecting the initial data to contain numerous errors, typos, and missing data. To resolve this, the data will be cleaned of erroneous or troublesome values, and properly formatted so that it is fully usable. 
# MAGIC 4. Transform data: outlier handling, scaling data, rebalancing data, adding features, join data sets, choosing features
# MAGIC     - With the data cleaned, we will then transform the data to be usable for our purposes. 
# MAGIC     - We will start by handling outliers. Depending on what patterns we observe with features containing outliers, we will implement different treatment techniques such as dropping the record and imputing values. 
# MAGIC     - Next we will perform data scaling by applying normalized scaling to features that have a highly variable range of values. Our initial EDA of each feature’s distributions will reveal which features have data ranges that should be normalized.
# MAGIC     - Afterwards, we will balance any features whose distributions are not normally distributed using the Synthetic Minority Oversampling Technique. SMOTE will allow us to artificially rebalance feature distributions, introducing slight noise but greatly increasing our analytical accuracy. An initial EDA will show what features have highly imbalanced distributions that would need to be rebalanced. 
# MAGIC     - We will also add additional derived features from the data that would be useful for our analysis. Lastly, we will be joining relevant data sets together. More information about joining the data sets can be found in the following section.
# MAGIC     - In order to reduce storage and computational complexity, we will select the features in our data set that are most relevant to our analysis; all other features will be dropped from our usable data sets. We will be determining what features to keep and which to ignore after an initial EDA assessment of the data sets.
# MAGIC 5. Model Training
# MAGIC     - After the data is imported, cleaned, and transformed successfully, the data is now ready to be used in our analysis. After saving the completed data sets to our HDFS instance, we will begin training our models in our DataBricks instance using Apache Spark. To minimize the amount of variance within the training set, we will be utilizing cross validation during our training and testing phases. More information about our cross validation methodology can be found in the “Train/Test Data Splits” section below. More information about the machine learning algorithms being used can be found in the section, “Machine Learning Algorithms to be Used” below.
# MAGIC 6. Model Evaluation
# MAGIC     - After the data is trained, we will apply our model against our training data set. As mentioned previously, we will be using cross validation for our test data set. For more information about what evaluation metrics will be used, please refer to the section, “The Desired Outcomes and Metrics”.
# MAGIC 7. Model Re-Training and Hyperparameter Tuning
# MAGIC     - After each iteration of training and evaluation, we will analyze and compare the performance of each model with the intention of improving our models’ desired metrics on the testing data. Each model will be retrained with different hyperparameters in order to increase our models’ performance.
# MAGIC 
# MAGIC Below is a visualization of the pipeline’s components and processes.
# MAGIC 
# MAGIC ![pipeline](/files/tables/image/Group1_Phase1_ProjectProposal_PipelineViz-1.jpg)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Joining Datasets
# MAGIC 
# MAGIC Before we can begin feature engineering and machine learning, we first have to join weather and airlines datasets and address missing values. Since our goal is to predict the departure delay, we will have to drop any rows that are missing values in all three of DEP_DELAY, DEP_DELAY_NEW, and DEP_DEL15 (if any one of these columns is available, we will have sufficient information for our outcome variable). Similarly, if we have a small proportion of rows that are missing any important features, these can likely safely be dropped. Doing so can help improve runtime once we begin joining the datasets.
# MAGIC 
# MAGIC The steps below detail how the join of the airlines and weather datasets can be conducted:
# MAGIC 1. First, collect distinct pairs of airport codes from the ORIGIN_AIRPORT_ID column, and ORIGIN_STATE_ABR.
# MAGIC 2. Confirm that all airport codes and their corresponding states are in the weather stations dataset (this information can be found in the neighbor_call and neighbor_state columns). Perform an inner join on these two fields such that we have a table containing: ORIGIN_AIRPORT_ID (neighbor_call), ORIGIN_STATE_ABR (neighbor_state), station_id, distance_to_neighbor. Confirm that all airport codes and states were matched to at least one weather station, and that all matched weather stations exist in the weather data.
# MAGIC 3. Create a new column called WEATHER_DATETIME by converting the DATE column in the weather data to a datetime type.
# MAGIC 4. Perform an inner join of the airport - weather station linking table onto the weather dataset. The resulting dataset will only contain weather data for the stations of interest, and it may contain duplicate entries of weather if the same weather station corresponds to more than one airport. We may also find that one airport may correspond to more than one weather station.
# MAGIC 5. Identify rows in the merged weather data with duplicated WEATHER_DATETIME and airport (ORIGIN_AIRPORT_ID). Address these duplicates by keeping the row with the smallest value in “distance_to_neighbor”. For any ties, just keep the first occurrence. This strategy allows us to keep more granular time in our weather data by making use of multiple stations.
# MAGIC 6. Create a departure date-time column, called AIRLINES_DATETIME, in the airlines dataset by combining the FL_DATE, and departure time (CRS_DEP_TIME) into a datetime. Add a column called AIRLINES_DATETIME_SHIFTED to the airlines dataset that is equal to the departure datetime minus 2 hours. We will need to round the time down to the hour to help make merging onto the weather data easier, which can be accomplished by removing the last two digits from the departure time before converting to a datetime.
# MAGIC 7. Finally, we can merge the airlines data and the weather data together as a one-sided (left) merge onto the airlines data using the columns: ORIGIN_AIRPORT_ID, AIRLINES_DATETIME_SHIFTED, and WEATHER_DATETIME. The resulting table should have exactly as many rows as the original airlines dataset.
# MAGIC 
# MAGIC With all three datasets merged, we would need to identify the number and proportion of flights that did not have any corresponding weather data. If this is a small enough set, we can drop rows that are missing weather data. If we find that we need to keep these rows, we can fill the missing values with averages for the given ORIGIN_AIRPORT_ID and MONTH. Any other important features with missing values can be handled similarly.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Machine Learning Algorithms to be Used
# MAGIC 
# MAGIC As a basic benchmark, we assume no flights will be deplayed. 
# MAGIC As time permits, we will develop a more sophisticated benchmark: we can predict that a flight will be delayed if the airlines and airport have had any other delays prior in the day. To create this prediction, we can create an hourly cumulative count of delayed flights for each airline and airport over the course of a day with a lag of 2 (in order to avoid data leakage). We can then convert this shifted cumulative sum to a binary feature of ‘greater than 0’ or ‘equal to 0’. 
# MAGIC 
# MAGIC Our goal will be to develop a machine learning algorithm that outperforms our benchmark. In selecting our machine learning algorithms, we weigh in heavily on their ability to parallelize. For example, while certain models are great candidates for time series （such as LSTM), given that they can not be parallelized we will not consider them for the project. As such, we will attempt to implement the following algorithms using PySpark’s MLlib library:
# MAGIC 
# MAGIC - [Logistic Regression](https://spark.apache.org/docs/latest/mllib-linear-methods.html#logistic-regression)
# MAGIC 
# MAGIC   - The logistic regression is one of the most simplistic classification models and always a good place to start. It is bounded between 0 and 1, represented as: \\( f(\bold{x}) = \frac{1}{1 + e^{-\bold{w}^T\bold{x}}} \\)
# MAGIC   - Its output can be interpreted as a likelihood, which gives us the flexibility to define a cutoff likelihood for positive and negative predictions. This means that we can optimize our model for precision or recall as desired.
# MAGIC 
# MAGIC   - The loss function for logistic regression is defined as: \\( L(\bold{w}; \bold{x},y) = \log(1 + \exp(-y\bold{w}^T\bold{x})) \\)
# MAGIC     Where \\( \bold{w} \\) is the trained weights, \\( \bold{x} \\) is the input data, and \\( y \\) is the true outcome value.
# MAGIC 
# MAGIC - [Linear SVM ](https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-support-vector-machine)
# MAGIC 
# MAGIC   - The linear support vector machine algorithm defines a linear decision boundary that best separates positive and negative classes. The loss function for SVM is the Hinge loss, which is defined as: \\( L(\bold{w}; \bold{x},y) = \max(0, 1 - y\bold{w}^T\bold{x})  \\)
# MAGIC     Where \\( \bold{w} \\) is the trained weights, \\( \bold{x} \\) is the input data, and \\( y \\) is the true outcome value. 
# MAGIC   - MLlib’s implementation performs L2 regularization by default and uses an OWLQN optimizer.
# MAGIC 
# MAGIC - [Random Forest](https://docs.google.com/document/d/1ZCUOfiGdChziaCCqxihUFIBQIjL8mRaQNTz-vA0Fhk0/edit#)
# MAGIC 
# MAGIC   - Random Forest is an ensemble model of decision trees. This ensemble approach helps reduce overfitting, which is a risk for decision tree models. Decision trees use a 0-1 loss function, which is just the proportion of predictions that are incorrect (similar to an accuracy score). 
# MAGIC   - In a distributed system, we can train each decision tree in parallel. 
# MAGIC 
# MAGIC As time permits, we may also explore the multilayer perceptron classifier in MLlib.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Resource Management and Performance Optimization
# MAGIC 
# MAGIC Working with a big data problem over Azure means that extensive computing resources will be required which could cause the project to quickly run over the budget (e.g. free Azure account credit) if resources are not carefully managed. This is even more true with certain time series machine learning models which requires multiple features to be engineered for sliding windows. To avoid unexpected budget increase, the following best practices will be followed throughout the project:
# MAGIC - Improving speed through memory: cache after major execution (e.g. after every data pipeline milestone including performing join and transforming data into training ready format)  to speed up processing. This can be achieved using the cache operation. Cached data will be cleared as the cluster gets destroyed.
# MAGIC - Improving reliability by persist to disk: Persist or checkpoint to blob storage after major execution for reliability in case of unexpected reader disconnect
# MAGIC   - Persist after performing data join - do not unpersist 
# MAGIC   - Optionally unpersist for other milestones along the data pipeline if needed to free up the blob storage for cost minimization 
# MAGIC - Spark API selection: use Spark DataFrame as opposed to RDD
# MAGIC - Continuous Improvement practice: Monitor memory and disk usage after every major data pipeline milestone to identify opportunities of code optimization along the data pipeline
# MAGIC - Code consciously: choose spark operations carefully with performance in mind, such as
# MAGIC   - Broadcast - broadcast variables and small lookup tables where applicable
# MAGIC   - Minimize shuffling - choose ByKey operations carefully (e.g. use combineByKey or reduceByKey instead of groupByKey)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Train/Test/Validation Data Splits
# MAGIC 
# MAGIC With time series data, data leakage can compromise the model result if data is not split appropriately. In fact, the traditional random splitting approach will not work as it will break the temporal nature of the data. Instead, an 80-20 train-test split will be performed on data that is sorted chronologically such that 80% of the data will be the ‘older’ part of the dataset that makes up the training data vs. the 20% is the ‘newer’ testing dataset. 
# MAGIC 
# MAGIC While performing the training/test data set on the full dataset (2015-2021) is desirable, balancing completeness of data with computing resources is required. As such, the following data splitting techniques will be used:
# MAGIC - Perform EDA on the full dataset to identify unusual temporal patterns for opportunities to exclude data from the problem. Examples of such unusual patterns may be excluding data that covers the period from when the pandemic started to when airport operations stabilized. We assume that operational lessons learned from the Pandemic can be replicated quickly by airports in the event of another unexpected event that could impact the global airline operations and therefore can consider dropping the reactive phase
# MAGIC - Create mini-model on 2019 data and observe how the model would perform differently as the model expands to include other period:
# MAGIC   - Initially conduct 80-20 split on 2019 with Jan to mid-Sep as the training data and mid-Spe-Dec as the testing data
# MAGIC   - Continue to fine tune the data through cross validation by including more data into the model
# MAGIC 
# MAGIC Zooming in on cross validation, we will use the Blocked Time Series split technique. The Blocked Time Series split technique is a variation of the time series split technique that reduces data leakage characterized by adding margins between the training and the testing dataset at each fold and across folds to prevent the model from observing lag values which are used twice. A visual illustration and sample implementation of the blocked time series technique are shown in below ([reference](https://goldinlocks.github.io/Time-Series-Cross-Validation/#:~:text=Blocked%20and%20Time%20Series%20Split,and%20another%20as%20a%20response.)).
# MAGIC 
# MAGIC ![Blocking Time Series Split](/files/tables/image/BlockingTimeSeriesSplit.JPG)
# MAGIC 
# MAGIC ![Blocking Time Series Split Code](/files/tables/image/BlockingTimeSeriesSplitCode.JPG)
# MAGIC 
# MAGIC Each fold from the blocked CV will follow the 80-20 split as discussed above. The number of folds (k) will be decided after conducting EDA to identify opportunities of leaving data out from the problem. 2 approaches for determining the folds will be evaluated after completing EDA:
# MAGIC 1. Fixed duration: each fold will be approximately the same size 
# MAGIC 2. Varying duration: each fold will be a specific period that is representative of special events such as optimal operations (e.g. pre-pandemic), disrupted & recovery from disrupted operations (e.g. onset of and recovery from the Pandemic), stabilized operations after major disruption (e.g. the new stabilized operations) 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Team Names and Photos
# MAGIC 
# MAGIC Section, Group Number: Section 4, Group 1
# MAGIC 
# MAGIC Team Name: Sparks and Stripes Forever
# MAGIC 
# MAGIC Team Members:
# MAGIC - Nashat Cabral (cabralnc96@berkeley.edu)
# MAGIC - Deanna Emery (deanna.emery@berkeley.edu)
# MAGIC - Nina Huang (ninahuang2002@berkeley.edu)
# MAGIC - Ryan S. Wong (ryanswong@berkeley.edu)
# MAGIC 
# MAGIC 
# MAGIC ![group photo](/files/tables/image/Group_Photo.JPG)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## GANTT Chart for Project Timeline and Tasks
# MAGIC 
# MAGIC The following picture was taken from the GANTT chart that our team is using in the project management program Asana. This chart shows the tasks and schedules for each of the project phases along with major deadlines and milestones.
# MAGIC 
# MAGIC ![GANTT Phase Overview](/files/tables/image/PhaseOverview.JPG)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Project Credit Assignment Plan
# MAGIC 
# MAGIC Per the instructions, a full table of each team member’s assigned tasks have been provided in table format. Our initial planned version of the table is given below. As the project progresses, adjustments to the plan will be made on the table at the following link:
# MAGIC 
# MAGIC https://docs.google.com/spreadsheets/d/1A4N3sV1ngaBsPdUqcJ8R4gY6e2g3gHeIUDyDes7f4SU/edit#gid=0
# MAGIC 
# MAGIC | Phase Number | Team Member   | Tasks Completed                                                                                                                                                                                                                                   |
# MAGIC | ------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# MAGIC | Phase 1      | Nashat Cabral | Explain Data Section<br>Initial EDA and visuals<br>Data Pipeline plan section<br>Initial Pipeline Creation                                                                                                                                        |
# MAGIC |              | Deanna Emery  | Join plan for data sets<br>Initial EDA and visuals<br>Model selection discussion section<br>Initial Pipeline Creation                                                                                                                             |
# MAGIC |              | Nina Huang    | Strategizing resource management section<br>Train/Validation/Test split section<br>Data Pipeline plan section<br>Initial Pipeline Creation                                                                                                        |
# MAGIC |              | Ryan S. Wong  | Outcome and metrics section<br>Data ingesting section<br>Phase leader plan<br>Credit assignment plan<br>GANTT chart<br>Initial Pipeline Creation                                                                                                  |
# MAGIC | Phase 2      | Nashat Cabral | Plans to contribute to building the pipeline for the data<br>Focus on data cleaning tasks<br>Implement parallel training pipeline<br>Implement basic model usage.<br>Efficiency checks to ensure efficient code in Spark.                         |
# MAGIC |              | Deanna Emery  | Plans to contribute to building the pipeline for the data<br>Focus on data cleaning tasks<br>Establish proper proper data set joins for data of interest.<br>Implement basic model usage.<br>Efficiency checks to ensure efficient code in Spark. |
# MAGIC |              | Nina Huang    | Plans to contribute to building the pipeline for the data<br>Focus on data cleaning tasks<br>Establish proper proper data set joins for data of interest.<br>Implement basic model usage.<br>Efficiency checks to ensure efficient code in Spark. |
# MAGIC |              | Ryan S. Wong  | Plans to contribute to building the pipeline for the data<br>Focus on data cleaning tasks<br>Implement parallel training pipeline<br>Implement parallel scoring pipeline<br>Efficiency checks to ensure efficient code in Spark.                  |
# MAGIC | Phase 3      | Nashat Cabral | Plans to investigate influence levels for currently selected features<br>Tuning and experimentation of hyperparameters per model<br>Efficiency checks to ensure efficient code in Spark.                                                          |
# MAGIC |              | Deanna Emery  | Plans to investigate influence levels for currently selected features<br>Tuning and experimentation of hyperparameters per model<br>Efficiency checks to ensure efficient code in Spark.                                                          |
# MAGIC |              | Nina Huang    | Research optimal hyperparameters per model<br>Tuning and experimentation of hyperparameters per model<br>Efficiency checks to ensure efficient code in Spark.                                                                                     |
# MAGIC |              | Ryan S. Wong  | Plans to investigate influence levels for currently selected features<br>Research optimal hyperparameters per model<br>Tuning and experimentation of hyperparameters per model                                                                    |
# MAGIC | Phase 4      | Nashat Cabral | Plans to implement loss functions<br>Implement F1, Recall, Accuracy metrics<br>Establish baseline algorithm<br>Additional model testing and tuning.<br>Optimal model selection<br>Final report write-up                                           |
# MAGIC |              | Deanna Emery  | Plans to implement loss functions<br>Additional model testing and tuning.<br>Optimal model selection<br>Final report write-up                                                                                                                     |
# MAGIC |              | Nina Huang    | Plans to implement loss functions<br>Implement F1, Recall, Accuracy metrics<br>Establish baseline algorithm<br>Optimal model selection<br>Final report write-up                                                                                   |
# MAGIC |              | Ryan S. Wong  | Plans to implement loss functions<br>Implement F1, Recall, Accuracy metrics<br>Establish baseline algorithm<br>Optimal model selection<br>Final report write-up<br>Final Presentation mock-up                                                     |
# MAGIC | Phase 5      | Nashat Cabral | Class Presentation Preparation                                                                                                                                                                                                                    |
# MAGIC |              | Deanna Emery  | Class Presentation Preparation                                                                                                                                                                                                                    |
# MAGIC |              | Nina Huang    | Class Presentation Preparation                                                                                                                                                                                                                    |
# MAGIC |              | Ryan S. Wong  | Class Presentation Preparation                                                                                                                                                                                                                    |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Conclusions and Next Steps
# MAGIC 
# MAGIC Upon reviewing the data and creating the macro solution (project pan), we would like to conclude our phase 1 analysis with the following summary:
# MAGIC - Problem statement: empowering customers to plan their trip by providing if their flight will be delayed (either a ‘1’ in “ArrDel15” or “DepDel15”) by minimizing false negatives such that the customer can plan for their itinerary after their flight landed
# MAGIC - Data pipeline: building efficient data pipelines by leveraging partitions with proper resource management techniques to solve the big-data, time-series, machine-learning problem feasible within the given budget (AWS free credit tier for Blob Storage). Key stages of the data pipeline include:
# MAGIC   - Data format conversion: convert from csv to Parquet
# MAGIC   - Joining datasets: bring together relevant datasets, which may include (depending on the result of the EDA)  the flight table, weather table, and airport table
# MAGIC   - Data transformation: all the steps that are involved to prepare data into machine learning model ready format. Example of such steps include cleaning, scaling, rebalancing, feature engineering
# MAGIC - Model training: train the data using training dataset and evaluate the model on the testing dataset. To avoid overfitting and perform parameter tuning, we will use the Blocked Time Series cross validation technique
# MAGIC - Model result evaluation: model will be evaluated by balancing the recall and F1 to minimize false negatives while also considering prediction accuracy

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Open Issues or Problems
# MAGIC 
# MAGIC As we are working with a big-data time series problem, optimizing computing resources such that running a model is technically feasible and operationally manageable will be a consistent problem. To address these problems, designing solutions to address the following open issues will continue to be focused on our subsequent phases:
# MAGIC - Missing Data: Among our chosen fields of interest for this project, we have uncovered a large amount of missing data, particularly from the weather dataset. Going forward we will need to handle each of these circumstances with caution as the meaning of missing data may be of significance, or it may be a detriment to any conclusions we attempt to gather from the data.
# MAGIC - Special period consideration: The data provided covers the Covid Pandemic outbreak (announced by the WHO on Jan 30, 2020). This means we may see segments of special periods with abnormal flight delay results, such as during the start of the pandemic, pandemic recovery, and stabilization after the pandemic. As such, historical data may not be sufficient to train a model that can respond to special periods. Creating a model to predict for these special periods will be a topic of continuous research and exploration
# MAGIC - Training data time span: In addition to considering special periods, opportunities to further refine/reduce the size of the dataset required to create a machine learning model exists. Methods to refine the dataset include:
# MAGIC   - Excluding data that may be no longer relevant - for example, does 2015 data help to provide 2021 result in light of the pandemic?
# MAGIC   - Representing special periods - for example, should cross validation data splits be performed on special period phases?
# MAGIC   - Parsimonious feature selection - keeping the number of features selected for the model to be lean (e.g no more than 25)

# COMMAND ----------


