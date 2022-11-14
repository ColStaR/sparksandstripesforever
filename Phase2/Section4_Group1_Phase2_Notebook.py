# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Link to Phase 2 Deliverable Instructions: https://digitalcampus.instructure.com/courses/4868/pages/phase-descriptions-and-deliverables?module_item_id=686692
# MAGIC 
# MAGIC Nina will be updating abstract with baseline info, model info.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Team 13 Phase 2 Notebook
# MAGIC ## Sparks and Stripes Forever
# MAGIC 
# MAGIC #### Members:
# MAGIC - Nashat Cabral
# MAGIC - Deanna Emery
# MAGIC - Nina Huang
# MAGIC - Ryan S. Wong

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
# MAGIC ### Phase 2 Contributions
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
# MAGIC [Link to Phase 1 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/364123876153624/command/4295587629775265)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Nina, Nash, Ryan
# MAGIC 
# MAGIC Each person show a blurb Highlights of most important findings.
# MAGIC 
# MAGIC Link to EDA notebooks acceptable in lieu of putting everything here.
# MAGIC 
# MAGIC # Pre-Join EDA on Raw Data Sets
# MAGIC 
# MAGIC Initial EDA was conducted on the Airlines, Station, and Weather data sets individually in order to establish a baseline understanding of the data being worked in. In particular, we were interested to know what features would be of use when joining the data sets, and what amount of data was missing or erroneous. This section will discuss our findings when exploring the unjoined data sets, addressing both the overall data hygiene and some remarkable findings. This will be a high level summary of the most important findings; for more information and details, please consult the notebooks linked in this section below. The data set joining task will be discussed in the section below.
# MAGIC 
# MAGIC ## Selected Features
# MAGIC 
# MAGIC Upon initial inspection, it became obvious that each of the data sets included many features that were not relevant or useful for our analysis. As such, we discussed and deliberated over the meaning and worth of each feature in each data set in order to determine which features should be selected to be maintained and used in our analysis, and which would be dropped. Below is a list of features that we are including in our final data sets.
# MAGIC 
# MAGIC ### Airlines Data Set Features
# MAGIC 
# MAGIC - QUARTER
# MAGIC - MONTH
# MAGIC - DAY_OF_MONTH
# MAGIC - DAY_OF_WEEK
# MAGIC - FL_DATE
# MAGIC - OP_UNIQUE_CARRIER
# MAGIC - TAIL_NUM
# MAGIC - OP_CARRIER_FL_NUM
# MAGIC - ORIGIN_AIRPORT_ID
# MAGIC - ORIGIN_AIRPORT_SEQ_ID
# MAGIC - ORIGIN
# MAGIC - ORIGIN_STATE_ABR
# MAGIC - ORIGIN_WAC
# MAGIC - DEST_AIRPORT_ID
# MAGIC - DEST_AIRPORT_SEQ_ID
# MAGIC - DEST_STATE_ABR
# MAGIC - DEST_WAC
# MAGIC - CRS_DEP_TIME
# MAGIC - DEP_TIME
# MAGIC - DEP_DEL15
# MAGIC - CANCELLED
# MAGIC - CANCELLATION_CODE
# MAGIC - CRS_ELAPSED_TIME
# MAGIC - DISTANCE
# MAGIC - YEAR
# MAGIC 
# MAGIC ### Weather Data Set Features
# MAGIC 
# MAGIC - STATION
# MAGIC - DATE
# MAGIC - ELEVATION
# MAGIC - SOURCE
# MAGIC - HourlyDewPointTemperature
# MAGIC - HourlyDryBulbTemperature
# MAGIC - HourlyRelativeHumidity
# MAGIC - HourlyVisibility
# MAGIC - HourlyWindSpeed
# MAGIC 
# MAGIC ### Stations Data Set Features
# MAGIC 
# MAGIC - neighbor_call
# MAGIC - neighbor_state
# MAGIC - station_id
# MAGIC - distance_to_neighbor
# MAGIC - neighbor_lat
# MAGIC - neighbor_lon
# MAGIC 
# MAGIC ## Missing Values
# MAGIC 
# MAGIC df_airlines.count() = 42430592
# MAGIC TAIL_NUM = 242827 = 0.572%
# MAGIC DEP_TIME = 852812 = 2.009%
# MAGIC DEP_DEL15 = 857939 = 2.021%
# MAGIC CANCELLATION_CODE = 41556551 = 97.940%
# MAGIC CRS_ELAPSED_TIME = 170 = 0.00004%
# MAGIC 
# MAGIC ## Remarkable Findings
# MAGIC 
# MAGIC ### Airlines Data Set
# MAGIC 
# MAGIC - For features that were indicative of flight departures ("DEP_TIME" and "DEP_DEL15"), having null values in those features were 1:1 associated with the flights being cancelled ("CANCELLED" == 1). It was confirmed that for every null value corresponding to a departure-related feature, the flight in question was indeed cancelled.
# MAGIC - In general, none of the Airlines features showed any strong correlation with the response variable DEP_DELAY15. However, the feature with the strongest correlation was "CRS_DEP_TIME" with an R value of 0.1565. While this is not an indicator of a strong correlation, the R value is significantly higher than any of the other features.
# MAGIC - Similar to the previous point, the feature "CANCELLATION_CODE" is null for the majority of cases where a flight is not cancelled. Therefore, it has an expected value of 97.940% null values.
# MAGIC 
# MAGIC ### Weather Data Set
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Stations Data Set
# MAGIC 
# MAGIC ## Links to complete pre-join EDA Notebooks:
# MAGIC 
# MAGIC [EDA for Airlines Data Set (Part 1)](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804817899/command/1020093804817918)
# MAGIC 
# MAGIC [EDA for Airlines Data Set (Part 2)](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4423519682322242/command/4423519682322243)
# MAGIC 
# MAGIC [EDA for Weather, Stations Data Sets](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/2260108212960246/command/2260108212960247)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Joining the Data Sets
# MAGIC 
# MAGIC Deanna
# MAGIC 
# MAGIC ## Joined Features

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # EDA on Joined Data Set
# MAGIC 
# MAGIC Nina, Nash, Ryan
# MAGIC 
# MAGIC Highlights of most important findings.
# MAGIC 
# MAGIC Link to EDA notebooks acceptable in lieu of putting everything here.
# MAGIC 
# MAGIC ## List of Features
# MAGIC 
# MAGIC ## Features Used in Analysis

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Notes on Data Cleaning
# MAGIC 
# MAGIC Nash and Ryan
# MAGIC 
# MAGIC ## Missing Data
# MAGIC 
# MAGIC Cancelled flights turned into delayed flights with DEP_DEL15 = 1.
# MAGIC 
# MAGIC Dropped/skipped all rows with null values that could not be confidently replaced with an inferred value ("TAIL_NUM", etc.). Counts are minimal, ~2% of total count of rows, so assumed to be insignificant impact on analysis.
# MAGIC  
# MAGIC ## Non-Numerical Features
# MAGIC 
# MAGIC StringIndexing and Tokenization of categorical features

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Ryan
# MAGIC 
# MAGIC # Baseline Model Pipeline
# MAGIC 
# MAGIC After successfully joining and cleaning the data, we went to work creating a data pipeline and implementing a basic baseline model.
# MAGIC 
# MAGIC ## Pipeline Diagram, Process
# MAGIC 
# MAGIC Below is an updated diagram of our pipeline, which follows and includes all of the major tasks and processes that our pipeline conducts thus far.
# MAGIC 
# MAGIC ![Data Pipeline Image](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/DataPipelinev2.png?raw=true)
# MAGIC 
# MAGIC ## Baseline Model, Features
# MAGIC 
# MAGIC After establishing the pipeline, we set about creating a baseline model that will be used for comparison against our own trained model. Due to its relative simplicity for a categorization task, a logistic regression model was decided upon to used as the baseline model.
# MAGIC 
# MAGIC Furthermore, the following features from the joined data set were being used with the baseline model:
# MAGIC 
# MAGIC #### Categorical Features
# MAGIC 
# MAGIC - ORIGIN
# MAGIC - OP_UNIQUE_CARRIER
# MAGIC - TAIL_NUM
# MAGIC - ORIGIN_STATE_ABR
# MAGIC - DEST_STATE_ABR
# MAGIC 
# MAGIC #### Numeric Features
# MAGIC 
# MAGIC - QUARTER
# MAGIC - MONTH
# MAGIC - DAY_OF_MONTH
# MAGIC - DAY_OF_WEEK
# MAGIC - OP_CARRIER_FL_NUM
# MAGIC - ORIGIN_AIRPORT_ID
# MAGIC - ORIGIN_AIRPORT_SEQ_ID
# MAGIC - ORIGIN_WAC
# MAGIC - DEST_AIRPORT_ID
# MAGIC - DEST_AIRPORT_SEQ_ID
# MAGIC - DEST_WAC
# MAGIC - DEP_TIME
# MAGIC - CANCELLED
# MAGIC - CRS_ELAPSED_TIME
# MAGIC - DISTANCE
# MAGIC - YEAR
# MAGIC - STATION
# MAGIC - ELEVATION
# MAGIC - SOURCE
# MAGIC - HourlyDewPointTemperature
# MAGIC - HourlyDryBulbTemperature
# MAGIC - HourlyRelativeHumidity
# MAGIC - HourlyVisibility
# MAGIC - HourlyWindSpeed
# MAGIC - distance_to_neighbor
# MAGIC 
# MAGIC ## Baseline Model Evaluation
# MAGIC 
# MAGIC Precision = 0.01718794406241851
# MAGIC Recall = 0.5598992135603573
# MAGIC Accuracy = 0.8082395453515033
# MAGIC 
# MAGIC ## Link to Data Pipeline Creation, Baseline Model Evaluation Notebook
# MAGIC [Data Pipeline Creation, Baseline Model Evaluation Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804822439/command/1020093804826247)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Notes about Planned Features
# MAGIC 
# MAGIC Nina
# MAGIC 
# MAGIC 2 Hour Delayed Time
# MAGIC 
# MAGIC Flight Tracker
# MAGIC 
# MAGIC Airport Tracker
# MAGIC 
# MAGIC Graph Database with GraphFrames

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Link to Phase 2 Presentation Video
# MAGIC 
# MAGIC [Click Here for the Video](https://drive.google.com/file/d/1Ubpv8pGEZStzTEzrSpBFVwnFjY6y3aSx/view?usp=sharing)

# COMMAND ----------


