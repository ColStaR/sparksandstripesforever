# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Team 13 Phase 3 Notebook
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
# MAGIC [Link to Official Spreadsheet (Requires UC Berkeley Account)](https://docs.google.com/spreadsheets/d/1Va1bwlEmrIrOc1eFo1ySYlPQpt4kZjDqahQABgw0la4/edit#gid=0)
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
# MAGIC 
# MAGIC [Link to Official Spreadsheet (Requires UC Berkeley Account)](https://docs.google.com/spreadsheets/d/1A4N3sV1ngaBsPdUqcJ8R4gY6e2g3gHeIUDyDes7f4SU/edit#gid=549854846)
# MAGIC 
# MAGIC ### Phase 3 Contributions
# MAGIC 
# MAGIC | Task                                      | Estimated Time (hours) | Nashat Cabral | Deanna Emery | Nina Huang | Ryan S. Wong |
# MAGIC |-------------------------------------------|------------------------|---------------|--------------|------------|--------------|

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Links to Previous Notebooks
# MAGIC 
# MAGIC [Link to Phase 1 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/364123876153624/command/4295587629775265)
# MAGIC 
# MAGIC [Link to Phase 1 Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase1/Section4_Group1_Phase1_ProjectProposal_Notebook.ipynb)
# MAGIC 
# MAGIC [Link to Phase 2 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804836156/command/1020093804836157)
# MAGIC 
# MAGIC [Link to Phase 2 Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/Section4_Group1_Phase2_Notebook_Final.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Project Abstract
# MAGIC 
# MAGIC In the flight industry, delays are a key issue for airline companies, airports, and customers. This project focuses on empowering customers to plan their itineary by predicting if their flight will be delayed 2 hours before their planned departure time. 
# MAGIC Our customer-focused objective would be achieved through minimizing false negatives by running machine models against airport, flight, and local weather data. Any analyses and methods applied will come from the perspective of benefiting the customer, and thus we chose a weighted combintaion of recall and F1 as our metrics (incorrectly identifying a false negative delay would be detrimental to the customer experience). Furthermore, we would evaluate the performance of our model against our prediction baseline, which will be a logistic regression model that attempts to predict the occurence of a flight delay given our variables of interest. 
# MAGIC This document outlines the high level plan for approaching this problem.

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
# MAGIC - DISTANCE  
# MAGIC - ELEVATION
# MAGIC - HourlyDewPointTemperature
# MAGIC - HourlyAltimeterSetting
# MAGIC - HourlyWetBulbTemperature
# MAGIC - HourlyPrecipitation 
# MAGIC - HourlyStationPressure 
# MAGIC - HourlyDewPointTemperature 
# MAGIC - HourlyDryBulbTemperature 
# MAGIC - HourlySeaLevelPressure 
# MAGIC - HourlyPressureChange 
# MAGIC - HourlyWindGustSpeed 
# MAGIC - HourlyRelativeHumidity 
# MAGIC - HourlyVisibility 
# MAGIC - HourlyWindSpeed 
# MAGIC - distance_to_neighbor 
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
# MAGIC 
# MAGIC ## Remarkable Findings
# MAGIC 
# MAGIC ### Airlines Data Set
# MAGIC 
# MAGIC he Airline On-Time Performance Data table contains the scheduled and actual departure/arrival times for U.S. Domestic flights for qualifying airline carriers. These carriers must account for at least one percentage of U.S Domestic scheduled passenger revenues in order to qualify. Our data ranges from 2015 to 2021
# MAGIC 
# MAGIC - More than 50% of the data in the Airlines dataset contain duplicate records (74,177,433 to 42,430,592 after performing de-duplication). Data duplication is performed before conducting further analysis
# MAGIC - The Airlines work with 20 unique carriers, 8465 tail numbers (aircraft ID), 7300 flight numbers, 388 origin airport id, and 386 dest airport id
# MAGIC - A flight from the Airlines dataset is uniquely identified by the natural composite key: ORIGIN_AIPORT_ID, DEST_AIRPORT_ID, FL_DATE, OP_CARRIER_FL_NUM, OP_UNIQUE_CARRIER, DEP_TIME
# MAGIC - For features that were indicative of flight departures ("DEP_TIME" and "DEP_DEL15"), having null values in those features were 1:1 associated with the flights being cancelled ("CANCELLED" == 1). It was confirmed that for every null value corresponding to a departure-related feature, the flight in question was indeed cancelled.
# MAGIC - In general, none of the Airlines features showed any strong correlation with the response variable DEP_DELAY15. However, the feature with the strongest correlation was "CRS_DEP_TIME" with an R value of 0.1565. While this is not an indicator of a strong correlation, the R value is significantly higher than any of the other features.
# MAGIC - Similar to the previous point, the feature "CANCELLATION_CODE" is null for the majority of cases where a flight is not cancelled. Therefore, it has an expected value of 97.940% null values.
# MAGIC - Airport_id columns (ORIGIN_AIRPORT_ID and DEST_AIRPORT_ID) uniquely match to their airport code (ORIGIN and DEST). This contradicts the flight dictionary documentation. Furthermore airport_seq_ids (ORIGIN_AIRPORT_SEQ_ID and DEST_AIRPORT_SEQ_ID) uniquely identifie an airport (1 to 1 relationship with the aiport_ids) whereas a single aiport_id can have multiple airport_seq_id. As such the airport_seq_IDs are more accurate tracker of airports. As time permit, we will further improve our model performance by improving our join algorithm to consider the movement of airport locations that is tracked by the airport_seq_id as opposed to solely relying on the airport_ids.
# MAGIC - Less than 0.6% of flights (236,132) departed within 2 hours from their assigned aircraft's previous flight departure
# MAGIC 
# MAGIC ### Weather Data Set
# MAGIC 
# MAGIC The Quality Controlled Local Climatological Data contains summary data from weather stations housed at airports. These stations log daily temperature highs/lows, precipitation, wind speed, visibility, and storm characteristics. The available data ranges from 2015 to 2021.
# MAGIC 
# MAGIC - Given documentation did not always match our given columns
# MAGIC - Several fields/columns within the weather dataset had large amounts of missing data.
# MAGIC - These fields would be kept in our analyses with the goal of seeing reduced portions of missing data after the join across datasets
# MAGIC   - This would be resolved post-join where most of the remaining data had its percentage of nulls reduced significantly. 
# MAGIC - Monthly data is included at this level, but accounts for only a minute proportion of the data.
# MAGIC   - Not every row contains monthly data despite most having hourly data of some sort. This brings to question the usability of these columns.
# MAGIC - Date fields will likely need to be adjusted for uniformity across all datasets.
# MAGIC - Dates/Times are in their own respective timezones
# MAGIC - HourlyPressureChange does not seem to change frequently nor drastically, is more or less a static field.
# MAGIC 
# MAGIC 
# MAGIC ### Stations Data Set
# MAGIC The final table, stations_data, houses information on airport location.
# MAGIC 
# MAGIC ## Links to complete pre-join EDA Notebooks:
# MAGIC 
# MAGIC [EDA for Airlines Data Set (Part 1)](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804817899/command/1020093804817918)
# MAGIC 
# MAGIC [EDA for Airlines Data Set in GitHub(Part 1)](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/NH_EDAPart2.py)
# MAGIC 
# MAGIC [EDA for Airlines Data Set (Part 2)](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4423519682322242/command/4423519682322243)
# MAGIC 
# MAGIC [EDA for Airlines Data Set in GitHub(Part 2)](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/RWong%20-%20EDA%20Notebook.py)
# MAGIC 
# MAGIC [EDA for Airlines Data Set (Part 3)](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804809577/command/1020093804815501)
# MAGIC 
# MAGIC [EDA for Airlines Data Set in GitHub(Part 3)](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/NH%20-%20EDA%20Notebook.py)
# MAGIC 
# MAGIC [EDA for Weather, Stations Data Sets](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/2260108212960246/command/2260108212960247)
# MAGIC 
# MAGIC [EDA for Weather, Stations Data Sets in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/Nash-%20EDA%20Weather%20FULL.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Joining the Data Sets
# MAGIC 
# MAGIC Before we began feature engineering and machine learning, we first had to join weather and airlines datasets and address missing values. Since our goal was to predict the departure delay, we had to drop any rows that were missing values in all four of DEP_DELAY, DEP_DELAY_NEW, DEP_DEL15, and CANCELLED (if any one of these columns was available, we had sufficient information for our outcome variable). Note that in our case, we treated cancellations as delays.
# MAGIC 
# MAGIC The steps below detail how the join of the airlines and weather datasets was conducted:
# MAGIC 1. First, we collected distinct pairs of airport codes from the ORIGIN column, and ORIGIN_STATE_ABR.
# MAGIC 2. Having confirmed that all airport codes and their corresponding states were in the weather stations dataset using the neighbor_call and neighbor_state columns, we performed an inner join on these two fields such that we had a table containing: ORIGIN (neighbor_call), ORIGIN_STATE_ABR (neighbor_state), station_id, distance_to_neighbor. We then confirmed that all airport codes and states were matched to at least one weather station, and that all matched weather stations existed in the weather data.
# MAGIC 3. Because each airport mapped to more than two hundred weather stations on average, for the sake of efficiency, we chose to narrow down to a set of two or three stations for each airport: we selected the airport weather stations (with a distance_to_neighbor equal to zero), the closest weather station from the airport with a distance greater than zero, and the furthest weather station from the airport.
# MAGIC 4. We then performed an inner join of the airport/station linking table onto the weather dataset. The resulting dataset only contained weather data for the stations of interest, and some duplicate entries of weather where the same weather station corresponded to more than one airport. We also found that some airports corresponded to more than one weather station.
# MAGIC 5. In the airplanes dataset we created a departure date-time column, called DEP_DATETIME, by combining the FL_DATE, and departure time (CRS_DEP_TIME) into a datetime. We added a column called DEP_DATETIME_LAG to the airlines dataset that was equal to the departure datetime minus 2 hours. We needed to round the time down to the hour to help make merging onto the weather data easier, which we accomplished by removing the last two digits from the departure time before converting to a datetime.
# MAGIC 6. In the weather dataset, we created a new column called DEP_DATETIME (to align with the airport dataset) by converting the DATE column in the weather data to a datetime type. Then, we created an additional column called DEP_DATETIME_LAG, which rounded the DEP_DATETIME up to the next hour for the sake of the merge.
# MAGIC 7. We then identified rows in the merged weather data with duplicated DEP_DATETIME and airport (ORIGIN). We aggregated these duplicated entries by averaging across the numeric features in the weather data. This strategy allowed us to keep more granular time in our weather data by making use of multiple stations.
# MAGIC 8. Finally, we were able to join the airlines data and the weather/station data together as an inner merge using the columns ORIGIN and DEP_DATETIME_LAG. The resulting table had 30,820,796 rows for the years 2015-2021.
# MAGIC 
# MAGIC ## Joined Features
# MAGIC - 'DEP_DATETIME_LAG',
# MAGIC  - 'QUARTER',
# MAGIC  - 'MONTH',
# MAGIC -  'DAY_OF_MONTH',
# MAGIC -  'DAY_OF_WEEK',
# MAGIC -  'FL_DATE',
# MAGIC -  'OP_UNIQUE_CARRIER',
# MAGIC -  'TAIL_NUM',
# MAGIC -  'OP_CARRIER_FL_NUM',
# MAGIC -  'ORIGIN_AIRPORT_ID',
# MAGIC -  'ORIGIN_AIRPORT_SEQ_ID',
# MAGIC -  'ORIGIN_STATE_ABR',
# MAGIC -  'ORIGIN_WAC',
# MAGIC -  'DEST_AIRPORT_ID',
# MAGIC -  'DEST_AIRPORT_SEQ_ID',
# MAGIC  - 'DEST_STATE_ABR',
# MAGIC - 'DEST_WAC',
# MAGIC - 'CRS_DEP_TIME',
# MAGIC - 'DEP_TIME',
# MAGIC - 'DEP_DEL15',
# MAGIC - 'CANCELLED',
# MAGIC - 'CANCELLATION_CODE',
# MAGIC - 'CRS_ELAPSED_TIME',
# MAGIC - 'DISTANCE',
# MAGIC - 'YEAR',
# MAGIC - 'DEP_HOUR',
# MAGIC - 'DEP_DATETIME',
# MAGIC - 'DATE',
# MAGIC -  'ELEVATION',
# MAGIC -  'SOURCE',
# MAGIC -  'HourlyAltimeterSetting',
# MAGIC -  'HourlyDewPointTemperature',
# MAGIC -  'HourlyWetBulbTemperature',
# MAGIC -  'HourlyDryBulbTemperature',
# MAGIC -  'HourlyPrecipitation',
# MAGIC -  'HourlyStationPressure',
# MAGIC -  'HourlySeaLevelPressure',
# MAGIC -  'HourlyPressureChange',
# MAGIC -  'HourlyRelativeHumidity',
# MAGIC -  'HourlyVisibility',
# MAGIC -  'HourlyWindSpeed',
# MAGIC -  'HourlyWindGustSpeed',
# MAGIC -  'MonthlyMeanTemperature',
# MAGIC -  'MonthlyMaximumTemperature',
# MAGIC -  'MonthlyGreatestSnowDepth',
# MAGIC -  'MonthlyGreatestSnowfall',
# MAGIC -  'MonthlyTotalSnowfall',
# MAGIC -  'MonthlyTotalLiquidPrecipitation',
# MAGIC  - 'MonthlyMinimumTemperature',
# MAGIC -  'DATE_HOUR',
# MAGIC -  'distance_to_neighbor',
# MAGIC -  'neighbor_lat',
# MAGIC -  'neighbor_lon',
# MAGIC  - 'time_zone_id',
# MAGIC  - 'UTC_DEP_DATETIME_LAG',
# MAGIC  - 'UTC_DEP_DATETIME',
# MAGIC  - 'flight_id'
# MAGIC  
# MAGIC  
# MAGIC ### Links to data joining notebooks
# MAGIC [Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4423519682321930/command/1020093804821142)
# MAGIC 
# MAGIC [Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/DLE_join_V1.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # EDA on Joined Data Set
# MAGIC 
# MAGIC - The Joined dataset contains 30,820,796 rows
# MAGIC - This EDA will ignore the monthly weather data in our current join as they are intended for use in a later iteration of our model.
# MAGIC 
# MAGIC Before conducting any feature engineering, we want to perform a broad brush analysis on how our features perform against flight delays. To achieve this, we created a correlation matrix for all non-string and non-id fields.
# MAGIC 
# MAGIC ![Correlation Matrix](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/Phase2/images/correlation_matrix.png)
# MAGIC 
# MAGIC Although some interesting relationships exist, no feature stands out as being able drive a strong machine learning performance (lack of strong correlation). As such, we will focus our efforts on engineering features that should have a stronger relationship with flight delays. We will define these engineered features through business intuition, such as tracking aircraft (TAIL_NUM) delay status across flights, weather systems, and airline management effectiveness. See our sections on "Features To Be Used in Future Analysis" and "Notes about Planned Features" for more.  
# MAGIC  
# MAGIC ## Notable Feature Characteristics
# MAGIC - About 80% of our data for DEP_DEL15 indicate as non-delayed flights
# MAGIC - Months with the most data in this dataset are December/January, likely due to the holiday season. This may impact our future cross validation decisions.
# MAGIC ![Month Visual](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/monthvis2.PNG?raw=true)
# MAGIC 
# MAGIC - Canceled flights are considered Delayed under DEP_DEL15 for this analysis
# MAGIC - Both HourlyWetBulbTemperature and HourlyDryBulbTemperature were normally distributed, with HourlyWetBulbTemperature appearing to have a slightly more narrow distribution.
# MAGIC ![Dry Visual](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/drytempvis.PNG?raw=true)
# MAGIC ![Wet Visual](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/wettempvis.PNG?raw=true)
# MAGIC 
# MAGIC - HourlyPressureChange does not seem to change significantly across the dataset, while also was missing from 24% of it, and will likely be dropped from future analyses.
# MAGIC   - HourlyStationPressure and HourlySeaLevelPressure showed a similarly tight distribution but were both missing from less than 1% of the dataset, and thus were deemed worthy to keep in our current model
# MAGIC - HourlyWindGustSpeed was missing from 64% of the dataset and will likely be dropped from future analyses
# MAGIC - HourlyWindSpeed displays outlier datapoints with records indicating a windspeed greater than 1000.
# MAGIC ![Wind Visual](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/windspeedprof.PNG?raw=true)
# MAGIC 
# MAGIC - Categorical Variables in our dataset seemed to be relatively free of missing/error data 
# MAGIC - HourlyPrecipitation was missing from about 4% of the joined dataset, and had 90% of its fields labeled with 0, indicating 90% of fields showing no precipitation.
# MAGIC 
# MAGIC ## Link to complete post-join EDA Notebook:
# MAGIC 
# MAGIC [NC Joined Data EDA wFullNulls](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804817787/command/1020093804817792)
# MAGIC 
# MAGIC [NC Joined Data EDA wFullNulls in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/NC%20Joined%20Data%20EDA%20wFullNulls.py)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features To Be Used in Future Analysis
# MAGIC The following features are engineered for further feature enhancement and or building the machine learning model:
# MAGIC 1. flight_id - The flight_id feature is created to uniquely identify a flight in the joined dataset
# MAGIC 2. is_prev_delayed - The previous flight delay indicator is created by looking at the status of the prior flight that was delivered by the same aircraft (TAIL_NUM). More than 99.5% of the flights had a departure time that is greater than 2 hours from their previous flight. The remaining 0.5% of the exception flight examines the departure status of the same aircraft from 2 flights prior. Several helper features were created to construct the is_prev_delayed. These helper features (is_1prev_delayed, is_2prev_delayed) are dropped from the datapipelin after the is_prev_delayed feature is created. 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Notes on Data Cleaning
# MAGIC 
# MAGIC After the joined data set was created, there were several rows that had erroneous entries or were not in the correct format to be used by a machine learning algorithm. Some rows had missing values for their features, preventing those rows from being used in any meaningful analysis. Other features were in a non-numerical format (primarily strings) that prevented them from being natively used in machine learning. This data therefore had to be cleaned in order to be made usable, as described below.
# MAGIC 
# MAGIC ## Missing Data
# MAGIC 
# MAGIC #### - Cancelled Flights
# MAGIC 
# MAGIC Flights that were cancelled were marked by the feature CANCELLED == 1. Furthermore, flight-related features being converted to null values, the most important being the DEP_DEL15 feature. Given that a cancelled flight illicits the same inconvenience as a delayed flight, it was decided that cancelled flights should be considered as a delayed flight. Thus, rows for cancelled flights had their DEP_DEL15 imputed to 1.
# MAGIC 
# MAGIC #### - HourlyPrecipitation Feature
# MAGIC 
# MAGIC The HourlyPrecipitation feature is a weather measurement that tracks the amount of precipitation a weather station received in the last hour. However, approximately 25% of this feature had null values. It was noted that a majority of the entries, including the average and median value, were set to 0. Therefore, it was decided to impute the null values in HourlyPrecipitation to be 0, thereby matching the mean and median value for that feature.
# MAGIC 
# MAGIC #### - Timezone Conversion
# MAGIC 
# MAGIC When importing the raw data, the times and dates for each row were localized to that location's time zone. However, multiple time zones makes consistent and meaningful analysis difficult, especially when accounting for flights that travel between time zones. As such, it was decided to standardize every row by converting their times and dates to the UTC time zone, thereby enforcing consistency. To accomplish this, we imported the open-source [Complete List of IATA Airports with Details dataset](https://github.com/lxndrblz/Airports) into our data, used it to map airports to their appropriate timezones, and then convert all times to UTC.
# MAGIC 
# MAGIC #### - HourlyWindSpeed Data Cleaning 
# MAGIC 
# MAGIC When conducting our EDA, we found HourlyWindSpeed had egregious outliers with wind speeds beyond a reasonable mean, for our model we will be filtering out records with windspeeds greater than 200 in order to better represent a field that is already left-tailed.
# MAGIC 
# MAGIC #### - Dropping Ambiguous Missing Values
# MAGIC 
# MAGIC For the above features, we were able to replace missing values with imputed values because we could infer values that would be appropriate to impute. However, not all features had missing data that could be easily inferred or assumed. For features whose null values were ambiguous and could not be confidently replaced with an inferred value, such as the TAIL_NUM feature, rows with null values were dropped. It should be noted that the count of such rows were minimal, approximately 2% of total rows, so dropping those rows is assumed to have an insignificant impact on the model and analysis.
# MAGIC  
# MAGIC ## Non-Numerical Features
# MAGIC 
# MAGIC StringIndexing and Tokenization of categorical features

# COMMAND ----------

# MAGIC %md
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
# MAGIC After the logistic regression model was created, it was applied to the data set. A custom version of time series cross validation was used, training the model on data from the years 2015 - 2020, before ultimately being tested and evaluated on data from the year 2021. After the model was trained and tested, the baseline logistic regression model performed with the following metrics:
# MAGIC 
# MAGIC | Metric    | Value       |
# MAGIC |-----------|-------------|
# MAGIC | Precision | 0.01718794406241851 |
# MAGIC | Recall    | 0.5598992135603573 |
# MAGIC | Accuracy  | 0.8082395453515033 |
# MAGIC 
# MAGIC 
# MAGIC Given that our primary metric for this model was precision, the low value of the precision metric surprises us. We have a number of possible reasons why the metric turned out so unusually low: high levels of variance in the features introducing noise, incorrect scaling for features prior to training, or perhaps linear regression does not perform well with a categorization task or data set like this. Further investigation will be done in order to determine whether this low precision value is the legitimate result of the circumstances or if a confounding error is at play.
# MAGIC 
# MAGIC ## Link to Data Pipeline Creation, Baseline Model Evaluation Notebook
# MAGIC 
# MAGIC [Data Pipeline Creation, Baseline Model Evaluation Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804822439)
# MAGIC 
# MAGIC [Data Pipeline Creation, Baseline Model Evaluation Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/Data%20Pipeline%20Creation%20-%20Cross%20Validation%20Testing.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Notes about Planned Features
# MAGIC 
# MAGIC Despite this phase focused on engineering features that are directly related to flight delays (i.e. tracking aricraft through their flight status and weather tracking), we believe data relating to airports (e.g. how busy and resilient an airport is) and airlines (e.g. management effectiveness) would also contribute to impacting how frequently flights are delayed. As such, we will continue to explore features around:
# MAGIC 
# MAGIC 1. Airport: 
# MAGIC   - Centrality measures: explore airport importance with centrality measures such as pagerank and betweenness.
# MAGIC   - Frequent diverted flight destination: airports that accept diverted flights need to account for unexpected increase in demand capacity, which can cause other flights which flying out from its airport to be delayed. As such, as time permits we would also like to explore patterns between diverted flights vs. destination airports for diverted flights.
# MAGIC   
# MAGIC 2. Airlines:
# MAGIC   - Our intuition suggests that some airlines may be more effective at mitigating delays than others as different airlines would have different KPIs and standard operating procedures. We would like to explore if there is value to further bucket the airlines by how frequently flights are delayed and one hot encode these buckets.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Link to Phase 2 Presentation Video
# MAGIC 
# MAGIC Per the project requirements, a 2 minute video providing a quick overview of the Phase 2 progress has been created.
# MAGIC 
# MAGIC [Click Here for the Video](https://drive.google.com/file/d/1Ubpv8pGEZStzTEzrSpBFVwnFjY6y3aSx/view?usp=sharing)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # -- Phase 3 --

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Nina, Deanna, Nash
# MAGIC 
# MAGIC # Added Features
# MAGIC 
# MAGIC ## Flight Previously Delayed Indicator
# MAGIC 
# MAGIC Was the flight in question previously delayed within a recent window of time?
# MAGIC 
# MAGIC ## Arrival Airport Weather Data
# MAGIC 
# MAGIC Adding weather data for a flight's arrival airport.
# MAGIC 
# MAGIC ## Incoming Flight Frequency
# MAGIC 
# MAGIC Is the airport experiencing an abnormal number of incoming flights?

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Ryan
# MAGIC 
# MAGIC # Updated Baseline Metrics, Evaluation with New Features
# MAGIC 
# MAGIC ## Baseline Model Evaluation over Training Data
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Baseline Model Evaluation over Test Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Deanna
# MAGIC 
# MAGIC # New Model: Random Forests
# MAGIC 
# MAGIC ## Model Evaluation over Training Data
# MAGIC 
# MAGIC ## Model Evaluation over Test Data
# MAGIC 
# MAGIC ## Comparison with Baseline Model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Assignment TBA
# MAGIC 
# MAGIC # Hyperparameter Tuning

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Assignment TBA
# MAGIC 
# MAGIC # Gap Analysis with Leaderboard Models
# MAGIC 
# MAGIC ## Identified Deficiencies
# MAGIC 
# MAGIC ## Recommended Steps
# MAGIC 
# MAGIC A gap analysis is a process that compares actual performance or results with what was expected or desired (versus other teams in this case). The method provides a way to identify suboptimal or missing strategies, structures, capabilities, processes, practices, technologies, or skills, and then recommends steps that will help the company meet its goals.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Link to Phase 3 Presentation Video
# MAGIC 
# MAGIC Per the project requirements, a 2 minute video providing a quick overview of the Phase 2 progress has been created.
# MAGIC 
# MAGIC [Click Here for the Video]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Link to In-Class Presentation Slides

# COMMAND ----------


