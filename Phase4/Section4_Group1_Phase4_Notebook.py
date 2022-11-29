# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Instructions: https://digitalcampus.instructure.com/courses/4868/pages/phase-descriptions-and-deliverables?module_item_id=686692
# MAGIC 
# MAGIC - Introduce the business case
# MAGIC - Introduce the dataset
# MAGIC - Summarize EDA and feature engineering
# MAGIC - Summarize algorithms tried, and justify the final prediction pipeline (all the steps) choice
# MAGIC - Describe novel approaches taken
# MAGIC - Discuss evaluation metrics in light of the business case
# MAGIC - Discuss performance and scalability concerns
# MAGIC - Summarize limitations, challenges, and future work

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Team 13 Phase 4 Notebook
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
# MAGIC | Task                   | Start Date | End Date | Estimated Time (hours) | Nashat Cabral | Deanna Emery | Nina Huang | Ryan S. Wong |
# MAGIC |------------------------|------------|----------|------------------------|---------------|--------------|------------|--------------|
# MAGIC | Data Pipeline Creation | 11/14/22   | 11/25/22 | 8                      |               | X            |            | X            |
# MAGIC | Model Building         | 11/18/22   | 11/27/22 | 12                     |               | X            |            | X            |
# MAGIC | Feature Engineering    | 11/14/22   | 11/25/22 | 10                     | X             | X            | X          |              |
# MAGIC | Notebook Writeup       | 11/19/22   | 11/27/22 | 3                      | X             | X            | X          | X            |
# MAGIC | Presentation Setup     | 11/14/22   | 11/16/22 | 4                      | X             | X            | X          | X            |
# MAGIC | Submission             | 11/27/22   | 11/27/22 | 1                      |               | X            |            |              |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Links to Project Notebooks
# MAGIC 
# MAGIC [Link to Phase 1 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/364123876153624/command/4295587629775265)
# MAGIC 
# MAGIC [Link to Phase 1 Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase1/Section4_Group1_Phase1_ProjectProposal_Notebook.ipynb)
# MAGIC 
# MAGIC [Link to Phase 2 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804836156/command/1020093804836157)
# MAGIC 
# MAGIC [Link to Phase 2 Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/Section4_Group1_Phase2_Notebook_Final.py)
# MAGIC 
# MAGIC [Link to Phase 3 Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/2647101326237439/command/2647101326237443)
# MAGIC 
# MAGIC [Link to Phase 3 Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/Section4_Group1_Phase3_Notebook_Final.py)
# MAGIC 
# MAGIC [Link to Phase 4 Notebook]()
# MAGIC 
# MAGIC [Link to Phase 4 Notebook in GitHub]()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1. Project Abstract
# MAGIC 
# MAGIC In the flight industry, delays are a key issue for airline companies, airports, and customers. This project focuses on empowering customers to plan their itinerary by predicting if their flight will be delayed 2 hours before their planned departure time. 
# MAGIC Our customer-focused objective would be achieved through minimizing false positives by running machine models against airport, flight, and local weather data. Any analyses and methods applied will come from the perspective of benefiting the customer, and thus we chose F-0.5 as our primary metric (highest priority is given towards minimizing false positives while secondarily minimizing false negatives) and precision as our secondary metrics (minimizing all instances of false positives, regardless of the number of false negatives). After creating and training our baseline logistic regression model on the data set and features created, our model returned F-0.5 of 0.423 and a precision of 0.429 in test evaluation. In later phases, we will be implementing more advanced classification models in order to create a machine learning model that optimally predicts future flight delays.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 2. Introduction
# MAGIC 
# MAGIC Some information about flight delays go here.
# MAGIC 
# MAGIC How much do flight delays cost users? Cost airlines?
# MAGIC 
# MAGIC Definitely get information about how delayed flights impact user experience.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3. Dataset of Interest

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 4. EDA on Raw and Joined Data Sets

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
# MAGIC ##### Count of Nulls in fields of interest in airline data
# MAGIC ![Airline Nulls](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/images/airlinenull.PNG?raw=true)
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
# MAGIC ##### Count of Nulls in fields of interest in weather data
# MAGIC ![Weather Nulls](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/images/weather%20null.PNG?raw=true)
# MAGIC 
# MAGIC 
# MAGIC ##### Statististics of fields of interest in weather data
# MAGIC ![Weather Stats](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/images/weatherstats.PNG?raw=true)
# MAGIC 
# MAGIC ### Stations Data Set Features
# MAGIC 
# MAGIC - neighbor_call
# MAGIC - neighbor_state
# MAGIC - station_id
# MAGIC - distance_to_neighbor
# MAGIC - neighbor_lat
# MAGIC - neighbor_lon
# MAGIC ##### Statististics on distance_to_neighbor field of station dataset
# MAGIC ![Station Stats](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/images/stationstats.PNG?raw=true)
# MAGIC 
# MAGIC 
# MAGIC ## Remarkable Findings
# MAGIC 
# MAGIC ### Airlines Data Set
# MAGIC 
# MAGIC The Airline On-Time Performance Data table contains the scheduled and actual departure/arrival times for U.S. Domestic flights for qualifying airline carriers. These carriers must account for at least one percentage of U.S Domestic scheduled passenger revenues in order to qualify. Our data ranges from 2015 to 2021
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
# MAGIC ##### Airlines Flight Delay Percentages
# MAGIC ![Airline Flight Delay Analysis](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-11-26_23-11-56.png)
# MAGIC 
# MAGIC - B6, more commonly known in the real world as JetBlue was the airline with the most delays in relation to its total number of delays
# MAGIC   - Although, Jet Blue has relatively less flights than other comparable airlines within this dataset.
# MAGIC - WN (Southwest) and AA (American Airlines) are two other airlines we see have a higher percentage of delayed flights, although they do not have the small number of total flights exhibited by Jetblue. This leads us to believe that Southwest and American Airlines appear to be the worst overall airlines.
# MAGIC - HA (Hawaiian Airlines) and QX (Horizon Air) display the least percentage of flights delayed but both have a relatively small number of flights within this dataset.
# MAGIC - DL (Delta Airlines) shows to have a considerable amount of flights within the dataset while also having a lower percentage of flights delayed than other airlines with similar quantities of total flights.
# MAGIC 
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
# MAGIC - Different airlines have different volumes of flight, with certain airlines that have low delay percentage and low flight volume (e.g. HA), low delay percentage and high flight volume (e.g. DL), and ones that have high delays. See the figure below for details. This observation suggests that including an airline effectiveness feature could improve the model performance. 
# MAGIC 
# MAGIC 
# MAGIC 
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
# MAGIC # EDA on Joined Data Set
# MAGIC 
# MAGIC - The Joined dataset contains close to 41 million rows
# MAGIC - This EDA will ignore the monthly weather data in our current join as they are intended for use in a later iteration of our model.
# MAGIC 
# MAGIC Before conducting any feature engineering, we want to perform a broad brush analysis on how our features perform against flight delays. To achieve this, we created a correlation matrix for all non-string and non-id fields calculated from both Pearson correlation to measure the linearity of columns against the target variable, and the Spearman correlation to measure the strength and direction of association (monotonic association).
# MAGIC 
# MAGIC ![Phase 3 Correlation Update](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-11-27_18-46-03.png)
# MAGIC 
# MAGIC 
# MAGIC Although some interesting relationships exist with the raw data, no feature stands out as being able drive a strong machine learning performance (lack of strong correlation). As such, focused our efforts on engineering features that should have a stronger relationship with flight delays defined through business intuitions (such as tracking aircraft delay status across flights), airline management effectiveness, airport importance, and special days effect. See our sections on "Features To Be Used in Future Analysis" and "Notes about Planned Features" for more. We conducted our correlation analysis on Pearson (calculating the strength of linearity) and Spearman (calculating the strength of association). It is interesting to note that the correlation with the target variable for the output variable is not as high as we would have hoped for. That said, the engineered features have a higher Pearson correlation than they do with Spearman. This suggests that while our engineered features can improve our logistic regression model results (assumes linearity), they may not be able to noticeably lift tree-based models results given the non-monotonic nature of the features.
# MAGIC 
# MAGIC  
# MAGIC ## Notable Feature Characteristics
# MAGIC - About 80% of our data for DEP_DEL15 indicate as non-delayed flights
# MAGIC - Months with the most data in this dataset are December/January, likely due to the holiday season. This may impact our future cross validation decisions.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ![Month Visual](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/monthvis2.PNG?raw=true)
# MAGIC 
# MAGIC - Canceled flights are considered Delayed under DEP_DEL15 for this analysis
# MAGIC - Both HourlyWetBulbTemperature and HourlyDryBulbTemperature were normally distributed, with HourlyWetBulbTemperature appearing to have a slightly more narrow distribution.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ![Dry Visual](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/drytempvis.PNG?raw=true)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ![Wet Visual](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/wettempvis.PNG?raw=true)
# MAGIC 
# MAGIC - HourlyPressureChange does not seem to change significantly across the dataset, while also was missing from 24% of it, and will likely be dropped from future analyses.
# MAGIC   - HourlyStationPressure and HourlySeaLevelPressure showed a similarly tight distribution but were both missing from less than 1% of the dataset, and thus were deemed worthy to keep in our current model
# MAGIC - HourlyWindGustSpeed was missing from 64% of the dataset and will likely be dropped from future analyses
# MAGIC - HourlyWindSpeed displays outlier datapoints with records indicating a windspeed greater than 1000.
# MAGIC 
# MAGIC 
# MAGIC 
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
# MAGIC 
# MAGIC [Correlation Analysis in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/NH/CorrelationAnalysis.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 5. Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Added Features
# MAGIC 
# MAGIC In addition to the pipeline changes, a number of new features were created and introduced to our joined data set. Many of these features contain highly predictive elements, leaning heavily into tracking the frequency of certain events occurring. These new features are described below.
# MAGIC 
# MAGIC ## 1. Flight Previously Delayed Indicator
# MAGIC 
# MAGIC The 'is_prev_delayed' field is an indicator field that is created to answer the question: Was the flight in question previously delayed within a recent window of time? We assume that if a flight's prior aircraft was delayed then it is very likely that the current flight will be delayed as well. 
# MAGIC We assume that the first flight of a day is always not delayed.
# MAGIC To avoid data leakage, we looked at flights that were at least 2 hours apart when creating this field. For example, if 2 flights departed within 2 hours, we would get the flight status from the next flight prior as an indicator for if the previous flight was delayed (as opposed to using the prior flight). 
# MAGIC 
# MAGIC 
# MAGIC ## 2. Flight Previously Diverted Indicator
# MAGIC Similar to the previous flight delay indicator, we created the 'is_prev_diverted' field to answer whether a flight in question was previously diverted? We assume that if a flight's prior aircraft was diverted then it is very likely that the current flight will be delayed as well. The methodology used to implement this field is the same for implementing the previously delayed fligth tracker field.
# MAGIC 
# MAGIC 
# MAGIC ## 3. Airline Efficacy Score
# MAGIC 
# MAGIC We created 2 fields to capture the effectiveness of airlines in managing delayed flights by looking at their historical delayed flight ratio:
# MAGIC 
# MAGIC 1. perc_delay: percentage of delayed flights of a carrier over the total number of flights scheduled for that carrier
# MAGIC 2. airline_type: categorical field with 3 values: 
# MAGIC   - BelowAverage: bottom 25 percentile of airlines by perc_delay
# MAGIC   - Average: airlines that fall within the 25-75th percentile by perc_delay
# MAGIC   - AboveAverage: top 25 percentile of airlines by perc_delay
# MAGIC For new airlines, we would impute their perc_delay as the average of the percentage delay across all airlines from the prior year.
# MAGIC To avoid data leakage, we used the airline efficiency from prior year to gauge their performance for the current year. For the beginning of the dataset (2015) where we didn't have the prior year data, we used 2015's airline values.
# MAGIC 
# MAGIC ## 4. Holiday and Special Event Day Tracker
# MAGIC 
# MAGIC We assume that certain days of the year are more busy or less busy than normal that would impact the volume of air travel during a particular day. Depending on how the air travel volume, the flight delay pattern may also be impacted. Furthermore, we recognize that multiple states had lock downs during the onset of Covid, which would operate in a way that is similar to days which are less busier than normal. With these assumptions in mind, we created a secondary dataset to track special dates (including holidays and shelter in place from Massachusetts due to Covid). The effect of these holidays are then joined to the dataset and summarized in the 'AssumedEffect' column.
# MAGIC 
# MAGIC ## 5. Incoming Flight Frequency with PageRank
# MAGIC 
# MAGIC After some contemplation, one factor that we sought to capture in our data was how busy an airport would be when it came to handling outbound air traffic; an airport with high levels of air traffic would likely lead to flights being delayed. To account for the factor of frequency and volume of outgoing flights for an airport, we created a column 'pagerank' that uses the DEST airport as nodes and ORIGIN to DEST connecting flights as edges. The PageRank values were computed using the GraphFrames library in PySpark.
# MAGIC 
# MAGIC To avoid data leakage, we used the pagerank value from the prior year to reflect incoming flight frequency of an airport. For the beginning of the dataset (2015), we do not have prior year data available. As such we decided to drop 2015 from further analysis for now given that we could not use 2014 data for the flights in 2015; it was decided that having this new feature would be more beneficial than having the 2015 data, which did not seem to be very influential when training our models. The dataset continues as normal from 2016 on, with each flight using the pagerank of the destination from the previous year. 
# MAGIC 
# MAGIC #### PageRank Scores by Destination Airport ID
# MAGIC 
# MAGIC Below is a visualization of the PageRank scores per airport per year. Please note the following:
# MAGIC 
# MAGIC - The truncated visual above displays the pagerank performance of destination airport ID's within the dataset, for visibility's sake not all data is shown.
# MAGIC - Destinations such as ATL, DEN, DFW, and ORD appear to have the highest pagerank scores over the years and thus are expected to be the busiest destination airports.
# MAGIC - Destinations with higher pagerank scores are expected to be more prone to delayed flights. 
# MAGIC 
# MAGIC ![PageRank](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/images/Pagerank.png?raw=true)
# MAGIC 
# MAGIC The correlation of these engineered fields is discussed further in the "EDA on Joined Data Set" section above. 
# MAGIC 
# MAGIC ## Updated Feature List
# MAGIC 
# MAGIC Below are the list of features that our pipeline now includes in our data set. This list contains 18 categorical features and 15 numeric features.
# MAGIC 
# MAGIC #### Categorical Features
# MAGIC 
# MAGIC - ORIGIN
# MAGIC - QUARTER
# MAGIC - MONTH
# MAGIC - DAY_OF_MONTH
# MAGIC - DAY_OF_WEEK
# MAGIC - FL_DATE
# MAGIC - OP_UNIQUE_CARRIER
# MAGIC - TAIL_NUM
# MAGIC - OP_CARRIER_FL_NUM
# MAGIC - ORIGIN_AIRPORT_SEQ_ID
# MAGIC - ORIGIN_STATE_ABR
# MAGIC - DEST_AIRPORT_SEQ_ID
# MAGIC - DEST_STATE_ABR
# MAGIC - CRS_DEP_TIME
# MAGIC - YEAR
# MAGIC - AssumedEffect
# MAGIC - is_prev_delayed
# MAGIC - is_prev_diverted
# MAGIC 
# MAGIC #### Numeric Features
# MAGIC 
# MAGIC - CRS_ELAPSED_TIME
# MAGIC - DISTANCE
# MAGIC - ELEVATION
# MAGIC - HourlyAltimeterSetting
# MAGIC - HourlyDewPointTemperature
# MAGIC - HourlyWetBulbTemperature
# MAGIC - HourlyDryBulbTemperature
# MAGIC - HourlyPrecipitation
# MAGIC - HourlyStationPressure
# MAGIC - HourlySeaLevelPressure
# MAGIC - HourlyRelativeHumidity
# MAGIC - HourlyVisibility
# MAGIC - HourlyWindSpeed
# MAGIC - perc_delay
# MAGIC - pagerank

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 6. Data Pipeline and Data Considerations

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
# MAGIC Given that our primary metric for this model was precision, the low value of the precision metric surprises us. We have a number of possible reasons why the metric turned out so unusually low: high levels of variance in the features introducing noise, incorrect scaling for features prior to training, or perhaps logistic regression does not perform well with a categorization task or data set like this. Further investigation will be done in order to determine whether this low precision value is the legitimate result of the circumstances or if a confounding error is at play.
# MAGIC 
# MAGIC ## Link to Data Pipeline Creation, Baseline Model Evaluation Notebook
# MAGIC 
# MAGIC [Data Pipeline Creation, Baseline Model Evaluation Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804822439)
# MAGIC 
# MAGIC [Data Pipeline Creation, Baseline Model Evaluation Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/Data%20Pipeline%20Creation%20-%20Cross%20Validation%20Testing.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Joining the Data Sets
# MAGIC 
# MAGIC ![Join Schema](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/data_join_gantt.png)
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
# MAGIC 8. Finally, we were able to join the airlines data and the weather/station data together as an inner merge using the columns ORIGIN and DEP_DATETIME_LAG. 
# MAGIC 
# MAGIC The resulting joined dataset had 40,933,735 rows for the years 2015-2021 and ran in 11 minutes on 4 cores.
# MAGIC 
# MAGIC ## Joined Dataset Schema
# MAGIC - ORIGIN : string
# MAGIC - DEP_DATETIME_LAG : timestamp
# MAGIC - QUARTER : integer
# MAGIC - MONTH : integer
# MAGIC - DAY_OF_MONTH : integer
# MAGIC - DAY_OF_WEEK : integer
# MAGIC - FL_DATE : string
# MAGIC - OP_UNIQUE_CARRIER : string
# MAGIC - TAIL_NUM : string
# MAGIC - OP_CARRIER_FL_NUM : integer
# MAGIC - ORIGIN_AIRPORT_ID : integer
# MAGIC - ORIGIN_AIRPORT_SEQ_ID : integer
# MAGIC - ORIGIN_STATE_ABR : string
# MAGIC - ORIGIN_WAC : integer
# MAGIC - DEST : string
# MAGIC - DEST_AIRPORT_ID : integer
# MAGIC - DEST_AIRPORT_SEQ_ID : integer
# MAGIC - DEST_STATE_ABR : string
# MAGIC - DEST_WAC : integer
# MAGIC - CRS_DEP_TIME : string
# MAGIC - DEP_TIME : integer
# MAGIC - DEP_DEL15 : double
# MAGIC - DEP_DELAY : double
# MAGIC - DIVERTED : double
# MAGIC - DIV_AIRPORT_LANDINGS : integer
# MAGIC - CANCELLED : double
# MAGIC - CANCELLATION_CODE : string
# MAGIC - CRS_ELAPSED_TIME : double
# MAGIC - DISTANCE : double
# MAGIC - YEAR : integer
# MAGIC - DEP_HOUR : string
# MAGIC - DEP_DAY : string
# MAGIC - DEP_DATETIME : string
# MAGIC - DATE : double
# MAGIC - ELEVATION : double
# MAGIC - SOURCE : double
# MAGIC - HourlyAltimeterSetting : double
# MAGIC - HourlyDewPointTemperature : double
# MAGIC - HourlyWetBulbTemperature : double
# MAGIC - HourlyDryBulbTemperature : double
# MAGIC - HourlyPrecipitation : double
# MAGIC - HourlyStationPressure : double
# MAGIC - HourlySeaLevelPressure : double
# MAGIC - HourlyRelativeHumidity : double
# MAGIC - HourlyVisibility : double
# MAGIC - HourlyWindSpeed : double
# MAGIC - DATE_HOUR : double
# MAGIC - distance_to_neighbor : double
# MAGIC - neighbor_lat : double
# MAGIC - neighbor_lon : double
# MAGIC - time_zone_id : string
# MAGIC - UTC_DEP_DATETIME_LAG : timestamp
# MAGIC - UTC_DEP_DATETIME : timestamp
# MAGIC - DEP_TIME_CLEANED : string
# MAGIC - flight_id : string
# MAGIC  
# MAGIC  
# MAGIC ### Links to data joining notebooks
# MAGIC [Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4423519682321930/command/1020093804821142)
# MAGIC 
# MAGIC [Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/DLE_join_V1.py)

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
# MAGIC Since the objective of our machine learning model is to help a customer plan their day, the impact of a canceled flight should be treated similarly to the impact of a delayed flight (bad experience for the customer either way that requires them to replan their day). As such, flights that were canceled were marked by the feature CANCELLED == 1. Furthermore, flight-related features being converted to null values, the most important being the DEP_DEL15 feature. Given that a canceled flight elicits the same inconvenience as a delayed flight, it was decided that canceled flights should be considered as a delayed flight. Thus, rows for canceled flights had their DEP_DEL15 imputed to 1.
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
# MAGIC Our current data set features 18 categorical features which we are including in our data pipeline for analysis. All 18 features are being converted with string indexing and encoded with one-hot encoding. While we recognize that this may not be ideal to be applied to every one of the categorical features in question, technical debugging issues with model training in the greater pipeline prevented us from taking a different approach with encoding and indexing those features. For now, we will continue to include those in our analysis, as we believe that their inclusion in a non-ideal format is preferable to dropping those feature entirely. In the future, we would like to investigate and resolve those technical issues in order to better format the categorical features.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 7. Algorithms and Evaluations

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
# MAGIC Precision, accuracy, F1, and F0.5 are described further with their equations in the bullet points below.
# MAGIC 
# MAGIC - Precision is the rate of true positivity within the model. It is computed using the formula below, where TP = true positives and FP = false positives.
# MAGIC 
# MAGIC \\( Precision  = \frac{TP}{TP + FP} \\)
# MAGIC 
# MAGIC - Accuracy is the rate of all correctly classified observations. It is computed using the formula below, where TP = true positives, TN = true negatives, FP = false positives, and FN = false negatives.
# MAGIC 
# MAGIC \\( Accuracy  = \frac{TP + TN}{TP + TN + FP + FN} \\)
# MAGIC 
# MAGIC - F1 is the harmonic mean of precision and recall, and is computed using the formula below.
# MAGIC 
# MAGIC \\( F_1 = 2 * \frac{precision \cdot recall}{precision + recall} = \frac{2 * TP}{2 * TP + FP + FN} \\)
# MAGIC 
# MAGIC 
# MAGIC - F0.5 is a weighted harmonic mean of precision and recall, and is computed using the formula below.
# MAGIC 
# MAGIC \\( F_{0.5} = \frac{1.25 * precision \cdot recall}{0.25 * precision + recall} \\)
# MAGIC 
# MAGIC For creating a proper comparison and target for our model, we will compare our model against a simple baseline model. This baseline will be a model that predicts that every flight will not be delayed. Any improvements in our model over this baseline model will represent an overall improvement in the ability to correctly predict what flights will be delayed. Therefore, our desired model would be a model that has a high F0.5 and precision score while matching or exceeding the accuracy score of the baseline evaluation.
# MAGIC 
# MAGIC This desired outcome may lead to some interesting alternatives that would be worthwhile to explore as well. One possibility is to change the predicted value from being a categorical one to a numerical one, and focus our efforts towards getting accurate predictions of flight delays. This alternative would require measuring different metrics for success and incorporating different models, but it would be able to be done with the same data. The intended audience and use case would be the same as well.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Updated Baseline Metrics, Evaluation with New Features
# MAGIC 
# MAGIC Using our baseline Logistic Regression model, we compared the performance of the original model without the above new features with an updated model that included the new features. Below is a table that shows the evaluation metrics for both the training data and the test data for both the old model and the updated model. Do note that these baseline models have not been hyperparameter tuned, so they will have the following default parameters: 
# MAGIC 
# MAGIC - regParam = 0.0
# MAGIC - elasticNetParam = 0
# MAGIC - maxIter = 10
# MAGIC - threshold = 0.5
# MAGIC 
# MAGIC | Metrics   | Old Model - Training | Old Model - Test | New Model - Training | New Model - Test |
# MAGIC |-----------|----------------------|------------------|----------------------|------------------|
# MAGIC | Precision | 0.652900256          | 0.616746361      | 0.635264338          | 0.544768051      |
# MAGIC | F0.5      | 0.62549383           | 0.478734702      | 0.634401578          | 0.476315543      |
# MAGIC | Recall    | 0.53556876           | 0.252617775      | 0.630973849          | 0.316990518      |
# MAGIC | Accuracy  | 0.604456197          | 0.58781616       | 0.658943611          | 0.695887596      |
# MAGIC 
# MAGIC As expected, the precision scores during the training data sets are consistently higher than that of the test sets. However, there is a significant difference between the metrics for the old model and the new model, which is best illustrated when comparing each model's test data set evaluation. The old model had a super precision level, being about 0.07 points higher than the new model. But the new model had significantly better recall and accuracy scores, which seemed to balance out with the lowered precision score to make the F0.5 score approximately equivalent between both models. This is likely due to the new features adding additional noise to the updated model, seeming to cause a greater number of false positive results while also decreasing the number of false negatives in equal proportion. Furthermore, because the new model does not contain values for the pagerank feature for the year 2015, the training data for that entire year is dropped to prevent data leakage. This means that the new model has approximately 14% less data to train with compared to the old model, and this lack of training data may explain the newfound performance gap.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Hyperparameter Tuning
# MAGIC 
# MAGIC Once we had our logistic regression model set up and fully functional with the new features, we set out to conduct hyperparameter tuning in order to optimize the performance of our model. For this logistic regression model, we opted to tune and evaluate the model based on four parameters: the regression parameter (regParam), the elastic net parameter (elasticNetParam), the maximum number of iterations (maxIter), and the probability threshold (threshold). For evaluating our models, our primary performance metric would be the F-0.5 score for the test data set evaluation, which is shown here under the "test_F0.5" column.
# MAGIC 
# MAGIC Using those parameters, we performed an exhaustive grid search with our baseline logistic regression model. The parameters we used to gridsearch through are listed below:
# MAGIC - regParam = [0.0, 0.01, 0.5, 2.0]
# MAGIC - elasticNetParam = [0.0, 0.5, 1.0]
# MAGIC - maxIter = [10, 50]
# MAGIC - thresholds = [0.5, 0.6, 0.7, 0.8] 
# MAGIC 
# MAGIC Below are the results of our gridsearch:
# MAGIC 
# MAGIC | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy | year | regParam | elasticNetParam | maxIter | threshold |
# MAGIC |----------------|-------------|-------------|-------------|---------------|---------------|-------------|-------------|-------------|--------------|------|----------|-----------------|---------|-----------|
# MAGIC | 0.186683521    | 1           | 0.222949167 | 0.314630679 | 0.186683521   | 0.47068126    | 1           | 0.526408876 | 0.640086023 | 0.47068126   | 2016 | 0.5      | 1               | 10      | 0.5       |
# MAGIC | 0.186683521    | 1           | 0.222949167 | 0.314630679 | 0.186683521   | 0.47068126    | 1           | 0.526408876 | 0.640086023 | 0.47068126   | 2016 | 2        | 1               | 10      | 0.5       |
# MAGIC | 0.186683521    | 1           | 0.222949167 | 0.314630679 | 0.186683521   | 0.47068126    | 1           | 0.526408876 | 0.640086023 | 0.47068126   | 2016 | 0.5      | 1               | 50      | 0.5       |
# MAGIC | 0.186683521    | 1           | 0.222949167 | 0.314630679 | 0.186683521   | 0.47068126    | 1           | 0.526408876 | 0.640086023 | 0.47068126   | 2016 | 2        | 1               | 50      | 0.5       |
# MAGIC | 0.439071124    | 0.397166678 | 0.42999745  | 0.417068971 | 0.792738767   | 0.725104372   | 0.484109444 | 0.659448273 | 0.580591902 | 0.676012827  | 2018 | 0        | 0.5             | 50      | 0.6       |
# MAGIC | 0.4515935      | 0.374470563 | 0.433728073 | 0.409431858 | 0.798329653   | 0.707584047   | 0.485156972 | 0.64815306  | 0.575630968 | 0.668639884  | 2018 | 0.01     | 0.5             | 50      | 0.5       |
# MAGIC | 0.426549347    | 0.405695253 | 0.422208758 | 0.415861022 | 0.787233194   | 0.728797604   | 0.479315894 | 0.660083519 | 0.578297115 | 0.676185866  | 2018 | 0        | 0.5             | 10      | 0.6       |
# MAGIC | 0.445742056    | 0.37766576  | 0.430231736 | 0.40888976  | 0.796152289   | 0.707598382   | 0.486246297 | 0.648550901 | 0.57640177  | 0.668942401  | 2018 | 0.01     | 0.5             | 10      | 0.5       |
# MAGIC | 0.439071124    | 0.397166678 | 0.42999745  | 0.417068971 | 0.792738767   | 0.725104372   | 0.484109444 | 0.659448273 | 0.580591902 | 0.676012827  | 2018 | 0        | 0               | 50      | 0.6       |
# MAGIC | 0.434604184    | 0.399075436 | 0.427001201 | 0.41608275  | 0.790895889   | 0.734204296   | 0.470663992 | 0.660263617 | 0.573612117 | 0.675872459  | 2018 | 0.01     | 0               | 50      | 0.6       |
# MAGIC | 0.34746515     | 0.533351724 | 0.37349999  | 0.420793793 | 0.725897142   | 0.621151394   | 0.610919662 | 0.619077721 | 0.615993043 | 0.64148751   | 2016 | 0.5      | 0               | 50      | 0.5       |
# MAGIC | 0.310108584    | 0.598232796 | 0.343163885 | 0.408474456 | 0.676544308   | 0.578952878   | 0.675669363 | 0.596015826 | 0.623583274 | 0.616058034  | 2016 | 2        | 0               | 50      | 0.5       |
# MAGIC | 0.186683521    | 1           | 0.222949167 | 0.314630679 | 0.186683521   | 0.47068126    | 1           | 0.526408876 | 0.640086023 | 0.47068126   | 2016 | 0.5      | 0.5             | 50      | 0.5       |
# MAGIC | 0.186683521    | 1           | 0.222949167 | 0.314630679 | 0.186683521   | 0.47068126    | 1           | 0.526408876 | 0.640086023 | 0.47068126   | 2016 | 2        | 0.5             | 50      | 0.5       |
# MAGIC | 0.439071124    | 0.397166678 | 0.42999745  | 0.417068971 | 0.792738767   | 0.725104372   | 0.484109444 | 0.659448273 | 0.580591902 | 0.676012827  | 2018 | 0        | 1               | 50      | 0.6       |
# MAGIC | 0.477447225    | 0.355912414 | 0.446924605 | 0.407817673 | 0.807039456   | 0.735711672   | 0.430761273 | 0.644464107 | 0.543374963 | 0.66463698   | 2018 | 0.01     | 1               | 50      | 0.5       |
# MAGIC | 0.428907331    | 0.402227907 | 0.423292013 | 0.415139414 | 0.788423921   | 0.736174812   | 0.467936762 | 0.660455482 | 0.572178302 | 0.675856728  | 2018 | 0.01     | 0               | 10      | 0.6       |
# MAGIC | 0.347447074    | 0.533359155 | 0.37348401  | 0.42078285  | 0.725881016   | 0.621130584   | 0.610930442 | 0.619063398 | 0.61598829  | 0.641473979  | 2016 | 0.5      | 0               | 10      | 0.5       |
# MAGIC | 0.310107158    | 0.598234654 | 0.34316261  | 0.408473652 | 0.676542227   | 0.578952878   | 0.675669363 | 0.596015826 | 0.623583274 | 0.616058034  | 2016 | 2        | 0               | 10      | 0.5       |
# MAGIC | 0.186683521    | 1           | 0.222949167 | 0.314630679 | 0.186683521   | 0.47068126    | 1           | 0.526408876 | 0.640086023 | 0.47068126   | 2016 | 0.5      | 0.5             | 10      | 0.5       |
# MAGIC | 0.186683521    | 1           | 0.222949167 | 0.314630679 | 0.186683521   | 0.47068126    | 1           | 0.526408876 | 0.640086023 | 0.47068126   | 2016 | 2        | 0.5             | 10      | 0.5       |
# MAGIC | 0.426549347    | 0.405695253 | 0.422208758 | 0.415861022 | 0.787233194   | 0.728797604   | 0.479315894 | 0.660083519 | 0.578297115 | 0.676185866  | 2018 | 0        | 0               | 10      | 0.6       |
# MAGIC | 0.426549347    | 0.405695253 | 0.422208758 | 0.415861022 | 0.787233194   | 0.728797604   | 0.479315894 | 0.660083519 | 0.578297115 | 0.676185866  | 2018 | 0        | 1               | 10      | 0.6       |
# MAGIC | 0.47775445     | 0.355723861 | 0.447080386 | 0.407805831 | 0.807132224   | 0.734715312   | 0.431268057 | 0.644078482 | 0.5435056   | 0.664419167  | 2018 | 0.01     | 1               | 10      | 0.5       |
# MAGIC 
# MAGIC As can be seen, the model with the best F0.5 score for the training validation set recieved a score of 0.660455482, and a test evaluation F0.5 score of 0.423292013. This model has the parameters regParam = 0.01, elasticNetParam = 0, maxIter = 10, and threshold = 0.6. 
# MAGIC With that being said, we did find a model with a different set of parameters that actually performed better than the previous model when evaluating the test data set. This second model has the parameters regParam = 0.01, elasticNetParam = 1, maxIter = 10, and threshold = 0.5, and it performed the best on the test evaluation with a F0.5 value of 0.447080386.  
# MAGIC 
# MAGIC 
# MAGIC We believe that this increase in performance is due to the regularization parameter helping combat some of the overfitting in the model, therefore making the model more generalizable and accurate to unseen data. With further testing on increasing both the regParam and the elasticNetParam, model performance might increase even more until hitting a point of diminishing returns as our model incorporates too much artificial noise and thus loses the influence of the training data.
# MAGIC 
# MAGIC After analyzing the data, we made the following notes regarding the influence of each parameter on the model's performance:
# MAGIC 
# MAGIC - Increasing the regularization parameter led to increases in the recall value, but this generally led to a decrease in the F0.5 score.
# MAGIC - Increasing the elastic net parameter on its own makes no difference in the model metrics. But increasing the elastic net parameter along with the regularization parameter does yield an increase in precision that offsets the decrease in recall, leading to an increase F0.5 score.
# MAGIC - Increasing the maximum number of iterations parameter makes very little difference. Precision increases and recall decreases in roughly equivalent amounts that net very small increases in the F0.5 score, only a few hundreths of points changed.
# MAGIC - Increasing the threshold parameter leads to a significant increase in precision at the cost of a significant decrease in recall. But when increasing from thresholds of 0.5 to 0.6, the change in precision and recall leads to a net increase for the F0.5 score.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 8. Discussion of Evaluations and Experimental Results

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Discussion of Experimental Results
# MAGIC 
# MAGIC Recall that the basis of our project is to find if a machine learning model can help improve flight delay predictions. To this end, we have successfully created a logistic regression model to serve as baseline for comparison against more advanced models.
# MAGIC 
# MAGIC As mentioned previously, we have two top models based on our baseline logistic regression model: Model 1 with the best F0.5 score for training validation with a score of 0.660, and Model 2 with the best F0.5 score for test evaluation with a score of 0.447.
# MAGIC 
# MAGIC Model Designation | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy | year | regParam | elasticNetParam | maxIter | threshold |
# MAGIC ------------------|----------------|-------------|-------------|-------------|---------------|---------------|-------------|-------------|-------------|--------------|------|----------|-----------------|---------|-----------|
# MAGIC Model 1           | 0.428907331    | 0.402227907 | 0.423292013 | 0.415139414 | 0.788423921   | 0.736174812   | 0.467936762 | 0.660455482 | 0.572178302 | 0.675856728  | 2018 | 0.01     | 0               | 10      | 0.6       |
# MAGIC Model 2           | 0.47775445     | 0.355723861 | 0.447080386 | 0.407805831 | 0.807132224   | 0.734715312   | 0.431268057 | 0.644078482 | 0.5435056   | 0.664419167  | 2018 | 0.01     | 1               | 10      | 0.5       |
# MAGIC 
# MAGIC Looking back on our project's business case, if these models were put into production, we would expect them to have an okay level of predictive power on a real-world data set. Model 1 may have the best F0.5 score during training validation, but during the test evaluation, it had an F0.5 score of 0.423292013, a precision score of 0.428907331, and a recall score of 0.402227907. These test evaluation metrics would be below the scores for Model 2, which had the following test evaluation scores: F0.5 at 0.447080386, precision at 0.47775445, and recall of 0.355723861. The F0.5 scores for Model 1 and Model 2 were relatively close, being within 0.02 points of each other, indicating that both models would have roughly equivalent predictive power for predicting fight delays. However, Model 2 has an increase of 0.05 points of precision at the cost of 0.05 points of recall. Because of our desire to minimize the occurance of false positives at all cost, it would be preferable for our model to have 0.05 more points in precision even if it reduces recall by 0.05. Therefore, while Model 1 and Model 2 would both be roughly suitable models to use, Model 2 might actually be the superior model given our business case.
# MAGIC 
# MAGIC With that being said, there was some discussion within our team whether or not Model 2 would be usable as a valid model or not given that we are picking that model based on test evaluation metrics. Would using the test evaluation metrics consistute a form of data leakage, given that we would only be able to view the training validation data? Further discussion and consideration would help clarrify what the best approach for this matter would be.
# MAGIC 
# MAGIC In all models, we find that the validation scores (for precision, recall, F0.5, and F1) are significantly higher than the test scores. This may be a result of over-fitting, or it could be a symptom of the blocking time series cross-validation method. Our cross-validation approach trains a model for each year - if we see significant variance in flight behavior across different years (for example, the COVID-era travel patterns), we may find that the models trained on prior years may be less applicable to the current time-period. In future iterations, we will employ ensemble methods to help address this concern, as well as weights to discount training data from COVID years.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 9. Performance and Scalability Concerns

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 10. Limitations, Challenges, and Future Work

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 11. Conclusion
# MAGIC 
# MAGIC Shit was crazy, yo.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 12. Appendix

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Link to In-Class Presentation Slides
# MAGIC 
# MAGIC [Presentation Link in Google Slides](https://docs.google.com/presentation/d/1-Yc9jgz9TPdmsvAWvPCFchSAAOvipdvYbo6E6HiaK_s/edit#slide=id.g18d7e4d3627_1_1247)
# MAGIC 
# MAGIC [Presentation Link in PDF Format](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/Phase_3_Presentation.pdf)
