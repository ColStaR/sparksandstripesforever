# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # "Allay Airway Delays!" - Creating a Machine Learning Model to Predict Flight Delays
# MAGIC # Team 13 Research Notebook
# MAGIC ### Section 4, Group 1
# MAGIC ### "Sparks and Stripes Forever"
# MAGIC 
# MAGIC #### Members:
# MAGIC - Nashat Cabral
# MAGIC - Deanna Emery
# MAGIC - Nina Huang
# MAGIC - Ryan S. Wong
# MAGIC 
# MAGIC Link to notebook: https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/632558266974488/command/632558266974545

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 1. Project Abstract
# MAGIC 
# MAGIC Flight delays: the bane of airline travelers everywhere. But can they be predicted and avoided? This project aims to help travelers by creating a machine learning model that predicts whether a flight will be delayed 2 hours before its departure time. We incorporated flight, weather, and weather station data across 27 features, including 10 newly created features such as a highly predictive previously-delayed flight tracker. The F0.5 metric was chosen as our primary metric in order to minimize false positives while balancing recall; precision is our secondary metric, as it focuses solely on minizing false positives.  Our baseline logistic regression model returned a test evaluation F0.5 score of 0.197 and precision of 0.328. Five models were trained with blocking time series cross validation: Logistic Regression, Gradient Boosted Tree, and Multilayer Perceptron Classifier Neural Network.  Gradient Boosted Trees demonstrated significant improvement over the baseline, having a test evaluation F0.5 score of 0.526 and a precision of 0.623. The most important features for this top-performing model were indicators for whether the flight was previously delayed or not, the hour of the flight's scheduled departure time, and the flight's airline carrier.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 2. Introduction
# MAGIC 
# MAGIC According to a [2010 report from UC Berkeley](https://news.berkeley.edu/2010/10/18/flight_delays/), in 2007, airling travelers incurred a total of $16.7 billion in expenses as a direct result of their flight being delayed. Not only do delayed flights waste valuable time for customers, but they also impose direct financial burdens on them due to added expenses such as food, travel, and accommodations. Flight delays hamper any airline traveler's experience, and are one of the major reasons why traveler satisfaction with airlines have [plummeted to their lowest levels](https://www.cntraveler.com/story/travelers-satisfaction-with-us-airlines-is-at-its-lowest-point-since-the-pandemic-began) since the COVID-19 pandemic. While airports and airlines may offer as many ameneties as they can, customers would ultimately prefer to be in control of their own time and schedule, as opposed to being beholden to the airline's tardy schedule. 
# MAGIC 
# MAGIC The universally-unpleasant experience of waiting for a delayed flight is what inspired this project, "Allay Airway Delays!". Inspired by our own discomfort of having to wait for a delayed flight, this project seeks to improve the flying experience of air travelers by leveraging airline data, weather data, and weather station data in order to create a machine learning model that can reliably predict which flights will be delayed. To accomplish this experiment, the project will consist of the following steps: create a joined dataset containing flight, weather, and station data; add additional highly-predictive features to the dataset; create and train multiple predictive machine learning models, including a basic baseline and more sophisticated models; and ultimately evaluate the models to see which one has the most effective predictive abilities. The project would be provided as a tool for air travellers, providing them with an accurate prediction of whether or not their flight will be delayed. This delay prediction tool would allow air travellers to schedule their time more optimally by reducing the amount of time they spend waiting for a delayed flight.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3. Dataset of Interest
# MAGIC 
# MAGIC For this project, we will be examining the data from two tables of considerable size, along with an additional smaller table:
# MAGIC 
# MAGIC - Flight Data: A table of over 898 million rows that contains a collection of the “on-time” performance data for passenger flights from the U.S. Department of Transportation. These flights will be limited to domestic flights within the United States and its territories. 
# MAGIC - Weather Data: A table of over 74 million rows that is composed of weather data that was provided by the U.S. National Oceanic and Atmospheric Administration. This data set was used to determine the effect of weather conditions on flight performance for airports within the region.
# MAGIC - Weather Station Data: A table of 5 million rows that houses metadata about weather stations and the major entities that the stations cater to, such as airports. This data set was provided to us by our organization.
# MAGIC 
# MAGIC This section will discuss the lineage of this project's data, starting with the origins of the original datasets, the process of joining the datasets together, and the features that were added.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3.1 Original Datasets
# MAGIC 
# MAGIC By exploring these datasets in conjunction with one another, we hope to develop a better understanding of the fields within and their possible relationships to one another. The information on the datasets below is obtained from the dataset documentation provided, as well as preliminary analyses on fields of interest.
# MAGIC 
# MAGIC ### Airline Flights Data Set
# MAGIC 
# MAGIC The Airline On-Time Performance Data table contains the scheduled and actual departure/arrival times for U.S. Domestic flights for qualifying airline carriers. These carriers must account for at least one percentage of U.S Domestic scheduled passenger revenues in order to qualify. Our data ranges from 2015 to 2021. For the purposes of this preliminary analysis of the data, we will be examining the data from the file ``“/parquet_airlines_data_3m/”``  which consists of flight data from the first quarter of 2015. Since our objective is to maximize a customer's experience in planning their itinerary, we treated canceled flights with the same regard as delayed flights as though they were cancelled indefinitely. Variables of interest within this dataset include: 
# MAGIC 
# MAGIC 
# MAGIC - ORIGIN_AIRPORT_ID: Identifier for the airport of departure.
# MAGIC - DEST_AIRPORT_ID: Identifier for the airport of arrival.
# MAGIC - FL_DATE: scheduled flight date .
# MAGIC - DEP_DELAY_NEW: numerical variable, difference in minutes between scheduled and actual departure time. Early departures are set to 0.
# MAGIC - DEP_DEL15: binary categorical variable that indicates if a flight departure was delayed by more than 15 minutes. This was our primary label feature for this experiment.
# MAGIC - ARR_DELAY_NEW:  numerical variable, difference in minutes between scheduled and actual arrival time. Early arrivals are set to 0.
# MAGIC - ARR_DEL15: binary categorical variable that indicates if a flight arrival was delayed by more than 15 minutes.
# MAGIC - CANCELLED: binary categorical variable indicating if flight was canceled.
# MAGIC - DIVERTED: binary categorical variable indicating if flight was diverted.
# MAGIC - CARRIER_DELAY: numerical variable, indicates time spent delayed due to carrier.
# MAGIC - WEATHER_DELAY: numerical variable, indicates time spent delayed due to weather.
# MAGIC - NAS_DELAY: numerical variable, indicates time spent delayed due to National Air System.
# MAGIC - SECURITY_DELAY: numerical variable, indicates time spent delayed due to security.
# MAGIC - LATE AIRCRAFT DELAY: numerical variable, indicates time spent delayed due to a late aircraft.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Weather Data Set
# MAGIC 
# MAGIC The Quality Controlled Local Climatological Data contains summary data from weather stations located near airports. These stations measure and record daily weather data such as temperature highs, temperature lows, precipitation, wind speed, visibility, and storm characteristics. The available data ranges from 2015 to 2021. For the purposes of this preliminary analysis of the data, we will be examining the data from the file ``“/parquet_weather_data_3m/”``  which consists of weather data from the first quarter of 2015. Any weather variables that were believed have a relationship with flight delays were features of interest, which are listed below:
# MAGIC 
# MAGIC - Station: identifier for each station.
# MAGIC - Date: Year-Month-Day-Hour-Minute-Second identifier for the date of a recorded row. The field providing data to the hour allows for the field to identify hourly data.
# MAGIC - HourlyWindSpeed: numerical variable, indicates wind speed in meters per second. 9999’s are considered missing values.
# MAGIC - HourlySkyConditions:  numeric variable, height in meters of the lowest cloud or obscuring item (max of 22,000).
# MAGIC - HourlyVisibility: numerical variable, distance in meters an object can be seen (max of 16000). 999999 is considered missing.
# MAGIC - HourlyDryBulbTemperature: numerical variable, temperature of air in celsius. 9999+ is considered missing.
# MAGIC - HourlySeaLevelPressure: numerical variable, air pressure relative to Mean Sea Level in hectopascals. 99999 is considered missing.
# MAGIC 
# MAGIC 
# MAGIC ### Stations Data Set
# MAGIC 
# MAGIC The final table, stations_data, houses valuable information on weather station location including fields such as: 
# MAGIC - lat: latitude.
# MAGIC - lon: longitude. 
# MAGIC - station_id: identifier for each station.
# MAGIC - distance_to_neighbor: numeric variable, distance to neighboring station in meters.
# MAGIC 
# MAGIC 
# MAGIC This dataset is considerably smaller in both row count and column count than our other datasets, as it contains information about static facilities. The information identifies the station and its location, while also giving us a station's distance to its neighbor.
# MAGIC Overall, compared with the other sources of data in the dataset, the stations table is relatively simple.

# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC 
# MAGIC ### 3.1.1. EDA on Raw Datasets
# MAGIC 
# MAGIC Initial EDA was conducted on the Airlines, Station, and Weather datasets individually in order to establish a baseline understanding of the data being worked in. In particular, we were interested to know what features would be of use when joining the datasets, and what amount of data was missing or erroneous. This section will discuss our findings when exploring the unjoined datasets, addressing both the overall data hygiene and some remarkable findings. This will be a high level summary of the most important findings; for more information and details, please consult the notebooks linked at the end of this section. The dataset joining task will be discussed in the section 3.2, "Joining the Data Sets".
# MAGIC 
# MAGIC ### 3.1.1.1 Selected Features
# MAGIC 
# MAGIC Upon initial inspection, it became obvious that each of the datasets included many features that were not relevant or useful for our analysis. As such, we discussed and deliberated over the meaning and worth of each feature in each dataset in order to determine which features should be selected to be used in our analysis, and which would be dropped. Below is a list of features from the airlines data set that we are including in our final datasets.
# MAGIC 
# MAGIC ### Airlines Dataset Analysis
# MAGIC 
# MAGIC 
# MAGIC ##### Count of Nulls in fields of interest in airline data
# MAGIC 
# MAGIC The two figures below display summary statistics for our numeric variables, as well as null value counts for our chosen variables.
# MAGIC 
# MAGIC ![Table 3.1.A: EDA Summary of Raw Airline Features(https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/Phase1/images/airlinestats.PNG)
# MAGIC 
# MAGIC *Table 3.1.A: EDA Summary of Raw Airline Features*
# MAGIC 
# MAGIC ![Table 3.1.B: EDA Missing Values of Raw Airline Features](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/Phase1/images/airlinenull.PNG)
# MAGIC 
# MAGIC *Figure 3.1.B: EDA Missing Values of Raw Airline Features*
# MAGIC 
# MAGIC The null values shown in Table 3.1.B may indicate flights without delays. These would be imputed or removed in a later step.
# MAGIC 
# MAGIC In addition to null analysis, preliminary study of the dataset shows that the dataset contains several duplicates. As such, we removed duplicate as one of the first data operations.
# MAGIC 
# MAGIC <center><img src="https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-12-03_01-37-29.png" width="1000"/></center>
# MAGIC 
# MAGIC *Figure 3.1.1.A: Size of Data Sets*
# MAGIC 
# MAGIC Further, from studying indicator attributes of the dataset, we were able to identify the unique record identifier of this dataset, which is a combination of DEST_AIRPORT_ID, FL_DATE, OP_CARRIER_FL_NUM, OP_UNIQUE_CARRIER, DEP_TIME. A 'flight_id' column was then created to guide analysis of flight related data and prediction. 
# MAGIC 
# MAGIC <center><img src="https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-12-03_02-14-51.png" width="400"/></center>
# MAGIC 
# MAGIC *Figure 3.1.1.B: Distinct Counts for ID fields from the Airlines Data*
# MAGIC 
# MAGIC ### Weather Dataset Analysis
# MAGIC 
# MAGIC 
# MAGIC The two figures below display summary statistics for our numeric variables, as well as null value counts for our chosen variables.
# MAGIC 
# MAGIC ![Table 3.1.C: EDA Summary of Raw Weather Features](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/Phase1/images/weatherstats.PNG)
# MAGIC 
# MAGIC *Table 3.1.C: EDA Summary of Raw Weather Features*
# MAGIC 
# MAGIC ![Table 3.1.D: EDA Missing Values of Raw Weather Features](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/Phase1/images/weather%20null.PNG)
# MAGIC 
# MAGIC *Table 3.1.D: EDA Missing Values of Raw Weather Features*
# MAGIC 
# MAGIC The statistics table shows the max being 9s, which is likely representative of missing data.
# MAGIC 
# MAGIC Figure 3.1.D indicates a large portion of data is missing from our dataset. These values may negatively affect any attempted analyses and will likely need to be filtered out.
# MAGIC 
# MAGIC ### Stations Dataset Analysis
# MAGIC 
# MAGIC 
# MAGIC ##### Statististics on distance_to_neighbor field of station dataset
# MAGIC 
# MAGIC ![Table 3.1.1.D: EDA Summary of Raw Station Feature](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/images/stationstats.PNG?raw=true)
# MAGIC 
# MAGIC *Table 3.1.1.D: EDA Summary of Raw Station Feature*
# MAGIC 
# MAGIC ## 3.1.1.2 Remarkable Findings
# MAGIC 
# MAGIC ### Airlines Dataset
# MAGIC 
# MAGIC The Airline On-Time Performance Data table contains the scheduled and actual departure/arrival times for U.S. Domestic flights for qualifying airline carriers. These carriers must account for at least one percentage of U.S Domestic scheduled passenger revenues in order to qualify. Our data ranges from 2015 to 2021
# MAGIC 
# MAGIC - More than 50% of the data in the Airlines dataset contain duplicate records (74,177,433 to 42,430,592 after performing de-duplication, meaning there were 31,746,841 duplicate entries). Duplicate data removal is performed before conducting further analysis.
# MAGIC - The Airlines work with 20 unique carriers, 8465 tail numbers (aircraft ID), 7300 flight numbers, 388 origin airport id, and 386 dest airport id.
# MAGIC - A flight from the airlines dataset is uniquely identified by the natural composite key: ORIGIN_AIPORT_ID, DEST_AIRPORT_ID, FL_DATE, OP_CARRIER_FL_NUM, OP_UNIQUE_CARRIER, DEP_TIME.
# MAGIC - For features that were indicative of flight departures ("DEP_TIME" and "DEP_DEL15"), having null values in those features were 1:1 associated with the flights being cancelled ("CANCELLED" == 1). It was confirmed that for every null value corresponding to a departure-related feature, the flight in question was indeed cancelled.
# MAGIC - In general, none of the Airlines features showed any strong correlation with the response variable DEP_DELAY15. However, the feature with the strongest correlation was "CRS_DEP_TIME" with an R value of 0.1565. While this is not an indicator of a strong correlation, the R value is significantly higher than any of the other features.
# MAGIC - Similar to the previous point, the feature "CANCELLATION_CODE" is null for the majority of cases where a flight is not cancelled. Therefore, it has an expected value of 97.940% null values.
# MAGIC - Airport_id columns (ORIGIN_AIRPORT_ID and DEST_AIRPORT_ID) uniquely match to their airport code (ORIGIN and DEST). This contradicts the flight dictionary documentation. Furthermore airport_seq_ids (ORIGIN_AIRPORT_SEQ_ID and DEST_AIRPORT_SEQ_ID) uniquely identifie an airport (1 to 1 relationship with the aiport_ids) whereas a single aiport_id can have multiple airport_seq_id. As such the airport_seq_IDs are more accurate tracker of airports. As time permit, we will further improve our model performance by improving our join algorithm to consider the movement of airport locations that is tracked by the airport_seq_id as opposed to solely relying on the airport_ids.
# MAGIC - Less than 0.6% of flights (236,132) departed within 2 hours from their assigned aircraft's previous flight departure.
# MAGIC 
# MAGIC ##### Airlines Flight Delay Percentages
# MAGIC 
# MAGIC <center><img src="https://github.com/ColStaR/sparksandstripesforever/blob/main/images/2022-12-05_21-57-09.png?raw=true" width="800"/></center>
# MAGIC 
# MAGIC *Figure 3.1.1.C: Total Flight Volume (Bottom), Percent Delays (Top) by OP_UNIQUE_CARRIER Bar Charts*
# MAGIC 
# MAGIC - B6, more commonly known in the real world as JetBlue was the airline with the most delays in relation to its total number of delays.
# MAGIC   - Although, Jet Blue has relatively less flights than other comparable airlines within this dataset.
# MAGIC - WN (Southwest) and AA (American Airlines) are two other airlines we see have a higher percentage of delayed flights, although they do not have the small number of total flights exhibited by Jetblue. This leads us to believe that Southwest and American Airlines appear to be the worst overall airlines.
# MAGIC - HA (Hawaiian Airlines) and QX (Horizon Air) display the least percentage of flights delayed but both have a relatively small number of flights within this dataset.
# MAGIC - DL (Delta Airlines) shows to have a considerable amount of flights within the dataset while also having a lower percentage of flights delayed than other airlines with similar quantities of total flights.
# MAGIC 
# MAGIC 
# MAGIC ### Weather Dataset
# MAGIC 
# MAGIC The Quality Controlled Local Climatological Data contains summary data from weather stations housed near airports. These stations log daily temperature highs, temperature lows, precipitation, wind speed, visibility, and storm characteristics. The available data ranges from 2015 to 2021.
# MAGIC 
# MAGIC - Given documentation did not always match our given columns. We found numerous instances of the data dictionary appearing incorrect, and data to be directly contratictory to what the documentation states.
# MAGIC - Several fields within the weather dataset had large amounts of missing data.
# MAGIC - These fields would be kept in our analyses with the goal of seeing reduced portions of missing data after the join across datasets.
# MAGIC   - This would be resolved post-join where most of the remaining data had its percentage of nulls reduced significantly. 
# MAGIC - Monthly data is included at this level, but accounts for only a minute proportion of the data.
# MAGIC   - Not every row contains monthly data despite most having hourly data of some sort. This brings to question the usability of these columns.
# MAGIC - Date fields will likely need to be adjusted for uniformity across all datasets.
# MAGIC - Dates/Times are in their own respective timezones.
# MAGIC - HourlyPressureChange does not seem to change frequently nor drastically, is more or less a static field.
# MAGIC - Different airlines have different volumes of flight, with certain airlines that have low delay percentage and low flight volume (e.g. HA), low delay percentage and high flight volume (e.g. DL), and ones that have high delays. See the figure below for details. This observation suggests that including an airline effectiveness feature could improve the model performance. 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Links to complete pre-join EDA Notebooks:
# MAGIC 
# MAGIC To view the full EDA code and the analyses that were conducted on the pre-joined data set, please view the notebooks linked in the appendix, section 13.1, "Links to Pre-Joined EDA Notebooks".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3.2 Joining the Datasets
# MAGIC 
# MAGIC Before we began feature engineering and machine learning, we first had to join weather and airlines datasets and address missing values. Since our goal was to predict the departure delay, we had to drop any rows that were missing values in all four of `DEP_DELAY`, `DEP_DELAY_NEW`, `DEP_DEL15`, and `CANCELLED` (if any one of these columns was available, we had sufficient information for our outcome variable). Note that in our case, we treated cancellations as delays. Similarly, if we have a small proportion of rows that are missing any important features, these rows can safely be dropped. Doing so can help improve runtime once we begin joining the datasets.
# MAGIC 
# MAGIC <center><img src="https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/data_join_gantt.png" style="float: center;"/></center>
# MAGIC 
# MAGIC *Figure 3.2.A: Wire Diagram for Joined Data Set*
# MAGIC 
# MAGIC The steps below detail how the join of the airlines and weather datasets was conducted:
# MAGIC 
# MAGIC 1. First, we collected distinct pairs of airport codes from the `ORIGIN` and `ORIGIN_STATE_ABR` columns in the airlines dataset. Having confirmed that all airport codes and their corresponding states were in the weather stations dataset using the `neighbor_call` and `neighbor_state` columns, we performed an inner join on these two fields such that we had a table containing: `ORIGIN` (`neighbor_call`), `ORIGIN_STATE_ABR` (`neighbor_state`), `station_id`, `distance_to_neighbor`.
# MAGIC 
# MAGIC 2. Because each airport mapped to more than two hundred weather stations on average, for the sake of efficiency, we chose to narrow down to a set of two or three stations for each airport: we selected the airport weather stations (with a `distance_to_neighbor` equal to zero), the closest weather station from the airport with a distance greater than zero, and the furthest weather station from the airport. We then performed an inner join of the airport/station linking table onto the weather dataset. The resulting dataset only contained weather data for the stations of interest, and some duplicate entries of weather where the same weather station corresponded to more than one airport. We also found that some airports corresponded to more than one weather station.
# MAGIC 
# MAGIC 3. In the airlines dataset we created a departure date-time column, called `DEP_DATETIME`, by combining the `FL_DATE`, and departure time (`CRS_DEP_TIME`) into a datetime. In order to account for and avoid data leakage, we added a column called `DEP_DATETIME_LAG` to the airlines dataset that was equal to the departure datetime minus 2 hours. We needed to round the time down to the hour to help make merging onto the weather data easier, which we accomplished by removing the last two digits from the departure time before converting to a datetime.
# MAGIC 
# MAGIC 4. Similarly, in the weather dataset, we created a new column called `DEP_DATETIME` (to align with the airport dataset) by converting the `DATE` column in the weather data to a datetime type. Then, we created an additional column called `DEP_DATETIME_LAG`, which rounded the `DEP_DATETIME` up to the next hour for the sake of the merge. We then identified rows in the merged weather data with duplicated `DEP_DATETIME` and airport (`ORIGIN`). We aggregated these duplicated entries by averaging across the numeric features in the weather data and concatenating the categorical features. This strategy allowed us to keep more granular time in our weather data by making use of multiple stations.
# MAGIC 
# MAGIC 5. Finally, we were able to join the airlines data and the weather/station data together as an inner merge using the columns `ORIGIN` and `DEP_DATETIME_LAG`. 
# MAGIC 
# MAGIC The resulting joined dataset had 40,933,735 rows and 56 columns for the years 2015-2021 and ran in 11 minutes on our cluster with 4 cores.
# MAGIC 
# MAGIC ### 3.2.1 Joined Dataset Schema
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
# MAGIC - AggHourlyPresentWeatherType : string
# MAGIC - time_zone_id : string
# MAGIC - UTC_DEP_DATETIME_LAG : timestamp
# MAGIC - UTC_DEP_DATETIME : timestamp
# MAGIC - DEP_TIME_CLEANED : string
# MAGIC - flight_id : string
# MAGIC 
# MAGIC ### Links to data joining notebooks
# MAGIC 
# MAGIC To view the full code base for joining the data set, please review the notebooks in the appendix, section 13.2, "Links to Joining Data Sets Notebooks".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3.2.1 EDA on Joined Dataset
# MAGIC 
# MAGIC EDA of the joined dataset is performed on the following:
# MAGIC 
# MAGIC - The Joined dataset contains close to 41 million rows
# MAGIC    - 35,249,720 non-delayed flights
# MAGIC    - 6,461,666 delated flights
# MAGIC - This EDA will ignore the monthly weather data in our current join as they are intended for use in a later iteration of our model.
# MAGIC - EDA performed on 2021 data is used for understanding of the dataset only as an inspiration for future proof considerations and machine learning approaches. It is not used for operations involving the training dataset to avoid data leakage
# MAGIC 
# MAGIC Prior to conducting any analysis, we closely examined the treatment of data from 2015:
# MAGIC 1. To prevent data leakage, we would like to build our engineered features based on the values obtained from the year before. Since 2015 is the beginning of the dataset with no prior year to look up to, it will be treated either as a special case or be dropped from the dataset
# MAGIC 2. Given the Covid pandemic in 2020 and changes in airline management responsiveness post Covid, 2015 is too distant to represent future years
# MAGIC As such, we decided to drop 2015 from our analysis.
# MAGIC 
# MAGIC Before brainstorming what features to engineer, we observed how flight delay percentage changed over the year (total flights delivered divided by total flights scheduled within a year) to get a sense of how consistent the flight delay patterns are year over year. As shown in the figures below, both the flight volume and flight percentage vary across years. It is also interesting to note that 2020 has the lowest flight volume (likely due to travel restrictions during COVID) but rebounded quickly in 2021.
# MAGIC 
# MAGIC 
# MAGIC ![Figure 3.2.1.A: Flight Volume by Year Graph](https://github.com/ColStaR/sparksandstripesforever/blob/main/images/flight_volume_by_year_graph.png?raw=true)
# MAGIC 
# MAGIC *Figure 3.2.1.A: Flight Volume by Year Graph*
# MAGIC 
# MAGIC ![Figure 3.2.1.B: Flight Percent Delay by Year Graph](https://github.com/ColStaR/sparksandstripesforever/blob/main/images/Flight_percent_delay_by_year_graph.png?raw=true)
# MAGIC 
# MAGIC *Figure 3.2.1.B: Flight Percent Delay by Year Graph*
# MAGIC 
# MAGIC Before selecting features for our machine learning model, we want to get a sense of how features relate to the target variable (`DEP_DELAY15`). To achieve this, we created a correlation matrix for all non-string and non-id fields calculated from both Pearson correlation to measure the linearity of columns against the target variable, and the Spearman correlation to measure the strength and direction of association (monotonic association). Furthermore, since we learned from the year over year percentage delay analysis that a major disruption took place in 2020, we anticipate that the strength of relationship with the target variable to change across years as well. As such, we ran a correlation analysis cross multiple years: across all of the dataset (2016-2021), years before Covid (2016-2019), Covid breakout (2020), Covid recovery (2021).
# MAGIC 
# MAGIC 
# MAGIC ![Figure 3.2.1.C: Pearson Correlation of Features](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-12-03_01-56-45.png)
# MAGIC 
# MAGIC *Figure 3.2.1.C: Pearson Correlation of Features*
# MAGIC 
# MAGIC ![Figure 3.2.1.D: Spearman Correlation of Features](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-12-03_03-00-24.png)
# MAGIC 
# MAGIC *Figure 3.2.1.D: Spearman Correlation of Features*
# MAGIC 
# MAGIC Although some interesting relationships exist with the raw data, no feature stood out as being able drive a strong machine learning performance (lack of strong correlation). As such, we focused our efforts on engineering features that should have a stronger relationship with flight delays defined through business intuitions (such as tracking aircraft delay status across flights), airline management effectiveness, airport importance, and special days effect.  It is interesting to note that among all the features engineered, only the flight tracker feature (`is_prev_delay`) showed a medium strength correlation with the output variable followed by airline effectiveness (`perc_delay`) with a weak correlation. Other features do not seem to be significantly associated with the output variable. In addition, the engineered features have a higher Pearson correlation than they do with Spearman. This suggests that while our engineered features can improve our logistic regression model results (assumes linearity), they may not be able to noticeably lift tree-based models results given the non-monotonic nature of the features.
# MAGIC 
# MAGIC  
# MAGIC ## Notable Feature Characteristics
# MAGIC - About 80% of our data for `DEP_DEL15` indicate as non-delayed flights
# MAGIC - Months with the most data in this dataset are December/January, likely due to the holiday season. This may impact our future cross validation decisions.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ![Figure 3.2.1.E: Volume of Flights per Month Bar Graph](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/monthvis2.PNG?raw=true)
# MAGIC 
# MAGIC *Figure 3.2.1.E: Volume of Flights per Month Bar Graph*
# MAGIC 
# MAGIC - Canceled flights are considered Delayed under `DEP_DEL15` for this analysis as previously discussed
# MAGIC - Both `HourlyWetBulbTemperature` and `HourlyDryBulbTemperature` were normally distributed, with `HourlyWetBulbTemperature` appearing to have a slightly more narrow distribution.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ![Figure 3.2.1.F: HourlyDryBulbTemperature Distribution Bar Graph](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/drytempvis.PNG?raw=true)
# MAGIC *Figure 3.2.1.F: HourlyDryBulbTemperature Distribution Bar Graph*
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ![Figure 3.2.1.G: HourlyWetBulbTemperature Distribution Bar Graph](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/wettempvis.PNG?raw=true)
# MAGIC *Figure 3.2.1.G: HourlyWetBulbTemperature Distribution Bar Graph*
# MAGIC 
# MAGIC - `HourlyPressureChange` does not seem to change significantly across the dataset, while also was missing from 24% of it, and will likely be dropped from future analyses.
# MAGIC   - `HourlyStationPressure` and `HourlySeaLevelPressure` showed a similarly tight distribution but were both missing from less than 1% of the dataset, and thus were deemed worthy to keep in our current model
# MAGIC - `HourlyWindGustSpeed` was missing from 64% of the dataset and will likely be dropped from future analyses
# MAGIC - `HourlyWindSpeed` displays outlier datapoints with records indicating a windspeed greater than 1000.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ![Figure 3.2.1.H: Hourly Wind Speed Summary Statistics](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/images/windspeedprof.PNG?raw=true)
# MAGIC *Figure 3.2.1.H: Hourly Wind Speed Summary Statistics*
# MAGIC 
# MAGIC - Categorical Variables in our dataset seemed to be relatively free of missing/error data 
# MAGIC - `HourlyPrecipitation` was missing from about 4% of the joined dataset, and had 90% of its fields labeled with 0, indicating 90% of fields showing no precipitation.
# MAGIC 
# MAGIC ## Link to complete post-join EDA Notebook:
# MAGIC 
# MAGIC To review the full code base and analysis for the post-join EDA, please review the notebooks listed in the appendix, section 13.3, "Links to Complete Post-Join EDA Notebooks".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3.3 Feature Engineering
# MAGIC 
# MAGIC In addition to the pipeline changes, a number of new features were created and introduced to our joined dataset. These new features are described below.
# MAGIC 
# MAGIC ### 3.3.1. Flight Previously Delayed Indicator
# MAGIC 
# MAGIC The `is_prev_delayed` field is an indicator field that is created to answer the question: Was the flight in question previously delayed within a recent window of time? We created this feature assuming that if a flight's prior aircraft was delayed then it is very likely that the current flight will be delayed as well.
# MAGIC Special considerations in constructing this field include:
# MAGIC 1. This record contains nulls, which are resulting from the first flight of the day. By assuming that the first flight of a day is always not delayed, we replaced the nulls with 0.
# MAGIC 2. To avoid data leakage, we looked at flights that were at least 2 hours apart when creating this field. For example, if 2 flights departed within 2 hours, we would get the flight status from the next flight prior as an indicator for if the previous flight was delayed (as opposed to using the prior flight).
# MAGIC 
# MAGIC This feature has the highest predictive power by Spearman Correlation analysis (see section 3.2.1).
# MAGIC 
# MAGIC ### 3.3.2. Flight Previously Diverted Indicator
# MAGIC Similar to the previous flight delay indicator, we created the `is_prev_diverted` field to answer whether a flight in question was previously diverted? We assume that if a flight's prior aircraft was diverted then it is very likely that the current flight will be delayed as well. The methodology used to implement this field is the same for implementing the previously delayed fligth tracker field. 
# MAGIC It is interesting to know that despite the methodology for creating this field is the same as `is_prev_delayed` (feature with the highest association with our target variable by Spearman analysis), the previously diverted indicator has a weak correlation with the target variable. In fact, when we added this feature to our best performing model, it actually caused our performance to drop slightly. As such we excluded this feature from our machine learning pipeline.
# MAGIC 
# MAGIC ### 3.3.3. Airline Efficacy Score
# MAGIC 
# MAGIC We created 2 fields to capture the effectiveness of airlines in managing delayed flights by looking at their historical delayed flight ratio (see section 3.1.1 for a detailed EDA discussion):
# MAGIC 
# MAGIC 1. `perc_delay`: percentage of delayed flights of a carrier over the total number of flights scheduled for that carrier
# MAGIC 2. `airline_type`: categorical field with 3 values: 
# MAGIC   - BelowAverage: bottom 25 percentile of airlines by perc_delay
# MAGIC   - Average: airlines that fall within the 25-75th percentile by perc_delay
# MAGIC   - AboveAverage: top 25 percentile of airlines by perc_delay
# MAGIC For new airlines, we would impute their perc_delay as the average of the percentage delay across all airlines from the prior year.
# MAGIC To avoid data leakage, we used the airline efficiency from prior year to gauge their performance for the current year. For the beginning of the dataset (2015) where we didn't have the prior year data, we used 2015's airline values.
# MAGIC By Spearman analysis, this feature has one of the higher association with our target variable (see section 3.2.1). That said, it is interesting to observe that the level of association it has with the target variable dropped in 2020 (year with lowest flight volume and affected by Covid) and 2021 (year after recovery from Covid). This observation matches our hypothesis where the effect of airline efficiency is less during a year with lower flight volume and after recovery from a major industry disruptive incident (airlines gaining experience in managing delays from weathering through disruptions).
# MAGIC 
# MAGIC ![Figure 3.3.3.A: Spearman Correlation on Features by Year](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-12-03_02-10-32.png)
# MAGIC 
# MAGIC *Figure 3.3.3.A: Spearman Correlation on Features by Year*
# MAGIC 
# MAGIC ### 3.3.4. Holiday and Special Event Day Tracker
# MAGIC 
# MAGIC We assume that certain days of the year are more busy or less busy than normal that would impact the volume of air travel during a particular day. Depending on the air travel volume, the flight delay pattern may also be impacted. Furthermore, we recognize that multiple states had lock downs during the onset of Covid (Massachusetts shelter in place dates are used as a high level guideline. The effect of the shelter in place dates are classified as `NotBusy`), which would operate in a way that is similar to days which are less busier than normal. With these assumptions in mind, we created a lookup table to track special days (including holidays and shelter in place from Massachusetts due to Covid) and their assumed effect. 
# MAGIC 
# MAGIC 3 columns were created in the lookup table: 1 attribute capturing the calendar day of the special date (which is used as the join key to combine this lookup table back the main dataframe on `FL_DATE`), and 2 attributes with a mapping of their date type (`SpecialDateType`) and assumed effect (`AssumedEffect_Text`). The mapping of the date type against their assumed effect is shown below:
# MAGIC 
# MAGIC | SpecialDateType      | AssumedEffect_Text |
# MAGIC |----------------------|--------------------|
# MAGIC | NewYear              | NotBusy            |
# MAGIC | Thanksgiving_m1      | Busy               |
# MAGIC | Thanksgiving         | NotBusy            |
# MAGIC | SunAfterThanksgiving | Busy               |
# MAGIC | Christmas_m3         | Christmas_m3       |
# MAGIC | Christmas_m2         | Christmas_m2       |
# MAGIC | Christmas_m1         | Christmas_m1       |
# MAGIC | Christmas            | NotBusy            |
# MAGIC | Christmas_p1         | Christmas_p1       |
# MAGIC | Christmas_p2         | Christmas_p2       |
# MAGIC | Christmas_p3         | Christmas_p3       |
# MAGIC | MA_ShelterInPlace    | NotBusy            |
# MAGIC 
# MAGIC *Table 3.3.4.A Mapping of special days and their assumed effect against airline travel volume*
# MAGIC 
# MAGIC For the `SpecialDateType`, a _m# signifies # days before a special day whereas _p# signifies # days after a special day.
# MAGIC 
# MAGIC After the effect of the holidays are joined to the dataset, the distribution of the `AssumedEffect_Text` column is visualized in the figure below.
# MAGIC 
# MAGIC ![Figure 3.3.4.A Holiday and Special Effect Count by Year Bar Chart](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-12-05_00-32-15.png)
# MAGIC 
# MAGIC *Figure 3.3.4.A Holiday and Special Effect Count by Year Bar Chart*
# MAGIC 
# MAGIC ### 3.3.5. Outgoing Flight Frequency with PageRank
# MAGIC 
# MAGIC After some contemplation, one factor that we sought to capture in our data was how busy an airport would be when it came to handling outbound air traffic; an airport with high levels of air traffic would likely lead to flights being delayed. To account for the factor of frequency and volume of outgoing flights for an airport, we created a column `pagerank` that uses the `DEST` airport as nodes and `ORIGIN` to `DEST` connecting flights as edges. The PageRank values were computed using the GraphFrames library in PySpark.
# MAGIC 
# MAGIC To avoid data leakage, we used the pagerank value from the prior year to reflect outgoing flight frequency of an airport. For the beginning of the dataset (2015), we do not have prior year data available. As such we decided to drop 2015 from further analysis for now given that we could not use 2014 data for the flights in 2015; it was decided that having this new feature would be more beneficial than having the 2015 data, which did not seem to be very influential when training our models. The dataset continues as normal from 2016 on, with each flight using the pagerank of the destination from the previous year. 
# MAGIC 
# MAGIC 
# MAGIC #### PageRank Scores by Destination Airport ID
# MAGIC 
# MAGIC Below is a visualization of the PageRank scores per airport per year. Please note the following:
# MAGIC 
# MAGIC - Pagerank tells us the performance of each airport as a destination within the dataset. Higher PageRanks indicate airports that receive a larger volume of outbound traffic.
# MAGIC - The truncated visual above displays the pagerank performance of destination airport ID's within the dataset, for visibility's sake not all data is shown.
# MAGIC - Destinations such as ATL, DEN, DFW, and ORD appear to have the highest pagerank scores over the years and thus are expected to be the busiest destination airports.
# MAGIC - Destinations with higher pagerank scores are expected to be more prone to delayed flights. 
# MAGIC 
# MAGIC ![Figure 3.3.5.A PageRank Values by Airport and Year](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/images/Pagerank.png?raw=true)
# MAGIC 
# MAGIC *Figure 3.3.5.A PageRank Values by Airport and Year*
# MAGIC 
# MAGIC ### 3.3.6. Extreme Weather Indicators
# MAGIC 
# MAGIC In efforts to further improve the predictive ability of our models using weather data, we decided on an approach that would take the occurence of extreme weather in the `HourlyPresentWeatherType` field, and create multiple features indicating the presence of each of the types of selected values for extreme weather. For these purposes, we determined the occurence of "Freezing Rain", "Blowing Snow", "Snow", "Rain", and "Thunder" as extreme weather that may contribute to the likelihood of a delayed flight. 
# MAGIC 
# MAGIC Their occurence was determined by the presence of the following extreme weather indicators in the `HourlyPresentWeatherType` column
# MAGIC - "FZRN" - Freezing Rain
# MAGIC - "BLSN" - Blowing Snow
# MAGIC - "SN" - Snow
# MAGIC - "RA" - Rain
# MAGIC - "TS" - Thunder
# MAGIC 
# MAGIC Each of the five above variables will have a column dedicated to their occurence in the `HourlyPresentWeatherType` column. Each column will return "1" to indicate the presence of its assigned value and a "0" for a lack thereof 
# MAGIC 
# MAGIC #### Extreme Weather Analysis
# MAGIC 
# MAGIC | Weather Type  | Count   | % of Total Delayed Rows |
# MAGIC |---------------|---------|-------------------------|
# MAGIC | Blowing_Snow  | 55930   | 0.2%                     |
# MAGIC | Freezing_Rain | 101508  | 0.3%                     |
# MAGIC | Rain          | 6590540 | 19.0%                   |
# MAGIC | Snow          | 2475951 | 7.0%                    |
# MAGIC | Thunder       | 614697  | 1.7%                    |
# MAGIC 
# MAGIC *Table 3.3.6.A Count and Percentage of Delayed Rows by Weather Type of Total Data Set*
# MAGIC 
# MAGIC The above figure shows the occurence of the specified values within the total 35,249,720 flights in our dataset
# MAGIC 
# MAGIC | Weather Type  | Count   | % of Total Delayed Rows |
# MAGIC |---------------|---------|-------------------------|
# MAGIC | Blowing_Snow  | 12337   | 0.2%                     |
# MAGIC | Freezing_Rain | 25869   | 0.4%                     |
# MAGIC | Rain          | 1439607 | 22.3%                   |
# MAGIC | Snow          | 514283  | 8.0%                    |
# MAGIC | Thunder       | 286287  | 4.4%                    |
# MAGIC 
# MAGIC *Table 3.3.6.B Count and Percentage of Delayed Rows by Weather Type of Delayed Flights*
# MAGIC 
# MAGIC The above figure shows the occurence of the specified values within the total 6,461,666 delayed flights in our dataset
# MAGIC 
# MAGIC While the presence of some of these values within the dataset may appear low, our hope is that they may be strongly related to delayed flights when they do occur. Additionally, we decided to track `blowing_snow` and `freezing_rain` in conjunction with `snow` and `rain` in efforts to see if the presence of the more extreme of the two would be strongly related with delays if the less extreme value was not
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 3.3.7. Departure Hour
# MAGIC 
# MAGIC Flights that depart around peak rush-hour tend to be more likely to be delayed. These sought-after departure times can lead to runway traffic, increasing odds of a delay. Furthermore, because flight typically only begin departure after 5:00am, there are few opportunities for planes to fall behind schedule in the early mornings, while flights departing towards the end of the day statistically have more opportunities to fall behind schedule. Thus, we hypothesize that the time of day may have a large impact on the likelihood of a flight being delayed.
# MAGIC 
# MAGIC To this end, we created an additional feature, called `DEP_HOUR`, which is the hour of the day when the flight is scheduled to depart. The feature is derived from the scheduled flight departure time (`CRS_DEP_TIME`) as opposed to the true departure time, and therefore avoids any concerns of data leakage. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 4. Data Pipeline and Data Considerations
# MAGIC 
# MAGIC Machine learning projects such as this require a number of intensive tasks processing the data before it can be used in analysis and model training. For this reason, the construction of an end-to-end data pipeline was crucial in the success of the project. This section details information about this data pipeline, including a general overview of its steps along with more detailed information about more specific and specialized tasks being conducted.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 4.1 Data Pipeline Overview
# MAGIC 
# MAGIC The raw data in question comes from multiple datasets on third party servers while being multiple gigabytes in size. In order for the data to be made usable, we will establish a data pipeline that conducts the necessary tasks to ingest, clean, transform, join, and make available the data we are interested in. To create this data pipeline, we will be leveraging the ML Pipelines API library, since that library is designed for machine learning pipelines and integrates natively with Apache Spark on our DataBricks instance.
# MAGIC 
# MAGIC The general process for data ingestion and processing through the pipeline is done through following steps:
# MAGIC 1. Import data into HDFS
# MAGIC     - The data in question is currently only available on a set of third party servers. We first copied the raw data files from their current location into our Microsoft Azure blob storage, uncompressing them so that they are fully accessible. 
# MAGIC 2. Convert CSV into Parquet
# MAGIC     - Once stored within our cloud storage, we converted the raw files from their native CSV format into the Parquet file format, as Parquet provides superior storage and computation performance compared to CSV. A copy of this converted raw data in Parque format will then be stored back onto our HDFS instance, which will then be transferred to our DataBricks instance for additional transformations.
# MAGIC 3. Join datasets
# MAGIC     - Only a subset of the data is needed, so we first join the datasets together in order to remove unnecessary data from our pipeline and analysis. 
# MAGIC     - Further information about the dataset joining process can be found in section 3.2, "Joining the Datasets".
# MAGIC 4. Clean joined data
# MAGIC     - As with all raw data, we are expecting the initial data to contain numerous errors, typos, and missing data. To resolve this, the data will be cleaned of erroneous or troublesome values, and properly formatted so that it is fully usable. 
# MAGIC     - Further detail is given below in section 4.3, "Process for Data Cleaning".
# MAGIC 5. Feature Engineering
# MAGIC     - Additional features were created and added to the dataset in order to provide greater predictive information on the data.
# MAGIC     - More information about the feature engineering can be found in section 3.3, "Feature Engineering".
# MAGIC 6. Feature Selection
# MAGIC     - 2015 is dropped from the joined dataset as it provides limited prediction value (patterns are less relevant to recent data). See section 3.2.1, "EDA on Joined Dataset" for detailed discussion. 
# MAGIC     - After all of the original features and newly created features are joined on the same dataset, we then select which features will be utilized in the machine learning model training. EDA and experimentation is done to find which features have low levels of importance in training, and which features introduce a high level of dimensionality to the models. Features that have either low levels of importance or high levels of dimensionality compared to their level of influence are dropped from the selected features in order to ensure that our models have optimal run-times without sacrificing model performance.
# MAGIC 7. Rebalance Data
# MAGIC     - Our EDA discovered that there is a substantial imbalance between the number of rows for delayed flights and non-delayed flights, with non-delayed flights accounting for about 80% of the data while delayed flights only accounted for about 20%. Because such imbalanced data will hinder our model's ability to train on the data, we opted to rebalance the data by down-sampling the data. 
# MAGIC     - More information about this process is given below in section 4.4, "Process for Downsampling Data".
# MAGIC 8. Transform Data with Pipeline
# MAGIC     - With the dataset primed with the selected features, the dataset can now be transformed into a format that can be readily used by machine learning models. MLLib provides a useful class called `Pipeline` that allows for data transformations to be easily staged and applied to our datasets. This `Pipeline` class provides the flexibility to easily change the characteristics of each transformation, including one-hot encoding, value imputation, and standard scaling. The final step of the pipeline is to apply a vector assembler to the transformed data, converting the dataset's features into a vector of values that the machine learning models can easily and space-efficiently train on.
# MAGIC     - All categorical features are transformed using string indexing. One-hot encoding is also applied to categorical variables for models where that would be appropriate.
# MAGIC     - Numeric values are transformed using a value imputer for models where that would be appropriate. Since our data is fully cleaned, value imputing was never needed for the current data and models.
# MAGIC     - A feature scaler is available to scale the feature vector for models where that would be appropriate.
# MAGIC 7. Model Training and Validation with Blocking Time Series Cross Validation
# MAGIC     - Because our data is time series data, special attention had to be taken for handling that time series aspect. As such, we implemented a custom cross validation method specifically designed for time series data called Blocking Time Series Cross Validation. This method allows the Dataset to be trained in a manner that is sensitive to time series data. Models are trained and validated per fold of this cross validation, training on the first 70% of each fold while the latter 30% is held out for training validation.
# MAGIC     - For this experiment, the models were trained on data from the years 2016 to 2020.
# MAGIC     - More information about Blocking Time Series Cross Validation is available in section 4.5, "Blocking Time Series Cross Validation".
# MAGIC     - More information about the machine learning models being used can be found in section 5.2, "Machine Learning Models Used".
# MAGIC 8. Model Evaluation on Held-out Dataset
# MAGIC     - The model with the best training validation metrics from the previous step is then evaluated on the held-out test dataset. The model is used to predict values from the held-out test dataset, and the model's performance is evaluated in order to determine its test evaluation metrics.
# MAGIC     - For this experiment, the models were tested on the held-out data from the year 2021.
# MAGIC 9. Model Hyperparameter Tuning
# MAGIC     - After each session of Blocking Time Series Cross Validation for a single model, the model is then re-run with a different set of parameters. This continues in the form of a custom grid search for every possible combination of parameters. Model validation and test evaluation metrics are gathered and saved throughout each step of each model's cross validation. 
# MAGIC     - Hyperparameter tuning and the parameters used are discussed in section 5.3, "Model Parameters Used".
# MAGIC 10. Final Model Selection
# MAGIC     - After every model with every combination of parameters is run through its cross validation, training validation, and test evaluation, the validation and evaluation metrics from the best models are compiled and saved for the user's consideration and determination as to which model will be considered to be the best performing model of the experiment.
# MAGIC     
# MAGIC Below is an overview visualization of the pipeline’s components and processes.
# MAGIC 
# MAGIC ![Figure 4.1.A Data Pipeline Image](https://github.com/ColStaR/sparksandstripesforever/blob/main/images/Data_Pipeline_v4.png?raw=true)
# MAGIC 
# MAGIC *Figure 4.1.A Data Pipeline Image*
# MAGIC 
# MAGIC ### 4.1.2 Links to Data Preparation Pipeline Notebook
# MAGIC 
# MAGIC The data pipeline's notebooks are split into two halves: one half of data preparation that conducts EDA and converts raw data into a usable format, and the individual model notebooks that contain the data pipelines for the individual models.
# MAGIC 
# MAGIC To view the full code for the EDA and data preparation portion of the pipeline, please review the notebook linked in the appendix section 14.3.1, "Link to Data Preparation Pipeline Notebook".
# MAGIC 
# MAGIC To view the code notebooks for the machine learning models, please refer to the model details in section 6 to find the location for the model's notebook in the appendix.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 4.2 Tools Used in Data Pipeline
# MAGIC 
# MAGIC The data pipeline for this project leverages a number of tools and platforms that proved to be invaluable in the construction and operation of our data pipeline. Below is a list of the tools used, along with a description of how they were used.
# MAGIC 
# MAGIC - Microsoft Azure: Microsoft's cloud platform offers cloud storage functionality that was used to store important data related to the project. Azure's blob storage provided a useful environment to build a data lake, which was then used to store data such as the raw datasets, the final datasets, and the data frames of metrics that were created after each model training validation and test evaluation. Do note that all of the data, with the exception of the raw datasets, were typically saved as parquet files; parquet files provided the storage and computational efficient storage format necessary for our analytics purposes.
# MAGIC 
# MAGIC - DataBricks: DataBricks provided the online development environment that was used to create almost the entirity of the project. DataBricks's native integration with tools like Azure, GitHub, Apache Spark, and most of the major Python libraries made using the platform incredibly easy and flexible to incorporate into the project's workflows. Furthermore, the online nature of the platform made development very consistent, efficient, and accessible for our project team.
# MAGIC 
# MAGIC - Apache Spark: Spark is a distributed systems engine that enables distributed systems to work efficiently on computation-heavy tasks. Spark proved to be a critical part of this project, as the sheer size of the data and computation that were required made usage of a distributed system a necessity.
# MAGIC 
# MAGIC - PySpark: PySpark is an open-source Python library that provides an interface between Python and Apache Spark. While Apache Spark was used as the back-end engine to power the project's large-scale computational needs, PySpark is the front-end interface that was used to execute those computational tasks. DataBricks's native integration with both Apache Spark and PySpark made integration and usage of those tools seamless.
# MAGIC 
# MAGIC - MLLib Library: MLLib is Apache Spark's machine learning library, designed specifically to work with optimally Spark's distributed systems. Due to our reliance on Apache Spark, MLLib was an obvious choice for implementing the necessary machine learning models that would need to run on our distributed system. MLLib provided all of the machine learning models that this project implemented, though some custom functionality would need to be created.
# MAGIC 
# MAGIC - Pandas Library: this open-source Python library provided many important analytical and data management functions. The DataFrame function within Pandas provided the convenience and flexibility necessary for conveniently logging, compiling, and storing data from the training data validation and test data evaluation.
# MAGIC 
# MAGIC - GitHub: GitHub is a version control platform that allowed our project team to easily collaborate on the same code base. Its native integration with DataBricks made its usage seamless, and GitHub's functionality as a code repository and version control proved to be very helpful during development.
# MAGIC 
# MAGIC ### 4.2.1 Cluster Computing Information
# MAGIC 
# MAGIC Our primary computing cluster utilized Apache Spark, using a PySpark interface on the DataBricks platform. This cluster had the following specifications:
# MAGIC 
# MAGIC - Number of Workers: 1 - 10
# MAGIC - Worker Type: Standard_D4s_v3
# MAGIC - Memory: 16 GB
# MAGIC - Number of Cores: 4 Cores
# MAGIC - DataBricks Runtime Version: 11.3 LTS ML 
# MAGIC - Apache Spark version: 3.3.0
# MAGIC - Scala Version: 2.12

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 4.3 Process for Data Cleaning
# MAGIC 
# MAGIC After the joined dataset was created, there were several rows that had erroneous entries or were not in the correct format to be used by a machine learning algorithm. Some rows had missing values for their features, preventing those rows from being used in any meaningful analysis. Other features were in a non-numerical format (primarily strings) that prevented them from being natively used in machine learning. This data therefore had to be cleaned in order to be made usable, as described below.
# MAGIC 
# MAGIC ### 4.3.1 Cleaning Missing Data
# MAGIC 
# MAGIC After joining the data, the amount of null values in our features dropped dramatically, but there were still a few null values that needed to be handled. For features that were missing values, how those missing values were handled depended on whether the feature was categorical or numeric, and whether there was an obvious value to impute or not. 
# MAGIC 
# MAGIC Numeric features with obvious values or were part of continuous ranges were simply handled by having the average value for that feature's remaining data imputed to fill in that missing value. For example, the HourlyPrecipitation feature had its null values replaced with the average value for the data, which was aproximately zero. Imputing the average value allowed us to keep that row for analysis, which was seen as more valuable than whatever minor shift in the feature's distribution caused by the imputation. However, numeric features with values that are not obvious or part of a continuous range would have rows with null values dropped. Imputing the average value may not make logical sense and risks introducing a disruptive influence into the feature's distribution, thereby it is believed that dropping such small number of rows was preferable to artifically shifting the feature's distribution.
# MAGIC 
# MAGIC For categorical features, nulls were handled by dropping the row, as the feature's categorical nature made imputing some sort of representative average impossible or not conceptually sound. As with numeric features that are not obvious or part of a continuous range, it is believed that imputing any value for a missing categorical value would risk creating a disruptive influence into the feature's distribution. As such, it was chosen that the preferred method for handling the relatively insignificant number of null categorical values was to drop the affected rows. It should be noted that in the final feature selection, none of the categorical features that were used in the final analysis required any null values to be dropped.
# MAGIC 
# MAGIC Through this process, a total of 125,000 rows with missing values were dropped, equivalent to 0.5% of the data. Given the small number and the similar proportions of delayed and non-delayed flights dropped, we deemed this acceptable.
# MAGIC 
# MAGIC ### 4.3.2 Cleaning Extreme Data
# MAGIC  
# MAGIC While a vast majority of the values in each feature was within acceptable ranges, there was only a single feaure of interest that had a range of values that exceeded reasonability. When conducting our EDA, we found that the feature `HourlyWindSpeed` had egregious outliers with wind speeds beyond a reasonable amount. For example, while the median of HourlyWindSpeed was about 20 units, the highest value in the feature was over 1100 units. Such values were clearly incorrect and could not be accounted for, but we also did not want to risk artificially swaying the distribution significantly by imputing artificial and potentially inaccurate values for that distribution. As such, for the HourlyWindSpeed feature, we opted to filter out records with windspeeds greater than 200 in order to better represent a field that is already left-tailed.
# MAGIC 
# MAGIC ### 4.3.1 Special Cases
# MAGIC 
# MAGIC There were a few special cases of data cleaning that came about due to conceptual issues that would need to be addressed within our dataset. While the data was seemingly valid on its own, the context of our business needs required us to make these changes in order for our analysis to make logical sense. These special cases are listed below.
# MAGIC 
# MAGIC #### 4.3.1.1 - Cancelled Flights
# MAGIC 
# MAGIC Since the objective of our project is to help a customer plan their time before their flight, the impact of a canceled flight is similar to that of a delayed flight: the customer's flight is not on-time, and thus they must adjust their plans to account for this negative experience. However, the dataset differentiated between flights that were delayed and cancelled such that cancelled flights would have null values for anything relating to delays, since the flights were not considered delayed. As such, it was decided to adjust that data such that flights that were marked as cancelled in the dataset will be marked as delayed; conceptually, they are being delayed indefinitely. Furthermore, flight-related features being converted to null values, the most important being the `DEP_DEL15` feature. Given that a canceled flight elicits the same inconvenience as a delayed flight, it was decided that canceled flights should be considered as a delayed flight. Thus, rows for canceled flights had their `DEP_DEL15` imputed to 1, indicating that those flights were delayed.
# MAGIC 
# MAGIC #### 4.3.1.2 - Timezone Conversion
# MAGIC 
# MAGIC When importing the raw data, the times and dates for each row were localized to that location's time zone. However, multiple time zones makes consistent and meaningful analysis difficult, especially when accounting for flights that travel between time zones. As such, it was decided to standardize every row by converting their times and dates to the UTC time zone, thereby enforcing consistency. To accomplish this, we imported the open-source [Complete List of IATA Airports with Details dataset](https://github.com/lxndrblz/Airports) into our data, used it to map airports to their appropriate timezones, and then convert all times to UTC.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 4.4 Process for Downsampling Data
# MAGIC 
# MAGIC In the training set of data from 2016 - 2020, 22% of the data represented delayed flights while the other 78% of the data were non-delayed flights. This nearly 1-5 ratio of delayed to non-delayed flights proved to be a serious hinderance when training models, as the imbalance of training examples influenced the models to predict that flights will be non-delayed more frequently than expected. To combat this imbalance, the data had to be artifically balanced such that there is an approximately equal number of delayed and non-delayed flights for the model to train with.
# MAGIC 
# MAGIC After implementing balancing the dataset by downsampling, model performance metrics across the board improved significantly enough to justify the decision. Figure 4.4.A shows the performance difference between two identical logistic regression models, with one being trained on the original imbalanced dataset and the other being trained on the downsampled dataset. Furthermore, there is a significant time savings with downsampling as well since much fewer rows need to be parsed through for every operation, cutting the required run time for model training in half. As a result of these two benefits, downsampling was fully implemented into our data pipeline.
# MAGIC 
# MAGIC | Metric    | With Downsampling | Without Downsampling |
# MAGIC |-----------|-------------------|----------------------|
# MAGIC | Precision | 0.625711615       | 0.029235972          |
# MAGIC | F0.5      | 0.491084249       | 0.054158388          |
# MAGIC | Recall    | 0.26393365        | 0.367069134          |
# MAGIC | Accuracy  | 0.604428916       | 0.809395177          |
# MAGIC 
# MAGIC *Table 4.4.A: Downsampling Metrics Comparison*
# MAGIC 
# MAGIC ### 4.4.1 Process
# MAGIC 
# MAGIC In order to downsample our original data, the following steps are followed:
# MAGIC 1. Count the number of rows for delayed flights and non-delayed flights. 
# MAGIC 2. Compute the ratio of delayed flights to non-delayed flights. Our EDA showed that there are consistently more non-delayed flights than delayed flights.
# MAGIC 3. Create a random sample of rows from the non-delayed flights data such that the number of rows for non-delayed flights is twice that of the number of rows for delayed flights. This establishes a ratio of two non-delayed flights to every one delayed flight.
# MAGIC 4. Join the delayed flights rows and the random sample of non-delayed flights together to create a unified dataset with an equal distribution of delayed flights and non-delayed flights.
# MAGIC 
# MAGIC ### 4.4.2 Upsampling VS Downsampling
# MAGIC 
# MAGIC After realizing the benenficial impact of rebalancing our data, we had initially sought to rebalance the data via artificial upsampling. After we had endless technical issues implementing SMOTE via the `imblearn` library, we successfully implemented artificial upsampling by using the MLLib's `sample()` function to randomly select and then add duplicate rows to the delayed flights data. With this method, we could increase the number of delayed flights in the distribution to match that of the non-delayed flights, thus creating an equivalent number of rows for the models to train on. While run-time for training models increased by about half, we believed that the increase in training rows would benefit the model's effectiveness.
# MAGIC 
# MAGIC After some experimentation with upsampling and downsampling, we soon realized that both methods resulted in very similar model performance. The table below shows the performance metrics for two identical logistic regression models with the same parameters on the same dataset, one with upsampling and another with downsampling. Note how similar each metric is.
# MAGIC 
# MAGIC | Metric    | Upsampling  | Downsampling |
# MAGIC |-----------|-------------|--------------|
# MAGIC | Precision | 0.622251122 | 0.625711615  |
# MAGIC | F0.5      | 0.49051794  | 0.491084249  |
# MAGIC | Recall    | 0.265601834 | 0.26393365   |
# MAGIC | Accuracy  | 0.608348208 | 0.604428916  |
# MAGIC 
# MAGIC *Table 4.4.2.A: Upsampling and Downsampling Metrics Comparison*
# MAGIC 
# MAGIC One benefit that downsampling had over upsampling is run-time performance. Because upsampling dramatically increases the size of the data while downsampling dramatically decreases it, downsampling had a far superior run-time compared to upsampling; a downsampled model evaluation would take 50% of the time that an identical upsampled evaluation would take. Because of the dramatic time savings paired with the similarity in model performance, downsampling was chosen over upsampling as our pipeline's method for redistributing data.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 4.5 Blocking Time Series Cross Validation
# MAGIC 
# MAGIC Since this project revolves around time series data, the standard K-Folds Cross Validation method is insufficient since it does not take into account the chronological aspect of the data. KFolds introduces time gaps in the data, tests on data occurring before the training data, and allows for data leakage when the model memorizes future data it should not have seen yet, as seen in the illustration below.
# MAGIC 
# MAGIC ![Figure 4.5.A: Image of K-Folds Cross Validation](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/images/kfoldssplit.JPG?raw=true)
# MAGIC 
# MAGIC *Figure 4.5.A: Image of KF-olds Cross Validation*
# MAGIC 
# MAGIC However, the MLLib library that we were working with did not offer any viable out-of-the-box for cross validating time series data. As such, we chose to build our own version of cross validation called `BlockingTimeSeriesSplit`. Blocking Time Series Split will split the training data into partitions based on an arbitrary number, builds a model for each partition, trains that model on data from the first 70% of that partition, and then tests that model on the data from the latter 30% of that partition. The model with the best average validation metrics across all folds is chosen as our best model to be evaluated against the 2021 test data. This custom method of cross validation should provide more conceptually and mathematically-sound model results due to the way it handles time series data.
# MAGIC 
# MAGIC <center><img src="https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/images/BlockingTimesSplitImage.JPG?raw=true" width="600"/></center>
# MAGIC 
# MAGIC *Figure 4.5.B: Image of Blocking Time Series Split Cross Validation*
# MAGIC 
# MAGIC ### 4.5.1 Process
# MAGIC 
# MAGIC In order to implement this Blocking Time Series Cross Validation method, the following steps are followed:
# MAGIC 1. The training dataset is passed into the Blocking Time Series Cross Validation function with a parameter for how many folds the training data should be broken into.
# MAGIC 2. The number of folds parameter is then used to segment the dataset into equivalent partitions, each representing one fold of the cross validation.
# MAGIC 3. The data within the first partition is then assigned a label equivalent to the relative age of that data within that partition. A 0% percentage indicates that that row is the oldest in the partition, while 100% percentage indicates that that row is the youngest in the partition. 
# MAGIC 4. The model is trained on the data from the first 70% of the partition, representing the 70% oldest data in that partition.
# MAGIC 5. The model is then validated on the data from the latter 30% of the partition, representing the 30% youngest data in the partition.
# MAGIC 6. Metrics are recorded for the training validation.
# MAGIC 7. Repeat steps 3-6 on each paritition in the cross validation until all partitions have been validated.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 4.6 Data Leakage Considerations
# MAGIC 
# MAGIC Data Leakage describes a mistake in machine learning such that data about the target variable is erroneously introduced into a model while it is training. In a sense, data leakage is the mistake of providing the model with data it should not know. For time series data, one serious possibility for data leakage is to provide the model with data from the future that it should not have seen yet.
# MAGIC 
# MAGIC In this project, we took great lengths to ensure that we did not introduce any possibility of data leakage into our dataset, especially in regards to our engineered features. Because we are working with time series data, we were especially mindful to prevent data from the unseen future from leaking into a model's training data. For this reason, we took the following steps to ensure no data leakage occurred in our dataset:
# MAGIC - All predictive data with a time component were always created with a two hour time lag. For example, when predicting weather conditions for a flight, the weather report could be no sooner than two hours behind the flight's scheduled departure time. The two hour window was determined to be sufficiently large enough to prevent any possiblity of data leakage in this context while still being small enough to have sufficient predictive power.
# MAGIC - Certain features from the original dataset are prevented from being used, as they provide information that inherently leaks future information. For example, we could not include any information regarding real departure times or flight cancellation reasons due to the fact that such information would not be available to the model when it would be expected to create a prediction.
# MAGIC 
# MAGIC Because of these efforts, we do not believe that there are any risks of data leakage within our current dataset. However, we will be mindful of any such leakage occurring either from the original dataset or through our engineered features.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 5. Metrics, Models, Features, and Parameters Used

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 5.1 Evaluation Metrics and Desired Outcomes
# MAGIC 
# MAGIC Our desired outcome is to create a machine learning model that will effectively predict what flights will be categorized as delayed. The ability to predict what flights will not arrive on time is highly valuable to individual air travelers. To this intended audience, the ability to accurately predict flight delays will allow them to better economy their time and improve travel scheduling. For evaluating the effectiveness of our model, it is important that we kept the audience’s intended usage for the model in mind. When predicting flight delays, a false positive prediction is far worse than a false negative outcome: a flight that is predicted to be on-time but arrives late means the flier will have to wait, but a flight that is predicted to be late but arrives on-time means the flier may miss their flight entirely.
# MAGIC 
# MAGIC For that reason, our primary metric for success will be the F0.5 score of the model, with the model’s precision being a secondary metric. F0.5 and precision are described in greater detail with their equations in the bullet points below.
# MAGIC 
# MAGIC ### 5.1.1 F0.5 Score
# MAGIC 
# MAGIC F0.5 is a weighted harmonic mean of precision and recall. By considering both precision and recall, F0.5 provides a measurement of how well a model avoids both false positive outcomes and false negative outcomes. However, unlike the F1 score where both recall and precision are weighted evenly in the calculation, F0.5 weighs precision more heavily, therefore making false positive outcomes more impactful than false negative ones. This weighing towards precision, while still accounting for recall, is the main reason why F0.5 score was chosen to be the primary metric, as it suits the need to minimize false positive occurrences while still tracking the positive predictive capabilities of the model. 
# MAGIC 
# MAGIC The mathematical equation for computing the F0.5 score is shown below:
# MAGIC 
# MAGIC \\( F_{0.5} = \frac{1.25 * precision \cdot recall}{0.25 * precision + recall} \\)
# MAGIC 
# MAGIC ### 5.1.2 Precision
# MAGIC 
# MAGIC Precision is the rate of true positives within the model. Precision solely measures the rate of false positive outcomes. Because it does not offer any ability to measure recall alongside precision like the F0.5 score can, precision does not provide insight into the overall predictive capabilities of a model, which is why it was not chosen to be the primary metric. However, being able to have a pure measure of the rate of false positive outcomes is valuable for the experiment, especially when comparing models with similar F0.5 scores, which is why precision is being used as a secondary metric.
# MAGIC 
# MAGIC The mathematical equation for computing precision is shown below, where \\( TP \\) is the number of true positive outcomes and \\( FP \\) is the number of false positive outcomes:
# MAGIC 
# MAGIC \\( Precision  = \frac{TP}{TP + FP} \\)
# MAGIC 
# MAGIC ## 5.1.3 Desired Outcomes
# MAGIC 
# MAGIC For creating a proper comparison and target for our model, we will create a basic model to act as a baseline model, and then create a number of more advanced models which will be used to compare against a simple baseline model. Any improvements in the more advanced model over this baseline model will represent an overall improvement in the ability to correctly predict what flights will be delayed. Therefore, our desired model would be a model that has as high of an F0.5 score as possible, followed by having as high of a precision score as possible.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 5.2 Machine Learning Models Used
# MAGIC 
# MAGIC For this project, our goal will be to develop an advanced machine learning model that outperforms our benchmark of a basic machine learning model. Due to the amount of data that would need to be computed, using a distributed system via Apache Spark was a necessity. Therefore, when selecting what models to implement, the model's ability to parallelize effectively on a distributed system was an essential requirement. While certain models, such as Long Short-Term Memory networks, are great candidates for time series data, their inability to parallelize well prevented them from being considered for this project. As such, we implemented the following algorithms using PySpark’s MLlib library:
# MAGIC 
# MAGIC - [Logistic Regression](https://spark.apache.org/docs/latest/mllib-linear-methods.html#logistic-regression)
# MAGIC 
# MAGIC   - Logistic regression is one of the most simplistic classification models and always a good place to start. It is bounded between 0 and 1, represented as: \\( f(\bold{x}) = \frac{1}{1 + e^{-\bold{w}^T\bold{x}}} \\)
# MAGIC   - Its output can be interpreted as a likelihood, which gives us the flexibility to define a cutoff likelihood for positive and negative predictions. This means that we can optimize our model for precision or recall as desired.
# MAGIC 
# MAGIC   - The loss function for logistic regression is defined as: \\( L(\bold{w}; \bold{x},y) = \log(1 + \exp(-y\bold{w}^T\bold{x})) \\)
# MAGIC     Where \\( \bold{w} \\) is the trained weights, \\( \bold{x} \\) is the input data, and \\( y \\) is the true outcome value.
# MAGIC   
# MAGIC - [Gradient Boosted Tree](https://spark.apache.org/docs/latest/mllib-ensembles.html#gradient-boosted-trees-gbts)
# MAGIC 
# MAGIC   - Gradient-Boosted Trees (GBTs) are ensembles of decision trees. GBTs iteratively train decision trees in order to minimize a loss function. The logistic loss function is used for the GBT Classifier, which is defined as: \\( L(F; x,y) = 2 \sum_{i=1}^N \log(1 + \exp(-2y_iF(x_i)) \\)
# MAGIC     Where \\( \bold{x} \\) is the input data, \\( F(\bold{x}) \\) is the predicted outcome, and \\( y \\) is the true outcome value.
# MAGIC   
# MAGIC   Like decision trees, GBTs handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions.
# MAGIC   
# MAGIC - [Multilayer Perceptron Classifier Neural Network](https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier)
# MAGIC 
# MAGIC   - Multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network. MLPC consists of multiple layers of nodes. Each layer is fully connected to the next layer in the network. Nodes in the input layer represent the input data. All other nodes map inputs to outputs by a linear combination of the inputs with the node’s weights w and bias b and applying an activation function. The loss function optimized is the logistic loss, equivalent to the loss function defined above for GBT.
# MAGIC 
# MAGIC Furthermore, we had initially attempted to implement the following algorithms as well. However, initial testing showed that both models were not performing up to par with what was needed. Given the limited time and resources available for the project, it was decided that these models would not be developed any further beyond initial testing. While these models did not see a full evaluation, their results have been recorded and kept in order to facilitate further discussion.
# MAGIC 
# MAGIC - [Linear SVM ](https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-support-vector-machine)
# MAGIC 
# MAGIC   - The linear support vector machine algorithm defines a linear decision boundary that best separates positive and negative classes. The loss function for SVM is the Hinge loss, which is defined as: \\( L(\bold{w}; \bold{x},y) = \max(0, 1 - y\bold{w}^T\bold{x})  \\)
# MAGIC     Where \\( \bold{w} \\) is the trained weights, \\( \bold{x} \\) is the input data, and \\( y \\) is the true outcome value. 
# MAGIC   - MLlib’s implementation performs L2 regularization by default and uses an OWLQN optimizer.
# MAGIC   - Do note that we believed that the complexity of our data would create a non-linear pattern between delayed and non-delayed groupings, and had intended to implement non-linear SVM in order to better fit this non-linear data. However, MLLib does not provide a native class for non-linear SVM, and we would have been unable to create our own custom version. As such, we opted to use linear SVM with the expectation that the results would not be ideal.
# MAGIC 
# MAGIC - [Random Forest](https://docs.google.com/document/d/1ZCUOfiGdChziaCCqxihUFIBQIjL8mRaQNTz-vA0Fhk0/edit#)
# MAGIC 
# MAGIC   - Random Forest is an ensemble model of decision trees. This ensemble approach helps reduce overfitting, which is a risk for decision tree models. Decision trees use a 0-1 loss function, which is just the proportion of predictions that are incorrect (similar to an accuracy score). 
# MAGIC   - In a distributed system, we can train each decision tree in parallel. 
# MAGIC 
# MAGIC As a baseline model against which to benchmark model performance, logistic regression was chosen to be the baseline model. This model was chosen to be the baseline due to its simplicity to implement, even though its classification effectiveness is not ideal. Gradient Boosted Trees and the Multilayer Perceptron Classifier Neural Network were chosen to be the more advanced models due to their more sophisticated nature and greater effectiveness with classification tasks. As mentioned in Section 5.1.3, "Desired Outcomes", these three models will be trained, validated, and evaluated against each other with the desire to have as high of F0.5 and precision scores as possible.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 5.3 Features Selected for Models
# MAGIC 
# MAGIC Below are the list of features from the dataset that were selected to be used when training and evaluating the models:
# MAGIC 
# MAGIC - 13 Categorical Features
# MAGIC   - QUARTER
# MAGIC   - MONTH
# MAGIC   - DAY_OF_WEEK
# MAGIC   - OP_UNIQUE_CARRIER
# MAGIC   - DEP_HOUR
# MAGIC   - AssumedEffect_Text
# MAGIC   - airline_type
# MAGIC   - is_prev_delayed
# MAGIC   - Blowing_Snow
# MAGIC   - Freezing_Rain
# MAGIC   - Rain
# MAGIC   - Snow
# MAGIC   - Thunder
# MAGIC 
# MAGIC - 13 Numeric Features
# MAGIC   - DISTANCE
# MAGIC   - ELEVATION
# MAGIC   - HourlyAltimeterSetting
# MAGIC   - HourlyDewPointTemperature
# MAGIC   - HourlyWetBulbTemperature
# MAGIC   - HourlyDryBulbTemperature
# MAGIC   - HourlyPrecipitation
# MAGIC   - HourlyStationPressure
# MAGIC   - HourlySeaLevelPressure
# MAGIC   - HourlyRelativeHumidity
# MAGIC   - HourlyVisibility
# MAGIC   - HourlyWindSpeed
# MAGIC   - perc_delay
# MAGIC   
# MAGIC - 1 Graph Feature
# MAGIC   - pagerank (of airports as measured by their departing flights)
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 5.4 Model Parameters Used
# MAGIC 
# MAGIC Each model can be trained with a different set of parameters, which will change how the model trains and predicts incoming data. In order to find the optimal parameters for each model, we conducted a grid search while building the models, training and evaluating every combination of parameters for each of the models used. The parameter values chosen were determined through a combination of researching other examples of models conducting similar tasks, and by applying mathematical intuition to create arbitrary but reasonable values. Below are the list of parameters and the range of parameters used to tune each model.
# MAGIC 
# MAGIC - Logistic Regression:
# MAGIC   - Regularization Parameter: 0.0, 0.01, 0.5, 1.0, 2.0
# MAGIC   - Elastic Net: 0.0, 0.5, 1.0
# MAGIC   - Maximum Iterations: 5, 10, 50
# MAGIC   - Threshold: 0.5, 0.6, 0.7, 0.8
# MAGIC 
# MAGIC - Gradient Boosted Trees:
# MAGIC   - Maximum Iterations: 5, 10,  50
# MAGIC   - Maximum Depth: 4, 8, 16
# MAGIC   - Maximum Bins: 32, 64, 128
# MAGIC   - Step Size: 0.1, 0.5
# MAGIC   - Threshold: 0.5, 0.6, 0.7, 0.8
# MAGIC   
# MAGIC - Multilayer Perceptron Classifier Neural Network:
# MAGIC   - Maximum Iterations: 100, 200
# MAGIC   - Block Size: 128, 256
# MAGIC   - Step Size: 0.03, 0.1
# MAGIC   - Threshold: 0.5, 0.6, 0.7, 0.8
# MAGIC   - Layer Architectures Used:
# MAGIC     - 90 Sigmoid Input Nodes, 30 Sigmoid Hidden Nodes, 15 Sigmoid Hidden Nodes, 2 Softmax Output Nodes
# MAGIC     - 90 Sigmoid Input Nodes, 15 Sigmoid Hidden Nodes, 2 Softmax Output Nodes
# MAGIC 
# MAGIC 
# MAGIC ### 5.4.1 A Note about RandomSearch
# MAGIC 
# MAGIC This grid search method is powerful in that it allows us to systematically train and evaluate a large number of parameters for our models. However, its reliance on user-defined values for each parameter means that it may not truly reach optimal parameter values: if the true optimal value exists between two pre-defined values, there is no way for the model to ever reach that optimal value. RandomSearch is an improvement upon GridSearch, as it tests parameter values randomly from within a range of values rather than relying on set user-defined values. This added flexibility allows RandomSearch to potentially find optimal parameter values that would otherwise exist between two unreachable pre-defined values.
# MAGIC 
# MAGIC We had originally intended to utilize random search instead of grid search for hyperparameter tuning. However, we experienced technical issues with implementing Random Search to work with our custom Blocking Time Series Cross Validation function. MLLib provides a ParamGridBuilder class that provides random search functionality, but this ParamGridBuilder class would not interface with our custom Blocking Time Series Cross Validation, making the usage of MLLib's built-in random search functionality impossible. We had also explored the possibility of using SciKit-Learn's built-in random search class, but SciKit-Learn could not fully integrate and operate with DataBricks's environment, thus preventing that from being an option as well. Due to time constraints, we opted to instead choose the less-optimal grid search hyperparameter tuning method rather than attempt to create our own custom random search class.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 6. Experiments and Results
# MAGIC 
# MAGIC With the finalized dataset created, the models chosen, and the pipeline prepared, the project proceeded with the experiments in question. Each type of model was trained, cross validated, hyperparameter tuned, and had its training validation and test evaluation metrics collected. This section will summarize the experiments that were run along with showing the results collected from each experiment. The final analysis and discussion of the results can be found in the following Section 7, "Discussion of Experimental Results".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.1 "Never Delayed" Naive Model
# MAGIC 
# MAGIC Given the relatively low occurrence of delayed flights, one possible approach coul be to predict that every fight will not be delayed (i.e, predict all zeros). If we were to employ this appraoch, given that 18.8% of all flights in 2021 were delayed, we would have an accuracy score of 81.2%. While this may look great at face-value, the naive model would have no true positives or false positives, meaning that precision, recall, and F0.5 scores would all be equal to zero. 
# MAGIC 
# MAGIC Ultimately, the naive model, while correct 81% of the time, offers no tangible value to airline travelers. Thus, we have selected to use a basic logistic regression for our baseline approach instead, which is further described in Section 6.2.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.2 Logistic Regression - Baseline Model
# MAGIC 
# MAGIC Logistic Regression was the first model that was created for this project, and was created with the expressed purpose of being used as a baseline model for comparison with more advanced models. This baseline version of logistic regression was trained using only the raw data, and included none of the constructed features.
# MAGIC 
# MAGIC For reference, below are a list of features that this baseline model was trained with:
# MAGIC 
# MAGIC | Feature Name              | Type        | Label Encoding | OneHot Encoding |
# MAGIC |---------------------------|-------------|----------------|-----------------|
# MAGIC | QUARTER                   | Categorical | 1              | 1               |
# MAGIC | MONTH                     | Categorical | 1              | 1               |
# MAGIC | DAY_OF_WEEK               | Categorical | 1              | 1               |
# MAGIC | OP_UNIQUE_CARRIER         | Categorical | 1              | 1               |
# MAGIC | DISTANCE                  | Numeric     | 0              | 0               |
# MAGIC | ELEVATION                 | Numeric     | 0              | 0               |
# MAGIC | HourlyAltimeterSetting    | Numeric     | 0              | 0               |
# MAGIC | HourlyDewPointTemperature | Numeric     | 0              | 0               |
# MAGIC | HourlyWetBulbTemperature  | Numeric     | 0              | 0               |
# MAGIC | HourlyDryBulbTemperature  | Numeric     | 0              | 0               |
# MAGIC | HourlyPrecipitation       | Numeric     | 0              | 0               |
# MAGIC | HourlyStationPressure     | Numeric     | 0              | 0               |
# MAGIC | HourlySeaLevelPressure    | Numeric     | 0              | 0               |
# MAGIC | HourlyRelativeHumidity    | Numeric     | 0              | 0               |
# MAGIC | HourlyVisibility          | Numeric     | 0              | 0               |
# MAGIC | HourlyWindSpeed           | Numeric     | 0              | 0               |
# MAGIC 
# MAGIC *Table 6.2.A: Features used in Logistic Regression - Baseline Model*
# MAGIC 
# MAGIC ### Data Encoding Methods Used
# MAGIC 
# MAGIC For the logistic regression model, all categorical features were string indexed and one-hot encoded. Furthermore, all numeric features were scaled using the built-in standard scaler in the `LogisticRegression` function. After processesing the features listed in the table above in our pipeline, the 16 unique features resulted in 49 distinct columns.
# MAGIC 
# MAGIC ### List of Parameters Used
# MAGIC 
# MAGIC Using those parameters, we performed an exhaustive grid search with our baseline logistic regression model. The parameters we used to gridsearch through are listed below:
# MAGIC 
# MAGIC - regParam = [0.0, 0.01, 0.5, 2.0]
# MAGIC - elasticNetParam = [0.0, 0.5, 1.0]
# MAGIC - maxIter = [5, 10]
# MAGIC - thresholds = [0.5, 0.6, 0.7]
# MAGIC 
# MAGIC ### Cross Validation Run-Times
# MAGIC 
# MAGIC For a full cross validation run across the full training dataset, logistic regression required approximately 10 minutes. The 24 cross-validation runs shown below took approximately 8.5 hours to complete.
# MAGIC 
# MAGIC ### Grid-Search Results
# MAGIC 
# MAGIC For the full table of the grid-search results for the logistic regression baseline model, please check the appendix for section 14.4.1, "Cross Validation Results Table".
# MAGIC 
# MAGIC ### 6.2.1 Top Performing Model
# MAGIC 
# MAGIC regParam = 0, elasticNetParam = 0, maxIter = 5, threshold = 0.5
# MAGIC 
# MAGIC | cv_fold | test_Precision | test_Recall | test_F0.5 | test_F1  | test_Accuracy | val_Precision | val_Recall | val_F0.5 | val_F1   | val_Accuracy |
# MAGIC |---------|----------------|-------------|-----------|----------|---------------|---------------|------------|----------|----------|--------------|
# MAGIC | 0       | 0.365994       | 0.084262    | 0.219328  | 0.136986 | 0.800715      | 0.565021      | 0.049388   | 0.182969 | 0.090837 | 0.640758     |
# MAGIC | 1       | 0.317176       | 0.094757    | 0.215847  | 0.145920 | 0.791792      | 0.526077      | 0.104016   | 0.290404 | 0.173690 | 0.652546     |
# MAGIC | 2       | 0.333326       | 0.106540    | 0.233793  | 0.161470 | 0.792298      | 0.552102      | 0.091188   | 0.274553 | 0.156523 | 0.613855     |
# MAGIC | 3       | 0.295717       | 0.034937    | 0.118625  | 0.062491 | 0.803236      | 0.339682      | 0.036459   | 0.127539 | 0.065850 | 0.844396     |
# MAGIC 
# MAGIC *Table 6.3.1.A: CV Folds Results Table for Baseline Logistic Regression*
# MAGIC 
# MAGIC 
# MAGIC Of all of the models created in the cross validation grid-search, the model that scored the highest average F0.5 score during training validation would be the model with the following parameters: regularization parameter of 0, elastic net parameter of 0, maximum iterations of 5, and a threshold of 0.5. This model had an average validation F0.5 score of 0.219 and an average precision score of 0.496, and it received an average test evaluation F0.5 score of 0.197 and an average precision score of 0.328.
# MAGIC 
# MAGIC ![Figure 6.2.1.A: ROC Graph for Logistic Regression - Baseline](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/BaselineLR_bestModel_ROC.jpg)
# MAGIC 
# MAGIC *Figure 6.2.1.A: ROC Graph for Logistic Regression - Baseline*
# MAGIC 
# MAGIC ![Figure 6.2.1.B: Scores by Threshold Graph for Logistic Regression - Baseline](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/BaselineLR_bestModel_ScoresByThresh.jpg)
# MAGIC 
# MAGIC *Figure 6.2.1.B: Scores by Threshold Graph for Logistic Regression - Baseline*
# MAGIC 
# MAGIC ![Figure 6.2.1.C: Loss Curve Graph for Logistic Regression - Baseline](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/BaselineLR_bestModel_LossCurve.jpg)
# MAGIC 
# MAGIC *Figure 6.2.1.C: Loss Curve Graph for Logistic Regression - Baseline*
# MAGIC 
# MAGIC ### 6.2.2 Most Influencial Features
# MAGIC 
# MAGIC The most influencial features for this baseline model include the `HourlyPrecipitation`, a number of OP Carrier classes, and the month of the year (specifically the major holiday months around the end of the year and fourth of July). The OP Carriers that are most indicitive of delays in the model include JetBlue (B6), Endeavor Air (9E), and Jetstream Intl (OH), while the carriers that are least indicitive of delays include Alaska Airlines (AS), Delta Airlines (DL).
# MAGIC 
# MAGIC | featureName                | coefficient | importance |
# MAGIC |----------------------------|-------------|------------|
# MAGIC | HourlyPrecipitation        | 4.95733     | 4.95733    |
# MAGIC | OP_UNIQUE_CARRIER_class_B6 | 0.588973    | 0.588973   |
# MAGIC | OP_UNIQUE_CARRIER_class_AS | -0.538183   | 0.538183   |
# MAGIC | OP_UNIQUE_CARRIER_class_9E | 0.430673    | 0.430673   |
# MAGIC | OP_UNIQUE_CARRIER_class_OH | 0.428061    | 0.428061   |
# MAGIC | MONTH_class_11             | -0.351882   | 0.351882   |
# MAGIC | OP_UNIQUE_CARRIER_class_DL | -0.327041   | 0.327041   |
# MAGIC | OP_UNIQUE_CARRIER_class_WN | 0.265175    | 0.265175   |
# MAGIC | MONTH_class_12             | 0.262774    | 0.262774   |
# MAGIC | OP_UNIQUE_CARRIER_class_UA | -0.257401   | 0.257401   |
# MAGIC | MONTH_class_7              | 0.251142    | 0.251142   |
# MAGIC 
# MAGIC *Table 6.2.2.A: Most Important Features for for Logistic Regression - Baseline*
# MAGIC 
# MAGIC ### 6.2.3 Model Notebook and Code
# MAGIC 
# MAGIC To view the full code for the Baseline Logistic Regression model, please review the notebooks in the appendix, section 14.4.2, "Link to Modelling Notebook".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.3 Logistic Regression - Feature Engineering
# MAGIC 
# MAGIC After the baseline logistic regression model shown above was created, a second experimental model was created in order to incorporate the added predictive power of the newly engineered features. This model is an iteration upon the baseline logistic regression model, allowing us to test newly created features for their predictive ability.
# MAGIC 
# MAGIC ### Data Encoding Methods Used
# MAGIC 
# MAGIC The logistic regression model was trained on an expanded set of features listed in the table below, which includes all of the same features from the baseline model, with the addition of 10 engineered features. All categorical and numeric variables were encoded using the same pipeline steps for the baseline model (categorical values were label encoded and one-hot encoded, and standard scaling was applied using the `LogisticRegression` function).
# MAGIC 
# MAGIC | Feature Name              | Type        | Label Encoding | OneHot Encoding |
# MAGIC |---------------------------|-------------|----------------|-----------------|
# MAGIC | QUARTER                   | Categorical | 1              | 1               |
# MAGIC | MONTH                     | Categorical | 1              | 1               |
# MAGIC | DAY_OF_WEEK               | Categorical | 1              | 1               |
# MAGIC | OP_UNIQUE_CARRIER         | Categorical | 1              | 1               |
# MAGIC | DEP_HOUR                  | Categorical | 1              | 1               |
# MAGIC | AssumedEffect_Text        | Categorical | 1              | 1               |
# MAGIC | airline_type              | Categorical | 1              | 1               |
# MAGIC | is_prev_delayed           | Categorical | 1              | 1               |
# MAGIC | Blowing_Snow              | Categorical | 1              | 1               |
# MAGIC | Freezing_Rain             | Categorical | 1              | 1               |
# MAGIC | Rain                      | Categorical | 1              | 1               |
# MAGIC | Snow                      | Categorical | 1              | 1               |
# MAGIC | Thunder                   | Categorical | 1              | 1               |
# MAGIC | DISTANCE                  | Numeric     | 0              | 0               |
# MAGIC | ELEVATION                 | Numeric     | 0              | 0               |
# MAGIC | HourlyAltimeterSetting    | Numeric     | 0              | 0               |
# MAGIC | HourlyDewPointTemperature | Numeric     | 0              | 0               |
# MAGIC | HourlyWetBulbTemperature  | Numeric     | 0              | 0               |
# MAGIC | HourlyDryBulbTemperature  | Numeric     | 0              | 0               |
# MAGIC | HourlyPrecipitation       | Numeric     | 0              | 0               |
# MAGIC | HourlyStationPressure     | Numeric     | 0              | 0               |
# MAGIC | HourlySeaLevelPressure    | Numeric     | 0              | 0               |
# MAGIC | HourlyRelativeHumidity    | Numeric     | 0              | 0               |
# MAGIC | HourlyVisibility          | Numeric     | 0              | 0               |
# MAGIC | HourlyWindSpeed           | Numeric     | 0              | 0               |
# MAGIC | perc_delay                | Numeric     | 0              | 0               |
# MAGIC | pagerank                  | Numeric     | 0              | 0               |
# MAGIC 
# MAGIC *Table 6.3.A: Features used in Logistic Regression - Feature Engineering Model*
# MAGIC 
# MAGIC After applying the data pipeline to these 27 unique features, our final feature set contained 90 columns.
# MAGIC 
# MAGIC ### List of Parameters Used
# MAGIC 
# MAGIC Using the parameters listed below, we performed an exhaustive grid search with our logistic regression model:
# MAGIC - regParam = [0.0, 0.01, 0.5, 2.0]
# MAGIC - elasticNetParam = [0.0, 0.5, 1.0]
# MAGIC - maxIter = [10, 50]
# MAGIC - thresholds = [0.5, 0.6, 0.7, 0.8]
# MAGIC 
# MAGIC ### Cross Validation Run-Times
# MAGIC 
# MAGIC For a full cross validation run across all of the training data, logistic regression would require between 7 - 20 minutes, with the mean value being approximately 15 minutes.
# MAGIC 
# MAGIC ### Grid-Search Results
# MAGIC 
# MAGIC For the full table of the grid-search results for the logistic regression baseline model, please check the appendix for section 14.4.2, "Cross Validation Results Table".
# MAGIC 
# MAGIC 
# MAGIC ### 6.3.1 Top Performing Model
# MAGIC 
# MAGIC regParam = 0.0, elasticNetParam = 0.0, maxIter = 5, threshold = 0.6
# MAGIC 
# MAGIC | cv_fold | test_Precision | test_Recall | test_F0.5 | test_F1  | test_Accuracy | val_Precision | val_Recall | val_F0.5 | val_F1   | val_Accuracy |
# MAGIC |---------|----------------|-------------|-----------|----------|---------------|---------------|------------|----------|----------|--------------|
# MAGIC | 0       | 0.645674       | 0.261823    | 0.499278  | 0.372568 | 0.834472      | 0.789250      | 0.338911   | 0.623540 | 0.474197 | 0.726894     |
# MAGIC | 1       | 0.608832       | 0.293249    | 0.501000  | 0.395839 | 0.831975      | 0.772133      | 0.366543   | 0.632219 | 0.497104 | 0.739633     |
# MAGIC | 2       | 0.624821       | 0.298145    | 0.512510  | 0.403671 | 0.834656      | 0.797583      | 0.356369   | 0.639286 | 0.492627 | 0.711578     |
# MAGIC | 3       | 0.605849       | 0.297409    | 0.501772  | 0.398967 | 0.831803      | 0.517809      | 0.223475   | 0.409849 | 0.312208 | 0.851886     |
# MAGIC 
# MAGIC *Table 6.3.1.A: CV Folds Results Table for Baseline Logistic Regression*
# MAGIC 
# MAGIC Of the models created in the cross validation grid-search, the model that scored the highest average F0.5 score during training validation was the model with the following parameters: regularization parameter of 0, elastic net parameter of 0, maximum iterations of 5, and a threshold of 0.6. This model had an average validation F0.5 score of 0.576 and an average precision score of 0.719, and it received an average test evaluation F0.5 score of 0.504 and an average precision score of 0.621. These are significant improvements on the original baseline model, indicating that the engineered features provide significant additional predictive power.
# MAGIC 
# MAGIC ![Figure 6.3.1.A: ROC Graph for Logistic Regression - Feature Engineering](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/LR_bestModel_ROC.jpg)
# MAGIC 
# MAGIC *Figure 6.3.1.A: ROC Graph for Logistic Regression - Feature Engineering*
# MAGIC 
# MAGIC ![Figure 6.3.1.B: Scores by Threshold Graph for Logistic Regression - Feature Engineering](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/LR_bestModel_ScoresByThresh.jpg)
# MAGIC 
# MAGIC *Figure 6.3.1.B: Scores by Threshold Graph for Logistic Regression - Feature Engineering*
# MAGIC 
# MAGIC ![Figure 6.2.1.C: Loss Curve Graph for Logistic Regression - Feature Engineering](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/LR_bestModel_LossCurve.jpg)
# MAGIC 
# MAGIC *Figure 6.2.1.C: Loss Curve Graph for Logistic Regression - Feature Engineering*
# MAGIC 
# MAGIC ### 6.3.2 Most Influential Feature
# MAGIC 
# MAGIC The table below identifies the most important features for this logistic regression model. Interestingly, `HourlyPrecipitation` remains the most predictive feature, but all of the following important features are engineered, including an indicator for whether the plane's previous flight was delayed, extreme weather indicators, holiday seasons, and the scheduled departure hour.
# MAGIC 
# MAGIC | featureName                           | coefficient | importance |
# MAGIC |---------------------------------------|-------------|------------|
# MAGIC | HourlyPrecipitation                   | 2.423615    | 2.423615   |
# MAGIC | is_prev_delayed_class_0.0             | -1.704681   | 1.704681   |
# MAGIC | Freezing_Rain_class_0                 | -1.344212   | 1.344212   |
# MAGIC | DEP_HOUR_class_05                     | -1.174979   | 1.174979   |
# MAGIC | DEP_HOUR_class_06                     | -1.138793   | 1.138793   |
# MAGIC | DEP_HOUR_class_04                     | -1.116434   | 1.116434   |
# MAGIC | DEP_HOUR_class_07                     | -0.870017   | 0.870017   |
# MAGIC | AssumedEffect_Text_class_Christmas_p2 | 0.843418    | 0.843418   |
# MAGIC | Thunder_class_0                       | -0.834798   | 0.834798   |
# MAGIC | AssumedEffect_Text_class_Christmas_p3 | 0.501366    | 0.501366   |
# MAGIC 
# MAGIC *Table 6.3.2.A: Most Important Features for Baseline Logistic Regression*
# MAGIC 
# MAGIC ### 6.3.3 Model Notebook and Code
# MAGIC 
# MAGIC To view the full code for the Logistic Regression Feature Engineering model, please review the notebooks in the appendix, section 14.5.2, "Link to Modelling Notebook".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.4 Gradient Boosted Trees Model
# MAGIC 
# MAGIC Gradient Boosted Trees (GBT) have proven themselves to be a reliable general-purpose machine learning model. This model was chosen to test its improved classification prediction capabilities from a non-neural network model.
# MAGIC 
# MAGIC ### Data Encoding Methods Used
# MAGIC 
# MAGIC Due to the significant improvement in the logistic regression performance with the added features, we elected to use all of the same features as those in the expanded logistic regression model. This time, however, categorical features were only label encoded and not one-hot encoded, and numeric features were not scaled. This is because tree models rely on identifying splitting points (or bucketing the data) to make predictions, which are irrespective of scale.
# MAGIC 
# MAGIC | Feature Name              | Type        | Label Encoding | OneHot Encoding |
# MAGIC |---------------------------|-------------|----------------|-----------------|
# MAGIC | QUARTER                   | Categorical | 1              | 0               |
# MAGIC | MONTH                     | Categorical | 1              | 0               |
# MAGIC | DAY_OF_WEEK               | Categorical | 1              | 0               |
# MAGIC | OP_UNIQUE_CARRIER         | Categorical | 1              | 0               |
# MAGIC | DEP_HOUR                  | Categorical | 1              | 0               |
# MAGIC | AssumedEffect_Text        | Categorical | 1              | 0               |
# MAGIC | airline_type              | Categorical | 1              | 0               |
# MAGIC | is_prev_delayed           | Categorical | 1              | 0               |
# MAGIC | Blowing_Snow              | Categorical | 1              | 0               |
# MAGIC | Freezing_Rain             | Categorical | 1              | 0               |
# MAGIC | Rain                      | Categorical | 1              | 0               |
# MAGIC | Snow                      | Categorical | 1              | 0               |
# MAGIC | Thunder                   | Categorical | 1              | 0               |
# MAGIC | DISTANCE                  | Numeric     | 0              | 0               |
# MAGIC | ELEVATION                 | Numeric     | 0              | 0               |
# MAGIC | HourlyAltimeterSetting    | Numeric     | 0              | 0               |
# MAGIC | HourlyDewPointTemperature | Numeric     | 0              | 0               |
# MAGIC | HourlyWetBulbTemperature  | Numeric     | 0              | 0               |
# MAGIC | HourlyDryBulbTemperature  | Numeric     | 0              | 0               |
# MAGIC | HourlyPrecipitation       | Numeric     | 0              | 0               |
# MAGIC | HourlyStationPressure     | Numeric     | 0              | 0               |
# MAGIC | HourlySeaLevelPressure    | Numeric     | 0              | 0               |
# MAGIC | HourlyRelativeHumidity    | Numeric     | 0              | 0               |
# MAGIC | HourlyVisibility          | Numeric     | 0              | 0               |
# MAGIC | HourlyWindSpeed           | Numeric     | 0              | 0               |
# MAGIC | perc_delay                | Numeric     | 0              | 0               |
# MAGIC | pagerank                  | Numeric     | 0              | 0               |
# MAGIC 
# MAGIC *Table 6.4.A: Features used in Gradient Boosted Trees*
# MAGIC 
# MAGIC Because we did not use any one-hot encoding, the 27 unique features, when applied to the data pipeline, returned 27 columns. 
# MAGIC 
# MAGIC ### List of Parameters Used
# MAGIC   - Maximum Iterations: [5, 10, 50]
# MAGIC   - Maximum Depth: [4, 8, 16]
# MAGIC   - Maximum Bins: [32, 64, 128]
# MAGIC   - Step Size: [0.1, 0.5]
# MAGIC   - Threshold: [0.5, 0.6, 0.7, 0.8]
# MAGIC 
# MAGIC ### Cross Validation Run-Times
# MAGIC 
# MAGIC For a full cross validation run on the training dataset, GBT would require between 5 - 10 minutes, with the mean value being approximately 7 minutes.
# MAGIC 
# MAGIC ### Grid-Search Results
# MAGIC 
# MAGIC For the full table of the grid-search results for the logistic regression baseline model, please check the appendix for section 14.4.3, "Cross Validation Results Table".
# MAGIC 
# MAGIC ### 6.4.1 Top Performing Model
# MAGIC 
# MAGIC maxIter = 5, maxDepth = 4, maxBins = 32, stepSize = 0.5, threshold = 0.6
# MAGIC 
# MAGIC | cv_fold | test_Precision | test_Recall | test_F0.5 | test_F1  | test_Accuracy | val_Precision | val_Recall | val_F0.5 | val_F1   | val_Accuracy |
# MAGIC |---------|----------------|-------------|-----------|----------|---------------|---------------|------------|----------|----------|--------------|
# MAGIC | 0       | 0.629992       | 0.323516    | 0.529643  | 0.427501 | 0.837357      | 0.816609      | 0.344118   | 0.640674 | 0.484196 | 0.733589     |
# MAGIC | 1       | 0.628752       | 0.320410    | 0.527270  | 0.424497 | 0.836928      | 0.789900      | 0.366295   | 0.641521 | 0.500498 | 0.743318     |
# MAGIC | 2       | 0.625221       | 0.327274    | 0.528917  | 0.429648 | 0.836904      | 0.806017      | 0.371477   | 0.653199 | 0.508566 | 0.717922     |
# MAGIC | 3       | 0.608611       | 0.322447    | 0.516869  | 0.421552 | 0.833899      | 0.522263      | 0.221592   | 0.410787 | 0.311161 | 0.852415     |
# MAGIC 
# MAGIC *Table 6.4.1.A: CV Folds Results Table for Gradient Boosted Trees*
# MAGIC 
# MAGIC The model with the best average metrics from cross-validation had the following parameters: maximum iterations of 5, maximum depth of 4, maximum bins of 32, step size of 0.5, and a threshold of 0.6. The average F0.5 score on the validation set for this model is 0.587 and the average precision is 0.734. When tested on the 2021 data, the model returned an F0.5 score of 0.526 and a precision of 0.623, marginally surpassing the performance of the logistic regression model. 
# MAGIC 
# MAGIC ![Figure 6.4.1.A: ROC Graph for Gradient Boosted Trees](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/GBT_bestModel_ROC.jpg)
# MAGIC 
# MAGIC *Figure 6.4.1.A: ROC Graph for Gradient Boosted Trees*
# MAGIC 
# MAGIC ![Figure 6.4.1.B: Scores by Threshold Graph for Gradient Boosted Trees](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/GBT_bestModel_ScoresByThresh.jpg)
# MAGIC 
# MAGIC *Figure 6.4.1.B: Scores by Threshold Graph for Gradient Boosted Trees*
# MAGIC 
# MAGIC ![Figure 6.4.1.C: Loss Curve Graph for Gradient Boosted Trees](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/GBT_bestModel_LossCurve.jpg)
# MAGIC 
# MAGIC *Figure 6.4.1.C: Loss Curve Graph for Gradient Boosted Trees*
# MAGIC 
# MAGIC 
# MAGIC ### 6.4.2 Most Influential Features
# MAGIC 
# MAGIC The following table lists the most important features in the GBT model:
# MAGIC 
# MAGIC | featureName               | coefficient | importance |
# MAGIC |---------------------------|-------------|------------|
# MAGIC | is_prev_delayed_idx       | 0.426188    | 0.426188   |
# MAGIC | DEP_HOUR_idx              | 0.249705    | 0.249705   |
# MAGIC | OP_UNIQUE_CARRIER_idx     | 0.086915    | 0.086915   |
# MAGIC | HourlyPrecipitation       | 0.051673    | 0.051673   |
# MAGIC | MONTH_idx                 | 0.041822    | 0.041822   |
# MAGIC | HourlyVisibility          | 0.02378     | 0.02378    |
# MAGIC | Thunder_idx               | 0.020908    | 0.020908   |
# MAGIC | AssumedEffect_Text_idx    | 0.01614     | 0.01614    |
# MAGIC | HourlyDewPointTemperature | 0.012712    | 0.012712   |
# MAGIC | Snow_idx                  | 0.011951    | 0.011951   |
# MAGIC | DAY_OF_WEEK_idx           | 0.011846    | 0.011846   |
# MAGIC | DISTANCE                  | 0.010155    | 0.010155   |
# MAGIC | HourlyRelativeHumidity    | 0.008347    | 0.008347   |
# MAGIC | pagerank                  | 0.007879    | 0.007879   |
# MAGIC 
# MAGIC *Table 6.2.2.A: Most Important Features for Gradient Boosted Trees*
# MAGIC 
# MAGIC For the first time, we find that the `HourlyPrecipitation` feature is no longer the most important, though it remains in the top three. Instead, the previous-delay indicator, the departure hour, and the carrier become the most important features. Interestingly, the PageRank for the departure airports becomes more important in the GBT model than it was in the logistic regression models, but it is not in the top 10 for this model. 
# MAGIC 
# MAGIC ### 6.4.3 Model Notebook and Code
# MAGIC 
# MAGIC To view the full code for the Logistic Regression Feature Engineering model, please review the notebooks in the appendix, section 14.6.2, "Link to Modelling Notebook".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.5 Multilayer Perceptron Neural Network Model
# MAGIC 
# MAGIC The Multilayer Perceptron Neural Network (MLP NN or MLP) was implemented due to its neural network architecture and its improved classification abilities. This model requires considerable amounts of time in order to train a model and will not provide insights into feature importance, but that was deemed to be acceptable given its classification capabilities. To investigate how model prediction power would be affected by the model's layer architecture, two models with different layer architectures were created and tested. Due to time constraints, we were unable to conduct hyperparameter tuning for parameters except for threshold; all other parameters were kept at the model's defaults.
# MAGIC 
# MAGIC ### Data Encoding Methods Used
# MAGIC 
# MAGIC | Feature Name              | Type        | Label Encoding | OneHot Encoding |
# MAGIC |---------------------------|-------------|----------------|-----------------|
# MAGIC | QUARTER                   | Categorical | 1              | 1               |
# MAGIC | MONTH                     | Categorical | 1              | 1               |
# MAGIC | DAY_OF_WEEK               | Categorical | 1              | 1               |
# MAGIC | OP_UNIQUE_CARRIER         | Categorical | 1              | 1               |
# MAGIC | DEP_HOUR                  | Categorical | 1              | 1               |
# MAGIC | AssumedEffect_Text        | Categorical | 1              | 1               |
# MAGIC | airline_type              | Categorical | 1              | 1               |
# MAGIC | is_prev_delayed           | Categorical | 1              | 1               |
# MAGIC | Blowing_Snow              | Categorical | 1              | 1               |
# MAGIC | Freezing_Rain             | Categorical | 1              | 1               |
# MAGIC | Rain                      | Categorical | 1              | 1               |
# MAGIC | Snow                      | Categorical | 1              | 1               |
# MAGIC | Thunder                   | Categorical | 1              | 1               |
# MAGIC | DISTANCE                  | Numeric     | 0              | 0               |
# MAGIC | ELEVATION                 | Numeric     | 0              | 0               |
# MAGIC | HourlyAltimeterSetting    | Numeric     | 0              | 0               |
# MAGIC | HourlyDewPointTemperature | Numeric     | 0              | 0               |
# MAGIC | HourlyWetBulbTemperature  | Numeric     | 0              | 0               |
# MAGIC | HourlyDryBulbTemperature  | Numeric     | 0              | 0               |
# MAGIC | HourlyPrecipitation       | Numeric     | 0              | 0               |
# MAGIC | HourlyStationPressure     | Numeric     | 0              | 0               |
# MAGIC | HourlySeaLevelPressure    | Numeric     | 0              | 0               |
# MAGIC | HourlyRelativeHumidity    | Numeric     | 0              | 0               |
# MAGIC | HourlyVisibility          | Numeric     | 0              | 0               |
# MAGIC | HourlyWindSpeed           | Numeric     | 0              | 0               |
# MAGIC | perc_delay                | Numeric     | 0              | 0               |
# MAGIC | pagerank                  | Numeric     | 0              | 0               |
# MAGIC 
# MAGIC *Table 6.5.A: Features used in Multilayer Perceptron Classifier*
# MAGIC 
# MAGIC ### List of Parameters, Layer Architectures Used
# MAGIC 
# MAGIC   - Maximum Iterations: 100
# MAGIC   - Block Size: 128
# MAGIC   - Step Size: 0.5
# MAGIC   - Threshold: 0.5, 0.6, 0.7, 0.8
# MAGIC 
# MAGIC Layer Architecture:
# MAGIC - num_layers = 4: 90 Sigmoid Input Nodes, 30 Sigmoid Hidden Nodes, 15 Sigmoid Hidden Nodes, 2 Softmax Output Nodes
# MAGIC - num_layers = 3: 90 Sigmoid Input Nodes, 15 Sigmoid Hidden Nodes, 2 Softmax Output Nodes
# MAGIC   
# MAGIC ### Cross Validation Run-Times
# MAGIC 
# MAGIC The run-time for the multilayer perceptron neural network model was highly dependent upon the layer architecture used. For the model with three layers, the single cross validation required 144 minutes. For the model with four layers, a single cross validation run required 322 minutes.
# MAGIC 
# MAGIC ### Experimental Results
# MAGIC 
# MAGIC For the full table of the grid-search results for the logistic regression baseline model, please check the appendix for section 14.4.4, "Cross Validation Results Table".
# MAGIC 
# MAGIC ### 6.5.1 Top Performing Model
# MAGIC 
# MAGIC maxIter = 100, blockSize = 128, stepSize = 0.5, threshold = 0.5
# MAGIC 
# MAGIC Layer architecture: 90 Sigmoid Input Nodes, 30 Sigmoid Hidden Nodes, 15 Sigmoid Hidden Nodes, 2 Softmax Output Nodes
# MAGIC 
# MAGIC | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy | cv_fold | maxIter | blockSize | stepSize | num_layers | threshold |
# MAGIC |----------------|-------------|-------------|-------------|---------------|---------------|-------------|-------------|-------------|--------------|---------|---------|-----------|----------|------------|-----------|
# MAGIC | 0.647378619    | 0.283098698 | 0.514874655 | 0.393931245 | 0.836491084   | 0.784178475   | 0.357852437 | 0.633285949 | 0.4914406   | 0.730873701  | 0       | 100     | 128       | 0.5      | 4          | 0.5       |
# MAGIC | 0.598651648    | 0.332025203 | 0.515809475 | 0.427145974 | 0.832836751   | 0.708765842   | 0.42892724  | 0.626958416 | 0.534430561 | 0.737635004  | 1       | 100     | 128       | 0.5      | 4          | 0.5       |
# MAGIC | 0.539930159    | 0.386778577 | 0.500308982 | 0.450699147 | 0.823034728   | 0.769602559   | 0.406263931 | 0.652831686 | 0.531798063 | 0.718930018  | 2       | 100     | 128       | 0.5      | 4          | 0.5       |
# MAGIC | 0.583216793    | 0.321987116 | 0.501795051 | 0.414908268 | 0.829543932   | 0.494001684   | 0.19200103  | 0.375786038 | 0.276526114 | 0.848871925  | 3       | 100     | 128       | 0.5      | 4          | 0.5       |
# MAGIC 
# MAGIC *Table 6.5.1.A: CV Folds Results Table for Multilayer Perceptron Classifier*
# MAGIC 
# MAGIC The model with the best average metrics from cross-validation had the following parameters: maximum iterations of 100, block size of 128, step size of 0.5, and a threshold of 0.6. The average F0.5 score on the validation set for this model is 0.572 and the average precision is 0.689. When tested on the 2021 data, the model returned an F0.5 score of 0.508 and a precision of 0.592, marginally surpassing the performance of the logistic regression model. 
# MAGIC 
# MAGIC ![Figure 6.5.1.A: ROC Graph for MultiLayer Perceptron Classifier](https://github.com/ColStaR/sparksandstripesforever/blob/main/images/MLPNN_bestModel_ROC.jpg?raw=true)
# MAGIC 
# MAGIC *Figure 6.5.1.A: ROC Graph for MultiLayer Perceptron Classifier*
# MAGIC 
# MAGIC ![Figure 6.5.1.B: Scores by Threshold Graph for MultiLayer Perceptron Classifier](https://github.com/ColStaR/sparksandstripesforever/blob/main/images/MLPNN_ScoresByThresh.jpg?raw=true)
# MAGIC 
# MAGIC *Figure 6.5.1.B: Scores by Threshold Graph for MultiLayer Perceptron Classifier*
# MAGIC 
# MAGIC ### 6.5.2 Most Influential Feature
# MAGIC 
# MAGIC Because this model operates on a neural network, there is no way to determine feature importances.
# MAGIC 
# MAGIC ### 6.5.3 Model Notebook and Code
# MAGIC 
# MAGIC To view the full code for the Logistic Regression Feature Engineering model, please review the notebooks in the appendix, section 14.7.2, "Link to Modelling Notebook".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.6 Random Forest Model
# MAGIC 
# MAGIC The Random Forest model was one of the first models that we had implemented after the Logistic Regression model. However, after conducting initial testing, experiencing numerous technical issues, and receiving some additional guidance, the decision was made to no longer pursue the random forest model any further in order to focus on more effective models. As such, only one cross validation of Random Forest was run on the final set of features, and no test evaluations or feature importance analyses were conducted for that model. While the model was not utilized to completion, it is being reported here for record-keeping and discussion's sake.
# MAGIC 
# MAGIC ### Data Encoding Methods Used
# MAGIC 
# MAGIC For the Random Forest Model, all categorical features were string indexed and one-hot encoded. While recognized as not being ideal, the categorical values were forced to be one-hot encoded due to an unknown technical issue at the time. 
# MAGIC 
# MAGIC All numeric features were not scaled nor normalized.
# MAGIC 
# MAGIC ### List of Parameters Used
# MAGIC 
# MAGIC - Number of Trees = 10
# MAGIC - Max Depth = 4
# MAGIC - Max Bins = 32
# MAGIC - Threshold = 0.5
# MAGIC 
# MAGIC ### Cross Validation Run-Times
# MAGIC 
# MAGIC The single run of cross-validation listed below took 2.53 hours to complete.
# MAGIC 
# MAGIC ### 6.6.1 Experimental Results
# MAGIC 
# MAGIC | val_Precision  | val_Recall  | val_F0.5  | val_F1   | val_Accuracy  | year  | numTrees  | maxDepth  | maxBins  | threshold  |
# MAGIC |----------------|-------------|-----------|----------|---------------|-------|-----------|-----------|----------|------------|
# MAGIC | 0.455658       | 0.982655    | 0.510404  | 0.622611 | 0.463598      | 2019  | 10        | 4         | 32       | 0.5        |
# MAGIC | 0.470379       | 0.970134    | 0.524407  | 0.633567 | 0.48018       | 2018  | 10        | 4         | 32       | 0.5        |
# MAGIC | 0.438627       | 0.984745    | 0.493347  | 0.606919 | 0.447544      | 2017  | 10        | 4         | 32       | 0.5        |
# MAGIC | 0.481061       | 0.9417      | 0.533227  | 0.636811 | 0.494418      | 2016  | 10        | 4         | 32       | 0.5        |
# MAGIC | 0.388531       | 0.999983    | 0.442666  | 0.559627 | 0.388564      | 2020  | 10        | 4         | 32       | 0.5        |
# MAGIC 
# MAGIC *Table 6.6.1.A: CV Folds Results Table for Random Forest*
# MAGIC 
# MAGIC ### 6.6.2 Top Performing Model
# MAGIC 
# MAGIC While there might not be much final development behind this random forest model, we can see that the best performing model for the random forest model would be the model created from the cross validation fold for the year 2016. This model trained on the 2016 training data had a training validation F0.5 score of 0.533227 and a precision score of 0.481061. While this score may not seem too bad at first, we would also expect the test validation score for this model to be lower than this test validation score. From previous iterations of this model, F0.5 training validation scores tended to be in the range between 0.20 and 0.55.
# MAGIC 
# MAGIC ### 6.6.3 Model Notebook and Code
# MAGIC 
# MAGIC To view the full code for the Logistic Regression Feature Engineering model, please review the notebooks in the appendix, section 14.8, "Link to Random Forest, Linear SVM Modelling Notebook".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.7 Linear Support Vector Machines Model
# MAGIC 
# MAGIC The Linear Support Vector Machines model was the third model that we had implemented, after Logistic Regression and Random Forest. Several versions of the model were created, but initial testing consistently showed that the model had poor predictive performance even on the latest dataset. As such, the decision was made to no longer pursue the linear support vector machine any further in order to focus on more effective models. While several cross validations were done as part of the model's hyperparameter tuning, no test evaluations or feature importance analyses were conducted for that model. While the model was not utilized to completion, it is being reported here for record-keeping and discussion's sake.
# MAGIC 
# MAGIC Recall that we had intended to implement non-linear SVM, but were unable to since MLLib does not offer that type of SVM. As such, we created this linear SVM with the expectation that its predictive performance would be diminished.
# MAGIC 
# MAGIC ### Data Encoding Methods Used
# MAGIC 
# MAGIC For the Linear Support Vector Machines model, all categorical features were string indexed and one-hot encoded.
# MAGIC 
# MAGIC All numeric features were not scaled nor normalized.
# MAGIC 
# MAGIC ### List of Parameters Used
# MAGIC 
# MAGIC - Regularization Parameter = 0.0, 0.01, 0.5
# MAGIC - Max Iterations = 1, 5, 10
# MAGIC - Thresholds = 0.5, 0.6
# MAGIC 
# MAGIC ### Cross Validation Run-Times
# MAGIC 
# MAGIC For a single cross validation run, logistic regression would require between 7 - 20 minutes, with the mean value being approximately 15 minutes. Completing all of the cross validations to build the models below took approximately 2.28 hours.
# MAGIC 
# MAGIC ### 6.7.1 Experimental Results
# MAGIC 
# MAGIC | precision   | f0.5        | recall      | accuracy    | regParam | maxIter | threshold |
# MAGIC |-------------|-------------|-------------|-------------|----------|---------|-----------|
# MAGIC | 0.034181175 | 0.042144283 | 0.61859136  | 0.815763131 | 0.5      | 1       | 0.6       |
# MAGIC | 0.068966094 | 0.083667986 | 0.568020992 | 0.816400023 | 0.5      | 1       | 0.5       |
# MAGIC | 0.288695319 | 0.311521653 | 0.455620087 | 0.802817204 | 0.5      | 5       | 0.5       |
# MAGIC | 0.240201706 | 0.267757819 | 0.494824155 | 0.812378394 | 0.5      | 5       | 0.6       |
# MAGIC | 0.363497292 | 0.376385597 | 0.438588705 | 0.794313225 | 0.01     | 5       | 0.5       |
# MAGIC | 0.337100693 | 0.356809432 | 0.465724596 | 0.804053535 | 0        | 5       | 0.6       |
# MAGIC | 0.366146333 | 0.378334453 | 0.43644756  | 0.793410166 | 0        | 5       | 0.5       |
# MAGIC | 0.333815399 | 0.354149656 | 0.468240536 | 0.804862785 | 0.01     | 5       | 0.6       |
# MAGIC | 0.328600912 | 0.351519053 | 0.487529163 | 0.810178141 | 0.01     | 10      | 0.6       |
# MAGIC | 0.220375696 | 0.244887107 | 0.441160338 | 0.802342266 | 0.01     | 1       | 0.6       |
# MAGIC | 0.332331119 | 0.354467808 | 0.483216736 | 0.809006834 | 0        | 10      | 0.6       |
# MAGIC | 0.267438901 | 0.28750648  | 0.410808351 | 0.791637167 | 0.01     | 1       | 0.5       |
# MAGIC | 0.400274575 | 0.403744073 | 0.418245127 | 0.784139386 | 0        | 10      | 0.5       |
# MAGIC | 0.356146481 | 0.373051717 | 0.460482829 | 0.801905129 | 0.01     | 10      | 0.5       |
# MAGIC | 0.359232075 | 0.375296667 | 0.457052942 | 0.800713361 | 0        | 10      | 0.5       |
# MAGIC | 0.248131181 | 0.279191923 | 0.559183315 | 0.823121813 | 0.5      | 10      | 0.5       |
# MAGIC | 0.267438901 | 0.28750648  | 0.410808351 | 0.791637167 | 0        | 1       | 0.5       |
# MAGIC | 0.220375696 | 0.244887107 | 0.441160338 | 0.802342266 | 0        | 1       | 0.6       |
# MAGIC | 0.228508586 | 0.258370448 | 0.541346501 | 0.819832792 | 0.5      | 10      | 0.6       |
# MAGIC 
# MAGIC *Table 6.7.2.A: CV Folds Results Table for Support Vector Machines*
# MAGIC 
# MAGIC ### 6.7.2 Top Performing Model
# MAGIC 
# MAGIC While this model was not evaluated against the test dataset, we can see that the best performing model for the linear SVM model would be the model created with the following parameters: regularization parameter of 0, maximum iterations of 10, and a threshold of 0.5 . This model has a training validation F0.5 score of 0.403744073 and a precision score of 0.400274575. While this score may not seem too bad at first, we would also expect the test validation score for this model to be lower than this test validation score. 
# MAGIC 
# MAGIC ### 6.7.3 Model Notebook and Code
# MAGIC 
# MAGIC To view the full code for the Logistic Regression Feature Engineering model, please review the notebooks in the appendix, section 14.8, "Link to Random Forest, Linear SVM Modelling Notebook".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.8 Logistic Regression - Non-Weather Feature Set Experiment
# MAGIC 
# MAGIC After recognizing how important weather was in feature importance for our baseline logistic regression model, we wanted to investigate the effect on a model's predictive powers if the model was trained without that weather data. As such, we used the best model from the baseline logistic regression model to create an experiment where the model would be trained on a feature set without any weather features.
# MAGIC 
# MAGIC ### Data Encoding Methods Used
# MAGIC 
# MAGIC For the logistic regression model, all categorical features were string indexed and one-hot encoded. Furthermore, all numeric features were scaled using the built-in standard scaler in the LogisticRegression function.
# MAGIC 
# MAGIC ### List of Parameters Used
# MAGIC 
# MAGIC regParam = 0.0, elasticNetParam = 0.0, maxIter = 5, threshold = 0.6
# MAGIC  
# MAGIC ### Cross Validation Run-Time
# MAGIC 
# MAGIC The cross validation for this run took 21 minutes.
# MAGIC 
# MAGIC ### 6.8.1 Experimental Model Results
# MAGIC 
# MAGIC | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy | cv_fold | regParam | elasticNetParam | maxIter | threshold |
# MAGIC |----------------|-------------|-------------|-------------|---------------|---------------|-------------|-------------|-------------|--------------|---------|----------|-----------------|---------|-----------|
# MAGIC | 0.657086484    | 0.249687135 | 0.495417838 | 0.361867699 | 0.834663524   | 0.810412024   | 0.334081917 | 0.630593461 | 0.473124396 | 0.729746786  | 0       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.618903725    | 0.292301636 | 0.505859662 | 0.397070911 | 0.833336303   | 0.782080871   | 0.343854599 | 0.623226409 | 0.47768653  | 0.735994016  | 1       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.626336197    | 0.303382515 | 0.516394738 | 0.408767616 | 0.835227823   | 0.8000814     | 0.342485099 | 0.631366832 | 0.479649907 | 0.707679891  | 2       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.614001492    | 0.298534513 | 0.506876355 | 0.401738968 | 0.833062885   | 0.558943608   | 0.187697482 | 0.400510122 | 0.281024736 | 0.855415451  | 3       | 0        | 0               | 5       | 0.6       |
# MAGIC 
# MAGIC *Table 6.8.1.A: Experimental Results for Logistic Regression - Non-Weather Feature Set*
# MAGIC 
# MAGIC ### 6.8.2 Most Important Features
# MAGIC 
# MAGIC | featureName                           | coefficient  | importance  |
# MAGIC |---------------------------------------|--------------|-------------|
# MAGIC | is_prev_delayed_class_0.0             | -1.714405699 | 1.714405699 |
# MAGIC | DEP_HOUR_class_04                     | -1.219327646 | 1.219327646 |
# MAGIC | DEP_HOUR_class_05                     | -1.21372786  | 1.21372786  |
# MAGIC | DEP_HOUR_class_06                     | -1.166412925 | 1.166412925 |
# MAGIC | DEP_HOUR_class_07                     | -0.884830124 | 0.884830124 |
# MAGIC | AssumedEffect_Text_class_Christmas_p2 | 0.858492179  | 0.858492179 |
# MAGIC | AssumedEffect_Text_class_Christmas_p3 | 0.800329133  | 0.800329133 |
# MAGIC | DEP_HOUR_class_08                     | -0.510369826 | 0.510369826 |
# MAGIC | MONTH_class_7                         | 0.45786144   | 0.45786144  |
# MAGIC | DEP_HOUR_class_19                     | 0.40117288   | 0.40117288  |
# MAGIC 
# MAGIC *Table 6.8.2.A: Most Important Features for Logistic Regression - Non-Weather Feature Set*
# MAGIC 
# MAGIC ### 6.8.3 Model Notebook and Code
# MAGIC 
# MAGIC To view the full code for the Logistic Regression Feature Non-Weather Feature Set model, please review the notebooks in the appendix, section 14.9, "Link to Logistic Regression - No-Weather Feature Set Experiment Modelling Notebook".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.9 Logistic Regression - Only-Weather Feature Set Experiment
# MAGIC 
# MAGIC After recognizing how important weather was in feature importance for our baseline logistic regression model, we wanted to investigate the effect on a model's predictive powers if the model was trained with only that weather data. As such, we used the best model from the baseline logistic regression model to create an experiment where the model would be trained on a feature set with only weather features.
# MAGIC 
# MAGIC ### Data Encoding Methods Used
# MAGIC 
# MAGIC For the logistic regression model, all categorical features were string indexed and one-hot encoded. Furthermore, all numeric features were scaled using the built-in standard scaler in the LogisticRegression function.
# MAGIC 
# MAGIC ### List of Parameters Used
# MAGIC 
# MAGIC regParam = 0.0, elasticNetParam = 0.0, maxIter = 5, threshold = 0.6
# MAGIC  
# MAGIC ### Cross Validation Run-Time
# MAGIC 
# MAGIC The cross validation for this run took 18 minutes.
# MAGIC 
# MAGIC ### 6.9.1 Experimental Model Results
# MAGIC 
# MAGIC | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy | cv_fold | regParam | elasticNetParam | maxIter | threshold |
# MAGIC |----------------|-------------|-------------|-------------|---------------|---------------|-------------|-------------|-------------|--------------|---------|----------|-----------------|---------|-----------|
# MAGIC | 0.508196923    | 0.038438925 | 0.147552428 | 0.071471871 | 0.813547825   | 0.582155296   | 0.025248464 | 0.107579194 | 0.048397879 | 0.63921744   | 0       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.488937251    | 0.038814174 | 0.147297998 | 0.07191907  | 0.812988442   | 0.719656244   | 0.045215901 | 0.180672894 | 0.085085868 | 0.658615992  | 1       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.483116024    | 0.047494141 | 0.170445947 | 0.08648602  | 0.812696612   | 0.754791804   | 0.070379858 | 0.256303986 | 0.128754155 | 0.625762246  | 2       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.51434062     | 0.039776448 | 0.151895106 | 0.073842313 | 0.813730413   | 0.408194162   | 0.046294011 | 0.159234023 | 0.083157037 | 0.846440871  | 3       | 0        | 0               | 5       | 0.6       |
# MAGIC 
# MAGIC *Table 6.9.1.A: Experimental Results for Logistic Regression - Only-Weather Feature Set*
# MAGIC 
# MAGIC ### 6.9.2 Most Important Features
# MAGIC 
# MAGIC | featureName            | coefficient  | importance  |
# MAGIC |------------------------|--------------|-------------|
# MAGIC | HourlyPrecipitation    | 2.78782843   | 2.78782843  |
# MAGIC | Freezing_Rain_class_0  | -1.572490817 | 1.572490817 |
# MAGIC | Thunder_class_0        | -1.024342543 | 1.024342543 |
# MAGIC | Blowing_Snow_class_0   | 0.397398254  | 0.397398254 |
# MAGIC | Snow_class_0           | -0.205575905 | 0.205575905 |
# MAGIC | HourlySeaLevelPressure | 0.204385586  | 0.204385586 |
# MAGIC | Rain_class_0           | -0.131849779 | 0.131849779 |
# MAGIC | HourlyAltimeterSetting | -0.058443236 | 0.058443236 |
# MAGIC | HourlyVisibility       | -0.057101717 | 0.057101717 |
# MAGIC | HourlyWindSpeed        | 0.018368722  | 0.018368722 |
# MAGIC 
# MAGIC *Table 6.9.2.A: Most Important Features for Logistic Regression - Only-Weather Feature Set*
# MAGIC 
# MAGIC ### 6.9.3 Model Notebook and Code
# MAGIC 
# MAGIC To view the full code for the Logistic Regression Feature Non-Weather Feature Set model, please review the notebooks in the appendix, section, "14.10 Link to Logistic Regression - Only-Weather Feature Set Experiment Modelling Notebook".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.10 Logistic Regression - Ensemble Prediction Experiment
# MAGIC 
# MAGIC Model ensembling is a method of machine learning where multiple models are trained and used in conjunction to predict an outcome. Different machine learning algorithms often have different strengths and weaknesses, which can mean that they may pick up on some patterns that other models would not. There are a number of ways in which ensembling can be implemented: majority vote, unanimous vote, and any vote (at least one positive prediction). In some cases, different models may perform better on sub-categories of data. In such instances, ensemble models can be implemented such that one model is used for each sub-category of data (eg. one model for long-distance flights, and one for short-distance flights, etc.). Such approaches can significantly improve predictive performance. 
# MAGIC 
# MAGIC ### Models Used In Ensembling
# MAGIC 
# MAGIC For this ensembling experiment, we used the top performing Logistic Regression, Gradient Boosted Tree, and Multilayer Perceptron models detailed in the previous sections. Furthermore, we attempted three different ensembling approaches: majority vote, unanimous vote, and "at-least-one" vote. We accomplished this by using each trained and validated model to predict on the 2021 data, and averaged their predictions together. We then identified flights with an average greater than 0.5 as a majority vote prediction, equal to one as a unanimous vote prediction, and greater than zero as an "at-least-one" vote prediction.
# MAGIC  
# MAGIC ### Ensemble Method Run-Times
# MAGIC 
# MAGIC Performing the averages across the predictions for the 2021 dataset takes approximately 30 seconds. That said, training each of these three models took nearly 3 hours in total. The ensemble appraoch may be better suited towards algorithms that are quick to train.
# MAGIC 
# MAGIC ### Results
# MAGIC 
# MAGIC The table below details the test performance for each of the three models on their own, as well as the performance of each of the three ensemble approaches.
# MAGIC 
# MAGIC | Model                 | test_Precision | test_Recall | test_F0.5 | test_F1  | test_Accuracy |
# MAGIC |-----------------------|----------------|-------------|-----------|----------|---------------|
# MAGIC | Logistic Regression   | 0.575585       | 0.342514    | 0.506635  | 0.429465 | 0.829182      |
# MAGIC | Gradient Boosted Tree | 0.60977        | 0.335231    | 0.523952  | 0.432621 | 0.834952      |
# MAGIC | Multilayer Perceptron | 0.53993        | 0.386779    | 0.500309  | 0.450699 | 0.823035      |
# MAGIC | Ensemble Majority     | 0.60694        | 0.338126    | 0.523675  | 0.434302 | 0.834662      |
# MAGIC | Ensemble At Least One | 0.509535       | 0.40544     | 0.484649  | 0.451567 | 0.815145      |
# MAGIC | Ensemble Unanimous    | 0.631332       | 0.320957    | 0.529017  | 0.425565 | 0.837361      |
# MAGIC 
# MAGIC *Table 6.10.A: Ensembling Method and Model Metrics Comparison*
# MAGIC 
# MAGIC We find that, of the ensemble models, the unanimous vote method helps to achieve slightly higher precision and F0.5 scores, though slightly at the expense of recall. That said, this ensemble method only offers a marginal increase in performance in comparison to the GBT model, suggesting that the additional time and resources to train the logistic regression and multilayer perceptron models may not be worthwhile.
# MAGIC 
# MAGIC ### 6.10.3 Model Notebook and Code
# MAGIC 
# MAGIC To view the full code for the Logistic Regression Feature Non-Weather Feature Set model, please review the notebooks in the appendix, section 14.11, "Link to Logistic Regression - Ensemble Prediction Experiment Modelling Notebook".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.11 Logistic Regression - Downsampling Ratio Experiment
# MAGIC 
# MAGIC The models detailed in the previous sections were all trained based on a downsampled dataset that had a final ratio of delays to non-delays of 1:2. Performing this downsampling helped to improve model performance in comprasion to models that were trained on the full dataset (which had a ratio of 1:4 delays to non-delays). We performed an additional experiment to adjust the downsampling rate such that the final training dataset would have a more even disribution of delays and non-delays (with a ratio of about 1:1.5).
# MAGIC 
# MAGIC After performing this enhanced downsampling approach, our training dataset was significantly smaller with approximately 6.5 million delays and 10 million non-delays between 2016-2020. 
# MAGIC 
# MAGIC ### Model Set-Up
# MAGIC 
# MAGIC For this experiment, we used the same features, encodings, and parameters as the best performing logistic regression model. All categorical features were string indexed and one-hot encoded. Furthermore, all numeric features were scaled using the built-in standard scaler in the `LogisticRegression` function. The model was trained using the following parameters:
# MAGIC - regParam = 0
# MAGIC - elasticNetParam = 0
# MAGIC - maxIter = 5
# MAGIC - threshold = 0.6
# MAGIC 
# MAGIC ### Model Results
# MAGIC 
# MAGIC The table below details the test and validation metrics for the logistic regression model that was trained on the increased downsampled dataset.
# MAGIC 
# MAGIC | cv_fold | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy |
# MAGIC |---------|----------------|-------------|-------------|-------------|---------------|---------------|-------------|-------------|-------------|--------------|
# MAGIC | 0       | 0.555845414    | 0.353866554 | 0.498893981 | 0.432433799 | 0.825643352   | 0.846718455   | 0.395883251 | 0.689643973 | 0.539515844 | 0.636086787  |
# MAGIC | 1       | 0.43004456     | 0.40097616  | 0.423898533 | 0.415001966 | 0.787810004   | 0.779942473   | 0.47338661  | 0.690510295 | 0.589173791 | 0.652641039  |
# MAGIC | 2       | 0.528596973    | 0.369470071 | 0.486675741 | 0.434935816 | 0.819800337   | 0.842309237   | 0.411923073 | 0.696719671 | 0.553273275 | 0.630528856  |
# MAGIC | 3       | 0.500832445    | 0.372577232 | 0.468572369 | 0.427288067 | 0.812529093   | 0.549208001   | 0.359174435 | 0.496653667 | 0.434313711 | 0.706825666  |
# MAGIC 
# MAGIC *Table 6.11.A: Logistic regression - Downsampling Ratio Model Metrics*
# MAGIC 
# MAGIC Interestingly, while we find that the validation metrics have improved on the smaller dataset, the test metrics have worsened. This indicates that the model is overfitting, likely as a result of the decreased variance in the non-delayed flights after downsampling. To address this, we attempted to train the model again using different regularization and elastic net parameters, but ultimately found that they had even worse validation and test performance. Thus, we have maintained the downsampling ratio resulting in a distribution of delays to non-delays of 1:2. 
# MAGIC 
# MAGIC For the full table of the grid-search results for the Logistic Regression - Downsampling Experiment model, please check the appendix for section 14.12.1, "Cross Validation Results Table".
# MAGIC 
# MAGIC ### Model Notebook and Code
# MAGIC 
# MAGIC To view the full code for the Logistic Regression - Downsampling Experiment model, please review the notebooks in the appendix, section 14.12.2, "Link to Modelling Notebook".

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 7. Discussion of Experimental Results

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 7.1 Overall Top Model and Performance
# MAGIC 
# MAGIC ### 7.1.1 The Top Model
# MAGIC 
# MAGIC After the conclusion of our experiments, we have found that the Gradient Boosted Tree model with the parameters maxIter = 5, maxDepth = 4, maxBins = 32, stepSize = 0.5, and threshold = 0.6 performed the best at predicting flight delays. As shown in Figure 6.3.5.A, this model boasts a maximum training validation F0.5 score of 0.587 and a test evaluation F0.5 score of 0.526. This model has the highest score for those metrics of all of the models tested, though we do note that this score is not overly impressive. With training validation precision of 0.734, a training validation recall of 0.326, test evaluation precision of 0.623, and a test evaluation recall of 0.323, the model apparently does avoid false positive cases, but utterly falters in avoiding false negatives. Furthermore, the difference in the recall and precision values suggests that the model is conservative in its predictions when labeling flights as delayed, which would lead to many cases of users being told that their flights are on-time when they were ultimately delayed. Also, while there is very little difference in the F0.5 scores between validation and test datasets, there is a more substantial difference in the precision values between the two, which may indicate the presence of over-fitting in the model.
# MAGIC 
# MAGIC ### 7.1.2 Top Model's Most Important Features
# MAGIC 
# MAGIC The best model's most important features are shown in figure 6.3.6. `is_prev_delayed_idx` is the most important feature with a coefficient of 0.426, followed by `DEP_HOUR_idx` with an importance of 0.250, and `OP_UNIQUE_CARRIER_idx` with an importance of 0.0870. Of the top ten most important metrics, four are related to weather, three are related to time and date, two are related to specific airlines, and one is related to tracking previous delays.
# MAGIC 
# MAGIC ### 7.1.3 Error Analysis for Top Model
# MAGIC 
# MAGIC As previously noted, our Gradient Boosted Tree provided the best results with a F0.5 of 0.526 and a precision of 0.623 for test evaluation. This is within the average range of strong team performance (see section 8 Gap Analysis for detail). Further analyzing our results reveal the following: 
# MAGIC 
# MAGIC 1. 5,665,055 records are in our blind dataset
# MAGIC 
# MAGIC 2. Features with the highest importance are: `is_prev_delayed` (flight tracker), `DEP_HOUR` (hour of the planned departure), `OP_UNIQUE_CARRIER`, and `HourlyPrecipitation`
# MAGIC 
# MAGIC 3. 1.84% of the blind dataset was dropped due to new values which were unseen to our training dataset. As a future opportunity, we would explore methods to include these records, such as creating clustering models to impute values for new categorical entry 
# MAGIC  
# MAGIC <center><img src="https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-12-04_19-33-02.png" width="800"/></center>
# MAGIC 
# MAGIC *Figure 7.1.3.A: Count and Percentage of Records Dropped*
# MAGIC 
# MAGIC 4. The percentage split of our results between true positives, true negatives, false postivies and false negatives fall within our range of expectation. Speficially, since we placed a heavier emphasis on minimizing false positives with our F0.5 and precisions, the fact that we have twice as many false negatives as we do false positives validates that our model is tuned well to our metrics.
# MAGIC 
# MAGIC 
# MAGIC <center><img src="https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-12-04_19-33-15.png" width="800"/></center>
# MAGIC 
# MAGIC *Figure 7.1.3.B: Result type percentage summary*
# MAGIC 
# MAGIC 
# MAGIC Expanding further upon our top 3 features, we observed the following:
# MAGIC 
# MAGIC `DEP_HOUR`: This family of features is created from the scheduled departure hour of a flight. As seen in the figure below, minimum delays are observed between 11PM – 5AM for both the predicted results (broken down into fn for false negative, fp for false positive, tn for true negative, and tp for true positive) and the actual results (calculated using `DEP_DEL15`). Between 6AM-11AM we observe moderate occurrences of flight delays, and we see that there are significantly more false negatives than there are false positives. During peak flight delay hours (2PM – 8PM), the proportion of false positives and false negatives are more balanced. We are not surprised by this result where we see a higher proportion of false negatives compared to false positives given our priority to maximize F0.5 and precision. That said, the proportion of false negatives to false positives changes during the day, which suggests certain hours of the days are more influential to the model than others and that other features might have a stronger pull on the model performance during less predictive hours.   
# MAGIC 
# MAGIC ![Figure 7.1.3.C: DEP Hour and Flights Delay Graph](https://github.com/ColStaR/sparksandstripesforever/blob/main/images/dep_hour_count_graph.png?raw=true)
# MAGIC 
# MAGIC *Figure 7.1.3.C: DEP Hour and Flights Delay Graph*
# MAGIC 
# MAGIC ![Figure 7.1.3.D: DEP Hour and Flight Delay Predictions Count Graph](https://github.com/ColStaR/sparksandstripesforever/blob/main/images/dep_hour_predictions_graph.png?raw=true)
# MAGIC 
# MAGIC *Figure 7.1.3.D: DEP Hour and Flight Delay Predictions Count Graph*
# MAGIC 
# MAGIC `is_prev_delayed`: This family of features is created to track whether the aircraft for a given flight is delayed due to a delay with the prior flight. We observe that when the previous flight is delayed, that is a highly predictive signal that the next flight will also be delayed.
# MAGIC Again we can see that there is a higher proportion of false negatives when the previous flight wasn’t delayed, which makes sense given our metrics priority. It is interesting that the proportion of true positives and false positives when the prior flight was not delayed is similar. This can be explained partly by our EDA analysis, such that the flight tracker is more predictive when the previous flight was delayed. When focusing solely on cases where the prior flight is delayed, we noticed only a small proportion of false negatives. As such, we believe that the `is_prev_delayed` variable behaves consistently between the holdout dataset and the training dataset. 
# MAGIC 
# MAGIC ![Figure 7.1.3.E: Flight_Tracker_Previous_Delay_Counts](https://github.com/ColStaR/sparksandstripesforever/blob/main/images/flight_tracker_actual_graph.png?raw=true) 
# MAGIC 
# MAGIC *Figure 7.1.3.E: Flight_Tracker_Previous_Delay_Counts*
# MAGIC 
# MAGIC ![Figure 7.1.3.E: Flight_Tracker_Previous_Delay_Counts](https://github.com/ColStaR/sparksandstripesforever/blob/main/images/Flight_tracker_predictions_graph.png?raw=true) 
# MAGIC 
# MAGIC *Figure 7.1.3.E: Flight_Tracker_Previous_Delay_Counts*
# MAGIC 
# MAGIC 
# MAGIC `OP_UNIQUE_CARRIER`: as noted previously from our correlation analysis, we noticed that the percentage effectiveness of airlines (percentage of flights delayed) against the outcome variable changes across the year. Specifically, we saw that the correlation strength dropped in 2020 and 2021 as a part of recovery from the major industry disruptor (Covid). As such, we were not surprised to see that although the effectiveness of the airline still played an important role in the modeling, the proportion of false positives was higher than true positives across big airlines such as AA and DL (see figure below). This confirms our correlation trend analysis, suggesting that our model performance could improve by further analyzing airline effectiveness across years for airlines of different sizes. 
# MAGIC 
# MAGIC ![Figure 7.1.3.F: Airline Effectiveness Predictions Graph](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/2022-12-04_19-32-10.png)
# MAGIC 
# MAGIC *Figure 7.1.3.F: Airline Effectiveness Predictions Graph*

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## 7.2 Model Performance Comparison
# MAGIC 
# MAGIC ### 7.2.1 Baseline Logistic Regression and Feature Engineered Logistic Regression
# MAGIC 
# MAGIC | Metrics        | Baseline Logistic Regression | FE Logistic Regression | Improvement over Baseline | Percent Improvement over Baseline |
# MAGIC |----------------|------------------------------|------------------------|---------------------------|-----------------------------------|
# MAGIC | test_Precision | 0.328                        | 0.575585               | 0.2476                    | 75%                               |
# MAGIC | test_Recall    | 0.081                        | 0.342514               | 0.2615                    | 323%                              |
# MAGIC | test_F0.5      | 0.197                        | 0.506635               | 0.3096                    | 157%                              |
# MAGIC 
# MAGIC *Table 7.2.1.A Test Evaluation Comparison for Baseline Logistic Regression and  Feature Engineered Logistic Regression*
# MAGIC 
# MAGIC The feature engineered logistic regression model's best model performed significantly better than the baseline logistic regression model. In test evaluation, the feature engineered logistic regression model's F0.5 score was 0.209 points higher and the precision score was 0.249 points higher. On average, every metric measured in the feature engineered logistic regression model was an improvement over the baseline model. However, the feature engineered logistic regression model took on average about 50% longer to train using cross validation than the baseline logistic regression.
# MAGIC 
# MAGIC ### 7.2.2 Baseline Logistic Regression and Gradient Boosted Trees
# MAGIC 
# MAGIC | Metrics        | Baseline Logistic Regression | Gradient Boosted Tree | Improvement over Baseline | Percent Improvement over Baseline     |
# MAGIC |----------------|------------------------------|-----------------------|---------------------------|------|
# MAGIC | test_Precision | 0.328                        | 0.60977               | 0.2818                    | 86%  |
# MAGIC | test_Recall    | 0.081                        | 0.335231              | 0.2542                    | 314% |
# MAGIC | test_F0.5      | 0.197                        | 0.523952              | 0.3270                    | 166% |
# MAGIC 
# MAGIC 
# MAGIC *Table 7.2.2.A Test Evaluation Comparison for Baseline LR and Gradient Boosted Trees*
# MAGIC 
# MAGIC The gradient boosted trees model's best model performed significantly better than the baseline logistic regression model. In test evaluation, the gradient boosted trees model's F0.5 score was 0.197 points higher and the precision score was 0.164 points higher. Every single metric measured in the gradient boosted trees model was an improvement over the baseline model. However, the gradient boosted tree model took on average about 50% longer to run a cross validation than the baseline logistic regression.
# MAGIC 
# MAGIC ### 7.2.3 Baseline Logistic Regression and Multiclass Preceptron Classifier Neural Network
# MAGIC 
# MAGIC | Metrics        | Baseline Logistic Regression | Multilayer Perceptron | Improvement over Baseline | Percent Improvement over Baseline |
# MAGIC |----------------|------------------------------|-----------------------|---------------------------|-----------------------------------|
# MAGIC | test_Precision | 0.328                        | 0.53993               | 0.2119                    | 65%                               |
# MAGIC | test_Recall    | 0.081                        | 0.386779              | 0.3058                    | 378%                              |
# MAGIC | test_F0.5      | 0.197                        | 0.500309              | 0.3033                    | 154%                              |
# MAGIC 
# MAGIC *Table 7.2.3.A Test Evaluation Comparison for Baseline LR and Multiclass Preceptron Classifier Neural Network*
# MAGIC 
# MAGIC The multiclass preceptron classifier neural network model's best model performed significantly better than the baseline logistic regression model. In test evaluation, the multiclass preceptron classifier model's F0.5 score was 0.225 points higher and the precision score was 0.250 points higher. On average, every metric measured in the multiclass preceptron classifier model was an improvement over the baseline model. However, the neural network model took dramatically more time to run a cross validation than the baseline logistic regression: the 3-layer architecture model took 1440% longer, and the 4-layer architecture model took 3220% longer.
# MAGIC 
# MAGIC ### 7.2.4 Gradient Boosted Trees and Multiclass Preceptron Classifier Neural Network
# MAGIC 
# MAGIC | Metrics        | Gradient Boosted Tree | Multilayer Perceptron | MLP Improvement over GBT | Percent Improvement MLP over GBT     |
# MAGIC |----------------|-----------------------|-----------------------|--------------------------|------|
# MAGIC | test_Precision | 0.60977               | 0.53993               | -0.0698                  | -11% |
# MAGIC | test_Recall    | 0.335231              | 0.386779              | 0.0515                   | 15%  |
# MAGIC | test_F0.5      | 0.523952              | 0.500309              | -0.0236                  | -5%  |
# MAGIC 
# MAGIC 
# MAGIC *Table 7.2.4.A Test Evaluation Comparison for Gradient Boosted Trees and Multiclass Preceptron Classifier Neural Network*
# MAGIC 
# MAGIC The gradient boosted trees model's best model performed slightly better than the multiclass preceptron classifier neural network. In training validation, the gradient boosted trees model's F0.5 score was 0.001 point higher and the precision score was 0.037 points lower. In test evaluation, the gradient boosted trees model's F0.5 score was 0.028 points higher and the precision score was 0.086 points higher. Furthermore, the GBT model took only 15 minutes to train while the 3-layer MLP took 144 minutes and the 4-layer MLP took 322 minutes, a 960% incrrease and a 2147% increase respectively. 
# MAGIC 
# MAGIC In test validation for F0.5 score, gradient boosted trees and multiclass preceptron classifiers were nearly identical, with gradient boosted trees succeeding with a very narrow margin. However, the reduction in precision score for training validation indicates that GBT has an increased number of false positives that is offset by a much larger reduction in false negatives. However, when analyzing the significant difference in each model's performance in test evaluation, it becomes clear that the GBT model fares better and is more generalizable than the MLP. And considering that the GBT model takes between 1/10th to 1/20th of the time to train and deploy for better performance, it becomes clear that the gradient boosted tree model performs better than the multiclass preceptron classifier neural network for this purpose.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 7.3 Features and Feature Importance
# MAGIC 
# MAGIC Every model in this experiment, with the exception of MLP NN and the unfinished models, have had their most important features and associated importance coefficients listed in the model's details. For our analysis, we focused our feature importance analysis against the logistic regression model (feature importance corresponds to the strength of a feature’s coefficient) and the gradient boosted tree model (feature importance corresponds to how frequently a feature was used to make key decisions within the decision tree). Given the differences in methods for calculating feature importance, we were not surprised to observe that although both methods provide a list of similar important features, their relative feature importance ranking differs. Specifically, where we saw more categorical fields rated as having higher importance for the gradient boosted tree model (e.g. is flight previously delayed indicator, scheduled departure hour, and airlines), the logistic regression model has numeric features such as hourly precipitation as higher ranked. Regardless of the ranking difference, we identified a few important patterns on the features that exert the greatest amount of influence across our models.
# MAGIC 
# MAGIC ## 7.3.1 Most Important Features and Patterns
# MAGIC 
# MAGIC Of the models used in this experiment that have feature importances listed, it is clear that some types of features and certain individual features are shown to consistently exert a high level of influence on the model's predictions. For example, the feature `HourlyPrecipitation` can be considered the most important feature in our models overall. Of the four models that include that feature, it is the most important feature in three models and the fourth-most important feature the other. The only model where it is not shown is in the no-weather feature set model, because `HourlyPrecipitation` was dropped due to it being a weather feature. Therefore, we can conclude that incidences of precipitation greatly increase the likelihood of flights being delayed.
# MAGIC 
# MAGIC The next most influential group of features are ones that track the recent history of a flight being delayed. The features `is_prev_delayed_class_0.0` and `is_prev_delayed_idx` appeared as the most important features in two of the four features that included it. This indicates that if a flight has not been previously delayed recently, then it is likely to stay on-time and not be delayed. The inverse implication of this condition is that flights that have been delayed previously are likely to continue being delayed for future flights as well. Therefore, knowing whether or not a flight was previously delayed is another valuable and predictive feature that can help determine if the next flight will be delayed.
# MAGIC 
# MAGIC The third most influential group of features describe the hour that the flight is scheduled to depart. For example, the features that indicate the flight is scheduled to depart in the hours between 4 AM, 5 AM, 6 AM, and 7 AM occupy the second through fifth most important features in two of the three models that include those features. For the Gradient Boosted Tree, `DEP_Hour_idx` is the second most important feature. And of the three models that one-hot encode individual flight hour times, hour indicator features account for six of the top ten features. This strongly suggests that there are both slow periods and rush hours that consistently influence whether or not a flight scheduled to depart at that time will be delayed or not.
# MAGIC 
# MAGIC The last of the most important group of features would be extreme weather indicators. Beyond the `HourlyPrecipitation` feature mentioned above, there are separate indicators for rain, snow, freezing rain, blowing snow, and thunder. This suggests that weather features that are suggestive of storms and extreme weather are associated with increased flight delays.
# MAGIC 
# MAGIC ## 7.3.2 Impact of Engineered Features on Model Performance
# MAGIC 
# MAGIC One of the most difficult and time-consuming tasks of this project was the creation of engineered features that would assist in our model's predictive abilities. However, whether or not the engineered features are of a benefit or not can only be understood by comparing the model performance with and without these features. In order to compare the impact of the engineered features on model performance, we can compare the model performance between the baseline logistic regression model and the feature engineered logistic regression model. Recall that the baseline model contains only a selection of features from the raw dataset, while the feature engineered logistic regression model includes the created features that were used for the other models. We can measure and compare the influence of the engineered features by comparing the training validation and test validation metrics between both models.
# MAGIC 
# MAGIC In training validation, the best feature engineered model had an F0.5 score that was 0.163 points higher, a precision score that was 0.22 points higher, and a recall score that was 0.59 points higher. In test evaluation, the best feature engineered model had an F0.5 score that was 0.209 points higher, a precision score that was 0.249 points higher, and a recall score that was 0.128 points higher. In nearly every metric, the feature engineered model improved over the baseline model performance. This indicates that the created features included in the feature set do indeed capture valuable information that increases the model's predictive power.
# MAGIC 
# MAGIC However, it should be noted that not all of the created features were beneficial when added to the analysis. In previous iterations of model building, several additional created features had been included in the analysis, but led to a net decrease in model performance metrics afterwards. Only after further research and experimentation were we able to isolate and remove the offending features (e.g. `is_prev_diverted`, see section 3.3, "Feature Engineering" for detail), thereby improving our model by limiting the feature selection to the features being used now. Thus, it is not always beneficial to add more and more created features, as the cost of doing so is both an increase in dimensionality, training time, and noise in the data, resulting in lower model effectiveness.
# MAGIC 
# MAGIC ## 7.3.3 Weather Features and Predictive Power
# MAGIC 
# MAGIC As mentioned in the section 7.3.1, "Most Important Features and Patterns", the weather features are shown to be very influencial in our trained models. Furthermore, historical and current weather data is very abundant and accessible compared to airline flight data or weather station data. In an effort to improve prediction accuracy or responsiveness, one might consider using only the weather data to predict flight delays. In contrast, the weather data was so dirty, messy, and of questionable quality that one might consider dropping the data and not using it at all. Both of these situations beg the question as to how the weather data influences the flight data performance metrics. In order to understand the value of the weather features, we will compare the model performance metrics of the best logistic regression model that includes the full feature set, an identical model that is trained on a feature set without weather features, and another identical model that is trained on a feature set containing only weather features.
# MAGIC 
# MAGIC | Metric         | Full Feature Set | No Weather | Only Weather |
# MAGIC |----------------|----------|------------|--------------|
# MAGIC | val_f0.5       | 0.639    | 0.631      | 0.256        |
# MAGIC | val_precision  | 0.798    | 0.800      | 0.755        |
# MAGIC | val_recall     | 0.356    | 0.342      | 0.070        |
# MAGIC | test_f0.5      | 0.513    | 0.516      | 0.170        |
# MAGIC | test_precision | 0.625    | 0.626      | 0.483        |
# MAGIC | test_recall    | 0.298    | 0.303      | 0.047        |
# MAGIC 
# MAGIC *Figure 7.3.3.A: Weather Feature Set Experiments Comparison Metrics*
# MAGIC 
# MAGIC First, we can compare the feature engineered logistic regression model with the no-weather feather logistic regression model. In training validation, the best feature engineered logistic regression model had an F0.5 score that was 0.008 points higher, a precision score that was 0.002 points lower, and a recall score that was 0.014 points higher. In test evaluation, the best feature engineered logistic regression model had an F0.5 score that was 0.04 points lower, a precision score that was 0.002 points lower, and a recall score that was 0.005 points lower. Considering the increase in F0.5 score in the training validation data and the accompanying decrease in the test evaluation data, this suggests that the model is not as generalizable as the no-weather data. This may be because the feature engineered model is overfit to its data, and the no-weather feature set has enough noise to avoid overfitting and thus be more generalizable. While the overall impact between the two models is minor, this would indicate that the inclusion of the weather features is relatively harmless at best, but may be diminishing the model's predictive powers at a very small scale at worst.
# MAGIC 
# MAGIC Secondly, we can compare the feature engineered logistic regression model with the only-weather feather logistic regression model. In training validation, the best feature engineered logistic regression model had an F0.5 score that was 0.383 points higher, a precision score that was 0.043 points higher, and a recall score that was 0.286 points higher. In test evaluation, the best feature engineered logistic regression model had an F0.5 score that was 0.342 points higher, a precision score that was 0.142 points higher, and a recall score that was 0.251 points higher. Given these dramatic metric differences, it may be tempting to assume that this shows the worth of the non-weather features, but given how poor the only-weather metrics are, this suggests the opposite is true: weather data has very poor predictive power on its own. 
# MAGIC 
# MAGIC With these findings in mind, it is clear that the feature engineered model performed the best, the no-weather model performed slightly worse, and the only-weather model performed generally poorly. The relatively similar metrics between the feature engineered model and the no-weather model indicates that the weather data, while introducing some useful signal to the model, also introduces an equivalent or slightly greater amount of noise to offset this. However, the weather data only works well when paired with another related data set, since the only-weather model had very poor performance if the weather features were not supported by other features. To this end, it is clear that the weather data is a worthy inclusion for the models to use when paired with other data sets, but it must be utilized properly and leveraged effectively lest it hinder the model's predictive powers.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 7.4 Influence of Hyperparameter Tuning on Performance
# MAGIC 
# MAGIC ### 7.4.1 Logistic Regression
# MAGIC 
# MAGIC For the logistic regression models, we adjusted the regularization parameter, elastic net parameter, maximum iteration, and threshold. In general, it appears that regularization, elastic net parameter, and maximum iteration only make minor changes to the metrics, though they generally seemed to diminish the model metrics. Threshold made the most dramatic influence, as increasing the threshold level dramatically decreased all of the metrics simultaneously. The addition of parameters diminishing performance metrics is exemplified in the fact that the best model for logistic regression used all of the lowest level of each parameter (regParam = 0, elasticNetParam = 0, maxIter = 5, threshold = 0.5). This trend in parameters suggests that the model may be underfitted to the data, and thus increasing the parameters to increase the amount of noise only worsens the tightness of the model. Further investigation into lower levels of parameters may improve model metrics.
# MAGIC 
# MAGIC 
# MAGIC ### 7.4.2 Multiclass Perceptron Classifier Neural Network
# MAGIC 
# MAGIC For the MLP NN model, we had intended to adjust the maximum iteration number, block size, step size, and threshold parameters, along with the model's layer architecture. However, due to time constraints, we were unable to conduct any tuning with the maximum iteration number, block size, step size parameters, but we could adjust the threshold parameter and the layer architecture. As threshold increases, precision increases decently, but recall drops much more dramatically, causing an overall drop in overall model metrics. For the model layer architectures, the 3-layer architecture performs substantially worse than the 4-layer architecture in nearly all metrics except for accuracy (which were quite similar), and run-time (4-layer cost 2.5x longer than 3-layer). Further investiation into lower threshold levels and more sophisticated neural network layer architectures may improve model metrics. 
# MAGIC 
# MAGIC ### 7.4.3 Gradient Boosted Trees
# MAGIC 
# MAGIC For the gradient boosted trees models, we adjusted the maximum iteration, maximum depth, maximum bins, step size, and threshold parameters. In general, it appears that maximum iteration, maximum depth, step size, and maximum bins only make minor changes to the metrics, though they generally seemed improve and diminish the model metrics in equal measure for a net neutral outcome. Threshold made the most dramatic influence, as increasing the threshold level dramatically decreased all of the metrics simultaneously. The only other change in model performance is the run-time, as the models that were run with a step size of 0.5 had a runtime 1.5x faster than models with a step size of 0.1. The best model for gradient boosted trees used all of the lowest level of each parameter (maxIter = 5, maxDepth = 4, maxBins = 32, stepSize = 0.5, threshold = 0.6) suggests that the model may be overfitting to the data; the increase of the threshold value to 0.6 indicates that the optimal tuning point is near, since dropping the threshold to 0.5 tuned the model past the point of optimality. Further investigation into lower levels of parameters may improve model metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 8. Gap Analysis with Similar Projects
# MAGIC 
# MAGIC As a part of our iterative improvement process, our models were continuously compared against the project leader board. As such, our final model approach focused on the following: 
# MAGIC 1. Join: Our post join dataset had around 41 million records, which is in the upper quantiles among the leaderboard teams. We are happy with our methodology (corrected for time zone and tracks weather across a wide range of stations at the vicinity of an airport) and result. As such, we are highly confident that our models are trained on data that represents the US domestic flight pattern well
# MAGIC 2. Metrics used: our original intention was to optimize precision as our primary metrics and F1 as our secondary metrics. However, this gave us non-meaningful results as our model had a tendency to predict no flight delay.  With a need to balance precision and recall and taking inspiration from the leaderboard seeing teams using F-beta measurement, we fine-tuned our metrics to use F0.5 as our primary metrics (better suited for our business need than F1 given it has more emphasis on precision than recall) and precision as our secondary metrics.
# MAGIC 3. Model selection: XGBoost was created as it was a common model among performing teams
# MAGIC 4. Engineered features: flight trackers (previously delayed flights) and special weather indicators were common features that stood out as having importance. As such, we focused our feature engineering tasks to capture weather and flight tracking such that there were no risks of data leakage.
# MAGIC 5. Number of features: most teams have features ranging from 20-50. We targeted our number of features within this range and used 27 families of features for our final model.
# MAGIC 6. Pipeline: Watching some of the presentations made us realize that we had not implemented feature scaling or dataset balancing in our pipeline. As such, we implemented these steps into our pipeline, improving our results.
# MAGIC 
# MAGIC When comparing our final model result (Gradient Boosted Tree provided the best 2021 test results, with an F0.5 of 0.526 and a precision of 0.623) against the leaderboard, we focused on comparable teams who had similar quality of joined rows as us. Specifically, we removed teams with fewer than 40 million post-join records, and teams who did not have F-score or precision in their metrics. In addition, we focused our analysis on the work produced from their latest phases. As such, our model performance evaluation was performed against the following 7 teams: FP_Section2_Group1, FP_Section3_Goup12, Team 2-1, The Puffins, Group 3 (Team 13), Team 10, FP_Section4_Group4. Our key findings are: 
# MAGIC 1. Although many teams used F0.5 score and precision from prior phases, only 1 of these 7 teams (FP_Section4_Group4) continued to use F0.5 and precision as their metrics. Most of the teams selected F1 as their core metrics, and a few selected precision as their supporting metrics
# MAGIC 2. Our model results (F0.5 and precision) are very similar to the team that had the same metrics as us, which is a strong indication that the approaches we took are comparable. That said, it is interesting to note that our choices of models and top features are different. Whereas our best model was the Gradient Boosted tree, their team used logistics regression and log loss. Furthermore, whereas our flight tracker feature has the highest feature importance, their top features are based on average weather statistics. This is an inspiration for future work where we can explore creating more engineered features with rolling average weather statistics.
# MAGIC 3. Our secondary metrics result, precision, is 0.623. This makes us stand around the 67th percentile among the 4 of the 7 selected teams who provided precision (ranging 0.28 to 0.79) and beat the average precision of 0.56. This further illustrates that among the teams with high retention joins, our model is performing well against benchmarks.
# MAGIC 
# MAGIC In conclusion, we feel comfortable with our joined data set, improved data pipeline, and modified metrics. Our results also place us well against the benchmarks of teams that had a similar quality of join. We did notice that between the final model leaderboard and interim model leaderboard, many teams switched models and metrics, which is another indication that machine learning is an interactive process. As such, we recognize there are opportunities to further improve our model performance (see section 11.2 Future Work for more discussion). 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 9. Performance and Scalability Concerns
# MAGIC 
# MAGIC Large data projects such as this require computation and storage of massive amounts of data, leading to performance and scalability being major concerns. Some of the issues can be mitigated with modern tools, such as cloud storage, cloud computing, and distributed systems. However, even mitigated risks can be dangerous, and as such it is essential that such performance and scalability concerns are properly identified and managed.
# MAGIC 
# MAGIC ## 9.1. Performance Concerns
# MAGIC 
# MAGIC On a sufficiently-large dataset, even the simpest computation and action on that dataset can take an exorbitant amount of time. With the amount of data this project handles, ensuring that the algorithms perform optimally with as minimal performance overhead can make the difference between training a model within a few minutes or over the span of several days. Maintaining a rapid run-time performance with the tasks involved in a data project such as this is essential to the project's feasibility. To that end, the following items have been identified as possible points of concern that could impact the program's run-time performance if neglected:
# MAGIC 
# MAGIC ### 9.1.1 Computationally-Expensive Spark Operations
# MAGIC 
# MAGIC Apache Spark and its built-in libraries have been optimized such that their distributed systems operate with as minimal amounts of computational and memory overhead as possible. However, the distributed nature of Spark means that certain operations are inevitably more expensive in computation and memory. Sorting operations are particularly costly for run-time, as these complicated operations require the traversal, organization, and storage of a large amount of data. Our algorithms make frequent use of expensive operations, such as sorting Spark data frames within each fold of a cross validation, and expensive operations such as these will naturally incur time costs.
# MAGIC 
# MAGIC We have done our best to mitigate this issue by attempting to minimize the amount of times that such expensive operations occur within our code. We have been successful in removing many examples of redundant operations. Furthermore, since the process of processing dataframes through our pipeline is quite time-intensive, but saving and loading files in the parquet format to and from our Azure blob storage, we could achieve significant time savings by saving and loading data into Azure blob storage. However, as the amount of data being introduced into the system grows, so too will the run-time for Spark operations on that data.
# MAGIC 
# MAGIC ### 9.1.2 Dimensionality of Feature Data
# MAGIC 
# MAGIC We experienced first-hand how influential dimensionality is on run-time when training models. In initial attempts to train the data, we used a selection of features that unknowingly featured an exorbidant amount of dimensionality. This caused our models to train so slowly that they appeared as though they had malfunctioned entirely. After discovering the dimensionality issue and dropping just three worst-offending features, the number of one-hot-encoded features dropped from 860 to only 90. This reduced amount of dimensionality caused models that used take hours to train to be fully trained within minutes. However, the risk still exists that as more data and new features are added to the dataset, the dimensionality of the data will naturally increase as a result. If not managed properly, the dimensionality could increase to the same point we were at before, thus causing such slow-down that the algorithms seemingly did not work.
# MAGIC 
# MAGIC As in our example, we have mitigated this issue by actively identifying features with high levels of dimensionality but low levels of influence. Once found, those features are removed from our analysis so that they no longer impact run-times.
# MAGIC 
# MAGIC ### 9.1.3 DataBricks Cluster Compute Resources
# MAGIC 
# MAGIC DataBricks relies on computing clusters to operate, and it assigns a range of worker nodes to each cluster for a user to use. This provides great flexibility if you need to upscale to do things more quickly by acquiring more workers, but the opposite is also true: workers can be removed from your cluster, thereby making your operations slower. We have seen workers being split between two running notebooks when being run simltaneously on the same cluster. We have also seen workers be reallocated elsewhere during peak times so that they can attend to other, higher-priority clusters. In both of these cases, the lack of workers greatly diminished the speed at which operations completed.
# MAGIC 
# MAGIC In an effort to maintain as many workers as possible, our team strategized and coordinated our efforts so that leverages our workers most effectively. Tasks that can be completed quickly or are of low priority are allowed to be run simultaneously across different notebooks, thereby splitting our workers to use them more optimally. For high-intensity tasks such as training models, our team ensures that the person assigned to train models has sole control over those workers. Furthermore, we have determined that the best time to run operations is both overnight and during the morning until about noon, as that is when DataBricks reports the least amount of activity, and therefore is least likely to allocate workers away from our cluster.
# MAGIC 
# MAGIC ### 9.1.4 Inherent Model Run-Times
# MAGIC 
# MAGIC All machine learning models are not created equally, and one aspect that makes them unique is how much time is required for them to train on a dataset. Simple models, such as logistic regression, can be trained on a large dataset in only a matter of minutes. Meanwhile, more complex models such as Multiclass Perceptron Classifier Neural Networks may require multiple hours to train on the same dataset. The amount of time required to train each model will vary drastically between models even with the same amounts of data, and that is a factor that must be accounted for.
# MAGIC 
# MAGIC Model training run-time could be adjusted by decreasing the number of data provided to it, though that is not an optimal solution beyond the downsampling amounts as this would cause needless information loss. Changing the parameters being used with the models, such as step size, did prove to be an effective tool for modifying the amount of time models required to run, but their effects were usually limited and did not provide dramatic improvements without changing model effectiveness. Perhaps the most effective method we created to account for the natural run-time of models is to simply get estimates of how long the average run for a model would take, and then schedule our time accordingly.
# MAGIC 
# MAGIC ## 9.2. Scalability Concerns
# MAGIC 
# MAGIC This data project already handles a large dataset. But as time goes on, more data can be accumulated and incorporated into the project's existing datasets. While this additional data may provide useful benefits for the project, it does present a major issue relating to scalability with so much new data. Seeing as storage and memory are vital resources to computers, running out of either will lead to catastrophic results for the project's programming. Ensuring that the algorithms perform optimally with minimal memory overhead can make the difference between conducting a successful Spark operation or model training, and having the entire cluster crash because it ran out of memory. To that end, the following items have been identified as possible points of concern that could impact the program's ability to function as the project's data increases in scale.
# MAGIC 
# MAGIC ### 9.2.1 Expensive Spark Operations
# MAGIC 
# MAGIC Similar to the above Section 9.1.1, "Computationally-Expensive Spark Operations", some of those same Spark operations are also expensive in terms of memory usage. For example, caching a Spark dataframe allows that dataframe to be stored and reserved in the system's memory, speeding up program run-times by making that dataframe readily available at all times. However, if not properly managed, a sufficiently-large amount of cached Spark data could consume all of the memory within a cluster, thus causing it to crash when the system attempts to do an operation without sufficient memory space.
# MAGIC 
# MAGIC As mentioned above, the best way that we addressed this issue is to be mindful of cached data, caching when valuable and freeing that cached data when no longer needed. Furthermore, the acquisition of a more powerful cluster also provided more memory resources, which did help mitigate the issue as well.
# MAGIC 
# MAGIC ### 9.2.2 Noisy, Undesirable Features
# MAGIC 
# MAGIC When beginning the project, we adopted a naive approach to the project that believed that additional features would only make our models more powerful and predictive. However, further investigation revealed that not all of the features we were analyzing were helpful. In fact, removing those features actually helped improve model performance, as those models were introducing more noise than useful information for the models. However, if the project continued with that same naive approach, the project would grow in scale with a lot of noisy and undesirable features that would only hurt our model's overall performance.
# MAGIC 
# MAGIC We mitigated this risk by analyzing the feature importance and dimensionality of each feature on our models. Using this information, we then decided what features were important and influential enough to justify keeping, and which ones should be removed.
# MAGIC 
# MAGIC ### 9.2.3 Azure Blob Storage Limitations
# MAGIC 
# MAGIC As mentioned in Section 4.2, "Tools Used in Data Pipeline", this project leverages Microsoft Azure to provide blob storage for our data. Blob storage is an excellent solution for providing accessible and reliable data storage for machine learning projects such as this, but it is still a limited computational resource that can be used up. As the project increases in scale and grows in the amount of storage used, it is possible for the blob storage to be filled to its limit, thus preventing any further data from being saved to the blob storage.
# MAGIC 
# MAGIC While the data needs for this project were far below the maximum of what the Azure instance could handle, we were nonetheless careful when saving and judicious when removing needless data from storage. In the event that the blob storage ever was filled to the breaking point, the Azure instance can also be upgraded with additional storage space. Lastly, storage space can be increased by removing or archiving data that is no longer needed for immediate access on the blob storage.
# MAGIC 
# MAGIC ### 9.2.4 Time Drift
# MAGIC 
# MAGIC Because of the time-series nature of this analysis, behavior and patterns can change over time. Models can quickly become irrelevant if they aren't maintained and updated with current data on a regular basis. There is sufficient evidence from the yearly correlation analyses (detailed in Section 3) to suggest that flight behavior can change significantly from year to year. Moreover, drastic events, such as COVID or 9/11, can have major unforseen effects that can cause model performance to drop overnight. Additional analysis must be conducted to understand the extent of drift in the data and model performance with time. 
# MAGIC 
# MAGIC ### 9.2.5 Data Streaming
# MAGIC 
# MAGIC The goal of this analysis is to assist travelers as they prepare to travel, leading up to two hours prior to their flight. This requires current data to be available on demand in order to accurately track delays. Although the data may not have to be updated every minute, it likely would need to be updated every ten minutes in order to provide relevant and timely predictions to users. Maintaining such a database, as well as generating updated predictions for each flight may be very costly. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 10. Project Challenges And Limitations

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 10.1 Project Challenges
# MAGIC 
# MAGIC As to be expected from a project of this scale, our group faced several challenges from beginning to end over a variety of topics.
# MAGIC 
# MAGIC ### 10.1.1. Learning Challenges
# MAGIC 
# MAGIC Tackling the problems presented by this project required a comprehensive approach which utilized the entirety of our data science experience within the program, self-study, and professional lives. When a solution to an issue at hand was not immediately apparent, we turned to independently researching possible solutions and seeking advice from peers, instructors, and teaching assistants.
# MAGIC 
# MAGIC #### 10.1.1.1 Learning new platforms: Azure, DataBricks, MLLib, etc.
# MAGIC 
# MAGIC Conducting analyses on the scale presented by this project required us to utlize platforms we were unfamiliar with. It was critical for us to learn and understand the DataBricks platform for housing and conducting any experiments we sought to execute. While using DataBricks was similar to our previous work with other data science IDE's/platforms (Jupyter NB, Google Colab, etc.), we found not all of our prior experience was seamlessly translated to the new platform. Such differences include the Pandas library not being applicable to Pyspark DataFrames without modification, running the notebook out of a cluster, and the newfound precaution to be wary of editing the notebook while other users are active on the page.
# MAGIC 
# MAGIC To store important data related to the project, our group was required to learn the Microsoft Azure cloud platform. Despite our unfamiliarity with the platform, we utilized Azure's cloud storage functionality to build data lakes that would house our raw datasets, joined and cleaned datasets, and dataframes of performance metrics. Each of these items would be saved as a parquet file within a notebook and then stored on the cloud's blob storage.
# MAGIC 
# MAGIC Because of our reliance on Apache Spark, we tasked ourselves with becoming familiar with MLLib. This allowed us to utilize machine learning models on distributed systems. However, because of the various intricacies and deficiencies of this library, we found ourselves having to create custom functionalities to address any use cases that would not work with the default MLLib functions.
# MAGIC 
# MAGIC #### 10.1.1.1 Learning big data pipeline process
# MAGIC 
# MAGIC Learning to create a pipeline from scratch was not explicity shown to us on a granular level prior to the project. Working to create one on datasets on the scale presented here was a difficult, but rewarding task.
# MAGIC 
# MAGIC ### 10.1.2. Data Challenges
# MAGIC 
# MAGIC With datasets that are sourced from various entities, we encountered data that was not always consistent, reliable, nor understandable. We were faced with the challenge of finely combing through the data itself, as well as the data documentation, to find features, definitions, and relationships of interest.
# MAGIC 
# MAGIC #### 10.1.2.1 Messy Raw Data
# MAGIC 
# MAGIC We found several fields with reliability concerns during our time working on this project, particularly in the weather dataset, such as:
# MAGIC - egregious outliers (HourlyWindSpeed)
# MAGIC - missing data (HourlyWindGustSpeed, HourlyPressureChange)
# MAGIC - incorrect data (HourlyWindGustSpeed) 
# MAGIC - ambigious data (HourlyDryBulbTemperature vs HourlyWetBulbTemperature)
# MAGIC 
# MAGIC #### 10.1.2.2 Incorrect data documentation
# MAGIC 
# MAGIC In examining the documentation for the provided for the datasets within the scope of the project, we found inconsistencies with the definitions and examples for features within the documentation when comparing with the features available in the actual dataset. There were several instances where a field outlined in the documentation would be recorded differently in the data then documented, or where the feature would be missing entirely. We often found ourselves having to go beyond the provided documentation and conducting our own research to discern the meanings of fields and their application, as well as how they might impact a flight's likelihood to be delayed.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 10.1.3. Technical Challenges
# MAGIC 
# MAGIC #### 10.1.3.1 Implementing custom cross validation and accompanying functions
# MAGIC 
# MAGIC The MLLib library we used did not include the cross validation functionality natively, forcing us to create it from scratch. This meant we had to implement custom solutions for other, auxiliary MLLib functions to facilitate our use of these custom functions.
# MAGIC 
# MAGIC #### 10.1.3.2 Implementing ROC curves
# MAGIC 
# MAGIC ROC Curve creation was not uniformly available as a built-in function across all of our models and alternatives such as SKlearn were used instead.
# MAGIC 
# MAGIC #### 10.1.3.2 GraphFrames nonfunctional
# MAGIC 
# MAGIC During the process for calculating the PageRank for the destination airports, we found an issue implementing GraphFrames, and subsequently PageRank into our notebooks and were forced to seek assistance in finding a solution. Ultimately we found that the issue stemmed from our current provided cluster having an incompatibility with the GraphFrames library. We ultimately found ourselves using a temporary, weaker, cluster that could successfully conduct the PageRank calculations we were seeking. This was a time consuming process that hampered further GraphFrames usability in our project.
# MAGIC 
# MAGIC #### 10.1.3.3 SMOTE nonfunctional
# MAGIC 
# MAGIC We found SMOTE was not usable for the problem seen in this project.
# MAGIC 
# MAGIC #### 10.1.3.4 Addressing dimensionality to improve run times.
# MAGIC 
# MAGIC With several users in the course competing for the same resources to conduct any calculations or experiments, as well as a general lack of available time, we were tasked with drastically limiting the dimensionality of our experiments in efforts to see and report results before given deadlines.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 10.2 Project Limitations
# MAGIC 
# MAGIC Similarly to a realistic project that could be taken in the data science industry, there were various limitations we encountered while taking on this project. Many of these limitations were similar to what could be encountered in a practical, professional setting.
# MAGIC 
# MAGIC ### 10.2.1 Time Constraints
# MAGIC 
# MAGIC With check-in phases due every week or few weeks, we had to ensure any time spent was used wisely while remaining cogniscent of required deliverables for any given phase. This meant that, while there was a nigh endless amount of desirable functionalites, features, and experiments to create, we had to remain focused on what could be completed and delivered by a deadline. This served well in protecting against scope creep within our group. Ultimately, given more time in each phase, we believe we could have put forth a more comprehensive approach to the problem of predicting airline delays than we already have.
# MAGIC 
# MAGIC ### 10.2.2 MLLib Missing Functionality
# MAGIC 
# MAGIC Because of our reliance on MLLib as the primary data science library used throughout the project, we had issues implementing several functions or models that might have benefitted our experimenting process. 
# MAGIC 
# MAGIC Such examples include:
# MAGIC - Non-Linear SVM 
# MAGIC - Lacking loss curves functions
# MAGIC - MulticlassMetricClassifier not having metrics for Random Forest 
# MAGIC 
# MAGIC ### 10.2.2 Data Quality Concerns
# MAGIC 
# MAGIC As mentioned throughout this project, the performance of our model is limited based on the quality of data we have been provided. When data such as that present in the weather, stations, and airport datasets, is collected from several sources, the likelihood of misinputted data increases, which may explain certain outlier data collected. Additionally, the weather dataset had features reporting unrealistic values, such as a wind speed greater than anything ever recorded, or other strange values. Our concern is that if we could see odd behaviors on the extremes, would it be possible for there to be other instances of incorrect data within what we deem to be normal data. Ultimately, if our data is unreliable, we will not be able to produce a model of applicable effectiveness to this problem of prediction.
# MAGIC 
# MAGIC ### 10.2.3 Limited Compute Resources
# MAGIC 
# MAGIC Throughout our time working on the project, we could see the number of available workers for our cluster fluctuating frequently, particularly during the later phases where these numbers were on the lower end of the range. This constant fluctation lead to inconsistencies in run-times, ultimately affecting our ability to plan tasks efficiently. Additionally, the throttling of cluster resources in later phases impacted run-times at what was considered a critical time in the project's execution. These limitations regarding our computational resources likely affected the amount of work able to be conducted during the project and can be considered one of our greater limitations. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 11. Open Issues and Future Work

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 11.1 Open Issues
# MAGIC 
# MAGIC As expected with a project of this scale, even at its conclusion we have areas of possible expansion and further work.
# MAGIC ### 11.1.1 Models Cannot Predict On Unseen Data Categories
# MAGIC 
# MAGIC One limitation of our model is not being able to predict flight delays on all unseen data categories. Given the time that we have, although were able to partly address this issue with some of our engineered features (e.g. new airline's perc_delay is imputed as the average perc_delay across all airlines from the prior year), we were not able to implement such data handling across all categorical features such as new airports. As such, these flights with new categorical values will have poor prediction result. 
# MAGIC 
# MAGIC ### 11.1.2 Messy Source Data
# MAGIC 
# MAGIC Due to the nature of the messy weather data, there were several entries and values that seemed infeasible. These may be the result of incorrect input or faulty instrument readings. While outliers are easy to distinguish, there is often no way to identify such occurences of incorrect data. Although the hope is that the occurence of faulty data is infrequent, there are few robust ways of confirming this or measuring its effects on model performance.
# MAGIC 
# MAGIC ### 11.1.3 Parameters Not Optimal
# MAGIC 
# MAGIC Because our hyperparameter tuning was completed in a user-input grid-search fashion, there is high likelihood that models could have been optimizer further using a different method, such as Random Search (as discussed in Section 5.4.1). With additional time and resources, further experimentation can be completed on testing different hyperparemeter tuning methods.
# MAGIC 
# MAGIC ### 11.1.4 Cancelled Flights Treated As Delayed Flights
# MAGIC 
# MAGIC Because our analysis caters to airline travelers and their flight experiences, we opted to treat cancelled flights as delayed flights. But there may be characteristics inherent to cancelled flights that makes them distinct from delayed flights, perhaps causing confusion to the models during training. We may find that treating these datapoints separately may help improve model performance. Alternatively, we may also consider creating a multi-class classification model that can predict different outcomes of travel experiences (for example, delayed arrivals, diversions, etc.)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 11.2 Future Work
# MAGIC 
# MAGIC ### 11.2.1 Feature Weighting Across Years
# MAGIC As observed in from our section 3.2.1 "EDA on Joined Dataset", the strengh of association of some of our key features against the target variable change overtime. This is particulary true with the airline effectiveness where its strength of association is less prevelant in 2020 (introduction of a major industry disruptor: pandemic travel restriction) and 2021 (recovery from a major industry disruptor) and effect of previous flight being delayed (much higher association in 2020 and 2021 compared to the pre-pandmic years). As such, we believe that exploring with feature weighting across years could also improve our model result. Most machine learning algorithms accept a parameter for weighting, which allows the datapoint to be considered more or less important relative to other datapoints. Thus, we could implement a scheme where flights are weighted by their temporal relevance (i.e. flights from the most recent years would be weighted more heavily than those from more distant years). Alternatively, we could consider behavioral relevance instead, where years that were most similar to the current year are weighted more heavily. This method would be best suited to situations where behavior is returning to normalcy from a deviation (e.g. post-COVID era).
# MAGIC 
# MAGIC ### 11.2.3 Improved Ensembling
# MAGIC 
# MAGIC We tested multiple ensembling methods using our three different trained algorithms in conjunction. That said, we did not try ensembling within each algorithm. One possible appraoch could be to use each trained model across the folds during cross-validation as an ensemble. This could help remove the effect from any outlier years (like COVID-era), and can help further reduce the risks of over-fitting. This kind of approach can be more easily scalable in comparison to ensembles across multiple algorithms since it relies on use of already-trained models from the cross-validation step, and does not require additional training time for other models. 
# MAGIC 
# MAGIC ### 11.2.4 Model Saving
# MAGIC 
# MAGIC Because the MLLib trained models did not seem to support storage to cloud services like Azure Blob Storage, and MLLib models were unsupported in pickle files, we were unable to maintain a log of the previous models trained. This meant that every time we wanted to access a prior model, we would have to spend additional time re-training them. With further time for debugging and testing, we may be able to better track and compare model iterations, which could help us more effectively compare different algorithms and feature sets.
# MAGIC 
# MAGIC ### 11.2.5 Additional Models
# MAGIC 
# MAGIC Although we were able to implement Support Vector Machine and Random Forest models, we did not have sufficient time to exhaustively test and hyperparameter-tune these models. Although we do not expect them to surpass the performance of the Gradient Boosted Tree or Multilayer Perceptron models, it may be worth exploring these algorithms as an addition to the ensemble modeling appraoch. Furthermore, there are numerous models available outside of the MLLib library; Sci-kit Learn, PyTorch, and Tensorflow all support many other algorithms that are much more powerful. Although these models may not be feasible to implement in PySpark, they can be applied to a smaller subset of data (in memory) with possibly equal or superior results. 
# MAGIC 
# MAGIC ### 11.2.6 Additional Features
# MAGIC 
# MAGIC With additional time, we would like to pursue the creation of the following additional features:
# MAGIC 
# MAGIC - *Weather Forecasting and rolling weather average* 
# MAGIC   - Using weather data from the previous several hours prior to the flight, we can calculate the rate of change per hour to forecast weather patterns closer to the flight departure time. Alternatively, many weather stations provide very robust weather predictions, which can be scraped from onlinen sources. This data could prove to be strongly predictive and can help improve predictions further than two hours in advance of flight departure times.
# MAGIC - *Destination Airport Weather*
# MAGIC   - Our existing data sources currently provide weather information for the flight destination locations. We could improve our data join by including the weather at the destination prior to the flight departure. This could help account for any flights that are delayed due to storms at the destination as opposed to the origin. 
# MAGIC - *Type of airplane*
# MAGIC   - Older planes may be more prone to malfunctions that could cause flight delays or cancelations. Data on the type of plane for a flight could be used to better predict flights that are delayed due to causes other than weather or airport traffic. 
# MAGIC - *Centrality of airports using GraphFrames*
# MAGIC   - With our graphical representation of airports and flights, we implemented PageRank as a feature for our models. We did not explore any other graph data measures, however, including centrality or betweenness. 
# MAGIC - *Binning of existing features*
# MAGIC   - We had numerous categorical and numeric features that we implemented in our models. Although we did so in some features (like `airline_type`), we did not fully explore binning of other features, particularly numeric features. For many of the weather metrics, we could split them into buckets for each standard deviation from the location's monthly average. This would help differentiate weather patterns that are atypical for a particular region so that the continuous weather metrics are not washed out by the large weather variance across the United States. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 12. Conclusion
# MAGIC 
# MAGIC Our project aimed to create a tool that would allow travelers to better optimize their time with flight delay information. This was achieved via creating machine learning models to predict whether a flight would be delayed. We trained our models on flight, weather, station data and measured our model success on F0.5 and precision. We were successful in creating a Gradient Boosted Trees model with accompanying parameters that offered a significant improvement over the baseline logistic regression model. The baseline model's test evaluation scores were 0.197 for F0.5 and 0.328 for precision; the best model's test evaluation scores were 0.526 for F0.5 and 0.623 for precision, an improvement of 166% and 86% respectively. Furthermore, we discovered that the previous flight delay indicators, scheduled flight departure time, and extreme weather indicators were highly important features when predicting flight delays.
# MAGIC 
# MAGIC In comparison to other groups conducting similar experiments, we found that our features, models, and overall metrics used were very similar. This indicates that our findings stand well against the benchmark without incurring major flaws in our data, pipeline, or overall process. Therefore, our model has proven to possess a decent level of predictive ability when predicting flight delays, especially in comparison to the baseline. With more time and resources, the models can further be improved through: additional hyperparameter tuning, including additional models (e.g.  Random Forest and Support Vector Machines), implementing new features such as predicted weather.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 13. Special Thanks, Project Team Credits, and Project Material

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 13.1 Special Thanks
# MAGIC 
# MAGIC Professor Vinicio De Sola, for instruction and assistance
# MAGIC 
# MAGIC Mai La, for helping us with clusters to run GraphFrames for PageRank
# MAGIC 
# MAGIC Max Eagle, for assistance with dimensionality

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 13.2 Project Team Credits and Information
# MAGIC 
# MAGIC #### Section, Group Number: Section 4, Group 1
# MAGIC 
# MAGIC #### Team Number: Team 13
# MAGIC 
# MAGIC #### Team Name: Sparks and Stripes Forever
# MAGIC 
# MAGIC #### Team Members:
# MAGIC - Nashat Cabral (cabralnc96@berkeley.edu)
# MAGIC - Deanna Emery (deanna.emery@berkeley.edu)
# MAGIC - Nina Huang (ninahuang2002@berkeley.edu)
# MAGIC - Ryan S. Wong (ryanswong@berkeley.edu)
# MAGIC 
# MAGIC UC Berkeley MIDS Program
# MAGIC 
# MAGIC W261, Fall 2022
# MAGIC 
# MAGIC ### Phase Leader Plan
# MAGIC 
# MAGIC Per the instructions, each weekly phase was lead by a different member of the team. Below is a table showing the planned leader for each of the project phases.
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
# MAGIC 
# MAGIC ### Credit Assignment Plan
# MAGIC 
# MAGIC [Link to Official Spreadsheet (Requires UC Berkeley Account)](https://docs.google.com/spreadsheets/d/1A4N3sV1ngaBsPdUqcJ8R4gY6e2g3gHeIUDyDes7f4SU/edit#gid=549854846)
# MAGIC 
# MAGIC ### Project Contributions
# MAGIC 
# MAGIC | Task                                                   | Start Date | End Date   | Estimated Time (hours) | Nashat Cabral | Deanna Emery | Nina Huang | Ryan S. Wong |
# MAGIC |--------------------------------------------------------|------------|------------|------------------------|---------------|--------------|------------|--------------|
# MAGIC | Project Abstract Section                               | 10/23/2022 | 10/30/2022 | 1                      | X             |              | X          |              |
# MAGIC | Data Overview Section                                  | 10/23/2022 | 10/30/2022 | 4                      | X             |              |            |              |
# MAGIC | The Desired Outcomes and Metrics Section               | 10/23/2022 | 10/30/2022 | 2                      |               |              |            | X            |
# MAGIC | Data Ingesting and Pipeline Section                    | 10/23/2022 | 10/30/2022 | 4                      |               |              |            | X            |
# MAGIC | Joining Datasets Section                               | 10/23/2022 | 10/30/2022 | 4                      |               | X            |            |              |
# MAGIC | Machine Learning Algorithms to be Used Section         | 10/23/2022 | 10/30/2022 | 2                      |               | X            |            |              |
# MAGIC | Resource Management & Performance Optimization Section | 10/23/2022 | 10/30/2022 | 4                      |               |              | X          |              |
# MAGIC | Train/Test Data Splits Section                         | 10/23/2022 | 10/30/2022 | 2                      |               |              | X          |              |
# MAGIC | Conclusions and Next Steps Section                     | 10/23/2022 | 10/30/2022 | 2                      | X             |              | X          |              |
# MAGIC | Open Issues or Problems Section                        | 10/23/2022 | 10/30/2022 | 2                      | X             |              | X          |              |
# MAGIC | Set up Databricks instance                             | 10/23/2022 | 10/30/2022 | 2                      |               |              |            | X            |
# MAGIC | Set up GitHub and Integrate with Databricks            | 10/23/2022 | 10/30/2022 | 1                      |               |              |            | X            |
# MAGIC | Phase 1 Submission                                     | 10/23/2022 | 10/30/2022 | 1                      |               |              |            | X            |
# MAGIC | Revise Phase 1 Project Proposal                        | 10/31/2022 | 11/13/2022 | 1                      | X             | X            |            |              |
# MAGIC | Raw Data EDA: Airlines                                 | 10/31/2022 | 11/13/2022 | 2                      |               |              | X          | X            |
# MAGIC | Raw Data EDA: Stations and Weather                     | 10/31/2022 | 11/13/2022 | 3                      | X             |              |            |              |
# MAGIC | Join Data Sets                                         | 10/31/2022 | 11/13/2022 | 6                      |               | X            |            |              |
# MAGIC | Initial Data Pipeline Creation                         | 10/31/2022 | 11/13/2022 | 6                      |               |              |            | X            |
# MAGIC | Model Implementation and Evaluation                    | 10/31/2022 | 11/13/2022 | 3                      |               |              |            | X            |
# MAGIC | Join Data Sets v2                                      | 10/31/2022 | 11/13/2022 | 6                      |               | X            |            |              |
# MAGIC | Implementing Time Series Cross Validation              | 10/31/2022 | 11/13/2022 | 3                      | X             |              |            | X            |
# MAGIC | Feature Engineering: Lag Time Window                   | 10/31/2022 | 11/13/2022 | 3                      |               | X            | X          |              |
# MAGIC | Feature Engineering: Time Zone Conversion              | 10/31/2022 | 11/13/2022 | 3                      |               | X            | X          |              |
# MAGIC | Feature Engineering: Airport Tracker                   | 10/31/2022 | 11/13/2022 | 3                      |               |              | X          |              |
# MAGIC | Feature Engineering: Flight Tracker                    | 10/31/2022 | 11/13/2022 | 4                      |               |              | X          |              |
# MAGIC | Feature Engineering: Previously Delayed                | 10/31/2022 | 11/13/2022 | 4                      |               |              | X          |              |
# MAGIC | Data Cleaning, Imputation for Null Values              | 10/31/2022 | 11/13/2022 | 3                      | X             |              |            | X            |
# MAGIC | Phase 2 Check-In Presentation Video                    | 10/31/2022 | 11/13/2022 | 2                      |               |              |            | X            |
# MAGIC | Phase 2 Submission                                     | 10/31/2022 | 11/13/2022 | 1                      |               |              | X          |              |
# MAGIC | Data Pipeline Creation                                 | 11/14/2022 | 11/25/2022 | 8                      |               | X            |            | X            |
# MAGIC | Model Building                                         | 11/18/2022 | 11/27/2022 | 12                     |               | X            |            | X            |
# MAGIC | Feature Engineering                                    | 11/14/2022 | 11/25/2022 | 10                     | X             | X            | X          |              |
# MAGIC | Notebook Writeup                                       | 11/19/2022 | 11/27/2022 | 3                      | X             | X            | X          | X            |
# MAGIC | Presentation Setup                                     | 11/14/2022 | 11/16/2022 | 4                      | X             | X            | X          | X            |
# MAGIC | Phase 3 Submission                                     | 11/27/2022 | 11/27/2022 | 1                      |               | X            |            |              |
# MAGIC | Data Pipeline Revision                                 | 11/27/2022 | 12/4/2022  | 12                     |               | X            |            | X            |
# MAGIC | Model Building                                         | 11/27/2022 | 12/4/2022  | 20                     |               | X            |            | X            |
# MAGIC | Feature Engineering                                    | 11/27/2022 | 12/2/2022  | 12                     | X             | X            | X          |              |
# MAGIC | Notebook Writeup                                       | 11/27/2022 | 12/4/2022  | 12                     | X             | X            | X          | X            |
# MAGIC | Presentation Setup                                     | 11/30/2022 | 12/4/2022  | 4                      | X             | X            | X          | X            |
# MAGIC | Gap Analysis                                           | 11/22/2022 | 12/4/2022  | 8                      |               |              | X          |              |
# MAGIC | Phase 4 Submission                                     | 12/5/2022  | 12/5/2022  | 1                      | X             | X            | X          | X            |
# MAGIC | Final Presentation Preparation                         | 12/5/2022  | 12/9/2022  | 12                     | X             | X            | X          | X            |
# MAGIC | Phase 5 Submission                                     | 12/9/2022  | 12/9/2022  | 1                      | X             | X            | X          | X            |
# MAGIC 
# MAGIC ![Figure 13.2.A: Sparks and Stripe Forever Group Photo](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/Phase1/images/Group_Photo.JPG)
# MAGIC 
# MAGIC *Figure 13.2.A: Sparks and Stripe Forever Group Photo*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 13.3 Links to Previous Phase Notebooks
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

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 13.4 Links to Project Presentation Slides
# MAGIC 
# MAGIC ### 13.4.1 Phase 2 Slides
# MAGIC 
# MAGIC [Presentation Link in Google Slides](https://docs.google.com/presentation/d/1ro1-ZlB1FuTDR-qS_yYoTgTQB4ZYefqxH7Ae3i-cQvs/edit?usp=sharing)
# MAGIC 
# MAGIC [Presentation Link in PDF Format](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/Phase%202%20Presentation.pdf)
# MAGIC 
# MAGIC ### 13.4.2 Phase 3 In-Class Presentation Slides
# MAGIC 
# MAGIC [Presentation Link in Google Slides](https://docs.google.com/presentation/d/1-Yc9jgz9TPdmsvAWvPCFchSAAOvipdvYbo6E6HiaK_s/edit#slide=id.g18d7e4d3627_1_1247)
# MAGIC 
# MAGIC [Presentation Link in PDF Format](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/Phase_3_Presentation.pdf)
# MAGIC 
# MAGIC ### 13.4.3 Phase 4 Slides
# MAGIC 
# MAGIC [Presentation Link in Google Slides](https://docs.google.com/presentation/d/1qyLrE24qPNWH_03JeAe8SLfe58Qn1EEc8Rd2ikpjRHo/edit?usp=sharing)
# MAGIC 
# MAGIC [Presentation Link in PDF Format](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/Phase%204%20Presentation.pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 14. Appendix

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.1. Links to Pre-Joined EDA Notebooks
# MAGIC 
# MAGIC [EDA for Airlines Dataset (Part 1)](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804817899/command/1020093804817918)
# MAGIC 
# MAGIC [EDA for Airlines Dataset in GitHub(Part 1)](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/NH_EDAPart2.py)
# MAGIC 
# MAGIC [EDA for Airlines Dataset (Part 2)](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4423519682322242/command/4423519682322243)
# MAGIC 
# MAGIC [EDA for Airlines Dataset in GitHub(Part 2)](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/RWong%20-%20EDA%20Notebook.py)
# MAGIC 
# MAGIC [EDA for Airlines Dataset (Part 3)](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804809577/command/1020093804815501)
# MAGIC 
# MAGIC [EDA for Airlines Dataset in GitHub(Part 3)](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/NH%20-%20EDA%20Notebook.py)
# MAGIC 
# MAGIC [EDA for Weather, Stations Datasets](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/2260108212960246/command/2260108212960247)
# MAGIC 
# MAGIC [EDA for Weather, Stations Datasets in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/Nash-%20EDA%20Weather%20FULL.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.2 Links to Joining Data Sets Notebooks
# MAGIC 
# MAGIC [Join Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4423519682321930/command/1020093804821142)
# MAGIC 
# MAGIC [Join Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/DLE_join_V1.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.3 Links to Complete Post-Join EDA Notebooks:
# MAGIC 
# MAGIC [NC Joined Data EDA wFullNulls](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1020093804817787/command/1020093804817792)
# MAGIC 
# MAGIC [NC Joined Data EDA wFullNulls in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase2/NC%20Joined%20Data%20EDA%20wFullNulls.py)
# MAGIC 
# MAGIC [Correlation Analysis in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/NH/CorrelationAnalysis.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.3.1 Link to Data Preparation Pipeline Notebook
# MAGIC 
# MAGIC [Join Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1694179314625737/command/3035635552858508)
# MAGIC 
# MAGIC [Join Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/GroupNotebook_Code.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.4 Logistic Regression - Baseline Model Information
# MAGIC 
# MAGIC ### 14.4.1 Cross Validation Results Table
# MAGIC 
# MAGIC | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy | cv_fold | regParam | elasticNetParam | maxIter | threshold |
# MAGIC |----------------|-------------|-------------|-------------|---------------|---------------|-------------|-------------|-------------|--------------|---------|----------|-----------------|---------|-----------|
# MAGIC | 0.365993636    | 0.084262002 | 0.219328092 | 0.136985987 | 0.800714556   | 0.565020712   | 0.049388448 | 0.182968981 | 0.090836849 | 0.640758376  | 0       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.413245699    | 0.02164104  | 0.089464689 | 0.041128257 | 0.810591071   | 0.669585399   | 0.014099243 | 0.065019812 | 0.027616965 | 0.63922309   | 0       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.459970694    | 0.009741854 | 0.04490504  | 0.019079614 | 0.811978348   | 0.661375661   | 0.005553507 | 0.026865197 | 0.011014526 | 0.637612735  | 0       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.317175514    | 0.094757135 | 0.21584651  | 0.14592018  | 0.791792136   | 0.526077256   | 0.104015643 | 0.290404154 | 0.173689511 | 0.652545921  | 1       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.417155249    | 0.019538252 | 0.082276887 | 0.037328168 | 0.810839965   | 0.671551473   | 0.021163863 | 0.093973111 | 0.041034528 | 0.652721771  | 1       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.471623216    | 0.008268209 | 0.038631958 | 0.016251506 | 0.812109856   | 0.741096571   | 0.010279788 | 0.048697023 | 0.020278294 | 0.651273828  | 1       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.333326468    | 0.106539709 | 0.233793233 | 0.161469587 | 0.792297515   | 0.55210199    | 0.091187727 | 0.274552906 | 0.156523334 | 0.613855265  | 2       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.409576774    | 0.0257648   | 0.102925508 | 0.048479926 | 0.810161243   | 0.713118449   | 0.025996997 | 0.113442614 | 0.050165202 | 0.61319825   | 2       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.46304128     | 0.009472892 | 0.043781718 | 0.018565962 | 0.81201277    | 0.789427969   | 0.013069371 | 0.061288228 | 0.02571305  | 0.610858341  | 2       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.295716685    | 0.034936756 | 0.118625061 | 0.062490696 | 0.803236332   | 0.339682064   | 0.036458948 | 0.127538571 | 0.065850042 | 0.84439643   | 3       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.410770087    | 0.00902431  | 0.041476702 | 0.017660629 | 0.8115607     | 0.415923856   | 0.014771374 | 0.064669955 | 0.028529532 | 0.848674987  | 3       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.49971001     | 0.004051347 | 0.019620453 | 0.008037531 | 0.812295732   | 0.453045685   | 0.00574651  | 0.027345146 | 0.011349066 | 0.849394133  | 3       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.375389378    | 0.072077867 | 0.203836305 | 0.120935179 | 0.803314531   | 0.58022561    | 0.04033401  | 0.157794201 | 0.075424906 | 0.640680886  | 0       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.426248084    | 0.018042037 | 0.077148207 | 0.034618747 | 0.811124693   | 0.676116609   | 0.011231412 | 0.05265811  | 0.022095778 | 0.63875411   | 0       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.470634712    | 0.00840269  | 0.039213019 | 0.0165106   | 0.812099794   | 0.658735128   | 0.004673831 | 0.022724229 | 0.009281807 | 0.637446453  | 0       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.325656486    | 0.083656369 | 0.206300041 | 0.133116949 | 0.795483539   | 0.534614035   | 0.090990136 | 0.270676496 | 0.155512402 | 0.653062179  | 1       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.42552146     | 0.016288146 | 0.070626883 | 0.031375305 | 0.811226369   | 0.688873743   | 0.018578981 | 0.083849225 | 0.036182128 | 0.652502361  | 1       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.477422942    | 0.007268538 | 0.034256533 | 0.014319075 | 0.812167578   | 0.748845799   | 0.009317063 | 0.044376788 | 0.018405131 | 0.65109959   | 1       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.342539635    | 0.093832699 | 0.223866432 | 0.147311898 | 0.79610401    | 0.561428162   | 0.080433537 | 0.255658853 | 0.140708357 | 0.614008622  | 2       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.41977192     | 0.02121973  | 0.088253562 | 0.040397349 | 0.810774123   | 0.729667435   | 0.022761495 | 0.101182225 | 0.044145889 | 0.612722842  | 2       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.467601078    | 0.008157239 | 0.038125797 | 0.016034754 | 0.812084437   | 0.797917253   | 0.011647805 | 0.055025999 | 0.022960439 | 0.610510462  | 2       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.312302012    | 0.027999248 | 0.103043153 | 0.051391061 | 0.805979289   | 0.35584933    | 0.031629948 | 0.11666891  | 0.058095989 | 0.845718497  | 3       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.436307995    | 0.007508346 | 0.035123963 | 0.014762645 | 0.811885145   | 0.429257481   | 0.01246955  | 0.055857328 | 0.024235093 | 0.848955059  | 3       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.504199328    | 0.003387408 | 0.016493792 | 0.006729603 | 0.812307206   | 0.452777778   | 0.004372928 | 0.021051381 | 0.008662196 | 0.849436104  | 3       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.513693964    | 0.003545399 | 0.017250752 | 0.007042194 | 0.812332096   | 0.695744681   | 0.001452797 | 0.007203817 | 0.00289954  | 0.636925004  | 0       | 0.5      | 0               | 5       | 0.5       |
# MAGIC | 0.422206991    | 0.001158603 | 0.005730115 | 0.002310864 | 0.812216475   | 0.633451957   | 0.000790819 | 0.003934449 | 0.001579667 | 0.636749036  | 0       | 0.5      | 0               | 5       | 0.6       |
# MAGIC | 0.300159659    | 0.000530399 | 0.002633383 | 0.001058927 | 0.812164048   | 0.622276029   | 0.000570901 | 0.002844066 | 0.001140754 | 0.636709483  | 0       | 0.5      | 0               | 5       | 0.7       |
# MAGIC | 0.512132163    | 0.003731603 | 0.018129616 | 0.007409219 | 0.812329801   | 0.78613396    | 0.004611429 | 0.02252854  | 0.009169073 | 0.650104179  | 1       | 0.5      | 0               | 5       | 0.5       |
# MAGIC | 0.413083425    | 0.001057037 | 0.005231635 | 0.002108678 | 0.812213121   | 0.833910035   | 0.001107478 | 0.005508131 | 0.002212019 | 0.649237026  | 1       | 0.5      | 0               | 5       | 0.6       |
# MAGIC | 0.285633163    | 0.000473033 | 0.002349602 | 0.000944502 | 0.812163342   | 0.787096774   | 0.000280316 | 0.001399587 | 0.000560433 | 0.64899745   | 1       | 0.5      | 0               | 5       | 0.7       |
# MAGIC | 0.51001065     | 0.004503691 | 0.021750188 | 0.008928538 | 0.812329801   | 0.778504184   | 0.006115611 | 0.029646493 | 0.012135888 | 0.608812232  | 2       | 0.5      | 0               | 5       | 0.5       |
# MAGIC | 0.433280507    | 0.00128556  | 0.006352408 | 0.002563514 | 0.8122223     | 0.68791627    | 0.001485249 | 0.00736266  | 0.002964099 | 0.607411838  | 2       | 0.5      | 0               | 5       | 0.6       |
# MAGIC | 0.294557097    | 0.000519114 | 0.002577401 | 0.001036402 | 0.812160694   | 0.781456954   | 0.000484812 | 0.002418058 | 0.000969022 | 0.60723023   | 2       | 0.5      | 0               | 5       | 0.7       |
# MAGIC | 0.433628319    | 0.001336343 | 0.00660035  | 0.002664474 | 0.812219828   | 0.446488294   | 0.001432603 | 0.007072248 | 0.002856043 | 0.849521659  | 3       | 0.5      | 0               | 5       | 0.5       |
# MAGIC | 0.278734037    | 0.000472093 | 0.00234458  | 0.000942589 | 0.812155928   | 0.459183673   | 0.00024145  | 0.001204716 | 0.000482646 | 0.849566858  | 3       | 0.5      | 0               | 5       | 0.6       |
# MAGIC | 0.230348599    | 0.000316923 | 0.001575942 | 0.000632975 | 0.81215734    | 0.548387097   | 9.12144E-05 | 0.000455769 | 0.000182399 | 0.849575736  | 3       | 0.5      | 0               | 5       | 0.7       |
# MAGIC 
# MAGIC ### 14.4.2 Link to Modelling Notebook
# MAGIC 
# MAGIC [Link to Modelling Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238260296/command/1215577238260564)
# MAGIC 
# MAGIC [Link to Modelling Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/DLE_pipeline_cleaned_LR_Baseline.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.5 Logistic Regression - Feature Engineering Model Information
# MAGIC 
# MAGIC ### 14.5.1 Cross Validation Results Table
# MAGIC 
# MAGIC | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy | cv_fold | regParam | elasticNetParam | maxIter | threshold |
# MAGIC |----------------|-------------|-------------|-------------|---------------|---------------|-------------|-------------|-------------|--------------|---------|----------|-----------------|---------|-----------|
# MAGIC | 0.62432625     | 0.304023134 | 0.515669653 | 0.40891851  | 0.83502455    | 0.796993527   | 0.353103103 | 0.636870035 | 0.489386509 | 0.710491332  | 2       | 0.01     | 0.5             | 50      | 0.6       |
# MAGIC | 0.706582135    | 0.092442752 | 0.303424565 | 0.163495275 | 0.822442854   | 0.838122615   | 0.164710643 | 0.461091935 | 0.275315391 | 0.659309553  | 2       | 0.01     | 0.5             | 50      | 0.7       |
# MAGIC | 0.824395328    | 0.017853952 | 0.082153001 | 0.03495097  | 0.814934012   | 0.92026442    | 0.032315982 | 0.14167911  | 0.062439346 | 0.618690054  | 2       | 0.01     | 0.5             | 50      | 0.8       |
# MAGIC | 0.553375125    | 0.339285278 | 0.491364662 | 0.420657245 | 0.824581933   | 0.482928943   | 0.239217917 | 0.40118491  | 0.319949479 | 0.847029264  | 3       | 0.01     | 0.5             | 50      | 0.5       |
# MAGIC | 0.60849577     | 0.303965769 | 0.50692298  | 0.405412999 | 0.832642755   | 0.553162873   | 0.200816638 | 0.409473233 | 0.294661345 | 0.855379755  | 3       | 0.01     | 0.5             | 50      | 0.6       |
# MAGIC | 0.661740574    | 0.063333804 | 0.229000418 | 0.115603444 | 0.818107856   | 0.615488678   | 0.158675566 | 0.390592393 | 0.252305671 | 0.858530762  | 3       | 0.01     | 0.5             | 50      | 0.7       |
# MAGIC | 0.807354046    | 0.017179668 | 0.079160521 | 0.033643438 | 0.814751843   | 0.778694158   | 0.012158348 | 0.057218176 | 0.023942858 | 0.850882468  | 3       | 0.01     | 0.5             | 50      | 0.8       |
# MAGIC | 0.622834165    | 0.296571214 | 0.510510039 | 0.401813364 | 0.83425386    | 0.769233984   | 0.368004247 | 0.631525377 | 0.497840058 | 0.730234402  | 0       | 0        | 1               | 50      | 0.5       |
# MAGIC | 0.667710712    | 0.19747308  | 0.452300414 | 0.304802037 | 0.830916734   | 0.792029482   | 0.323693926 | 0.614276633 | 0.459567543 | 0.723364359  | 0       | 0        | 1               | 50      | 0.6       |
# MAGIC | 0.717600851    | 0.07869281  | 0.273496485 | 0.141832167 | 0.821254692   | 0.808541281   | 0.18706877  | 0.485776058 | 0.303839486 | 0.688507229  | 0       | 0        | 1               | 50      | 0.7       |
# MAGIC | 0.808753605    | 0.019514741 | 0.088985079 | 0.038109913 | 0.81509341    | 0.753584848   | 0.043661671 | 0.177233658 | 0.082541029 | 0.647305541  | 0       | 0        | 1               | 50      | 0.8       |
# MAGIC | 0.568419656    | 0.334325481 | 0.498596376 | 0.421020656 | 0.827403794   | 0.715092982   | 0.408808818 | 0.621905294 | 0.520216831 | 0.735265861  | 1       | 0        | 1               | 50      | 0.5       |
# MAGIC | 0.620456784    | 0.267153806 | 0.490675917 | 0.373491243 | 0.831767388   | 0.770797262   | 0.369263573 | 0.633109759 | 0.499319584 | 0.740015439  | 1       | 0        | 1               | 50      | 0.6       |
# MAGIC | 0.654247833    | 0.153881601 | 0.396435449 | 0.249159849 | 0.825916253   | 0.809610753   | 0.312786778 | 0.614423323 | 0.451240371 | 0.732913658  | 1       | 0        | 1               | 50      | 0.7       |
# MAGIC | 0.71946824     | 0.040360182 | 0.164817651 | 0.076432701 | 0.816918459   | 0.861461045   | 0.114070258 | 0.3728616   | 0.201463722 | 0.682532474  | 1       | 0        | 1               | 50      | 0.8       |
# MAGIC | 0.494407154    | 0.395798185 | 0.470941144 | 0.43964116  | 0.810615784   | 0.770194935   | 0.390811587 | 0.644972529 | 0.518517509 | 0.714829729  | 2       | 0        | 1               | 50      | 0.5       |
# MAGIC | 0.578098051    | 0.339797809 | 0.506987914 | 0.428014679 | 0.829529634   | 0.803622121   | 0.350672882 | 0.638641032 | 0.488278099 | 0.711205655  | 2       | 0        | 1               | 50      | 0.6       |
# MAGIC | 0.641272485    | 0.265798655 | 0.500007784 | 0.375823585 | 0.834278749   | 0.838675834   | 0.198704978 | 0.510099471 | 0.321288115 | 0.670147877  | 2       | 0        | 1               | 50      | 0.7       |
# MAGIC | 0.722352722    | 0.116545822 | 0.354163404 | 0.200708876 | 0.825764269   | 0.901885551   | 0.050701847 | 0.206968206 | 0.096006442 | 0.624846945  | 2       | 0        | 1               | 50      | 0.8       |
# MAGIC | 0.612648797    | 0.269108008 | 0.488042651 | 0.373955032 | 0.830872251   | 0.390661342   | 0.286585039 | 0.364208155 | 0.330626192 | 0.82544196   | 3       | 0        | 1               | 50      | 0.5       |
# MAGIC | 0.640018069    | 0.123914986 | 0.34916512  | 0.207630314 | 0.822473568   | 0.503626786   | 0.231346647 | 0.407667172 | 0.317051918 | 0.850074538  | 3       | 0        | 1               | 50      | 0.6       |
# MAGIC | 0.705356803    | 0.034821084 | 0.145394816 | 0.066365908 | 0.816102403   | 0.565301197   | 0.193551676 | 0.408414889 | 0.288369453 | 0.856299875  | 3       | 0        | 1               | 50      | 0.7       |
# MAGIC | 0.793391789    | 0.012849015 | 0.060336468 | 0.025288482 | 0.814080358   | 0.642785291   | 0.092571925 | 0.293680082 | 0.161836644 | 0.85575991   | 3       | 0        | 1               | 50      | 0.8       |
# MAGIC | 0.596815561    | 0.320811586 | 0.50919958  | 0.417305323 | 0.831833583   | 0.775678575   | 0.362990541 | 0.631977995 | 0.494549262 | 0.730383733  | 0       | 0.01     | 1               | 50      | 0.5       |
# MAGIC | 0.632984043    | 0.248193915 | 0.483167308 | 0.356574484 | 0.831871535   | 0.804652713   | 0.343222278 | 0.634143358 | 0.481193056 | 0.731067427  | 0       | 0.01     | 1               | 50      | 0.6       |
# MAGIC | 0.802311734    | 0.021541355 | 0.097261263 | 0.041956223 | 0.815343717   | 0.894852331   | 0.033518746 | 0.145755331 | 0.064617111 | 0.647376574  | 0       | 0.01     | 1               | 50      | 0.7       |
# MAGIC | 0.844432983    | 0.012317675 | 0.058192953 | 0.024281163 | 0.81418274    | 0.93217807    | 0.006930777 | 0.033653037 | 0.013759253 | 0.638963173  | 0       | 0.01     | 1               | 50      | 0.8       |
# MAGIC | 0.588432818    | 0.321849814 | 0.504807962 | 0.416105914 | 0.830454779   | 0.731841524   | 0.380159137 | 0.617578137 | 0.500388627 | 0.733486382  | 1       | 0.01     | 1               | 50      | 0.5       |
# MAGIC | 0.622629079    | 0.255084403 | 0.483342017 | 0.361901623 | 0.831156979   | 0.776654547   | 0.350930443 | 0.625010742 | 0.483425598 | 0.736698478  | 1       | 0.01     | 1               | 50      | 0.6       |
# MAGIC | 0.801757177    | 0.020853905 | 0.094443526 | 0.040650481 | 0.8152431     | 0.840061342   | 0.206416021 | 0.520500309 | 0.331401569 | 0.707596018  | 1       | 0.01     | 1               | 50      | 0.7       |
# MAGIC | 0.856964097    | 0.008372596 | 0.040288495 | 0.016583173 | 0.81360587    | 0.911181796   | 0.025669599 | 0.11534959  | 0.049932512 | 0.65705915   | 1       | 0.01     | 1               | 50      | 0.8       |
# MAGIC | 0.569591276    | 0.333446184 | 0.498924113 | 0.420642655 | 0.827590553   | 0.749442629   | 0.388779898 | 0.632155266 | 0.511970589 | 0.708777767  | 2       | 0.01     | 1               | 50      | 0.5       |
# MAGIC | 0.613995766    | 0.310950299 | 0.513840239 | 0.412828756 | 0.833969485   | 0.793108197   | 0.354941689 | 0.636066585 | 0.490409287 | 0.710172511  | 2       | 0.01     | 1               | 50      | 0.6       |
# MAGIC | 0.739941047    | 0.047923073 | 0.190312259 | 0.090016154 | 0.818130451   | 0.838112835   | 0.134185999 | 0.408998979 | 0.231334245 | 0.649631902  | 2       | 0.01     | 1               | 50      | 0.7       |
# MAGIC | 0.840940306    | 0.014970612 | 0.069877181 | 0.029417526 | 0.814575145   | 0.930180031   | 0.028764121 | 0.127989265 | 0.055802646 | 0.617546332  | 2       | 0.01     | 1               | 50      | 0.8       |
# MAGIC | 0.537511151    | 0.33487751  | 0.479484187 | 0.412661016 | 0.821069875   | 0.497461629   | 0.225551847 | 0.400821151 | 0.310377035 | 0.849227059  | 3       | 0.01     | 1               | 50      | 0.5       |
# MAGIC | 0.59964457     | 0.313509193 | 0.50708307  | 0.411746833 | 0.83185406    | 0.564080673   | 0.187885649 | 0.402784999 | 0.28188141  | 0.855994783  | 3       | 0.01     | 1               | 50      | 0.6       |
# MAGIC | 0.721165334    | 0.029518033 | 0.126825736 | 0.056714676 | 0.815694993   | 0.636507102   | 0.097384828 | 0.302063016 | 0.168924421 | 0.855856765  | 3       | 0.01     | 1               | 50      | 0.7       |
# MAGIC | 0.831978028    | 0.014528612 | 0.067900171 | 0.028558515 | 0.81447294    | 0.782954998   | 0.010548682 | 0.05004633  | 0.020816899 | 0.850720236  | 3       | 0.01     | 1               | 50      | 0.8       |
# MAGIC | 0.736008101    | 0.025290826 | 0.111173487 | 0.048901298 | 0.815341069   | 0.650646112   | 0.076995718 | 0.261294857 | 0.13769677  | 0.854936645  | 3       | 0.01     | 1               | 5       | 0.7       |
# MAGIC | 0.81835073     | 0.014082851 | 0.065879428 | 0.027689204 | 0.814353259   | 0.78465841    | 0.010537951 | 0.050003564 | 0.020796603 | 0.850723464  | 3       | 0.01     | 1               | 5       | 0.8       |
# MAGIC | 0.603708808    | 0.326273569 | 0.515962671 | 0.423608517 | 0.833337893   | 0.768413071   | 0.369168262 | 0.631765783 | 0.498731317 | 0.73034418   | 0       | 0        | 0               | 10      | 0.5       |
# MAGIC | 0.653988859    | 0.253944609 | 0.497305659 | 0.365835055 | 0.834743705   | 0.791381271   | 0.329291861 | 0.617949706 | 0.465069438 | 0.724740628  | 0       | 0        | 0               | 10      | 0.6       |
# MAGIC | 0.711127124    | 0.131785395 | 0.378416125 | 0.222362741 | 0.826984734   | 0.80871996    | 0.199222065 | 0.501725322 | 0.319690729 | 0.691897451  | 0       | 0        | 0               | 10      | 0.7       |
# MAGIC | 0.797383019    | 0.028253162 | 0.123729674 | 0.054572685 | 0.816252269   | 0.751946108   | 0.044632424 | 0.180344173 | 0.084263325 | 0.647496039  | 0       | 0        | 0               | 10      | 0.8       |
# MAGIC | 0.55231409     | 0.34609865  | 0.493505198 | 0.425539738 | 0.824603115   | 0.718531241   | 0.406589266 | 0.622944575 | 0.519316976 | 0.735752273  | 1       | 0        | 0               | 10      | 0.5       |
# MAGIC | 0.613915226    | 0.289424931 | 0.501470062 | 0.393389733 | 0.832457584   | 0.771671075   | 0.368705238 | 0.633252119 | 0.498991718 | 0.740067872  | 1       | 0        | 0               | 10      | 0.6       |
# MAGIC | 0.6507694      | 0.183464522 | 0.431137971 | 0.286234098 | 0.828253212   | 0.810647594   | 0.311203682 | 0.613673414 | 0.449750375 | 0.732661176  | 1       | 0        | 0               | 10      | 0.7       |
# MAGIC | 0.712659301    | 0.05374618  | 0.20645158  | 0.099954178 | 0.818317386   | 0.862287247   | 0.108606393 | 0.361105042 | 0.192914864 | 0.680965145  | 1       | 0        | 0               | 10      | 0.8       |
# MAGIC | 0.567301954    | 0.348622749 | 0.504065341 | 0.431857261 | 0.827823031   | 0.765227551   | 0.394223757 | 0.644011724 | 0.520368347 | 0.7144649    | 2       | 0        | 0               | 10      | 0.5       |
# MAGIC | 0.619997587    | 0.304487704 | 0.513566217 | 0.408403776 | 0.834420142   | 0.801188607   | 0.352269062 | 0.638461854 | 0.489370294 | 0.711156419  | 2       | 0        | 0               | 10      | 0.6       |
# MAGIC | 0.679635564    | 0.176712277 | 0.433109542 | 0.280493375 | 0.829830778   | 0.836399124   | 0.209512579 | 0.523264942 | 0.335087823 | 0.673310265  | 2       | 0        | 0               | 10      | 0.7       |
# MAGIC | 0.774590289    | 0.04791555  | 0.192056011 | 0.090248404 | 0.818673252   | 0.893535749   | 0.056223769 | 0.224591216 | 0.10579088  | 0.626551633  | 2       | 0        | 0               | 10      | 0.8       |
# MAGIC | 0.569587145    | 0.336685005 | 0.50036198  | 0.42320941  | 0.827738301   | 0.410451291   | 0.274694968 | 0.373530931 | 0.329123608 | 0.831543001  | 3       | 0        | 0               | 10      | 0.5       |
# MAGIC | 0.614681951    | 0.262368928 | 0.484549776 | 0.367763031 | 0.830672959   | 0.505947539   | 0.230273536 | 0.40820927  | 0.316498217 | 0.850387701  | 3       | 0        | 0               | 10      | 0.6       |
# MAGIC | 0.649967861    | 0.106505854 | 0.321681893 | 0.183021249 | 0.821521945   | 0.568944873   | 0.191212294 | 0.407818787 | 0.286228319 | 0.856544433  | 3       | 0        | 0               | 10      | 0.7       |
# MAGIC | 0.75932565     | 0.02410119  | 0.106930006 | 0.046719491 | 0.815386611   | 0.652357801   | 0.088553124 | 0.286956219 | 0.155938641 | 0.855795423  | 3       | 0        | 0               | 10      | 0.8       |
# MAGIC | 0.606873419    | 0.322947289 | 0.516121527 | 0.421561111 | 0.833646981   | 0.770703577   | 0.367240085 | 0.631865444 | 0.497446853 | 0.730370818  | 0       | 0.01     | 0               | 10      | 0.5       |
# MAGIC | 0.660858759    | 0.237056472 | 0.486801134 | 0.34894351  | 0.833958188   | 0.794194885   | 0.318435866 | 0.611478906 | 0.454598501 | 0.72235375   | 0       | 0.01     | 0               | 10      | 0.6       |
# MAGIC | 0.723185597    | 0.105619034 | 0.333353913 | 0.184318865 | 0.824533213   | 0.807876002   | 0.160822896 | 0.447656631 | 0.268246323 | 0.681169013  | 0       | 0.01     | 0               | 10      | 0.7       |
# MAGIC | 0.799588291    | 0.0215517   | 0.097271293 | 0.041972105 | 0.815328007   | 0.735291011   | 0.030926369 | 0.132363064 | 0.059356214 | 0.64382007   | 0       | 0.01     | 0               | 10      | 0.8       |
# MAGIC | 0.555856431    | 0.343233178 | 0.49458068  | 0.424403458 | 0.825244592   | 0.723535873   | 0.402575232 | 0.624031684 | 0.517315957 | 0.736255625  | 1       | 0.01     | 0               | 10      | 0.5       |
# MAGIC | 0.617508677    | 0.277585931 | 0.496025358 | 0.383002466 | 0.832126784   | 0.776704735   | 0.364410888 | 0.63338304  | 0.496075343 | 0.740080778  | 1       | 0.01     | 0               | 10      | 0.6       |
# MAGIC | 0.653878571    | 0.16050971  | 0.404940388 | 0.257748944 | 0.826476883   | 0.819432825   | 0.285816696 | 0.596646531 | 0.423809426 | 0.727157376  | 1       | 0.01     | 0               | 10      | 0.7       |
# MAGIC | 0.730680272    | 0.040404382 | 0.165430673 | 0.076574433 | 0.817085271   | 0.872868781   | 0.071636839 | 0.269659695 | 0.132406963 | 0.670412499  | 1       | 0.01     | 0               | 10      | 0.8       |
# MAGIC | 0.576642808    | 0.34359148  | 0.507761839 | 0.430606767 | 0.829440491   | 0.766452584   | 0.393231537 | 0.644173864 | 0.51978521  | 0.714517364  | 2       | 0.01     | 0               | 10      | 0.5       |
# MAGIC | 0.629767614    | 0.290434006 | 0.510481606 | 0.397534468 | 0.834763122   | 0.803532767   | 0.346139071 | 0.635563908 | 0.483849524 | 0.709840775  | 2       | 0.01     | 0               | 10      | 0.6       |
# MAGIC | 0.695232535    | 0.13886867  | 0.385965701 | 0.23149713  | 0.82693619    | 0.841902709   | 0.179758293 | 0.484770052 | 0.296260684 | 0.664458322  | 2       | 0.01     | 0               | 10      | 0.7       |
# MAGIC | 0.78790691     | 0.031138383 | 0.134439513 | 0.059909135 | 0.816568065   | 0.90596166    | 0.044173324 | 0.184820371 | 0.084239269 | 0.622647479  | 2       | 0.01     | 0               | 10      | 0.8       |
# MAGIC | 0.569976755    | 0.338509428 | 0.501406067 | 0.424756059 | 0.827898229   | 0.445812029   | 0.257310569 | 0.388840419 | 0.326293444 | 0.840163878  | 3       | 0.01     | 0               | 10      | 0.5       |
# MAGIC | 0.617507106    | 0.259459256 | 0.48394147  | 0.365391288 | 0.830831651   | 0.515921222   | 0.225594772 | 0.410312011 | 0.313922104 | 0.851667798  | 3       | 0.01     | 0               | 10      | 0.6       |
# MAGIC | 0.658380386    | 0.090686039 | 0.292353709 | 0.159414192 | 0.820486297   | 0.585985242   | 0.178533486 | 0.40233999  | 0.273683258 | 0.857454868  | 3       | 0.01     | 0               | 10      | 0.7       |
# MAGIC | 0.537207457    | 0.334982837 | 0.479333967 | 0.412651412 | 0.821006504   | 0.497468714   | 0.225659158 | 0.400892588 | 0.310480001 | 0.849227866  | 3       | 0.01     | 1               | 10      | 0.5       |
# MAGIC | 0.5994517      | 0.313588188 | 0.507014028 | 0.411769464 | 0.831827405   | 0.564060436   | 0.187891015 | 0.402781676 | 0.281884921 | 0.855993169  | 3       | 0.01     | 1               | 10      | 0.6       |
# MAGIC | 0.71924013     | 0.030015517 | 0.128609006 | 0.057626164 | 0.815731357   | 0.636233703   | 0.097926749 | 0.303053952 | 0.169729378 | 0.855881786  | 3       | 0.01     | 1               | 10      | 0.7       |
# MAGIC | 0.830161446    | 0.014603846 | 0.068218919 | 0.028702766 | 0.814477      | 0.782954998   | 0.010548682 | 0.05004633  | 0.020816899 | 0.850720236  | 3       | 0.01     | 1               | 10      | 0.8       |
# MAGIC | 0.622834165    | 0.296571214 | 0.510510039 | 0.401813364 | 0.83425386    | 0.769233984   | 0.368004247 | 0.631525377 | 0.497840058 | 0.730234402  | 0       | 0        | 0               | 50      | 0.5       |
# MAGIC | 0.667710712    | 0.19747308  | 0.452300414 | 0.304802037 | 0.830916734   | 0.792029482   | 0.323693926 | 0.614276633 | 0.459567543 | 0.723364359  | 0       | 0        | 0               | 50      | 0.6       |
# MAGIC | 0.717600851    | 0.07869281  | 0.273496485 | 0.141832167 | 0.821254692   | 0.808541281   | 0.18706877  | 0.485776058 | 0.303839486 | 0.688507229  | 0       | 0        | 0               | 50      | 0.7       |
# MAGIC | 0.808753605    | 0.019514741 | 0.088985079 | 0.038109913 | 0.81509341    | 0.753584848   | 0.043661671 | 0.177233658 | 0.082541029 | 0.647305541  | 0       | 0        | 0               | 50      | 0.8       |
# MAGIC | 0.568419656    | 0.334325481 | 0.498596376 | 0.421020656 | 0.827403794   | 0.715092982   | 0.408808818 | 0.621905294 | 0.520216831 | 0.735265861  | 1       | 0        | 0               | 50      | 0.5       |
# MAGIC | 0.620456784    | 0.267153806 | 0.490675917 | 0.373491243 | 0.831767388   | 0.770797262   | 0.369263573 | 0.633109759 | 0.499319584 | 0.740015439  | 1       | 0        | 0               | 50      | 0.6       |
# MAGIC | 0.654247833    | 0.153881601 | 0.396435449 | 0.249159849 | 0.825916253   | 0.809610753   | 0.312786778 | 0.614423323 | 0.451240371 | 0.732913658  | 1       | 0        | 0               | 50      | 0.7       |
# MAGIC | 0.71946824     | 0.040360182 | 0.164817651 | 0.076432701 | 0.816918459   | 0.861461045   | 0.114070258 | 0.3728616   | 0.201463722 | 0.682532474  | 1       | 0        | 0               | 50      | 0.8       |
# MAGIC | 0.494407154    | 0.395798185 | 0.470941144 | 0.43964116  | 0.810615784   | 0.770194935   | 0.390811587 | 0.644972529 | 0.518517509 | 0.714829729  | 2       | 0        | 0               | 50      | 0.5       |
# MAGIC | 0.578098051    | 0.339797809 | 0.506987914 | 0.428014679 | 0.829529634   | 0.803622121   | 0.350672882 | 0.638641032 | 0.488278099 | 0.711205655  | 2       | 0        | 0               | 50      | 0.6       |
# MAGIC | 0.641272485    | 0.265798655 | 0.500007784 | 0.375823585 | 0.834278749   | 0.838675834   | 0.198704978 | 0.510099471 | 0.321288115 | 0.670147877  | 2       | 0        | 0               | 50      | 0.7       |
# MAGIC | 0.722352722    | 0.116545822 | 0.354163404 | 0.200708876 | 0.825764269   | 0.901885551   | 0.050701847 | 0.206968206 | 0.096006442 | 0.624846945  | 2       | 0        | 0               | 50      | 0.8       |
# MAGIC | 0.612648797    | 0.269108008 | 0.488042651 | 0.373955032 | 0.830872251   | 0.390661342   | 0.286585039 | 0.364208155 | 0.330626192 | 0.82544196   | 3       | 0        | 0               | 50      | 0.5       |
# MAGIC | 0.640018069    | 0.123914986 | 0.34916512  | 0.207630314 | 0.822473568   | 0.503626786   | 0.231346647 | 0.407667172 | 0.317051918 | 0.850074538  | 3       | 0        | 0               | 50      | 0.6       |
# MAGIC | 0.705356803    | 0.034821084 | 0.145394816 | 0.066365908 | 0.816102403   | 0.565301197   | 0.193551676 | 0.408414889 | 0.288369453 | 0.856299875  | 3       | 0        | 0               | 50      | 0.7       |
# MAGIC | 0.793391789    | 0.012849015 | 0.060336468 | 0.025288482 | 0.814080358   | 0.642785291   | 0.092571925 | 0.293680082 | 0.161836644 | 0.85575991   | 3       | 0        | 0               | 50      | 0.8       |
# MAGIC | 0.607791197    | 0.321503738 | 0.51591115  | 0.420549245 | 0.833701703   | 0.770597452   | 0.367086808 | 0.631717614 | 0.497284122 | 0.730307856  | 0       | 0.01     | 0               | 50      | 0.5       |
# MAGIC | 0.661147319    | 0.23401326  | 0.484338858 | 0.345674828 | 0.833709117   | 0.794215647   | 0.317696139 | 0.610942421 | 0.453847594 | 0.722158409  | 0       | 0.01     | 0               | 50      | 0.6       |
# MAGIC | 0.723335852    | 0.102620022 | 0.327340348 | 0.179740209 | 0.824191292   | 0.807700085   | 0.159707752 | 0.445880374 | 0.26668373  | 0.68084452   | 0       | 0.01     | 0               | 50      | 0.7       |
# MAGIC | 0.798478006    | 0.021116283 | 0.095481165 | 0.041144474 | 0.815259869   | 0.733102617   | 0.031057432 | 0.13278564  | 0.059590356 | 0.643804733  | 0       | 0.01     | 0               | 50      | 0.8       |
# MAGIC | 0.559175927    | 0.341372079 | 0.49589701  | 0.423935309 | 0.825858707   | 0.722849452   | 0.402929073 | 0.623792799 | 0.517432254 | 0.736146727  | 1       | 0.01     | 0               | 50      | 0.5       |
# MAGIC | 0.618485905    | 0.274252128 | 0.494379561 | 0.380001902 | 0.832020342   | 0.776510849   | 0.364426972 | 0.633289599 | 0.496050691 | 0.740043672  | 1       | 0.01     | 0               | 50      | 0.6       |
# MAGIC | 0.654538992    | 0.15585273  | 0.399122406 | 0.25175896  | 0.826110603   | 0.819300931   | 0.286572631 | 0.59724828  | 0.424622183 | 0.727344519  | 1       | 0.01     | 0               | 50      | 0.7       |
# MAGIC | 0.734062388    | 0.038506606 | 0.159140905 | 0.073174698 | 0.816905926   | 0.873215274   | 0.072509955 | 0.272153547 | 0.133901049 | 0.670685954  | 1       | 0.01     | 0               | 50      | 0.8       |
# MAGIC | 0.563515005    | 0.350530869 | 0.502456088 | 0.432208952 | 0.827128598   | 0.7679506     | 0.392165362 | 0.644445075 | 0.519195727 | 0.714618257  | 2       | 0.01     | 0               | 50      | 0.5       |
# MAGIC | 0.620339829    | 0.306155076 | 0.514699977 | 0.409975676 | 0.834592427   | 0.804185958   | 0.346116474 | 0.635875477 | 0.483945788 | 0.709971532  | 2       | 0.01     | 0               | 50      | 0.6       |
# MAGIC | 0.685921395    | 0.168097992 | 0.424430814 | 0.270021995 | 0.82940148    | 0.842395441   | 0.177340397 | 0.481359966 | 0.292998905 | 0.663735121  | 2       | 0.01     | 0               | 50      | 0.7       |
# MAGIC | 0.778799301    | 0.039831664 | 0.16533426  | 0.075787195 | 0.817649608   | 0.907103589   | 0.043047575 | 0.180898874 | 0.082194521 | 0.622274579  | 2       | 0.01     | 0               | 50      | 0.8       |
# MAGIC | 0.578599749    | 0.332255607 | 0.503881254 | 0.422115344 | 0.829240669   | 0.446020906   | 0.25756275  | 0.389082698 | 0.326552131 | 0.840195356  | 3       | 0.01     | 0               | 50      | 0.5       |
# MAGIC | 0.603687087    | 0.307366342 | 0.506103819 | 0.407337453 | 0.832115134   | 0.555330269   | 0.198391407 | 0.408381231 | 0.292343332 | 0.855520195  | 3       | 0.01     | 0.5             | 5       | 0.6       |
# MAGIC | 0.659007061    | 0.062490243 | 0.226528931 | 0.11415569  | 0.817956931   | 0.615663062   | 0.145100711 | 0.37344576  | 0.234851215 | 0.857774488  | 3       | 0.01     | 0.5             | 5       | 0.7       |
# MAGIC | 0.804253434    | 0.017070579 | 0.078673405 | 0.03343156  | 0.814720952   | 0.781707753   | 0.011739835 | 0.055372779 | 0.023132265 | 0.850846147  | 3       | 0.01     | 0.5             | 5       | 0.8       |
# MAGIC | 0.595555238    | 0.328855034 | 0.512438152 | 0.4237325   | 0.832104543   | 0.764669012   | 0.371276374 | 0.630959985 | 0.499854203 | 0.730019688  | 0       | 0        | 1               | 5       | 0.5       |
# MAGIC | 0.645674423    | 0.261822542 | 0.499278365 | 0.372567899 | 0.834472393   | 0.789250149   | 0.338910535 | 0.623539913 | 0.474196972 | 0.726894226  | 0       | 0        | 1               | 5       | 0.6       |
# MAGIC | 0.702639512    | 0.134808859 | 0.381367025 | 0.22621581  | 0.826891884   | 0.805336645   | 0.19590551  | 0.496456855 | 0.31514831  | 0.690607552  | 0       | 0        | 1               | 5       | 0.7       |
# MAGIC | 0.771137989    | 0.025241924 | 0.111597751 | 0.048883721 | 0.815628445   | 0.715537883   | 0.037175175 | 0.153894114 | 0.070678318 | 0.644766103  | 0       | 0        | 1               | 5       | 0.8       |
# MAGIC | 0.547571062    | 0.346724973 | 0.490719598 | 0.424594439 | 0.823604713   | 0.718588154   | 0.403599994 | 0.621568046 | 0.516886896 | 0.735129537  | 1       | 0        | 1               | 5       | 0.5       |
# MAGIC | 0.608831816    | 0.293248695 | 0.50100031  | 0.395838583 | 0.831975329   | 0.7721326     | 0.366543128 | 0.632219211 | 0.497103594 | 0.739633085  | 1       | 0        | 1               | 5       | 0.6       |
# MAGIC | 0.645629632    | 0.182627545 | 0.428407614 | 0.284717737 | 0.827761072   | 0.811324663   | 0.306010942 | 0.609899958 | 0.444404032 | 0.731374562  | 1       | 0        | 1               | 5       | 0.7       |
# MAGIC | 0.704799332    | 0.049248131 | 0.19245045  | 0.092063302 | 0.817668849   | 0.866798986   | 0.090310025 | 0.318722095 | 0.163577267 | 0.675759001  | 1       | 0        | 1               | 5       | 0.8       |
# MAGIC | 0.575585365    | 0.342513754 | 0.506635003 | 0.429465403 | 0.829181888   | 0.759343819   | 0.396621109 | 0.6419306   | 0.521074265 | 0.713539914  | 2       | 0        | 1               | 5       | 0.5       |
# MAGIC | 0.624821146    | 0.298145484 | 0.512510419 | 0.403671372 | 0.834656151   | 0.79758347    | 0.356369418 | 0.639285964 | 0.492627316 | 0.711577748  | 2       | 0        | 1               | 5       | 0.6       |
# MAGIC | 0.686644629    | 0.14494475  | 0.392939344 | 0.239362206 | 0.827087292   | 0.833215113   | 0.217179177 | 0.531622273 | 0.344550564 | 0.67534346   | 2       | 0        | 1               | 5       | 0.7       |
# MAGIC | 0.775194506    | 0.03195185  | 0.137147544 | 0.061373996 | 0.816554826   | 0.891425025   | 0.053111525 | 0.214449593 | 0.100250102 | 0.62541921   | 2       | 0        | 1               | 5       | 0.8       |
# MAGIC | 0.555394157    | 0.345545681 | 0.495242419 | 0.426030783 | 0.825234706   | 0.4581992     | 0.252020132 | 0.393770099 | 0.325182426 | 0.842656273  | 3       | 0        | 1               | 5       | 0.5       |
# MAGIC | 0.605848715    | 0.297409132 | 0.501772268 | 0.398966786 | 0.831803045   | 0.517809411   | 0.223475377 | 0.409848971 | 0.312208359 | 0.851885721  | 3       | 0        | 1               | 5       | 0.6       |
# MAGIC | 0.640961223    | 0.13796022  | 0.370669833 | 0.227050243 | 0.823686619   | 0.584284905   | 0.18257375  | 0.405738311 | 0.278213163 | 0.857496838  | 3       | 0        | 1               | 5       | 0.7       |
# MAGIC | 0.744059419    | 0.026472939 | 0.115873949 | 0.051126834 | 0.815556424   | 0.668794803   | 0.064075461 | 0.231615451 | 0.11694658  | 0.85443865   | 3       | 0        | 1               | 5       | 0.8       |
# MAGIC | 0.592127303    | 0.321801853 | 0.506955099 | 0.416985632 | 0.831092549   | 0.774045151   | 0.36330598  | 0.631300537 | 0.494509083 | 0.730106057  | 0       | 0.01     | 1               | 5       | 0.5       |
# MAGIC | 0.627221241    | 0.249751258 | 0.481634372 | 0.357250185 | 0.831313906   | 0.802427358   | 0.343966448 | 0.633542111 | 0.481524039 | 0.730841413  | 0       | 0.01     | 1               | 5       | 0.6       |
# MAGIC | 0.797176309    | 0.021770819 | 0.098133971 | 0.04238413  | 0.815343364   | 0.893655547   | 0.02850504  | 0.126398241 | 0.055247834 | 0.645753303  | 0       | 0.01     | 1               | 5       | 0.7       |
# MAGIC | 0.843517444    | 0.01161894  | 0.055060975 | 0.022922142 | 0.814072944   | 0.931275851   | 0.006502046 | 0.031626968 | 0.012913928 | 0.638816264  | 0       | 0.01     | 1               | 5       | 0.8       |
# MAGIC | 0.582642139    | 0.323830347 | 0.502345236 | 0.416288876 | 0.829539872   | 0.733955711   | 0.378258961 | 0.61777118  | 0.499229748 | 0.733586408  | 1       | 0.01     | 1               | 5       | 0.5       |
# MAGIC | 0.621409573    | 0.275516998 | 0.496696051 | 0.38176793  | 0.832504715   | 0.776846167   | 0.350622554 | 0.624914514 | 0.483170455 | 0.736660566  | 1       | 0.01     | 1               | 5       | 0.6       |
# MAGIC | 0.787595823    | 0.023478629 | 0.104886291 | 0.04559796  | 0.815515119   | 0.841460132   | 0.201108397 | 0.514081488 | 0.32463036  | 0.706227126  | 1       | 0.01     | 1               | 5       | 0.7       |
# MAGIC | 0.851960211    | 0.009584803 | 0.045860249 | 0.018956341 | 0.813783097   | 0.910707005   | 0.025660409 | 0.115306386 | 0.049914411 | 0.657051083  | 1       | 0.01     | 1               | 5       | 0.8       |
# MAGIC | 0.567959301    | 0.333869375 | 0.498110094 | 0.420532684 | 0.827293822   | 0.748359304   | 0.388623772 | 0.631455937 | 0.511582312 | 0.708441995  | 2       | 0.01     | 1               | 5       | 0.5       |
# MAGIC | 0.612144512    | 0.3113086   | 0.512996702 | 0.412724477 | 0.833706645   | 0.792121456   | 0.355416229 | 0.635862666 | 0.490672898 | 0.710090989  | 2       | 0.01     | 1               | 5       | 0.6       |
# MAGIC | 0.738500111    | 0.047090798 | 0.187603498 | 0.088536054 | 0.818005827   | 0.838645662   | 0.126749482 | 0.394970496 | 0.220216362 | 0.647312172  | 2       | 0.01     | 1               | 5       | 0.7       |
# MAGIC | 0.84029678     | 0.015017633 | 0.070078447 | 0.029507906 | 0.814579735   | 0.929723963   | 0.028644972 | 0.127510384 | 0.055577588 | 0.617497096  | 2       | 0.01     | 1               | 5       | 0.8       |
# MAGIC | 0.536240833    | 0.334831429 | 0.478656128 | 0.412251181 | 0.820791678   | 0.502722323   | 0.221452563 | 0.400887775 | 0.30746481  | 0.849934098  | 3       | 0.01     | 1               | 5       | 0.5       |
# MAGIC | 0.599478806    | 0.313686933 | 0.507081158 | 0.41186098  | 0.831837996   | 0.564855652   | 0.187601275 | 0.402838906 | 0.281657691 | 0.856053703  | 3       | 0.01     | 1               | 5       | 0.6       |
# MAGIC | 0.5            | 5.45446E-05 | 0.000272604 | 0.000109077 | 0.812296615   | 0.75          | 6.89302E-06 | 3.44638E-05 | 1.37859E-05 | 0.648927271  | 1       | 2        | 0               | 5       | 0.6       |
# MAGIC | 0.425787106    | 0.000801241 | 0.003976277 | 0.001599473 | 0.812244188   | 0.91202346    | 0.001277766 | 0.006353227 | 0.002551957 | 0.607546631  | 2       | 2        | 0               | 5       | 0.5       |
# MAGIC | 0.297619048    | 0.000164574 | 0.000821055 | 0.000328967 | 0.812254603   | 0.72          | 3.69772E-05 | 0.000184848 | 7.39505E-05 | 0.607101895  | 2       | 2        | 0               | 5       | 0.6       |
# MAGIC | 0.243569132    | 0.000284949 | 0.001418106 | 0.000569231 | 0.812183995   | 0.7           | 7.51178E-05 | 0.000375428 | 0.000150219 | 0.849579772  | 3       | 2        | 0               | 5       | 0.5       |
# MAGIC | 0.595555238    | 0.328855034 | 0.512438152 | 0.4237325   | 0.832104543   | 0.764669012   | 0.371276374 | 0.630959985 | 0.499854203 | 0.730019688  | 0       | 0        | 0.5             | 5       | 0.5       |
# MAGIC | 0.645674423    | 0.261822542 | 0.499278365 | 0.372567899 | 0.834472393   | 0.789250149   | 0.338910535 | 0.623539913 | 0.474196972 | 0.726894226  | 0       | 0        | 0.5             | 5       | 0.6       |
# MAGIC | 0.702639512    | 0.134808859 | 0.381367025 | 0.22621581  | 0.826891884   | 0.805336645   | 0.19590551  | 0.496456855 | 0.31514831  | 0.690607552  | 0       | 0        | 0.5             | 5       | 0.7       |
# MAGIC | 0.771137989    | 0.025241924 | 0.111597751 | 0.048883721 | 0.815628445   | 0.715537883   | 0.037175175 | 0.153894114 | 0.070678318 | 0.644766103  | 0       | 0        | 0.5             | 5       | 0.8       |
# MAGIC | 0.547571062    | 0.346724973 | 0.490719598 | 0.424594439 | 0.823604713   | 0.718588154   | 0.403599994 | 0.621568046 | 0.516886896 | 0.735129537  | 1       | 0        | 0.5             | 5       | 0.5       |
# MAGIC | 0.608831816    | 0.293248695 | 0.50100031  | 0.395838583 | 0.831975329   | 0.7721326     | 0.366543128 | 0.632219211 | 0.497103594 | 0.739633085  | 1       | 0        | 0.5             | 5       | 0.6       |
# MAGIC | 0.645629632    | 0.182627545 | 0.428407614 | 0.284717737 | 0.827761072   | 0.811324663   | 0.306010942 | 0.609899958 | 0.444404032 | 0.731374562  | 1       | 0        | 0.5             | 5       | 0.7       |
# MAGIC | 0.704799332    | 0.049248131 | 0.19245045  | 0.092063302 | 0.817668849   | 0.866798986   | 0.090310025 | 0.318722095 | 0.163577267 | 0.675759001  | 1       | 0        | 0.5             | 5       | 0.8       |
# MAGIC | 0.575585365    | 0.342513754 | 0.506635003 | 0.429465403 | 0.829181888   | 0.759343819   | 0.396621109 | 0.6419306   | 0.521074265 | 0.713539914  | 2       | 0        | 0.5             | 5       | 0.5       |
# MAGIC | 0.624821146    | 0.298145484 | 0.512510419 | 0.403671372 | 0.834656151   | 0.79758347    | 0.356369418 | 0.639285964 | 0.492627316 | 0.711577748  | 2       | 0        | 0.5             | 5       | 0.6       |
# MAGIC | 0.686644629    | 0.14494475  | 0.392939344 | 0.239362206 | 0.827087292   | 0.833215113   | 0.217179177 | 0.531622273 | 0.344550564 | 0.67534346   | 2       | 0        | 0.5             | 5       | 0.7       |
# MAGIC | 0.775194506    | 0.03195185  | 0.137147544 | 0.061373996 | 0.816554826   | 0.891425025   | 0.053111525 | 0.214449593 | 0.100250102 | 0.62541921   | 2       | 0        | 0.5             | 5       | 0.8       |
# MAGIC | 0.555394157    | 0.345545681 | 0.495242419 | 0.426030783 | 0.825234706   | 0.4581992     | 0.252020132 | 0.393770099 | 0.325182426 | 0.842656273  | 3       | 0        | 0.5             | 5       | 0.5       |
# MAGIC | 0.605848715    | 0.297409132 | 0.501772268 | 0.398966786 | 0.831803045   | 0.517809411   | 0.223475377 | 0.409848971 | 0.312208359 | 0.851885721  | 3       | 0        | 0.5             | 5       | 0.6       |
# MAGIC | 0.640961223    | 0.13796022  | 0.370669833 | 0.227050243 | 0.823686619   | 0.584284905   | 0.18257375  | 0.405738311 | 0.278213163 | 0.857496838  | 3       | 0        | 0.5             | 5       | 0.7       |
# MAGIC | 0.744059419    | 0.026472939 | 0.115873949 | 0.051126834 | 0.815556424   | 0.668794803   | 0.064075461 | 0.231615451 | 0.11694658  | 0.85443865   | 3       | 0        | 0.5             | 5       | 0.8       |
# MAGIC | 0.593297738    | 0.328547515 | 0.510950837 | 0.422905031 | 0.831692014   | 0.7812675     | 0.362601796 | 0.63470048  | 0.495317078 | 0.73149847   | 0       | 0.01     | 0.5             | 5       | 0.5       |
# MAGIC | 0.63569969     | 0.265950063 | 0.497394465 | 0.375011189 | 0.833608853   | 0.808385267   | 0.327914592 | 0.625178936 | 0.466569317 | 0.727539175  | 0       | 0.01     | 0.5             | 5       | 0.6       |
# MAGIC | 0.73978414     | 0.057562421 | 0.219496294 | 0.106813694 | 0.819300783   | 0.867041154   | 0.081947548 | 0.297330234 | 0.149742345 | 0.661839098  | 0       | 0.01     | 0.5             | 5       | 0.7       |
# MAGIC | 0.81586582     | 0.016925754 | 0.078144131 | 0.033163505 | 0.814756609   | 0.904820766   | 0.008130334 | 0.03924125  | 0.016115858 | 0.639271522  | 0       | 0.01     | 0.5             | 5       | 0.8       |
# MAGIC | 0.581052836    | 0.331244651 | 0.504898922 | 0.42194711  | 0.829642784   | 0.737514126   | 0.386866043 | 0.624336721 | 0.507513703 | 0.736405663  | 1       | 0.01     | 0.5             | 5       | 0.5       |
# MAGIC | 0.62345599     | 0.275829219 | 0.497944409 | 0.38245348  | 0.832801094   | 0.779483139   | 0.354622803 | 0.628811923 | 0.487472087 | 0.738203695  | 1       | 0.01     | 0.5             | 5       | 0.6       |
# MAGIC | 0.672721915    | 0.085881413 | 0.28425312  | 0.152317572 | 0.820574381   | 0.833044483   | 0.245180057 | 0.563044072 | 0.378855955 | 0.717750984  | 1       | 0.01     | 0.5             | 5       | 0.7       |
# MAGIC | 0.828770038    | 0.016481873 | 0.076336875 | 0.032320975 | 0.814751137   | 0.901334313   | 0.033835528 | 0.147090846 | 0.065222639 | 0.659504118  | 1       | 0.01     | 0.5             | 5       | 0.8       |
# MAGIC | 0.568974138    | 0.340304697 | 0.501567924 | 0.425886019 | 0.82778349    | 0.757531536   | 0.392428311 | 0.63868834  | 0.517021219 | 0.711928855  | 2       | 0.01     | 0.5             | 5       | 0.5       |
# MAGIC | 0.619917873    | 0.306943151 | 0.514911853 | 0.410589162 | 0.834586602   | 0.794985319   | 0.353741986 | 0.636257088 | 0.489619571 | 0.710237889  | 2       | 0.01     | 0.5             | 5       | 0.6       |
# MAGIC | 0.706682916    | 0.10088682  | 0.321082026 | 0.176566775 | 0.823373471   | 0.841134473   | 0.16675055  | 0.465009699 | 0.278324676 | 0.660236154  | 2       | 0.01     | 0.5             | 5       | 0.7       |
# MAGIC | 0.821731894    | 0.018106926 | 0.083201247 | 0.035433082 | 0.814958019   | 0.92211801    | 0.031911288 | 0.140155295 | 0.061687777 | 0.618572212  | 2       | 0.01     | 0.5             | 5       | 0.8       |
# MAGIC | 0.54564099     | 0.341023181 | 0.487178461 | 0.419721993 | 0.823005249   | 0.48361164    | 0.238531126 | 0.40117385  | 0.319483721 | 0.847141454  | 3       | 0.01     | 0.5             | 5       | 0.5       |
# MAGIC | 0.8242419      | 0.017918841 | 0.082426472 | 0.035075156 | 0.814942838   | 0.920257235   | 0.032336525 | 0.141757939 | 0.062477674 | 0.618697319  | 2       | 0.01     | 0.5             | 10      | 0.8       |
# MAGIC | 0.552936835    | 0.339573988 | 0.491209044 | 0.420752245 | 0.824501086   | 0.482873634   | 0.23925011  | 0.401172478 | 0.319966131 | 0.847020385  | 3       | 0.01     | 0.5             | 10      | 0.5       |
# MAGIC | 0.608820269    | 0.30523346  | 0.507806804 | 0.406611365 | 0.832777793   | 0.553107219   | 0.200811272 | 0.409444374 | 0.294647672 | 0.855374105  | 3       | 0.01     | 0.5             | 10      | 0.6       |
# MAGIC | 0.661369784    | 0.064386138 | 0.231703025 | 0.117348119 | 0.818194175   | 0.615309331   | 0.158546793 | 0.390378528 | 0.2521278   | 0.858512198  | 3       | 0.01     | 0.5             | 10      | 0.7       |
# MAGIC | 0.807443863    | 0.017280293 | 0.079588315 | 0.033836445 | 0.81476667    | 0.779945243   | 0.0122281   | 0.05753249  | 0.024078692 | 0.850893767  | 3       | 0.01     | 0.5             | 10      | 0.8       |
# MAGIC | 0.603708808    | 0.326273569 | 0.515962671 | 0.423608517 | 0.833337893   | 0.768413071   | 0.369168262 | 0.631765783 | 0.498731317 | 0.73034418   | 0       | 0        | 1               | 10      | 0.5       |
# MAGIC | 0.653988859    | 0.253944609 | 0.497305659 | 0.365835055 | 0.834743705   | 0.791381271   | 0.329291861 | 0.617949706 | 0.465069438 | 0.724740628  | 0       | 0        | 1               | 10      | 0.6       |
# MAGIC | 0.711127124    | 0.131785395 | 0.378416125 | 0.222362741 | 0.826984734   | 0.80871996    | 0.199222065 | 0.501725322 | 0.319690729 | 0.691897451  | 0       | 0        | 1               | 10      | 0.7       |
# MAGIC | 0.797383019    | 0.028253162 | 0.123729674 | 0.054572685 | 0.816252269   | 0.751946108   | 0.044632424 | 0.180344173 | 0.084263325 | 0.647496039  | 0       | 0        | 1               | 10      | 0.8       |
# MAGIC | 0.55231409     | 0.34609865  | 0.493505198 | 0.425539738 | 0.824603115   | 0.718531241   | 0.406589266 | 0.622944575 | 0.519316976 | 0.735752273  | 1       | 0        | 1               | 10      | 0.5       |
# MAGIC | 0.613915226    | 0.289424931 | 0.501470062 | 0.393389733 | 0.832457584   | 0.771671075   | 0.368705238 | 0.633252119 | 0.498991718 | 0.740067872  | 1       | 0        | 1               | 10      | 0.6       |
# MAGIC | 0.6507694      | 0.183464522 | 0.431137971 | 0.286234098 | 0.828253212   | 0.810647594   | 0.311203682 | 0.613673414 | 0.449750375 | 0.732661176  | 1       | 0        | 1               | 10      | 0.7       |
# MAGIC | 0.712659301    | 0.05374618  | 0.20645158  | 0.099954178 | 0.818317386   | 0.862287247   | 0.108606393 | 0.361105042 | 0.192914864 | 0.680965145  | 1       | 0        | 1               | 10      | 0.8       |
# MAGIC | 0.567301954    | 0.348622749 | 0.504065341 | 0.431857261 | 0.827823031   | 0.765227551   | 0.394223757 | 0.644011724 | 0.520368347 | 0.7144649    | 2       | 0        | 1               | 10      | 0.5       |
# MAGIC | 0.619997587    | 0.304487704 | 0.513566217 | 0.408403776 | 0.834420142   | 0.801188607   | 0.352269062 | 0.638461854 | 0.489370294 | 0.711156419  | 2       | 0        | 1               | 10      | 0.6       |
# MAGIC | 0.679635564    | 0.176712277 | 0.433109542 | 0.280493375 | 0.829830778   | 0.836399124   | 0.209512579 | 0.523264942 | 0.335087823 | 0.673310265  | 2       | 0        | 1               | 10      | 0.7       |
# MAGIC | 0.774590289    | 0.04791555  | 0.192056011 | 0.090248404 | 0.818673252   | 0.893535749   | 0.056223769 | 0.224591216 | 0.10579088  | 0.626551633  | 2       | 0        | 1               | 10      | 0.8       |
# MAGIC | 0.569587145    | 0.336685005 | 0.50036198  | 0.42320941  | 0.827738301   | 0.410451291   | 0.274694968 | 0.373530931 | 0.329123608 | 0.831543001  | 3       | 0        | 1               | 10      | 0.5       |
# MAGIC | 0.614681951    | 0.262368928 | 0.484549776 | 0.367763031 | 0.830672959   | 0.505947539   | 0.230273536 | 0.40820927  | 0.316498217 | 0.850387701  | 3       | 0        | 1               | 10      | 0.6       |
# MAGIC | 0.649967861    | 0.106505854 | 0.321681893 | 0.183021249 | 0.821521945   | 0.568944873   | 0.191212294 | 0.407818787 | 0.286228319 | 0.856544433  | 3       | 0        | 1               | 10      | 0.7       |
# MAGIC | 0.75932565     | 0.02410119  | 0.106930006 | 0.046719491 | 0.815386611   | 0.652357801   | 0.088553124 | 0.286956219 | 0.155938641 | 0.855795423  | 3       | 0        | 1               | 10      | 0.8       |
# MAGIC | 0.596898018    | 0.32084356  | 0.50926371  | 0.417352531 | 0.83184947    | 0.775630711   | 0.362990541 | 0.631952577 | 0.494539534 | 0.730373239  | 0       | 0.01     | 1               | 10      | 0.5       |
# MAGIC | 0.633399345    | 0.247547844 | 0.482869975 | 0.355972853 | 0.831868711   | 0.804733728   | 0.343499953 | 0.634373118 | 0.481480386 | 0.731159448  | 0       | 0.01     | 1               | 10      | 0.6       |
# MAGIC | 0.80295203     | 0.021486811 | 0.097046306 | 0.041853628 | 0.81534001    | 0.894522463   | 0.03268572  | 0.14258802  | 0.063066982 | 0.647104549  | 0       | 0.01     | 1               | 10      | 0.7       |
# MAGIC | 0.8450087      | 0.012330841 | 0.058253911 | 0.024306981 | 0.814186623   | 0.932198327   | 0.006932998 | 0.033663533 | 0.013763632 | 0.638963981  | 0       | 0.01     | 1               | 10      | 0.8       |
# MAGIC | 0.588653946    | 0.321785865 | 0.504906657 | 0.416107725 | 0.83048973    | 0.731801817   | 0.380163732 | 0.617557941 | 0.500383326 | 0.733477509  | 1       | 0.01     | 1               | 10      | 0.5       |
# MAGIC | 0.622825451    | 0.254201345 | 0.482800958 | 0.361044994 | 0.83111585    | 0.7767053     | 0.350921252 | 0.625031205 | 0.483426708 | 0.736706545  | 1       | 0.01     | 1               | 10      | 0.6       |
# MAGIC | 0.801928007    | 0.02088776  | 0.094584282 | 0.04071502  | 0.815248925   | 0.839982488   | 0.207199528 | 0.521470512 | 0.332404438 | 0.707810588  | 1       | 0.01     | 1               | 10      | 0.7       |
# MAGIC | 0.858028879    | 0.008326515 | 0.040076913 | 0.016492979 | 0.813600927   | 0.911308747   | 0.025662706 | 0.115323378 | 0.049919661 | 0.657058343  | 1       | 0.01     | 1               | 10      | 0.8       |
# MAGIC | 0.569588329    | 0.333450886 | 0.498924409 | 0.420645593 | 0.8275902     | 0.749436694   | 0.388779898 | 0.632151888 | 0.511969204 | 0.708776152  | 2       | 0.01     | 1               | 10      | 0.5       |
# MAGIC | 0.614003911    | 0.310955001 | 0.51384737  | 0.412834741 | 0.833971074   | 0.793133047   | 0.354956069 | 0.636088608 | 0.490427763 | 0.710182197  | 2       | 0.01     | 1               | 10      | 0.6       |
# MAGIC | 0.739941632    | 0.047925895 | 0.190321188 | 0.090021135 | 0.818130804   | 0.838079343   | 0.134099719 | 0.40883225  | 0.231204742 | 0.649602038  | 2       | 0.01     | 1               | 10      | 0.7       |
# MAGIC | 0.84095711     | 0.014972493 | 0.069885469 | 0.029421167 | 0.814575498   | 0.930308265   | 0.028766175 | 0.127999342 | 0.055806743 | 0.617548753  | 2       | 0.01     | 1               | 10      | 0.8       |
# MAGIC | 0.428571429    | 8.18169E-05 | 0.000408772 | 0.000163603 | 0.812291496   | 1             | 1.77712E-05 | 8.88498E-05 | 3.55418E-05 | 0.636634414  | 0       | 2        | 0               | 10      | 0.7       |
# MAGIC | 0.491552695    | 0.000574599 | 0.002859625 | 0.001147857 | 0.812292908   | 0.905923345   | 0.000597395 | 0.002979116 | 0.001194002 | 0.649113608  | 1       | 2        | 0               | 10      | 0.5       |
# MAGIC | 0.504273504    | 5.5485E-05  | 0.000277303 | 0.000110958 | 0.812296791   | 0.75          | 6.89302E-06 | 3.44638E-05 | 1.37859E-05 | 0.648927271  | 1       | 2        | 0               | 10      | 0.6       |
# MAGIC | 0.4244857      | 0.000795599 | 0.003948393 | 0.001588221 | 0.812243482   | 0.911634757   | 0.001271603 | 0.00632274  | 0.002539664 | 0.607544209  | 2       | 2        | 0               | 10      | 0.5       |
# MAGIC | 0.297619048    | 0.000164574 | 0.000821055 | 0.000328967 | 0.812254603   | 0.72          | 3.69772E-05 | 0.000184848 | 7.39505E-05 | 0.607101895  | 2       | 2        | 0               | 10      | 0.6       |
# MAGIC | 0.24433657     | 0.000284008 | 0.001413469 | 0.000567357 | 0.812185054   | 0.7           | 7.51178E-05 | 0.000375428 | 0.000150219 | 0.849579772  | 3       | 2        | 0               | 10      | 0.5       |
# MAGIC | 0.603708808    | 0.326273569 | 0.515962671 | 0.423608517 | 0.833337893   | 0.768413071   | 0.369168262 | 0.631765783 | 0.498731317 | 0.73034418   | 0       | 0        | 0.5             | 10      | 0.5       |
# MAGIC | 0.653988859    | 0.253944609 | 0.497305659 | 0.365835055 | 0.834743705   | 0.791381271   | 0.329291861 | 0.617949706 | 0.465069438 | 0.724740628  | 0       | 0        | 0.5             | 10      | 0.6       |
# MAGIC | 0.711127124    | 0.131785395 | 0.378416125 | 0.222362741 | 0.826984734   | 0.80871996    | 0.199222065 | 0.501725322 | 0.319690729 | 0.691897451  | 0       | 0        | 0.5             | 10      | 0.7       |
# MAGIC | 0.797383019    | 0.028253162 | 0.123729674 | 0.054572685 | 0.816252269   | 0.751946108   | 0.044632424 | 0.180344173 | 0.084263325 | 0.647496039  | 0       | 0        | 0.5             | 10      | 0.8       |
# MAGIC | 0.55231409     | 0.34609865  | 0.493505198 | 0.425539738 | 0.824603115   | 0.718531241   | 0.406589266 | 0.622944575 | 0.519316976 | 0.735752273  | 1       | 0        | 0.5             | 10      | 0.5       |
# MAGIC | 0.613915226    | 0.289424931 | 0.501470062 | 0.393389733 | 0.832457584   | 0.771671075   | 0.368705238 | 0.633252119 | 0.498991718 | 0.740067872  | 1       | 0        | 0.5             | 10      | 0.6       |
# MAGIC | 0.6507694      | 0.183464522 | 0.431137971 | 0.286234098 | 0.828253212   | 0.810647594   | 0.311203682 | 0.613673414 | 0.449750375 | 0.732661176  | 1       | 0        | 0.5             | 10      | 0.7       |
# MAGIC | 0.712659301    | 0.05374618  | 0.20645158  | 0.099954178 | 0.818317386   | 0.862287247   | 0.108606393 | 0.361105042 | 0.192914864 | 0.680965145  | 1       | 0        | 0.5             | 10      | 0.8       |
# MAGIC | 0.567301954    | 0.348622749 | 0.504065341 | 0.431857261 | 0.827823031   | 0.765227551   | 0.394223757 | 0.644011724 | 0.520368347 | 0.7144649    | 2       | 0        | 0.5             | 10      | 0.5       |
# MAGIC | 0.619997587    | 0.304487704 | 0.513566217 | 0.408403776 | 0.834420142   | 0.801188607   | 0.352269062 | 0.638461854 | 0.489370294 | 0.711156419  | 2       | 0        | 0.5             | 10      | 0.6       |
# MAGIC | 0.679635564    | 0.176712277 | 0.433109542 | 0.280493375 | 0.829830778   | 0.836399124   | 0.209512579 | 0.523264942 | 0.335087823 | 0.673310265  | 2       | 0        | 0.5             | 10      | 0.7       |
# MAGIC | 0.774590289    | 0.04791555  | 0.192056011 | 0.090248404 | 0.818673252   | 0.893535749   | 0.056223769 | 0.224591216 | 0.10579088  | 0.626551633  | 2       | 0        | 0.5             | 10      | 0.8       |
# MAGIC | 0.569587145    | 0.336685005 | 0.50036198  | 0.42320941  | 0.827738301   | 0.410451291   | 0.274694968 | 0.373530931 | 0.329123608 | 0.831543001  | 3       | 0        | 0.5             | 10      | 0.5       |
# MAGIC | 0.614681951    | 0.262368928 | 0.484549776 | 0.367763031 | 0.830672959   | 0.505947539   | 0.230273536 | 0.40820927  | 0.316498217 | 0.850387701  | 3       | 0        | 0.5             | 10      | 0.6       |
# MAGIC | 0.649967861    | 0.106505854 | 0.321681893 | 0.183021249 | 0.821521945   | 0.568944873   | 0.191212294 | 0.407818787 | 0.286228319 | 0.856544433  | 3       | 0        | 0.5             | 10      | 0.7       |
# MAGIC | 0.75932565     | 0.02410119  | 0.106930006 | 0.046719491 | 0.815386611   | 0.652357801   | 0.088553124 | 0.286956219 | 0.155938641 | 0.855795423  | 3       | 0        | 0.5             | 10      | 0.8       |
# MAGIC | 0.600240115    | 0.32583157  | 0.513712523 | 0.422380216 | 0.832723954   | 0.784924041   | 0.361542187 | 0.635973447 | 0.495057155 | 0.732004582  | 0       | 0.01     | 0.5             | 10      | 0.5       |
# MAGIC | 0.643415038    | 0.243178634 | 0.484072541 | 0.352957155 | 0.832645049   | 0.811059147   | 0.327456983 | 0.626122399 | 0.466549351 | 0.72789757   | 0       | 0.01     | 0.5             | 10      | 0.6       |
# MAGIC | 0.745317507    | 0.046590492 | 0.18635545  | 0.087698848 | 0.818053488   | 0.870733404   | 0.075085635 | 0.279143171 | 0.13824964  | 0.659861469  | 0       | 0.01     | 0.5             | 10      | 0.7       |
# MAGIC | 0.821195449    | 0.01656369  | 0.076635438 | 0.032472405 | 0.814728718   | 0.925247525   | 0.008303604 | 0.040079258 | 0.016459492 | 0.639401481  | 0       | 0.01     | 0.5             | 10      | 0.8       |
# MAGIC | 0.584483549    | 0.329695773 | 0.506239513 | 0.421584148 | 0.830186821   | 0.734782477   | 0.388885698 | 0.623811823 | 0.508595691 | 0.736174153  | 1       | 0.01     | 0.5             | 10      | 0.5       |
# MAGIC | 0.624661817    | 0.266423097 | 0.492276516 | 0.373531934 | 0.832256704   | 0.778858694   | 0.355401714 | 0.628975343 | 0.488084945 | 0.738271454  | 1       | 0.01     | 0.5             | 10      | 0.6       |
# MAGIC | 0.668616695    | 0.075882823 | 0.260950612 | 0.136296992 | 0.819480658   | 0.827426673   | 0.256420272 | 0.572467983 | 0.391510948 | 0.720172559  | 1       | 0.01     | 0.5             | 10      | 0.7       |
# MAGIC | 0.829222665    | 0.015840504 | 0.073580161 | 0.031087155 | 0.814657581   | 0.897787584   | 0.034591462 | 0.149860939 | 0.066616223 | 0.659687228  | 1       | 0.01     | 0.5             | 10      | 0.8       |
# MAGIC | 0.574922077    | 0.33842573  | 0.504422643 | 0.42605549  | 0.82885303    | 0.758756178   | 0.392337922 | 0.639336404 | 0.517227605 | 0.712233148  | 2       | 0.01     | 0.5             | 10      | 0.5       |
# MAGIC | 0.624040926    | 0.30434664  | 0.515699805 | 0.409149726 | 0.835006898   | 0.796953867   | 0.35289562  | 0.636714739 | 0.489179727 | 0.710421918  | 2       | 0.01     | 0.5             | 10      | 0.6       |
# MAGIC | 0.70558921     | 0.093872196 | 0.306338715 | 0.165699578 | 0.822564653   | 0.838439585   | 0.16415804  | 0.460300984 | 0.274559995 | 0.65916346   | 2       | 0.01     | 0.5             | 10      | 0.7       |
# MAGIC | 0.623358096    | 0.239572107 | 0.472100467 | 0.346121186 | 0.8300945     | 0.515805743   | 0.225616234 | 0.410267768 | 0.313921499 | 0.85165327   | 3       | 0.01     | 0               | 50      | 0.6       |
# MAGIC | 0.66599103     | 0.072614849 | 0.252814139 | 0.130951674 | 0.8190909     | 0.586170456   | 0.178571045 | 0.40244799  | 0.273747589 | 0.85747101   | 3       | 0.01     | 0               | 50      | 0.7       |
# MAGIC | 0.778136656    | 0.019031363 | 0.086677152 | 0.037154028 | 0.814850341   | 0.676996805   | 0.056848058 | 0.212773221 | 0.104888528 | 0.854044774  | 3       | 0.01     | 0               | 50      | 0.8       |
# MAGIC | 0.750207854    | 0.028851272 | 0.12502384  | 0.055565618 | 0.815908936   | 0.78861268    | 0.030091122 | 0.130532653 | 0.05797027  | 0.644631301  | 0       | 0.5      | 0               | 50      | 0.5       |
# MAGIC | 0.704638956    | 0.004671087 | 0.022752133 | 0.009280651 | 0.812805877   | 0.860162602   | 0.007050732 | 0.034134464 | 0.013986815 | 0.638773482  | 0       | 0.5      | 0               | 50      | 0.6       |
# MAGIC | 0.437275986    | 0.000917854 | 0.004551059 | 0.001831863 | 0.812247189   | 0.668845316   | 0.000681971 | 0.003396003 | 0.001362552 | 0.636753072  | 0       | 0.5      | 0               | 50      | 0.7       |
# MAGIC | 0.267015707    | 0.000335731 | 0.001670257 | 0.00067062  | 0.812186642   | 0.592948718   | 0.00041096  | 0.002049117 | 0.00082135  | 0.636674774  | 0       | 0.5      | 0               | 50      | 0.8       |
# MAGIC | 0.662773325    | 0.075509475 | 0.259354703 | 0.135573159 | 0.819258418   | 0.844712889   | 0.161632083 | 0.457782087 | 0.271343739 | 0.695238887  | 1       | 0.5      | 0               | 50      | 0.5       |
# MAGIC | 0.756440954    | 0.00742747  | 0.03573387  | 0.014710498 | 0.813241884   | 0.915139064   | 0.013305823 | 0.062872529 | 0.026230266 | 0.653163818  | 1       | 0.5      | 0               | 50      | 0.6       |
# MAGIC | 0.631607629    | 0.001089952 | 0.005412398 | 0.002176148 | 0.812381874   | 0.931623932   | 0.001502678 | 0.007465225 | 0.003000516 | 0.64941449   | 1       | 0.5      | 0               | 50      | 0.7       |
# MAGIC | 0.553846154    | 0.000135421 | 0.000676444 | 0.000270776 | 0.812301558   | 0.910714286   | 0.000117181 | 0.000585605 | 0.000234332 | 0.648962764  | 1       | 0.5      | 0               | 50      | 0.8       |
# MAGIC | 0.722837597    | 0.052961866 | 0.204789985 | 0.098692587 | 0.818425946   | 0.851652026   | 0.115116057 | 0.373590461 | 0.202817666 | 0.644444391  | 2       | 0.5      | 0               | 50      | 0.5       |
# MAGIC | 0.745698454    | 0.007213994 | 0.034726183 | 0.014289746 | 0.813188928   | 0.923094609   | 0.016495921 | 0.076977202 | 0.032412621 | 0.6130344    | 2       | 0.5      | 0               | 50      | 0.6       |
# MAGIC | 0.504406211    | 0.00113039  | 0.005601734 | 0.002255724 | 0.812300322   | 0.923379175   | 0.001931029 | 0.009575051 | 0.003853999 | 0.607788774  | 2       | 0.5      | 0               | 50      | 0.7       |
# MAGIC | 0.349770642    | 0.000286829 | 0.001429458 | 0.000573189 | 0.812250367   | 0.870229008   | 0.000234189 | 0.001169684 | 0.000468251 | 0.607171309  | 2       | 0.5      | 0               | 50      | 0.8       |
# MAGIC | 0.72523214     | 0.027103023 | 0.117891919 | 0.052253263 | 0.815456514   | 0.636133195   | 0.059246461 | 0.215827619 | 0.108397306 | 0.853387776  | 3       | 0.5      | 0               | 50      | 0.5       |
# MAGIC | 0.684912959    | 0.003330042 | 0.016332574 | 0.006627859 | 0.812634123   | 0.749849669   | 0.006690847 | 0.032301348 | 0.013263347 | 0.850244034  | 3       | 0.5      | 0               | 50      | 0.6       |
# MAGIC | 0.327439424    | 0.000470212 | 0.002337633 | 0.000939076 | 0.812203588   | 0.780487805   | 0.000343396 | 0.001713961 | 0.000686489 | 0.849610442  | 3       | 0.5      | 0               | 50      | 0.7       |
# MAGIC | 0.256689792    | 0.00024357  | 0.001213244 | 0.000486678 | 0.812209943   | 0.333333333   | 1.07311E-05 | 5.36486E-05 | 2.14615E-05 | 0.8495717    | 3       | 0.5      | 0               | 50      | 0.8       |
# MAGIC | 0.699171902    | 0.004605257 | 0.022435187 | 0.009150244 | 0.812789108   | 0.848951049   | 0.005393566 | 0.026299485 | 0.010719032 | 0.638239119  | 0       | 1        | 0               | 50      | 0.5       |
# MAGIC | 0.367970005    | 0.000646071 | 0.003207828 | 0.001289878 | 0.81220959    | 0.630922693   | 0.000562015 | 0.002800097 | 0.001123029 | 0.636712712  | 0       | 1        | 0               | 50      | 0.6       |
# MAGIC | 0.260380623    | 0.000283068 | 0.00140921  | 0.000565521 | 0.812198822   | 0.57195572    | 0.000344317 | 0.001717452 | 0.000688221 | 0.636659437  | 0       | 1        | 0               | 50      | 0.7       |
# MAGIC | 0.292644757    | 0.000175859 | 0.000877188 | 0.000351507 | 0.812249837   | 0.727272727   | 0.000106627 | 0.000532824 | 0.000213223 | 0.636652172  | 0       | 1        | 0               | 50      | 0.8       |
# MAGIC | 0.744163265    | 0.008572906 | 0.040976311 | 0.016950539 | 0.813352562   | 0.907666836   | 0.016375513 | 0.076366551 | 0.032170626 | 0.654089856  | 1       | 1        | 0               | 50      | 0.5       |
# MAGIC | 0.54           | 0.000736352 | 0.003661787 | 0.001470699 | 0.812317091   | 0.927536232   | 0.000882306 | 0.00439481  | 0.001762936 | 0.649211213  | 1       | 1        | 0               | 50      | 0.6       |
# MAGIC | 0.46031746     | 8.18169E-05 | 0.000408794 | 0.000163605 | 0.812293967   | 0.818181818   | 2.06791E-05 | 0.000103385 | 4.13571E-05 | 0.648931305  | 1       | 1        | 0               | 50      | 0.7       |
# MAGIC | 0.754484954    | 0.009808624 | 0.046618859 | 0.019365488 | 0.813538615   | 0.91457761    | 0.021884315 | 0.099863324 | 0.042745794 | 0.614888408  | 2       | 1        | 0               | 50      | 0.5       |
# MAGIC | 0.444716243    | 0.000854846 | 0.004241614 | 0.001706411 | 0.812256721   | 0.916442049   | 0.001396915 | 0.006942247 | 0.002789578 | 0.607591831  | 2       | 1        | 0               | 50      | 0.6       |
# MAGIC | 0.331938633    | 0.000223821 | 0.001116094 | 0.00044734  | 0.812254073   | 0.784615385   | 0.000104769 | 0.000523563 | 0.000209509 | 0.60712288   | 2       | 1        | 0               | 50      | 0.7       |
# MAGIC | 0.45625        | 6.8651E-05  | 0.000343048 | 0.000137281 | 0.812294144   | 0.7           | 1.438E-05   | 7.18941E-05 | 2.87594E-05 | 0.607096245  | 2       | 1        | 0               | 50      | 0.8       |
# MAGIC | 0.664625434    | 0.003062021 | 0.015033067 | 0.006095957 | 0.812581343   | 0.739910314   | 0.006197216 | 0.029981622 | 0.012291484 | 0.850177849  | 3       | 1        | 0               | 50      | 0.5       |
# MAGIC | 0.270170244    | 0.000343255 | 0.001707596 | 0.000685639 | 0.812186996   | 0.8           | 0.00019316  | 0.000964868 | 0.000386227 | 0.849595107  | 3       | 1        | 0               | 50      | 0.6       |
# MAGIC | 0.295572917    | 0.000213476 | 0.001064307 | 0.000426644 | 0.812241187   | 0.4           | 1.07311E-05 | 5.36498E-05 | 2.14616E-05 | 0.849572507  | 3       | 1        | 0               | 50      | 0.7       |
# MAGIC | 0.595555238    | 0.328855034 | 0.512438152 | 0.4237325   | 0.832104543   | 0.764669012   | 0.371276374 | 0.630959985 | 0.499854203 | 0.730019688  | 0       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.645674423    | 0.261822542 | 0.499278365 | 0.372567899 | 0.834472393   | 0.789250149   | 0.338910535 | 0.623539913 | 0.474196972 | 0.726894226  | 0       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.702639512    | 0.134808859 | 0.381367025 | 0.22621581  | 0.826891884   | 0.805336645   | 0.19590551  | 0.496456855 | 0.31514831  | 0.690607552  | 0       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.771137989    | 0.025241924 | 0.111597751 | 0.048883721 | 0.815628445   | 0.715537883   | 0.037175175 | 0.153894114 | 0.070678318 | 0.644766103  | 0       | 0        | 0               | 5       | 0.8       |
# MAGIC | 0.547571062    | 0.346724973 | 0.490719598 | 0.424594439 | 0.823604713   | 0.718588154   | 0.403599994 | 0.621568046 | 0.516886896 | 0.735129537  | 1       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.608831816    | 0.293248695 | 0.50100031  | 0.395838583 | 0.831975329   | 0.7721326     | 0.366543128 | 0.632219211 | 0.497103594 | 0.739633085  | 1       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.645629632    | 0.182627545 | 0.428407614 | 0.284717737 | 0.827761072   | 0.811324663   | 0.306010942 | 0.609899958 | 0.444404032 | 0.731374562  | 1       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.704799332    | 0.049248131 | 0.19245045  | 0.092063302 | 0.817668849   | 0.866798986   | 0.090310025 | 0.318722095 | 0.163577267 | 0.675759001  | 1       | 0        | 0               | 5       | 0.8       |
# MAGIC | 0.575585365    | 0.342513754 | 0.506635003 | 0.429465403 | 0.829181888   | 0.759343819   | 0.396621109 | 0.6419306   | 0.521074265 | 0.713539914  | 2       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.624821146    | 0.298145484 | 0.512510419 | 0.403671372 | 0.834656151   | 0.79758347    | 0.356369418 | 0.639285964 | 0.492627316 | 0.711577748  | 2       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.686644629    | 0.14494475  | 0.392939344 | 0.239362206 | 0.827087292   | 0.833215113   | 0.217179177 | 0.531622273 | 0.344550564 | 0.67534346   | 2       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.775194506    | 0.03195185  | 0.137147544 | 0.061373996 | 0.816554826   | 0.891425025   | 0.053111525 | 0.214449593 | 0.100250102 | 0.62541921   | 2       | 0        | 0               | 5       | 0.8       |
# MAGIC | 0.555394157    | 0.345545681 | 0.495242419 | 0.426030783 | 0.825234706   | 0.4581992     | 0.252020132 | 0.393770099 | 0.325182426 | 0.842656273  | 3       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.605848715    | 0.297409132 | 0.501772268 | 0.398966786 | 0.831803045   | 0.517809411   | 0.223475377 | 0.409848971 | 0.312208359 | 0.851885721  | 3       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.640961223    | 0.13796022  | 0.370669833 | 0.227050243 | 0.823686619   | 0.584284905   | 0.18257375  | 0.405738311 | 0.278213163 | 0.857496838  | 3       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.744059419    | 0.026472939 | 0.115873949 | 0.051126834 | 0.815556424   | 0.668794803   | 0.064075461 | 0.231615451 | 0.11694658  | 0.85443865   | 3       | 0        | 0               | 5       | 0.8       |
# MAGIC | 0.600508521    | 0.32494381  | 0.513427338 | 0.42169979  | 0.832713716   | 0.767329246   | 0.369252676 | 0.631228744 | 0.498579771 | 0.730118973  | 0       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.653683485    | 0.2444783   | 0.489739372 | 0.355863342 | 0.833874164   | 0.792190153   | 0.326499558 | 0.616364423 | 0.462415507 | 0.724146532  | 0       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.714951554    | 0.10977853  | 0.340042739 | 0.190332164 | 0.824686962   | 0.804564489   | 0.158346032 | 0.442990775 | 0.264613568 | 0.680189885  | 0       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.783347584    | 0.020677105 | 0.0935122   | 0.040290704 | 0.815104355   | 0.688121678   | 0.029044841 | 0.124246916 | 0.055737083 | 0.642398598  | 0       | 0.01     | 0               | 5       | 0.8       |
# MAGIC | 0.548988356    | 0.344657921 | 0.490794818 | 0.423463266 | 0.82384231    | 0.721873423   | 0.4010266   | 0.622297841 | 0.515612145 | 0.735471558  | 1       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.612079942    | 0.279374618 | 0.494338963 | 0.383641764 | 0.831501371   | 0.776716057   | 0.363259754 | 0.6326921   | 0.495009949 | 0.739795223  | 1       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.64705584     | 0.155952415 | 0.39701259  | 0.251329722 | 0.825602223   | 0.820076468   | 0.276476657 | 0.588613682 | 0.413535824 | 0.724693855  | 1       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.720287177    | 0.038117271 | 0.157291232 | 0.072403007 | 0.816672918   | 0.876565065   | 0.062574818 | 0.243378451 | 0.116810925 | 0.667800553  | 1       | 0.01     | 0               | 5       | 0.8       |
# MAGIC | 0.579988947    | 0.340490901 | 0.508459795 | 0.429082635 | 0.829925217   | 0.761330369   | 0.395333072 | 0.642386715 | 0.520426362 | 0.713727978  | 2       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.63207563     | 0.283069544 | 0.507045015 | 0.391022901 | 0.834501519   | 0.80075032    | 0.349023289 | 0.636095638 | 0.486148766 | 0.710103904  | 2       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.700120279    | 0.11933418  | 0.35478272  | 0.203911952 | 0.825101786   | 0.839974015   | 0.183277286 | 0.489319832 | 0.300899998 | 0.665384923  | 2       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.785656331    | 0.025497719 | 0.112840127 | 0.049392454 | 0.815776899   | 0.903468307   | 0.04345227  | 0.182208171 | 0.082916666 | 0.622341572  | 2       | 0.01     | 0               | 5       | 0.8       |
# MAGIC | 0.558854417    | 0.343803075 | 0.496714683 | 0.425711566 | 0.825888893   | 0.46820324    | 0.248398382 | 0.397801288 | 0.324590187 | 0.844498127  | 3       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.61153204     | 0.283499318 | 0.496608436 | 0.387403    | 0.831707018   | 0.524644423   | 0.220282872 | 0.411054775 | 0.310285799 | 0.852686387  | 3       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.647534658    | 0.106521841 | 0.32123309  | 0.182948053 | 0.821407736   | 0.595303042   | 0.170962688 | 0.397820061 | 0.265637896 | 0.85780758   | 3       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.760573453    | 0.022002163 | 0.098601308 | 0.042767142 | 0.81512642    | 0.69622905    | 0.042795669 | 0.171749972 | 0.080634889 | 0.853202138  | 3       | 0.01     | 0               | 5       | 0.8       |
# MAGIC | 0.749062876    | 0.029880096 | 0.128842401 | 0.057467803 | 0.816026323   | 0.789867816   | 0.030928591 | 0.133701702 | 0.059526329 | 0.644876689  | 0       | 0.5      | 0               | 5       | 0.5       |
# MAGIC | 0.770973154    | 0.020741995 | 0.093633627 | 0.040397158 | 0.815033393   | 0.679086229   | 0.056622705 | 0.212304956 | 0.104529651 | 0.85406576   | 3       | 0.01     | 0               | 10      | 0.8       |
# MAGIC | 0.750207854    | 0.028851272 | 0.12502384  | 0.055565618 | 0.815908936   | 0.78861268    | 0.030091122 | 0.130532653 | 0.05797027  | 0.644631301  | 0       | 0.5      | 0               | 10      | 0.5       |
# MAGIC | 0.704638956    | 0.004671087 | 0.022752133 | 0.009280651 | 0.812805877   | 0.860162602   | 0.007050732 | 0.034134464 | 0.013986815 | 0.638773482  | 0       | 0.5      | 0               | 10      | 0.6       |
# MAGIC | 0.437275986    | 0.000917854 | 0.004551059 | 0.001831863 | 0.812247189   | 0.668845316   | 0.000681971 | 0.003396003 | 0.001362552 | 0.636753072  | 0       | 0.5      | 0               | 10      | 0.7       |
# MAGIC | 0.267015707    | 0.000335731 | 0.001670257 | 0.00067062  | 0.812186642   | 0.592948718   | 0.00041096  | 0.002049117 | 0.00082135  | 0.636674774  | 0       | 0.5      | 0               | 10      | 0.8       |
# MAGIC | 0.662765167    | 0.075510415 | 0.259355922 | 0.135574504 | 0.819258242   | 0.844700881   | 0.161629785 | 0.45777558  | 0.271339882 | 0.695237273  | 1       | 0.5      | 0               | 10      | 0.5       |
# MAGIC | 0.756440954    | 0.00742747  | 0.03573387  | 0.014710498 | 0.813241884   | 0.915139064   | 0.013305823 | 0.062872529 | 0.026230266 | 0.653163818  | 1       | 0.5      | 0               | 10      | 0.6       |
# MAGIC | 0.631607629    | 0.001089952 | 0.005412398 | 0.002176148 | 0.812381874   | 0.931623932   | 0.001502678 | 0.007465225 | 0.003000516 | 0.64941449   | 1       | 0.5      | 0               | 10      | 0.7       |
# MAGIC | 0.553846154    | 0.000135421 | 0.000676444 | 0.000270776 | 0.812301558   | 0.910714286   | 0.000117181 | 0.000585605 | 0.000234332 | 0.648962764  | 1       | 0.5      | 0               | 10      | 0.8       |
# MAGIC | 0.722841154    | 0.052962806 | 0.204793025 | 0.098694253 | 0.818426123   | 0.85165428    | 0.115118111 | 0.373595135 | 0.202820918 | 0.644445198  | 2       | 0.5      | 0               | 10      | 0.5       |
# MAGIC | 0.745698454    | 0.007213994 | 0.034726183 | 0.014289746 | 0.813188928   | 0.923094609   | 0.016495921 | 0.076977202 | 0.032412621 | 0.6130344    | 2       | 0.5      | 0               | 10      | 0.6       |
# MAGIC | 0.504406211    | 0.00113039  | 0.005601734 | 0.002255724 | 0.812300322   | 0.923379175   | 0.001931029 | 0.009575051 | 0.003853999 | 0.607788774  | 2       | 0.5      | 0               | 10      | 0.7       |
# MAGIC | 0.349770642    | 0.000286829 | 0.001429458 | 0.000573189 | 0.812250367   | 0.870229008   | 0.000234189 | 0.001169684 | 0.000468251 | 0.607171309  | 2       | 0.5      | 0               | 10      | 0.8       |
# MAGIC | 0.72523214     | 0.027103023 | 0.117891919 | 0.052253263 | 0.815456514   | 0.636154156   | 0.059251827 | 0.21584379  | 0.108406591 | 0.853388584  | 3       | 0.5      | 0               | 10      | 0.5       |
# MAGIC | 0.684912959    | 0.003330042 | 0.016332574 | 0.006627859 | 0.812634123   | 0.749849669   | 0.006690847 | 0.032301348 | 0.013263347 | 0.850244034  | 3       | 0.5      | 0               | 10      | 0.6       |
# MAGIC | 0.327439424    | 0.000470212 | 0.002337633 | 0.000939076 | 0.812203588   | 0.780487805   | 0.000343396 | 0.001713961 | 0.000686489 | 0.849610442  | 3       | 0.5      | 0               | 10      | 0.7       |
# MAGIC | 0.256689792    | 0.00024357  | 0.001213244 | 0.000486678 | 0.812209943   | 0.333333333   | 1.07311E-05 | 5.36486E-05 | 2.14615E-05 | 0.8495717    | 3       | 0.5      | 0               | 10      | 0.8       |
# MAGIC | 0.699171902    | 0.004605257 | 0.022435187 | 0.009150244 | 0.812789108   | 0.848951049   | 0.005393566 | 0.026299485 | 0.010719032 | 0.638239119  | 0       | 1        | 0               | 10      | 0.5       |
# MAGIC | 0.367970005    | 0.000646071 | 0.003207828 | 0.001289878 | 0.81220959    | 0.630922693   | 0.000562015 | 0.002800097 | 0.001123029 | 0.636712712  | 0       | 1        | 0               | 10      | 0.6       |
# MAGIC | 0.260380623    | 0.000283068 | 0.00140921  | 0.000565521 | 0.812198822   | 0.57195572    | 0.000344317 | 0.001717452 | 0.000688221 | 0.636659437  | 0       | 1        | 0               | 10      | 0.7       |
# MAGIC | 0.292644757    | 0.000175859 | 0.000877188 | 0.000351507 | 0.812249837   | 0.727272727   | 0.000106627 | 0.000532824 | 0.000213223 | 0.636652172  | 0       | 1        | 0               | 10      | 0.8       |
# MAGIC | 0.744163265    | 0.008572906 | 0.040976311 | 0.016950539 | 0.813352562   | 0.907666836   | 0.016375513 | 0.076366551 | 0.032170626 | 0.654089856  | 1       | 1        | 0               | 10      | 0.5       |
# MAGIC | 0.54           | 0.000736352 | 0.003661787 | 0.001470699 | 0.812317091   | 0.927536232   | 0.000882306 | 0.00439481  | 0.001762936 | 0.649211213  | 1       | 1        | 0               | 10      | 0.6       |
# MAGIC | 0.46031746     | 8.18169E-05 | 0.000408794 | 0.000163605 | 0.812293967   | 0.818181818   | 2.06791E-05 | 0.000103385 | 4.13571E-05 | 0.648931305  | 1       | 1        | 0               | 10      | 0.7       |
# MAGIC | 0.754484954    | 0.009808624 | 0.046618859 | 0.019365488 | 0.813538615   | 0.91457761    | 0.021884315 | 0.099863324 | 0.042745794 | 0.614888408  | 2       | 1        | 0               | 10      | 0.5       |
# MAGIC | 0.444716243    | 0.000854846 | 0.004241614 | 0.001706411 | 0.812256721   | 0.916442049   | 0.001396915 | 0.006942247 | 0.002789578 | 0.607591831  | 2       | 1        | 0               | 10      | 0.6       |
# MAGIC | 0.331938633    | 0.000223821 | 0.001116094 | 0.00044734  | 0.812254073   | 0.784615385   | 0.000104769 | 0.000523563 | 0.000209509 | 0.60712288   | 2       | 1        | 0               | 10      | 0.7       |
# MAGIC | 0.45625        | 6.8651E-05  | 0.000343048 | 0.000137281 | 0.812294144   | 0.7           | 1.438E-05   | 7.18941E-05 | 2.87594E-05 | 0.607096245  | 2       | 1        | 0               | 10      | 0.8       |
# MAGIC | 0.664625434    | 0.003062021 | 0.015033067 | 0.006095957 | 0.812581343   | 0.739910314   | 0.006197216 | 0.029981622 | 0.012291484 | 0.850177849  | 3       | 1        | 0               | 10      | 0.5       |
# MAGIC | 0.270170244    | 0.000343255 | 0.001707596 | 0.000685639 | 0.812186996   | 0.8           | 0.00019316  | 0.000964868 | 0.000386227 | 0.849595107  | 3       | 1        | 0               | 10      | 0.6       |
# MAGIC | 0.295572917    | 0.000213476 | 0.001064307 | 0.000426644 | 0.812241187   | 0.4           | 1.07311E-05 | 5.36498E-05 | 2.14616E-05 | 0.849572507  | 3       | 1        | 0               | 10      | 0.7       |
# MAGIC | 0.323874152    | 0.000493723 | 0.002453652 | 0.000985942 | 0.812195822   | 0.589970501   | 0.000444281 | 0.002214732 | 0.000887892 | 0.636677195  | 0       | 2        | 0               | 10      | 0.5       |
# MAGIC | 0.292810458    | 0.000210655 | 0.001050253 | 0.000421007 | 0.812240658   | 0.620481928   | 0.000228804 | 0.001142337 | 0.00045744  | 0.636660244  | 0       | 2        | 0               | 10      | 0.6       |
# MAGIC | 0.323874152    | 0.000493723 | 0.002453652 | 0.000985942 | 0.812195822   | 0.589970501   | 0.000444281 | 0.002214732 | 0.000887892 | 0.636677195  | 0       | 2        | 0               | 50      | 0.5       |
# MAGIC | 0.292810458    | 0.000210655 | 0.001050253 | 0.000421007 | 0.812240658   | 0.620481928   | 0.000228804 | 0.001142337 | 0.00045744  | 0.636660244  | 0       | 2        | 0               | 50      | 0.6       |
# MAGIC | 0.428571429    | 8.18169E-05 | 0.000408772 | 0.000163603 | 0.812291496   | 1             | 1.77712E-05 | 8.88498E-05 | 3.55418E-05 | 0.636634414  | 0       | 2        | 0               | 50      | 0.7       |
# MAGIC | 0.491552695    | 0.000574599 | 0.002859625 | 0.001147857 | 0.812292908   | 0.905923345   | 0.000597395 | 0.002979116 | 0.001194002 | 0.649113608  | 1       | 2        | 0               | 50      | 0.5       |
# MAGIC | 0.504273504    | 5.5485E-05  | 0.000277303 | 0.000110958 | 0.812296791   | 0.75          | 6.89302E-06 | 3.44638E-05 | 1.37859E-05 | 0.648927271  | 1       | 2        | 0               | 50      | 0.6       |
# MAGIC | 0.4244857      | 0.000795599 | 0.003948393 | 0.001588221 | 0.812243482   | 0.911634757   | 0.001271603 | 0.00632274  | 0.002539664 | 0.607544209  | 2       | 2        | 0               | 50      | 0.5       |
# MAGIC | 0.297619048    | 0.000164574 | 0.000821055 | 0.000328967 | 0.812254603   | 0.72          | 3.69772E-05 | 0.000184848 | 7.39505E-05 | 0.607101895  | 2       | 2        | 0               | 50      | 0.6       |
# MAGIC | 0.24433657     | 0.000284008 | 0.001413469 | 0.000567357 | 0.812185054   | 0.7           | 7.51178E-05 | 0.000375428 | 0.000150219 | 0.849579772  | 3       | 2        | 0               | 50      | 0.5       |
# MAGIC | 0.622834165    | 0.296571214 | 0.510510039 | 0.401813364 | 0.83425386    | 0.769233984   | 0.368004247 | 0.631525377 | 0.497840058 | 0.730234402  | 0       | 0        | 0.5             | 50      | 0.5       |
# MAGIC | 0.667710712    | 0.19747308  | 0.452300414 | 0.304802037 | 0.830916734   | 0.792029482   | 0.323693926 | 0.614276633 | 0.459567543 | 0.723364359  | 0       | 0        | 0.5             | 50      | 0.6       |
# MAGIC | 0.717600851    | 0.07869281  | 0.273496485 | 0.141832167 | 0.821254692   | 0.808541281   | 0.18706877  | 0.485776058 | 0.303839486 | 0.688507229  | 0       | 0        | 0.5             | 50      | 0.7       |
# MAGIC | 0.808753605    | 0.019514741 | 0.088985079 | 0.038109913 | 0.81509341    | 0.753584848   | 0.043661671 | 0.177233658 | 0.082541029 | 0.647305541  | 0       | 0        | 0.5             | 50      | 0.8       |
# MAGIC | 0.568419656    | 0.334325481 | 0.498596376 | 0.421020656 | 0.827403794   | 0.715092982   | 0.408808818 | 0.621905294 | 0.520216831 | 0.735265861  | 1       | 0        | 0.5             | 50      | 0.5       |
# MAGIC | 0.620456784    | 0.267153806 | 0.490675917 | 0.373491243 | 0.831767388   | 0.770797262   | 0.369263573 | 0.633109759 | 0.499319584 | 0.740015439  | 1       | 0        | 0.5             | 50      | 0.6       |
# MAGIC | 0.654247833    | 0.153881601 | 0.396435449 | 0.249159849 | 0.825916253   | 0.809610753   | 0.312786778 | 0.614423323 | 0.451240371 | 0.732913658  | 1       | 0        | 0.5             | 50      | 0.7       |
# MAGIC | 0.71946824     | 0.040360182 | 0.164817651 | 0.076432701 | 0.816918459   | 0.861461045   | 0.114070258 | 0.3728616   | 0.201463722 | 0.682532474  | 1       | 0        | 0.5             | 50      | 0.8       |
# MAGIC | 0.494407154    | 0.395798185 | 0.470941144 | 0.43964116  | 0.810615784   | 0.770194935   | 0.390811587 | 0.644972529 | 0.518517509 | 0.714829729  | 2       | 0        | 0.5             | 50      | 0.5       |
# MAGIC | 0.578098051    | 0.339797809 | 0.506987914 | 0.428014679 | 0.829529634   | 0.803622121   | 0.350672882 | 0.638641032 | 0.488278099 | 0.711205655  | 2       | 0        | 0.5             | 50      | 0.6       |
# MAGIC | 0.641272485    | 0.265798655 | 0.500007784 | 0.375823585 | 0.834278749   | 0.838675834   | 0.198704978 | 0.510099471 | 0.321288115 | 0.670147877  | 2       | 0        | 0.5             | 50      | 0.7       |
# MAGIC | 0.722352722    | 0.116545822 | 0.354163404 | 0.200708876 | 0.825764269   | 0.901885551   | 0.050701847 | 0.206968206 | 0.096006442 | 0.624846945  | 2       | 0        | 0.5             | 50      | 0.8       |
# MAGIC | 0.612648797    | 0.269108008 | 0.488042651 | 0.373955032 | 0.830872251   | 0.390661342   | 0.286585039 | 0.364208155 | 0.330626192 | 0.82544196   | 3       | 0        | 0.5             | 50      | 0.5       |
# MAGIC | 0.640018069    | 0.123914986 | 0.34916512  | 0.207630314 | 0.822473568   | 0.503626786   | 0.231346647 | 0.407667172 | 0.317051918 | 0.850074538  | 3       | 0        | 0.5             | 50      | 0.6       |
# MAGIC | 0.705356803    | 0.034821084 | 0.145394816 | 0.066365908 | 0.816102403   | 0.565301197   | 0.193551676 | 0.408414889 | 0.288369453 | 0.856299875  | 3       | 0        | 0.5             | 50      | 0.7       |
# MAGIC | 0.793391789    | 0.012849015 | 0.060336468 | 0.025288482 | 0.814080358   | 0.642785291   | 0.092571925 | 0.293680082 | 0.161836644 | 0.85575991   | 3       | 0        | 0.5             | 50      | 0.8       |
# MAGIC | 0.60095563     | 0.325147882 | 0.513790703 | 0.421981879 | 0.832802153   | 0.784614643   | 0.361495537 | 0.635782076 | 0.494951876 | 0.731926284  | 0       | 0.01     | 0.5             | 50      | 0.5       |
# MAGIC | 0.644218513    | 0.240025391 | 0.481913811 | 0.349742418 | 0.832468529   | 0.810851997   | 0.327414776 | 0.625992773 | 0.466472239 | 0.727848331  | 0       | 0.01     | 0.5             | 50      | 0.6       |
# MAGIC | 0.747309378    | 0.044730333 | 0.180448549 | 0.084408388 | 0.817853666   | 0.870577563   | 0.076745023 | 0.283690972 | 0.141055425 | 0.660369195  | 0       | 0.01     | 0.5             | 50      | 0.7       |
# MAGIC | 0.821308287    | 0.016282503 | 0.075430828 | 0.031931953 | 0.814687942   | 0.924053785   | 0.008243626 | 0.039797954 | 0.016341467 | 0.639377265  | 0       | 0.01     | 0.5             | 50      | 0.8       |
# MAGIC | 0.586539189    | 0.328273852 | 0.506796094 | 0.420950446 | 0.830479139   | 0.734240778   | 0.38905113  | 0.623584428 | 0.50860725  | 0.736074128  | 1       | 0.01     | 0.5             | 50      | 0.5       |
# MAGIC | 0.625038692    | 0.260162693 | 0.488121444 | 0.367400576 | 0.831834819   | 0.778655112   | 0.355507407 | 0.628935292 | 0.488144609 | 0.738256127  | 1       | 0.01     | 0.5             | 50      | 0.6       |
# MAGIC | 0.671361885    | 0.069655335 | 0.246130329 | 0.12621552  | 0.818971043   | 0.827076441   | 0.257353127 | 0.573261447 | 0.392557921 | 0.720385516  | 1       | 0.01     | 0.5             | 50      | 0.7       |
# MAGIC | 0.830740607    | 0.015116377 | 0.070453896 | 0.029692463 | 0.814555905   | 0.897566463   | 0.03483042  | 0.15075212  | 0.067058605 | 0.659758214  | 1       | 0.01     | 0.5             | 50      | 0.8       |
# MAGIC | 0.57568746     | 0.338103165 | 0.504750117 | 0.426009519 | 0.828984008   | 0.758932829   | 0.392348193 | 0.63944219  | 0.517277569 | 0.712283191  | 2       | 0.01     | 0.5             | 50      | 0.5       |
# MAGIC | 0.706865339    | 0.004773593 | 0.023240183 | 0.009483144 | 0.812821058   | 0.860032362   | 0.007084053 | 0.03429047  | 0.014052358 | 0.638783169  | 0       | 0.5      | 0               | 5       | 0.6       |
# MAGIC | 0.437779767    | 0.000919735 | 0.004560351 | 0.001835613 | 0.812247542   | 0.668845316   | 0.000681971 | 0.003396003 | 0.001362552 | 0.636753072  | 0       | 0.5      | 0               | 5       | 0.7       |
# MAGIC | 0.266516517    | 0.000333851 | 0.001660931 | 0.000666866 | 0.812186819   | 0.592948718   | 0.00041096  | 0.002049117 | 0.00082135  | 0.636674774  | 0       | 0.5      | 0               | 5       | 0.8       |
# MAGIC | 0.662303283    | 0.076938919 | 0.262648463 | 0.137862527 | 0.819374746   | 0.843867305   | 0.165289978 | 0.463389654 | 0.27643423  | 0.696218164  | 1       | 0.5      | 0               | 5       | 0.5       |
# MAGIC | 0.756882476    | 0.007498002 | 0.036061063 | 0.014848904 | 0.813251945   | 0.914751553   | 0.01353559  | 0.063896066 | 0.026676448 | 0.653234803  | 1       | 0.5      | 0               | 5       | 0.6       |
# MAGIC | 0.632497274    | 0.001090892 | 0.005417088 | 0.002178027 | 0.812382404   | 0.93220339    | 0.001516464 | 0.007533301 | 0.003028002 | 0.64941933   | 1       | 0.5      | 0               | 5       | 0.7       |
# MAGIC | 0.558139535    | 0.000135421 | 0.000676449 | 0.000270776 | 0.812301911   | 0.909090909   | 0.000114884 | 0.000574128 | 0.000229738 | 0.648961957  | 1       | 0.5      | 0               | 5       | 0.8       |
# MAGIC | 0.722354585    | 0.054459021 | 0.209206094 | 0.101282272 | 0.818589758   | 0.851004564   | 0.117990004 | 0.379488738 | 0.207245809 | 0.645335477  | 2       | 0.5      | 0               | 5       | 0.5       |
# MAGIC | 0.746153846    | 0.007297691 | 0.035114711 | 0.014454016 | 0.813200401   | 0.923644202   | 0.016723947 | 0.077972504 | 0.032853041 | 0.613120764  | 2       | 0.5      | 0               | 5       | 0.6       |
# MAGIC | 0.506094998    | 0.001132271 | 0.005611139 | 0.002259486 | 0.812301734   | 0.923753666   | 0.001941301 | 0.00962559  | 0.003874459 | 0.607792809  | 2       | 0.5      | 0               | 5       | 0.7       |
# MAGIC | 0.354534747    | 0.000283068 | 0.001410833 | 0.000565684 | 0.812253014   | 0.869230769   | 0.000232134 | 0.001159433 | 0.000464145 | 0.607170502  | 2       | 0.5      | 0               | 5       | 0.8       |
# MAGIC | 0.724216921    | 0.027353176 | 0.118815554 | 0.052715331 | 0.815475754   | 0.635315943   | 0.06063614  | 0.219414739 | 0.110706204 | 0.853458803  | 3       | 0.5      | 0               | 5       | 0.5       |
# MAGIC | 0.685581934    | 0.003340387 | 0.016382645 | 0.00664838  | 0.812636064   | 0.751489869   | 0.006765965 | 0.032653843 | 0.013411184 | 0.850254526  | 3       | 0.5      | 0               | 5       | 0.6       |
# MAGIC | 0.327653997    | 0.000470212 | 0.002337641 | 0.000939076 | 0.812203765   | 0.780487805   | 0.000343396 | 0.001713961 | 0.000686489 | 0.849610442  | 3       | 0.5      | 0               | 5       | 0.7       |
# MAGIC | 0.261977574    | 0.000241689 | 0.001204002 | 0.000482932 | 0.81221418    | 0.333333333   | 1.07311E-05 | 5.36486E-05 | 2.14615E-05 | 0.8495717    | 3       | 0.5      | 0               | 5       | 0.8       |
# MAGIC | 0.701080095    | 0.00470024  | 0.022887424 | 0.009337876 | 0.8128027     | 0.8495637     | 0.005406894 | 0.026363333 | 0.010745402 | 0.638244769  | 0       | 1        | 0               | 5       | 0.5       |
# MAGIC | 0.369296833    | 0.000647012 | 0.003212545 | 0.00129176  | 0.812210649   | 0.630922693   | 0.000562015 | 0.002800097 | 0.001123029 | 0.636712712  | 0       | 1        | 0               | 5       | 0.6       |
# MAGIC | 0.260680035    | 0.000281187 | 0.001399894 | 0.000561768 | 0.812199705   | 0.567164179   | 0.000337653 | 0.001684255 | 0.000674905 | 0.636657015  | 0       | 1        | 0               | 5       | 0.7       |
# MAGIC | 0.292063492    | 0.000173038 | 0.000863145 | 0.000345871 | 0.812250367   | 0.75          | 0.000106627 | 0.000532834 | 0.000213224 | 0.636653787  | 0       | 1        | 0               | 5       | 0.8       |
# MAGIC | 0.744157839    | 0.008654723 | 0.041349978 | 0.017110448 | 0.813362624   | 0.907023244   | 0.016676508 | 0.07767036  | 0.03275086  | 0.654180202  | 1       | 1        | 0               | 5       | 0.5       |
# MAGIC | 0.54           | 0.000736352 | 0.003661787 | 0.001470699 | 0.812317091   | 0.927710843   | 0.000884604 | 0.004406214 | 0.001767523 | 0.64921202   | 1       | 1        | 0               | 5       | 0.6       |
# MAGIC | 0.462765957    | 8.18169E-05 | 0.000408795 | 0.000163605 | 0.812294144   | 0.818181818   | 2.06791E-05 | 0.000103385 | 4.13571E-05 | 0.648931305  | 1       | 1        | 0               | 5       | 0.7       |
# MAGIC | 0.755536505    | 0.009945926 | 0.04724204  | 0.019633396 | 0.813559445   | 0.914249256   | 0.022077418 | 0.100663723 | 0.04311372  | 0.614953787  | 2       | 1        | 0               | 5       | 0.5       |
# MAGIC | 0.44596577     | 0.000857667 | 0.004255597 | 0.001712041 | 0.812257604   | 0.915775401   | 0.001407186 | 0.00699295  | 0.002810055 | 0.607595059  | 2       | 1        | 0               | 5       | 0.6       |
# MAGIC | 0.333800842    | 0.000223821 | 0.001116111 | 0.000447342 | 0.81225478    | 0.796875      | 0.000104769 | 0.000523568 | 0.00020951  | 0.607123687  | 2       | 1        | 0               | 5       | 0.7       |
# MAGIC | 0.45625        | 6.8651E-05  | 0.000343048 | 0.000137281 | 0.812294144   | 0.7           | 1.438E-05   | 7.18941E-05 | 2.87594E-05 | 0.607096245  | 2       | 1        | 0               | 5       | 0.8       |
# MAGIC | 0.665581774    | 0.003077068 | 0.015105991 | 0.006125815 | 0.812583991   | 0.741280913   | 0.006272334 | 0.030334956 | 0.012439412 | 0.850187535  | 3       | 1        | 0               | 5       | 0.5       |
# MAGIC | 0.268401487    | 0.000339493 | 0.00168892  | 0.000678128 | 0.812186642   | 0.8           | 0.00019316  | 0.000964868 | 0.000386227 | 0.849595107  | 3       | 1        | 0               | 5       | 0.6       |
# MAGIC | 0.292267366    | 0.000209715 | 0.001045572 | 0.000419128 | 0.812240658   | 0.4           | 1.07311E-05 | 5.36498E-05 | 2.14616E-05 | 0.849572507  | 3       | 1        | 0               | 5       | 0.7       |
# MAGIC | 0.324290999    | 0.000494663 | 0.002458316 | 0.000987819 | 0.812195998   | 0.589970501   | 0.000444281 | 0.002214732 | 0.000887892 | 0.636677195  | 0       | 2        | 0               | 5       | 0.5       |
# MAGIC | 0.294195251    | 0.000209715 | 0.001045592 | 0.00041913  | 0.81224154    | 0.632911392   | 0.00022214  | 0.001109144 | 0.000444125 | 0.636661858  | 0       | 2        | 0               | 5       | 0.6       |
# MAGIC | 0.439393939    | 8.18169E-05 | 0.00040878  | 0.000163603 | 0.812292378   | 1             | 1.77712E-05 | 8.88498E-05 | 3.55418E-05 | 0.636634414  | 0       | 2        | 0               | 5       | 0.7       |
# MAGIC | 0.492369478    | 0.00057648  | 0.002868964 | 0.001151612 | 0.812293261   | 0.90625       | 0.000599693 | 0.002990547 | 0.001198592 | 0.649114415  | 1       | 2        | 0               | 5       | 0.5       |
# MAGIC 
# MAGIC 
# MAGIC ### 14.5.2 Link to Modelling Notebook
# MAGIC 
# MAGIC [Link to Modelling Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238240075/command/1215577238240090)
# MAGIC 
# MAGIC [Link to Modelling Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/DLE_pipeline_cleaned_LR.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.6 Gradient Boosted Trees Model Information
# MAGIC 
# MAGIC ### 14.6.1 Cross Validation Results Table
# MAGIC 
# MAGIC | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | maxIter | maxDepth | maxBins | stepSize | threshold | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy | cv_fold |
# MAGIC |----------------|-------------|-------------|-------------|---------------|---------|----------|---------|----------|-----------|---------------|-------------|-------------|-------------|--------------|---------|
# MAGIC | 0.609770199    | 0.335231109 | 0.523951665 | 0.432621497 | 0.834951647   | 5       | 4        | 32      | 0.5      | 0.5       | 0.799987112   | 0.382543084 | 0.656670807 | 0.517584308 | 0.719817876  | 2       |
# MAGIC | 0.609770199    | 0.335231109 | 0.523951665 | 0.432621497 | 0.834951647   | 10      | 4        | 32      | 0.5      | 0.5       | 0.799987112   | 0.382543084 | 0.656670807 | 0.517584308 | 0.719817876  | 2       |
# MAGIC | 0.609767154    | 0.335228288 | 0.523948487 | 0.432618381 | 0.83495094    | 10      | 4        | 128     | 0.5      | 0.5       | 0.799971647   | 0.382547192 | 0.656664892 | 0.517584832 | 0.719815455  | 2       |
# MAGIC | 0.609767154    | 0.335228288 | 0.523948487 | 0.432618381 | 0.83495094    | 5       | 4        | 128     | 0.5      | 0.5       | 0.799971647   | 0.382547192 | 0.656664892 | 0.517584832 | 0.719815455  | 2       |
# MAGIC | 0.614859413    | 0.332490713 | 0.525587989 | 0.431593429 | 0.8356136     | 5       | 4        | 32      | 0.1      | 0.5       | 0.798543159   | 0.383073089 | 0.656203307 | 0.517766061 | 0.719633847  | 2       |
# MAGIC | 0.614859413    | 0.332490713 | 0.525587989 | 0.431593429 | 0.8356136     | 5       | 4        | 128     | 0.1      | 0.5       | 0.798543159   | 0.383073089 | 0.656203307 | 0.517766061 | 0.719633847  | 2       |
# MAGIC | 0.614859413    | 0.332490713 | 0.525587989 | 0.431593429 | 0.8356136     | 10      | 4        | 128     | 0.1      | 0.5       | 0.798543159   | 0.383073089 | 0.656203307 | 0.517766061 | 0.719633847  | 2       |
# MAGIC | 0.614859413    | 0.332490713 | 0.525587989 | 0.431593429 | 0.8356136     | 10      | 4        | 32      | 0.1      | 0.5       | 0.798543159   | 0.383073089 | 0.656203307 | 0.517766061 | 0.719633847  | 2       |
# MAGIC | 0.635244093    | 0.315411671 | 0.528136367 | 0.421526715 | 0.837505726   | 5       | 4        | 32      | 0.5      | 0.6       | 0.815545592   | 0.357217839 | 0.649004709 | 0.496822166 | 0.715702251  | 2       |
# MAGIC | 0.635244093    | 0.315411671 | 0.528136367 | 0.421526715 | 0.837505726   | 10      | 4        | 32      | 0.5      | 0.6       | 0.815545592   | 0.357217839 | 0.649004709 | 0.496822166 | 0.715702251  | 2       |
# MAGIC | 0.635477531    | 0.315021395 | 0.528046302 | 0.421229341 | 0.837508727   | 5       | 4        | 128     | 0.5      | 0.6       | 0.815599309   | 0.357063767 | 0.648930173 | 0.496683093 | 0.715666737  | 2       |
# MAGIC | 0.635477531    | 0.315021395 | 0.528046302 | 0.421229341 | 0.837508727   | 10      | 4        | 128     | 0.5      | 0.6       | 0.815599309   | 0.357063767 | 0.648930173 | 0.496683093 | 0.715666737  | 2       |
# MAGIC | 0.609629508    | 0.335359007 | 0.523931008 | 0.432692553 | 0.834936466   | 5       | 8        | 32      | 0.1      | 0.6       | 0.785332363   | 0.381380357 | 0.648051249 | 0.513426025 | 0.715979909  | 2       |
# MAGIC | 0.609629508    | 0.335359007 | 0.523931008 | 0.432692553 | 0.834936466   | 10      | 8        | 32      | 0.1      | 0.6       | 0.785332363   | 0.381380357 | 0.648051249 | 0.513426025 | 0.715979909  | 2       |
# MAGIC | 0.609629508    | 0.335359007 | 0.523931008 | 0.432692553 | 0.834936466   | 5       | 8        | 32      | 0.5      | 0.6       | 0.785332363   | 0.381380357 | 0.648051249 | 0.513426025 | 0.715979909  | 2       |
# MAGIC | 0.609629508    | 0.335359007 | 0.523931008 | 0.432692553 | 0.834936466   | 10      | 8        | 32      | 0.5      | 0.6       | 0.785332363   | 0.381380357 | 0.648051249 | 0.513426025 | 0.715979909  | 2       |
# MAGIC | 0.641381169    | 0.308858795 | 0.527745336 | 0.416939347 | 0.837855237   | 5       | 4        | 128     | 0.1      | 0.6       | 0.819558927   | 0.346208917 | 0.643574905 | 0.486784071 | 0.713171856  | 2       |
# MAGIC | 0.641381169    | 0.308858795 | 0.527745336 | 0.416939347 | 0.837855237   | 10      | 4        | 128     | 0.1      | 0.6       | 0.819558927   | 0.346208917 | 0.643574905 | 0.486784071 | 0.713171856  | 2       |
# MAGIC | 0.642981624    | 0.307413363 | 0.527762045 | 0.415955779 | 0.837959561   | 10      | 4        | 32      | 0.1      | 0.6       | 0.82010753    | 0.344060133 | 0.642353347 | 0.484752008 | 0.712623806  | 2       |
# MAGIC | 0.642981624    | 0.307413363 | 0.527762045 | 0.415955779 | 0.837959561   | 5       | 4        | 32      | 0.1      | 0.6       | 0.82010753    | 0.344060133 | 0.642353347 | 0.484752008 | 0.712623806  | 2       |
# MAGIC | 0.614866299    | 0.332449335 | 0.525571333 | 0.431560263 | 0.835611834   | 5       | 4        | 32      | 0.5      | 0.5       | 0.778480413   | 0.377611018 | 0.64214172  | 0.508545904 | 0.743772027  | 1       |
# MAGIC | 0.614866299    | 0.332449335 | 0.525571333 | 0.431560263 | 0.835611834   | 10      | 4        | 32      | 0.5      | 0.5       | 0.778480413   | 0.377611018 | 0.64214172  | 0.508545904 | 0.743772027  | 1       |
# MAGIC | 0.614866299    | 0.332449335 | 0.525571333 | 0.431560263 | 0.835611834   | 10      | 4        | 128     | 0.5      | 0.5       | 0.778480413   | 0.377611018 | 0.64214172  | 0.508545904 | 0.743772027  | 1       |
# MAGIC | 0.614866299    | 0.332449335 | 0.525571333 | 0.431560263 | 0.835611834   | 5       | 4        | 128     | 0.5      | 0.5       | 0.778480413   | 0.377611018 | 0.64214172  | 0.508545904 | 0.743772027  | 1       |
# MAGIC | 0.614866299    | 0.332449335 | 0.525571333 | 0.431560263 | 0.835611834   | 5       | 4        | 32      | 0.1      | 0.5       | 0.778480413   | 0.377611018 | 0.64214172  | 0.508545904 | 0.743772027  | 1       |
# MAGIC | 0.614866299    | 0.332449335 | 0.525571333 | 0.431560263 | 0.835611834   | 5       | 4        | 128     | 0.1      | 0.5       | 0.778480413   | 0.377611018 | 0.64214172  | 0.508545904 | 0.743772027  | 1       |
# MAGIC | 0.614866299    | 0.332449335 | 0.525571333 | 0.431560263 | 0.835611834   | 10      | 4        | 128     | 0.1      | 0.5       | 0.778480413   | 0.377611018 | 0.64214172  | 0.508545904 | 0.743772027  | 1       |
# MAGIC | 0.614866299    | 0.332449335 | 0.525571333 | 0.431560263 | 0.835611834   | 10      | 4        | 32      | 0.1      | 0.5       | 0.778480413   | 0.377611018 | 0.64214172  | 0.508545904 | 0.743772027  | 1       |
# MAGIC | 0.625167348    | 0.324089905 | 0.527212029 | 0.426882019 | 0.836655778   | 5       | 8        | 32      | 0.1      | 0.6       | 0.815084738   | 0.345832426 | 0.641104623 | 0.485620751 | 0.733784448  | 0       |
# MAGIC | 0.625167348    | 0.324089905 | 0.527212029 | 0.426882019 | 0.836655778   | 10      | 8        | 32      | 0.1      | 0.6       | 0.815084738   | 0.345832426 | 0.641104623 | 0.485620751 | 0.733784448  | 0       |
# MAGIC | 0.625167348    | 0.324089905 | 0.527212029 | 0.426882019 | 0.836655778   | 5       | 8        | 32      | 0.5      | 0.6       | 0.815084738   | 0.345832426 | 0.641104623 | 0.485620751 | 0.733784448  | 0       |
# MAGIC | 0.625167348    | 0.324089905 | 0.527212029 | 0.426882019 | 0.836655778   | 10      | 8        | 32      | 0.5      | 0.6       | 0.815084738   | 0.345832426 | 0.641104623 | 0.485620751 | 0.733784448  | 0       |
# MAGIC | 0.621339669    | 0.327833733 | 0.526979821 | 0.429207356 | 0.836330803   | 5       | 4        | 32      | 0.5      | 0.5       | 0.809273899   | 0.349739874 | 0.64086376  | 0.488407227 | 0.733762654  | 0       |
# MAGIC | 0.621339669    | 0.327833733 | 0.526979821 | 0.429207356 | 0.836330803   | 10      | 4        | 32      | 0.5      | 0.5       | 0.809273899   | 0.349739874 | 0.64086376  | 0.488407227 | 0.733762654  | 0       |
# MAGIC | 0.621339669    | 0.327833733 | 0.526979821 | 0.429207356 | 0.836330803   | 10      | 4        | 128     | 0.5      | 0.5       | 0.809273899   | 0.349739874 | 0.64086376  | 0.488407227 | 0.733762654  | 0       |
# MAGIC | 0.621339669    | 0.327833733 | 0.526979821 | 0.429207356 | 0.836330803   | 5       | 4        | 128     | 0.5      | 0.5       | 0.809273899   | 0.349739874 | 0.64086376  | 0.488407227 | 0.733762654  | 0       |
# MAGIC | 0.621339669    | 0.327833733 | 0.526979821 | 0.429207356 | 0.836330803   | 5       | 4        | 32      | 0.1      | 0.5       | 0.809273899   | 0.349739874 | 0.64086376  | 0.488407227 | 0.733762654  | 0       |
# MAGIC | 0.621339669    | 0.327833733 | 0.526979821 | 0.429207356 | 0.836330803   | 5       | 4        | 128     | 0.1      | 0.5       | 0.809273899   | 0.349739874 | 0.64086376  | 0.488407227 | 0.733762654  | 0       |
# MAGIC | 0.621339669    | 0.327833733 | 0.526979821 | 0.429207356 | 0.836330803   | 10      | 4        | 128     | 0.1      | 0.5       | 0.809273899   | 0.349739874 | 0.64086376  | 0.488407227 | 0.733762654  | 0       |
# MAGIC | 0.621339669    | 0.327833733 | 0.526979821 | 0.429207356 | 0.836330803   | 10      | 4        | 32      | 0.1      | 0.5       | 0.809273899   | 0.349739874 | 0.64086376  | 0.488407227 | 0.733762654  | 0       |
# MAGIC | 0.628291496    | 0.319081206 | 0.526289747 | 0.423225216 | 0.836755689   | 5       | 4        | 32      | 0.5      | 0.6       | 0.793793248   | 0.357678707 | 0.638170177 | 0.493147821 | 0.741877197  | 1       |
# MAGIC | 0.628291496    | 0.319081206 | 0.526289747 | 0.423225216 | 0.836755689   | 10      | 4        | 32      | 0.5      | 0.6       | 0.793793248   | 0.357678707 | 0.638170177 | 0.493147821 | 0.741877197  | 1       |
# MAGIC | 0.628291496    | 0.319081206 | 0.526289747 | 0.423225216 | 0.836755689   | 5       | 4        | 128     | 0.5      | 0.6       | 0.793793248   | 0.357678707 | 0.638170177 | 0.493147821 | 0.741877197  | 1       |
# MAGIC | 0.628291496    | 0.319081206 | 0.526289747 | 0.423225216 | 0.836755689   | 10      | 4        | 128     | 0.5      | 0.6       | 0.793793248   | 0.357678707 | 0.638170177 | 0.493147821 | 0.741877197  | 1       |
# MAGIC | 0.628291496    | 0.319081206 | 0.526289747 | 0.423225216 | 0.836755689   | 5       | 4        | 128     | 0.1      | 0.6       | 0.793793248   | 0.357678707 | 0.638170177 | 0.493147821 | 0.741877197  | 1       |
# MAGIC | 0.628291496    | 0.319081206 | 0.526289747 | 0.423225216 | 0.836755689   | 10      | 4        | 128     | 0.1      | 0.6       | 0.793793248   | 0.357678707 | 0.638170177 | 0.493147821 | 0.741877197  | 1       |
# MAGIC | 0.628291496    | 0.319081206 | 0.526289747 | 0.423225216 | 0.836755689   | 10      | 4        | 32      | 0.1      | 0.6       | 0.793793248   | 0.357678707 | 0.638170177 | 0.493147821 | 0.741877197  | 1       |
# MAGIC | 0.628291496    | 0.319081206 | 0.526289747 | 0.423225216 | 0.836755689   | 5       | 4        | 32      | 0.1      | 0.6       | 0.793793248   | 0.357678707 | 0.638170177 | 0.493147821 | 0.741877197  | 1       |
# MAGIC | 0.629461192    | 0.322024733 | 0.528541634 | 0.426074768 | 0.837160098   | 5       | 4        | 32      | 0.5      | 0.6       | 0.817736475   | 0.338550668 | 0.637322713 | 0.478852042 | 0.732228175  | 0       |
# MAGIC | 0.629461192    | 0.322024733 | 0.528541634 | 0.426074768 | 0.837160098   | 10      | 4        | 32      | 0.5      | 0.6       | 0.817736475   | 0.338550668 | 0.637322713 | 0.478852042 | 0.732228175  | 0       |
# MAGIC | 0.629461192    | 0.322024733 | 0.528541634 | 0.426074768 | 0.837160098   | 5       | 4        | 128     | 0.5      | 0.6       | 0.817736475   | 0.338550668 | 0.637322713 | 0.478852042 | 0.732228175  | 0       |
# MAGIC | 0.629461192    | 0.322024733 | 0.528541634 | 0.426074768 | 0.837160098   | 10      | 4        | 128     | 0.5      | 0.6       | 0.817736475   | 0.338550668 | 0.637322713 | 0.478852042 | 0.732228175  | 0       |
# MAGIC | 0.629461192    | 0.322024733 | 0.528541634 | 0.426074768 | 0.837160098   | 5       | 4        | 128     | 0.1      | 0.6       | 0.817736475   | 0.338550668 | 0.637322713 | 0.478852042 | 0.732228175  | 0       |
# MAGIC | 0.629461192    | 0.322024733 | 0.528541634 | 0.426074768 | 0.837160098   | 10      | 4        | 128     | 0.1      | 0.6       | 0.817736475   | 0.338550668 | 0.637322713 | 0.478852042 | 0.732228175  | 0       |
# MAGIC | 0.629461192    | 0.322024733 | 0.528541634 | 0.426074768 | 0.837160098   | 10      | 4        | 32      | 0.1      | 0.6       | 0.817736475   | 0.338550668 | 0.637322713 | 0.478852042 | 0.732228175  | 0       |
# MAGIC | 0.629461192    | 0.322024733 | 0.528541634 | 0.426074768 | 0.837160098   | 5       | 4        | 32      | 0.1      | 0.6       | 0.817736475   | 0.338550668 | 0.637322713 | 0.478852042 | 0.732228175  | 0       |
# MAGIC | 0.635666797    | 0.311016128 | 0.525880015 | 0.417674431 | 0.837215526   | 5       | 8        | 32      | 0.1      | 0.6       | 0.802733787   | 0.342870207 | 0.632949244 | 0.480503736 | 0.739717784  | 1       |
# MAGIC | 0.635666797    | 0.311016128 | 0.525880015 | 0.417674431 | 0.837215526   | 10      | 8        | 32      | 0.1      | 0.6       | 0.802733787   | 0.342870207 | 0.632949244 | 0.480503736 | 0.739717784  | 1       |
# MAGIC | 0.635666797    | 0.311016128 | 0.525880015 | 0.417674431 | 0.837215526   | 5       | 8        | 32      | 0.5      | 0.6       | 0.802733787   | 0.342870207 | 0.632949244 | 0.480503736 | 0.739717784  | 1       |
# MAGIC | 0.635666797    | 0.311016128 | 0.525880015 | 0.417674431 | 0.837215526   | 10      | 8        | 32      | 0.5      | 0.6       | 0.802733787   | 0.342870207 | 0.632949244 | 0.480503736 | 0.739717784  | 1       |
# MAGIC | 0.61506071     | 0.319651103 | 0.519111938 | 0.420674761 | 0.834745117   | 5       | 4        | 32      | 0.5      | 0.6       | 0.53883575    | 0.210978999 | 0.411075334 | 0.303229637 | 0.854148086  | 3       |
# MAGIC | 0.61506071     | 0.319651103 | 0.519111938 | 0.420674761 | 0.834745117   | 10      | 4        | 32      | 0.5      | 0.6       | 0.53883575    | 0.210978999 | 0.411075334 | 0.303229637 | 0.854148086  | 3       |
# MAGIC | 0.614774529    | 0.319868341 | 0.51906329  | 0.420795824 | 0.834714932   | 5       | 4        | 128     | 0.5      | 0.6       | 0.538396278   | 0.211113138 | 0.410972354 | 0.303298465 | 0.854102887  | 3       |
# MAGIC | 0.614774529    | 0.319868341 | 0.51906329  | 0.420795824 | 0.834714932   | 10      | 4        | 128     | 0.5      | 0.6       | 0.538396278   | 0.211113138 | 0.410972354 | 0.303298465 | 0.854102887  | 3       |
# MAGIC | 0.604572171    | 0.330923026 | 0.518774462 | 0.427723954 | 0.833784668   | 5       | 4        | 128     | 0.1      | 0.6       | 0.513401785   | 0.225895243 | 0.409232295 | 0.313744048 | 0.85134737   | 3       |
# MAGIC | 0.604572171    | 0.330923026 | 0.518774462 | 0.427723954 | 0.833784668   | 10      | 4        | 128     | 0.1      | 0.6       | 0.513401785   | 0.225895243 | 0.409232295 | 0.313744048 | 0.85134737   | 3       |
# MAGIC | 0.604572171    | 0.330923026 | 0.518774462 | 0.427723954 | 0.833784668   | 10      | 4        | 32      | 0.1      | 0.6       | 0.513401785   | 0.225895243 | 0.409232295 | 0.313744048 | 0.85134737   | 3       |
# MAGIC | 0.604572171    | 0.330923026 | 0.518774462 | 0.427723954 | 0.833784668   | 5       | 4        | 32      | 0.1      | 0.6       | 0.513401785   | 0.225895243 | 0.409232295 | 0.313744048 | 0.85134737   | 3       |
# MAGIC | 0.582671864    | 0.337778718 | 0.508882783 | 0.427647414 | 0.830288144   | 5       | 4        | 32      | 0.5      | 0.5       | 0.495457706   | 0.23702877  | 0.406760684 | 0.320655014 | 0.848919545  | 3       |
# MAGIC | 0.582671864    | 0.337778718 | 0.508882783 | 0.427647414 | 0.830288144   | 10      | 4        | 32      | 0.5      | 0.5       | 0.495457706   | 0.23702877  | 0.406760684 | 0.320655014 | 0.848919545  | 3       |
# MAGIC | 0.5829011      | 0.337562421 | 0.508924364 | 0.427535697 | 0.830319388   | 10      | 4        | 128     | 0.5      | 0.5       | 0.495383718   | 0.236937556 | 0.40666706  | 0.32055605  | 0.848909053  | 3       |
# MAGIC | 0.5829011      | 0.337562421 | 0.508924364 | 0.427535697 | 0.830319388   | 5       | 4        | 128     | 0.5      | 0.5       | 0.495383718   | 0.236937556 | 0.40666706  | 0.32055605  | 0.848909053  | 3       |
# MAGIC | 0.609177277    | 0.320074294 | 0.515968836 | 0.419653822 | 0.833831446   | 5       | 8        | 32      | 0.1      | 0.6       | 0.532738829   | 0.20867181  | 0.406484776 | 0.299881253 | 0.853431361  | 3       |
# MAGIC | 0.609177277    | 0.320074294 | 0.515968836 | 0.419653822 | 0.833831446   | 10      | 8        | 32      | 0.1      | 0.6       | 0.532738829   | 0.20867181  | 0.406484776 | 0.299881253 | 0.853431361  | 3       |
# MAGIC | 0.597576488    | 0.323386467 | 0.510935077 | 0.419665413 | 0.8321199     | 5       | 8        | 32      | 0.5      | 0.6       | 0.521741414   | 0.213227167 | 0.404646408 | 0.302732515 | 0.852246505  | 3       |
# MAGIC | 0.597576488    | 0.323386467 | 0.510935077 | 0.419665413 | 0.8321199     | 10      | 8        | 32      | 0.5      | 0.6       | 0.521741414   | 0.213227167 | 0.404646408 | 0.302732515 | 0.852246505  | 3       |
# MAGIC | 0.571713065    | 0.336935158 | 0.501784003 | 0.423992975 | 0.828162657   | 5       | 4        | 32      | 0.1      | 0.5       | 0.478481779   | 0.237484842 | 0.397754437 | 0.317423093 | 0.846360159  | 3       |
# MAGIC | 0.571713065    | 0.336935158 | 0.501784003 | 0.423992975 | 0.828162657   | 5       | 4        | 128     | 0.1      | 0.5       | 0.478481779   | 0.237484842 | 0.397754437 | 0.317423093 | 0.846360159  | 3       |
# MAGIC | 0.571713065    | 0.336935158 | 0.501784003 | 0.423992975 | 0.828162657   | 10      | 4        | 128     | 0.1      | 0.5       | 0.478481779   | 0.237484842 | 0.397754437 | 0.317423093 | 0.846360159  | 3       |
# MAGIC | 0.571713065    | 0.336935158 | 0.501784003 | 0.423992975 | 0.828162657   | 10      | 4        | 32      | 0.1      | 0.5       | 0.478481779   | 0.237484842 | 0.397754437 | 0.317423093 | 0.846360159  | 3       |
# MAGIC 
# MAGIC 
# MAGIC ### 14.6.2 Link to Modelling Notebook
# MAGIC 
# MAGIC [Link to Modelling Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238241208/command/1215577238241209)
# MAGIC 
# MAGIC [Link to Modelling Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/RW_pipeline_cleaned_MLPNN.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.7 Multilayer Perceptron Neural Network Model Information
# MAGIC 
# MAGIC ### 14.7.1 Cross Validation Results Table
# MAGIC 
# MAGIC | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy | cv_fold | maxIter | blockSize | stepSize | num_layers | threshold |
# MAGIC |----------------|-------------|-------------|-------------|---------------|---------------|-------------|-------------|-------------|--------------|---------|---------|-----------|----------|------------|-----------|
# MAGIC | 0.496715608    | 0.270014577 | 0.425300208 | 0.349850462 | 0.811626365   | 0.791593695   | 0.33435666  | 0.621587825 | 0.470135513 | 0.726137076  | 0       | 100     | 128       | 0.5      | 3          | 0.6       |
# MAGIC | 0.564322387    | 0.187504585 | 0.40253274  | 0.281482412 | 0.820319838   | 0.813521037   | 0.261616826 | 0.57212937  | 0.395913489 | 0.709901143  | 0       | 100     | 128       | 0.5      | 3          | 0.7       |
# MAGIC | 0.605977334    | 0.095186909 | 0.292285648 | 0.164529524 | 0.818545981   | 0.820555976   | 0.127797301 | 0.393712104 | 0.221151425 | 0.672910594  | 0       | 100     | 128       | 0.5      | 3          | 0.8       |
# MAGIC | 0.467660865    | 0.349292331 | 0.437976483 | 0.399901376 | 0.803229095   | 0.547737803   | 0.518524986 | 0.541634846 | 0.532731217 | 0.680657003  | 1       | 100     | 128       | 0.5      | 3          | 0.5       |
# MAGIC | 0.630521026    | 0.244227206 | 0.478995673 | 0.352079337 | 0.831275778   | 0.70022473    | 0.389460116 | 0.6038568   | 0.500529315 | 0.727119463  | 1       | 100     | 128       | 0.5      | 3          | 0.7       |
# MAGIC | 0.693179895    | 0.119411294 | 0.353483669 | 0.20372731  | 0.824789521   | 0.796074279   | 0.288504973 | 0.588872558 | 0.423521634 | 0.724266328  | 1       | 100     | 128       | 0.5      | 3          | 0.8       |
# MAGIC | 0.49839751     | 0.386665726 | 0.471167584 | 0.435479019 | 0.811829894   | 0.635034444   | 0.496527229 | 0.60147777  | 0.557303946 | 0.690060915  | 2       | 100     | 128       | 0.5      | 3          | 0.6       |
# MAGIC | 0.556783775    | 0.345697089 | 0.496188121 | 0.426554264 | 0.825531967   | 0.38255283    | 0.215341196 | 0.331128778 | 0.275565168 | 0.82968339   | 3       | 100     | 128       | 0.5      | 3          | 0.5       |
# MAGIC | 0.618925238    | 0.301217849 | 0.511107778 | 0.405222474 | 0.834024559   | 0.50502481    | 0.172566989 | 0.364557593 | 0.257236321 | 0.850089873  | 3       | 100     | 128       | 0.5      | 3          | 0.6       |
# MAGIC | 0.664051796    | 0.235638313 | 0.486977322 | 0.347844315 | 0.834150419   | 0.588097794   | 0.132293131 | 0.348175805 | 0.215997442 | 0.85553553   | 3       | 100     | 128       | 0.5      | 3          | 0.7       |
# MAGIC | 0.714163851    | 0.118885597 | 0.356826724 | 0.203838551 | 0.825680421   | 0.623390331   | 0.063377939 | 0.225277114 | 0.115058299 | 0.85334742   | 3       | 100     | 128       | 0.5      | 3          | 0.8       |
# MAGIC | 0.647378619    | 0.283098698 | 0.514874655 | 0.393931245 | 0.836491084   | 0.784178475   | 0.357852437 | 0.633285949 | 0.4914406   | 0.730873701  | 0       | 100     | 128       | 0.5      | 4          | 0.5       |
# MAGIC | 0.680142308    | 0.222753562 | 0.482142028 | 0.335596001 | 0.834445032   | 0.806085429   | 0.325264458 | 0.622147875 | 0.463501068 | 0.726387307  | 0       | 100     | 128       | 0.5      | 4          | 0.6       |
# MAGIC | 0.714668985    | 0.143041332 | 0.397204218 | 0.23837233  | 0.828426379   | 0.82396276    | 0.266592768 | 0.581015134 | 0.402845169 | 0.712803818  | 0       | 100     | 128       | 0.5      | 4          | 0.7       |
# MAGIC | 0.75485051     | 0.059382141 | 0.225844341 | 0.110102781 | 0.819822932   | 0.860321881   | 0.142734014 | 0.428983456 | 0.244846166 | 0.680072841  | 0       | 100     | 128       | 0.5      | 4          | 0.8       |
# MAGIC | 0.598651648    | 0.332025203 | 0.515809475 | 0.427145974 | 0.832836751   | 0.708765842   | 0.42892724  | 0.626958416 | 0.534430561 | 0.737635004  | 1       | 100     | 128       | 0.5      | 4          | 0.5       |
# MAGIC | 0.689627445    | 0.228253162 | 0.491094806 | 0.342985011 | 0.835858257   | 0.816058252   | 0.326777307 | 0.627998781 | 0.46668012  | 0.737789881  | 1       | 100     | 128       | 0.5      | 4          | 0.7       |
# MAGIC | 0.738349094    | 0.120937603 | 0.365331011 | 0.207833241 | 0.826952607   | 0.843217869   | 0.249288296 | 0.571092295 | 0.384811276 | 0.720171753  | 1       | 100     | 128       | 0.5      | 4          | 0.8       |
# MAGIC | 0.616369846    | 0.328926506 | 0.52466982  | 0.428945651 | 0.835609716   | 0.811645572   | 0.361745486 | 0.649972428 | 0.500445474 | 0.716241423  | 2       | 100     | 128       | 0.5      | 4          | 0.6       |
# MAGIC | 0.668133896    | 0.273789439 | 0.518711707 | 0.388413786 | 0.838161501   | 0.833705375   | 0.313800492 | 0.626205748 | 0.455975284 | 0.705794564  | 2       | 100     | 128       | 0.5      | 4          | 0.7       |
# MAGIC | 0.718526059    | 0.185193022 | 0.455924694 | 0.294485344 | 0.833440629   | 0.859895528   | 0.214062824 | 0.536293465 | 0.342791068 | 0.677496112  | 2       | 100     | 128       | 0.5      | 4          | 0.8       |
# MAGIC | 0.670418539    | 0.216888137 | 0.47271964  | 0.327746498 | 0.832993678   | 0.604270189   | 0.101135351 | 0.302896389 | 0.173270763 | 0.854823648  | 3       | 100     | 128       | 0.5      | 4          | 0.7       |
# MAGIC | 0.43557598     | 0.340495604 | 0.412536546 | 0.382211408 | 0.793390708   | 0.761027279   | 0.375868013 | 0.631587317 | 0.5032052   | 0.730319964  | 0       | 100     | 128       | 0.5      | 3          | 0.5       |
# MAGIC | 0.546699844    | 0.309717403 | 0.474141332 | 0.395420471 | 0.822228557   | 0.607159012   | 0.453376775 | 0.568586872 | 0.519118351 | 0.705109911  | 1       | 100     | 128       | 0.5      | 3          | 0.6       |
# MAGIC | 0.571435124    | 0.328057554 | 0.497603277 | 0.416820756 | 0.827692229   | 0.727728579   | 0.398905476 | 0.62473337  | 0.515331334 | 0.705185978  | 2       | 100     | 128       | 0.5      | 3          | 0.7       |
# MAGIC | 0.674923718    | 0.226944092 | 0.483888109 | 0.339672729 | 0.834377424   | 0.819716254   | 0.28842389  | 0.599027909 | 0.426707311 | 0.695492991  | 2       | 100     | 128       | 0.5      | 3          | 0.8       |
# MAGIC | 0.651162693    | 0.291465651 | 0.522258987 | 0.402685872 | 0.837697251   | 0.780511625   | 0.369100438 | 0.638232698 | 0.501190257 | 0.742067567  | 1       | 100     | 128       | 0.5      | 4          | 0.6       |
# MAGIC | 0.539930159    | 0.386778577 | 0.500308982 | 0.450699147 | 0.823034728   | 0.769602559   | 0.406263931 | 0.652831686 | 0.531798063 | 0.718930018  | 2       | 100     | 128       | 0.5      | 4          | 0.5       |
# MAGIC | 0.583216793    | 0.321987116 | 0.501795051 | 0.414908268 | 0.829543932   | 0.494001684   | 0.19200103  | 0.375786038 | 0.276526114 | 0.848871925  | 3       | 100     | 128       | 0.5      | 4          | 0.5       |
# MAGIC | 0.725979157    | 0.114385668 | 0.350824162 | 0.197632286 | 0.825663122   | 0.650160716   | 0.040155816 | 0.16100302  | 0.075639892 | 0.852363538  | 3       | 100     | 128       | 0.5      | 4          | 0.8       |
# MAGIC | 0.632976332    | 0.278011943 | 0.504219339 | 0.386338628 | 0.834222263   | 0.56864096    | 0.150573578 | 0.365615017 | 0.23809948  | 0.855041571  | 3       | 100     | 128       | 0.5      | 4          | 0.6       |
# MAGIC | 0.432078624    | 0.452310152 | 0.435978832 | 0.441962977 | 0.785604553   | 0.587863309   | 0.57118822  | 0.584450854 | 0.579405814 | 0.674178752  | 2       | 100     | 128       | 0.5      | 3          | 0.5       |
# MAGIC 
# MAGIC ### 14.7.2 Link to Modelling Notebook
# MAGIC 
# MAGIC [Link to Modelling Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238241208/command/1215577238241209)
# MAGIC 
# MAGIC [Link to Modelling Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/RW_pipeline_cleaned_MLPNN.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.8 Link to Random Forest, Linear SVM Modelling Notebook
# MAGIC 
# MAGIC Please note that both the Random Forest and Linear SVM models are stored in the same notebook. The preceeding pipeline code should be run before either model creation cell is run. The notebook operates on an older version of the data pipeline, so its performance will not be optimal, but it will be functional.
# MAGIC 
# MAGIC [Link to Modelling Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/2647101326240378/command/1860389250617137)
# MAGIC 
# MAGIC [Link to Modelling Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/Data%20Pipeline%20v2.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.9 Link to Logistic Regression - No-Weather Feature Set Experiment Modelling Notebook
# MAGIC 
# MAGIC [Link to Modelling Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238253136/command/1215577238253137)
# MAGIC 
# MAGIC [Link to Modelling Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/RW_pipeline_experiment_no_weather_features.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.10 Link to Logistic Regression - Only-Weather Feature Set Experiment Modelling Notebook
# MAGIC 
# MAGIC [Link to Modelling Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238253335/command/1215577238253336)
# MAGIC 
# MAGIC [Link to Modelling Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/RW_pipeline_experiment_only_weather_features.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.11 Link to Logistic Regression - Ensemble Prediction Experiment Modelling Notebook
# MAGIC 
# MAGIC [Link to Modelling Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238260548/command/1215577238260549)
# MAGIC 
# MAGIC [Link to Modelling Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/DLE_pipeline_ensemble.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.12 Logistic Regression Downsampling Experiment Model Information
# MAGIC 
# MAGIC ### 14.12.1 Cross Validation Results Table
# MAGIC 
# MAGIC | test_Precision | test_Recall | test_F0.5   | test_F1     | test_Accuracy | val_Precision | val_Recall  | val_F0.5    | val_F1      | val_Accuracy | cv_fold | regParam | elasticNetParam | maxIter | threshold |
# MAGIC |----------------|-------------|-------------|-------------|---------------|---------------|-------------|-------------|-------------|--------------|---------|----------|-----------------|---------|-----------|
# MAGIC | 0.597109435    | 0.32935722  | 0.513602423 | 0.424542648 | 0.832404981   | 0.878181789   | 0.372070855 | 0.690366898 | 0.522687716 | 0.622559814  | 2       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.361778443    | 0.479012555 | 0.38039827  | 0.41222234  | 0.743592604   | 0.379841434   | 0.640754341 | 0.413518052 | 0.476947002 | 0.559632209  | 3       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.500832445    | 0.372577232 | 0.468572369 | 0.427288067 | 0.812529093   | 0.549208001   | 0.359174435 | 0.496653667 | 0.434313711 | 0.706825666  | 3       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.573209938    | 0.341215028 | 0.504594365 | 0.427783257 | 0.828656739   | 0.709042798   | 0.268249157 | 0.5336586   | 0.389239125 | 0.736220219  | 3       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.605058754    | 0.322257018 | 0.514718827 | 0.420535131 | 0.833302413   | 0.88091796    | 0.367273498 | 0.688374841 | 0.518410567 | 0.620986341  | 2       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.362180942    | 0.478813185 | 0.380729001 | 0.412409563 | 0.743897279   | 0.378795048   | 0.643025593 | 0.41271328  | 0.476746898 | 0.557716628  | 3       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.505885646    | 0.37053275  | 0.471442732 | 0.427757337 | 0.813914958   | 0.556530524   | 0.353104573 | 0.499031441 | 0.432071    | 0.709135521  | 3       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.580944334    | 0.336246767 | 0.507132949 | 0.425954098 | 0.829884441   | 0.712841179   | 0.265277536 | 0.532993067 | 0.386662168 | 0.73629585   | 3       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.597109435    | 0.32935722  | 0.513602423 | 0.424542648 | 0.832404981   | 0.878181789   | 0.372070855 | 0.690366898 | 0.522687716 | 0.622559814  | 2       | 0        | 0.5             | 5       | 0.7       |
# MAGIC | 0.361778443    | 0.479012555 | 0.38039827  | 0.41222234  | 0.743592604   | 0.379841434   | 0.640754341 | 0.413518052 | 0.476947002 | 0.559632209  | 3       | 0        | 0.5             | 5       | 0.5       |
# MAGIC | 0.500832445    | 0.372577232 | 0.468572369 | 0.427288067 | 0.812529093   | 0.549208001   | 0.359174435 | 0.496653667 | 0.434313711 | 0.706825666  | 3       | 0        | 0.5             | 5       | 0.6       |
# MAGIC | 0.573209938    | 0.341215028 | 0.504594365 | 0.427783257 | 0.828656739   | 0.709042798   | 0.268249157 | 0.5336586   | 0.389239125 | 0.736220219  | 3       | 0        | 0.5             | 5       | 0.7       |
# MAGIC | 0.59510506     | 0.327128415 | 0.511330796 | 0.422183276 | 0.83192255    | 0.877571775   | 0.367927582 | 0.687194741 | 0.51847937  | 0.620415972  | 2       | 0.01     | 0.5             | 5       | 0.7       |
# MAGIC | 0.431287174    | 0.411540885 | 0.427187764 | 0.421182715 | 0.787682379   | 0.432710229   | 0.497906807 | 0.444346904 | 0.463024768 | 0.638135055  | 3       | 0.01     | 0.5             | 5       | 0.5       |
# MAGIC | 0.508891582    | 0.364464193 | 0.47152133  | 0.424735864 | 0.814687236   | 0.694901348   | 0.28485225  | 0.539560311 | 0.404069376 | 0.736726081  | 3       | 0.01     | 0.5             | 5       | 0.6       |
# MAGIC | 0.579268503    | 0.331681949 | 0.504022243 | 0.421829542 | 0.829335638   | 0.72757323    | 0.257799022 | 0.533235556 | 0.380704179 | 0.737191027  | 3       | 0.01     | 0.5             | 5       | 0.7       |
# MAGIC | 0.590375987    | 0.333975643 | 0.511793061 | 0.42661514  | 0.831489544   | 0.88140182    | 0.367016776 | 0.688430587 | 0.518238443 | 0.620990061  | 2       | 0        | 0               | 10      | 0.7       |
# MAGIC | 0.354061188    | 0.4855438   | 0.374334749 | 0.409507369 | 0.737164953   | 0.361217507   | 0.706015258 | 0.400318322 | 0.47791837  | 0.516666171  | 3       | 0        | 0               | 10      | 0.5       |
# MAGIC | 0.49805519     | 0.374622655 | 0.467263903 | 0.427609704 | 0.811747459   | 0.471859582   | 0.436345579 | 0.464301714 | 0.45340822  | 0.670350285  | 3       | 0        | 0               | 10      | 0.6       |
# MAGIC | 0.576819103    | 0.337113838 | 0.50500255  | 0.425531662 | 0.82915082    | 0.676617436   | 0.281520552 | 0.528323492 | 0.397608103 | 0.732710181  | 3       | 0        | 0               | 10      | 0.7       |
# MAGIC | 0.590375987    | 0.333975643 | 0.511793061 | 0.42661514  | 0.831489544   | 0.88140182    | 0.367016776 | 0.688430587 | 0.518238443 | 0.620990061  | 2       | 0        | 0.5             | 10      | 0.7       |
# MAGIC | 0.354061188    | 0.4855438   | 0.374334749 | 0.409507369 | 0.737164953   | 0.361217507   | 0.706015258 | 0.400318322 | 0.47791837  | 0.516666171  | 3       | 0        | 0.5             | 10      | 0.5       |
# MAGIC | 0.49805519     | 0.374622655 | 0.467263903 | 0.427609704 | 0.811747459   | 0.471859582   | 0.436345579 | 0.464301714 | 0.45340822  | 0.670350285  | 3       | 0        | 0.5             | 10      | 0.6       |
# MAGIC | 0.576819103    | 0.337113838 | 0.50500255  | 0.425531662 | 0.82915082    | 0.676617436   | 0.281520552 | 0.528323492 | 0.397608103 | 0.732710181  | 3       | 0        | 0.5             | 10      | 0.7       |
# MAGIC | 0.596492734    | 0.326727794 | 0.511953228 | 0.422197621 | 0.832138258   | 0.878269365   | 0.367237781 | 0.687054527 | 0.517915443 | 0.6202709    | 2       | 0.01     | 0.5             | 10      | 0.7       |
# MAGIC | 0.427250979    | 0.416700992 | 0.425098461 | 0.421910045 | 0.78566051    | 0.417402368   | 0.529621247 | 0.435873388 | 0.466863042 | 0.620979141  | 3       | 0.01     | 0.5             | 10      | 0.5       |
# MAGIC | 0.509355911    | 0.364406827 | 0.471820933 | 0.424858519 | 0.814809388   | 0.691442239   | 0.285976005 | 0.53868824  | 0.40460855  | 0.736278492  | 3       | 0.01     | 0.5             | 10      | 0.6       |
# MAGIC | 0.57910722     | 0.333164057 | 0.504606653 | 0.422983197 | 0.829381709   | 0.72477785    | 0.259160191 | 0.533188319 | 0.381799581 | 0.737027366  | 3       | 0.01     | 0.5             | 10      | 0.7       |
# MAGIC | 0.599525071    | 0.327413363 | 0.514075864 | 0.42352871  | 0.832701006   | 0.883240163   | 0.363288723 | 0.686680085 | 0.514823515 | 0.619674493  | 2       | 0.01     | 0               | 10      | 0.7       |
# MAGIC | 0.35492234     | 0.485110265 | 0.375052749 | 0.409928066 | 0.737856031   | 0.363601062   | 0.696554344 | 0.402035712 | 0.477793912 | 0.522905136  | 3       | 0.01     | 0               | 10      | 0.5       |
# MAGIC | 0.504336897    | 0.371720506 | 0.470747753 | 0.427991056 | 0.813496603   | 0.494081059   | 0.409240911 | 0.474410937 | 0.447676885 | 0.683585768  | 3       | 0.01     | 0               | 10      | 0.6       |
# MAGIC | 0.583639477    | 0.333021113 | 0.507286737 | 0.424070305 | 0.830212593   | 0.695296565   | 0.272463241 | 0.530607527 | 0.391507798 | 0.734618322  | 3       | 0.01     | 0               | 10      | 0.7       |
# MAGIC | 0.43004456     | 0.40097616  | 0.423898533 | 0.415001966 | 0.787810004   | 0.779942473   | 0.47338661  | 0.690510295 | 0.589173791 | 0.652641039  | 1       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.568639602    | 0.329631824 | 0.496621972 | 0.417338688 | 0.827233804   | 0.853283061   | 0.391894885 | 0.690657262 | 0.537107597 | 0.644584661  | 1       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.452348272    | 0.435449288 | 0.448864352 | 0.443737946 | 0.795076129   | 0.760223417   | 0.502806875 | 0.689612719 | 0.605283282 | 0.646858906  | 0       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.560953339    | 0.351269102 | 0.501125622 | 0.432012121 | 0.826625514   | 0.849211714   | 0.393257056 | 0.689358831 | 0.537572463 | 0.635662893  | 0       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.43004456     | 0.40097616  | 0.423898533 | 0.415001966 | 0.787810004   | 0.779942473   | 0.47338661  | 0.690510295 | 0.589173791 | 0.652641039  | 1       | 0        | 0.5             | 5       | 0.6       |
# MAGIC | 0.568639602    | 0.329631824 | 0.496621972 | 0.417338688 | 0.827233804   | 0.853283061   | 0.391894885 | 0.690657262 | 0.537107597 | 0.644584661  | 1       | 0        | 0.5             | 5       | 0.7       |
# MAGIC | 0.467126805    | 0.399131048 | 0.451735326 | 0.430460308 | 0.801752145   | 0.788316204   | 0.450883722 | 0.685685503 | 0.573658757 | 0.639098667  | 0       | 0.01     | 0.5             | 5       | 0.5       |
# MAGIC | 0.549639235    | 0.351673485 | 0.494020005 | 0.428915605 | 0.824219712   | 0.856351015   | 0.380487446 | 0.685007571 | 0.526876905 | 0.632018889  | 0       | 0.01     | 0.5             | 5       | 0.6       |
# MAGIC | 0.445834278    | 0.395759628 | 0.434830633 | 0.419307237 | 0.794246305   | 0.772085237   | 0.4846106   | 0.690198994 | 0.595467382 | 0.65355341   | 1       | 0        | 0               | 10      | 0.6       |
# MAGIC | 0.575624005    | 0.329133399 | 0.500637699 | 0.418801956 | 0.828529467   | 0.852099923   | 0.393541729 | 0.691055229 | 0.538416288 | 0.644965228  | 1       | 0        | 0               | 10      | 0.7       |
# MAGIC | 0.451645014    | 0.438212254 | 0.448892981 | 0.444827247 | 0.794683723   | 0.757292694   | 0.508853336 | 0.689923874 | 0.608699004 | 0.647693061  | 0       | 0.01     | 0               | 10      | 0.5       |
# MAGIC | 0.564526874    | 0.351425213 | 0.503467123 | 0.433186364 | 0.827376257   | 0.849561469   | 0.394619639 | 0.690379147 | 0.538914533 | 0.636369383  | 0       | 0.01     | 0               | 10      | 0.6       |
# MAGIC | 0.446784071    | 0.394313255 | 0.435201712 | 0.418912    | 0.794665189   | 0.777437832   | 0.477740501 | 0.690770648 | 0.591809991 | 0.65324722   | 1       | 0.01     | 0               | 10      | 0.6       |
# MAGIC | 0.582456487    | 0.321506559 | 0.501111353 | 0.414316895 | 0.829383122   | 0.856873755   | 0.389029987 | 0.690738615 | 0.535112906 | 0.644341693  | 1       | 0.01     | 0               | 10      | 0.7       |
# MAGIC | 0.445834278    | 0.395759628 | 0.434830633 | 0.419307237 | 0.794246305   | 0.772085237   | 0.4846106   | 0.690198994 | 0.595467382 | 0.65355341   | 1       | 0        | 0.5             | 10      | 0.6       |
# MAGIC | 0.575624005    | 0.329133399 | 0.500637699 | 0.418801956 | 0.828529467   | 0.852099923   | 0.393541729 | 0.691055229 | 0.538416288 | 0.644965228  | 1       | 0        | 0.5             | 10      | 0.7       |
# MAGIC | 0.484951499    | 0.388250341 | 0.46194048  | 0.431246422 | 0.807773799   | 0.801349429   | 0.439571063 | 0.688086782 | 0.567723754 | 0.63952628   | 0       | 0.01     | 0.5             | 10      | 0.5       |
# MAGIC | 0.560753632    | 0.3484704   | 0.499852963 | 0.429830351 | 0.826469822   | 0.861482267   | 0.377822123 | 0.685879699 | 0.525273793 | 0.632239513  | 0       | 0.01     | 0.5             | 10      | 0.6       |
# MAGIC | 0.547938562    | 0.349543424 | 0.492079324 | 0.426812625 | 0.823776998   | 0.804356057   | 0.43042964  | 0.685289908 | 0.560775346 | 0.64522927   | 1       | 0.01     | 0.5             | 10      | 0.6       |
# MAGIC | 0.604206175    | 0.306220906 | 0.505772221 | 0.406447844 | 0.832123077   | 0.861596158   | 0.376185069 | 0.684855301 | 0.523710657 | 0.639978182  | 1       | 0.01     | 0.5             | 10      | 0.7       |
# MAGIC | 0.450940261    | 0.435954295 | 0.447861208 | 0.443320669 | 0.794491316   | 0.760219612   | 0.503548009 | 0.689888742 | 0.605818769 | 0.647130347  | 0       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.555845414    | 0.353866554 | 0.498893981 | 0.432433799 | 0.825643352   | 0.846718455   | 0.395883251 | 0.689643973 | 0.539515844 | 0.636086787  | 0       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.397175153    | 0.480158932 | 0.411395077 | 0.434742478 | 0.765630343   | 0.751591894   | 0.531207376 | 0.694006723 | 0.622468639 | 0.642099899  | 2       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.528596973    | 0.369470071 | 0.486675741 | 0.434935816 | 0.819800337   | 0.842309237   | 0.411923073 | 0.696719671 | 0.553273275 | 0.630528856  | 2       | 0        | 0               | 5       | 0.6       |
# MAGIC | 0.42739359     | 0.401450134 | 0.421940069 | 0.414015837 | 0.786694216   | 0.782862031   | 0.470071717 | 0.690913863 | 0.587423396 | 0.65257162   | 1       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.574343591    | 0.320685569 | 0.495894413 | 0.411570281 | 0.827879694   | 0.858067576   | 0.387892039 | 0.690638791 | 0.534267046 | 0.644174342  | 1       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.396626991    | 0.480527578 | 0.410978439 | 0.434564703 | 0.765280655   | 0.747835515   | 0.538938063 | 0.69403279  | 0.626430369 | 0.642974051  | 2       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.53441691     | 0.366877322 | 0.48969199  | 0.435075334 | 0.821166432   | 0.844028659   | 0.410766706 | 0.696995501 | 0.552598267 | 0.630558614  | 2       | 0.01     | 0               | 5       | 0.6       |
# MAGIC | 0.450940261    | 0.435954295 | 0.447861208 | 0.443320669 | 0.794491316   | 0.760219612   | 0.503548009 | 0.689888742 | 0.605818769 | 0.647130347  | 0       | 0        | 0.5             | 5       | 0.5       |
# MAGIC | 0.555845414    | 0.353866554 | 0.498893981 | 0.432433799 | 0.825643352   | 0.846718455   | 0.395883251 | 0.689643973 | 0.539515844 | 0.636086787  | 0       | 0        | 0.5             | 5       | 0.6       |
# MAGIC | 0.397175153    | 0.480158932 | 0.411395077 | 0.434742478 | 0.765630343   | 0.751591894   | 0.531207376 | 0.694006723 | 0.622468639 | 0.642099899  | 2       | 0        | 0.5             | 5       | 0.5       |
# MAGIC | 0.528596973    | 0.369470071 | 0.486675741 | 0.434935816 | 0.819800337   | 0.842309237   | 0.411923073 | 0.696719671 | 0.553273275 | 0.630528856  | 2       | 0        | 0.5             | 5       | 0.6       |
# MAGIC | 0.393385414    | 0.464325011 | 0.405784571 | 0.425921573 | 0.765055238   | 0.750467648   | 0.524827271 | 0.691046921 | 0.617685967 | 0.639146381  | 2       | 0.01     | 0.5             | 5       | 0.5       |
# MAGIC | 0.532508867    | 0.360473974 | 0.486110019 | 0.429919991 | 0.820557965   | 0.841063152   | 0.406328761 | 0.692813756 | 0.547940298 | 0.627606337  | 2       | 0.01     | 0.5             | 5       | 0.6       |
# MAGIC | 0.449162785    | 0.440357361 | 0.447373641 | 0.44471649  | 0.793586117   | 0.756355795   | 0.51103531  | 0.690099852 | 0.609953023 | 0.648041348  | 0       | 0        | 0               | 10      | 0.5       |
# MAGIC | 0.55944354     | 0.35432266  | 0.501391449 | 0.433860484 | 0.826430105   | 0.846591743   | 0.398431188 | 0.691116408 | 0.541851151 | 0.637172551  | 0       | 0        | 0               | 10      | 0.6       |
# MAGIC | 0.449162785    | 0.440357361 | 0.447373641 | 0.44471649  | 0.793586117   | 0.756355795   | 0.51103531  | 0.690099852 | 0.609953023 | 0.648041348  | 0       | 0        | 0.5             | 10      | 0.5       |
# MAGIC | 0.55944354     | 0.35432266  | 0.501391449 | 0.433860484 | 0.826430105   | 0.846591743   | 0.398431188 | 0.691116408 | 0.541851151 | 0.637172551  | 0       | 0        | 0.5             | 10      | 0.6       |
# MAGIC | 0.531886204    | 0.356110406 | 0.484096298 | 0.426601205 | 0.820311012   | 0.806423994   | 0.426066326 | 0.684254537 | 0.557554251 | 0.644206573  | 1       | 0.01     | 0.5             | 5       | 0.6       |
# MAGIC | 0.595860557    | 0.316904124 | 0.506662176 | 0.413755422 | 0.831435882   | 0.86069485    | 0.375237956 | 0.683771582 | 0.522626108 | 0.639319937  | 1       | 0.01     | 0.5             | 5       | 0.7       |
# MAGIC | 0.371015467    | 0.519072742 | 0.39346118  | 0.432730181 | 0.744551818   | 0.763127875   | 0.509243116 | 0.693935163 | 0.610855827 | 0.639621275  | 2       | 0        | 0               | 10      | 0.5       |
# MAGIC | 0.509499672    | 0.38330465  | 0.478023808 | 0.437483531 | 0.814979554   | 0.848615485   | 0.406040785 | 0.69673139  | 0.549269957 | 0.629863012  | 2       | 0        | 0               | 10      | 0.6       |
# MAGIC | 0.371015467    | 0.519072742 | 0.39346118  | 0.432730181 | 0.744551818   | 0.763127875   | 0.509243116 | 0.693935163 | 0.610855827 | 0.639621275  | 2       | 0        | 0.5             | 10      | 0.5       |
# MAGIC | 0.509499672    | 0.38330465  | 0.478023808 | 0.437483531 | 0.814979554   | 0.848615485   | 0.406040785 | 0.69673139  | 0.549269957 | 0.629863012  | 2       | 0        | 0.5             | 10      | 0.6       |
# MAGIC | 0.4013927      | 0.455024216 | 0.41108318  | 0.426529171 | 0.77033268    | 0.753891786   | 0.514062796 | 0.689551606 | 0.611295902 | 0.636883506  | 2       | 0.01     | 0.5             | 10      | 0.5       |
# MAGIC | 0.535206019    | 0.35929092  | 0.487471145 | 0.429950411 | 0.82116908    | 0.84262289    | 0.404710295 | 0.692714069 | 0.546795615 | 0.627371989  | 2       | 0.01     | 0.5             | 10      | 0.6       |
# MAGIC | 0.376654997    | 0.51064372  | 0.397515984 | 0.43353271  | 0.749519996   | 0.757613916   | 0.519782121 | 0.694095749 | 0.616557679 | 0.640904605  | 2       | 0.01     | 0               | 10      | 0.5       |
# MAGIC | 0.520611158    | 0.375905393 | 0.483394373 | 0.436579875 | 0.817883498   | 0.849448895   | 0.405679142 | 0.696967303 | 0.54911322  | 0.629958487  | 2       | 0.01     | 0               | 10      | 0.6       |
# MAGIC | 0.616605896    | 0.304387079 | 0.511644185 | 0.407575025 | 0.833905937   | 0.879065856   | 0.360847379 | 0.682916481 | 0.511662593 | 0.629080137  | 0       | 0        | 0               | 5       | 0.7       |
# MAGIC | 0.283944143    | 0.548386702 | 0.314251788 | 0.374156967 | 0.655649592   | 0.659014021   | 0.701725535 | 0.667135247 | 0.679699454 | 0.652018743  | 1       | 0        | 0               | 5       | 0.5       |
# MAGIC | 0.624872933    | 0.292512343 | 0.509167013 | 0.398486984 | 0.834240974   | 0.882512145   | 0.354994257 | 0.68032186  | 0.50631939  | 0.627211036  | 0       | 0.01     | 0               | 5       | 0.7       |
# MAGIC | 0.278876317    | 0.555554615 | 0.309726417 | 0.371345354 | 0.646928229   | 0.658006192   | 0.704979173 | 0.666893253 | 0.680683261 | 0.651982794  | 1       | 0.01     | 0               | 5       | 0.5       |
# MAGIC | 0.616605896    | 0.304387079 | 0.511644185 | 0.407575025 | 0.833905937   | 0.879065856   | 0.360847379 | 0.682916481 | 0.511662593 | 0.629080137  | 0       | 0        | 0.5             | 5       | 0.7       |
# MAGIC | 0.283944143    | 0.548386702 | 0.314251788 | 0.374156967 | 0.655649592   | 0.659014021   | 0.701725535 | 0.667135247 | 0.679699454 | 0.652018743  | 1       | 0        | 0.5             | 5       | 0.5       |
# MAGIC | 0.607335123    | 0.318282786 | 0.513979746 | 0.417676263 | 0.833413444   | 0.885530889   | 0.355951748 | 0.682460264 | 0.507790054 | 0.628398436  | 0       | 0.01     | 0.5             | 5       | 0.7       |
# MAGIC | 0.376304178    | 0.443856679 | 0.388118057 | 0.40729845  | 0.757524331   | 0.680332495   | 0.64751866  | 0.673506336 | 0.66352013  | 0.654454623  | 1       | 0.01     | 0.5             | 5       | 0.5       |
# MAGIC | 0.628017526    | 0.292093854 | 0.510578881 | 0.398734466 | 0.834648913   | 0.883548205   | 0.35239108  | 0.678890597 | 0.50383463  | 0.626247978  | 0       | 0.01     | 0               | 10      | 0.7       |
# MAGIC | 0.289391642    | 0.548298303 | 0.31957205  | 0.378834549 | 0.662497716   | 0.652722177   | 0.717195039 | 0.664672437 | 0.683441455 | 0.650427054  | 1       | 0.01     | 0               | 10      | 0.5       |
# MAGIC | 0.615054833    | 0.315031739 | 0.516648026 | 0.416653244 | 0.834419789   | 0.888218517   | 0.354094308 | 0.682360716 | 0.506334822 | 0.628179052  | 0       | 0.01     | 0.5             | 10      | 0.7       |
# MAGIC | 0.397701614    | 0.421721917 | 0.40228425  | 0.409359704 | 0.771573621   | 0.668716579   | 0.671472124 | 0.669265878 | 0.670091519 | 0.652115435  | 1       | 0.01     | 0.5             | 10      | 0.5       |
# MAGIC | 0.620728829    | 0.303172051 | 0.513215566 | 0.407376237 | 0.834432675   | 0.880330108   | 0.35748005  | 0.681096005 | 0.508479348 | 0.627830765  | 0       | 0        | 0               | 10      | 0.7       |
# MAGIC | 0.294519904    | 0.542211878 | 0.324133926 | 0.381704612 | 0.670284225   | 0.65265005    | 0.716789807 | 0.664542972 | 0.683217888 | 0.650263422  | 1       | 0        | 0               | 10      | 0.5       |
# MAGIC | 0.620728829    | 0.303172051 | 0.513215566 | 0.407376237 | 0.834432675   | 0.880330108   | 0.35748005  | 0.681096005 | 0.508479348 | 0.627830765  | 0       | 0        | 0.5             | 10      | 0.7       |
# MAGIC | 0.294519904    | 0.542211878 | 0.324133926 | 0.381704612 | 0.670284225   | 0.65265005    | 0.716789807 | 0.664542972 | 0.683217888 | 0.650263422  | 1       | 0        | 0.5             | 10      | 0.5       |
# MAGIC 
# MAGIC ### 14.12.2 Link to Modelling Notebook
# MAGIC 
# MAGIC [Link to Modelling Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238260297/command/1215577238260576)
# MAGIC [Link to Modelling Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/DLE_pipeline_LR_downsampleExperiment.py)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 14.13 Links to Feature Engineering Notebooks
# MAGIC 
# MAGIC ### 14.13.1 PageRank
# MAGIC [Link to PageRank Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1694179314626606/command/1694179314626607)
# MAGIC 
# MAGIC [Link to PageRank Notebook in GitHub](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase3/NC%20graphframes.py)
# MAGIC 
# MAGIC ### 14.13.2 Post-Pipeline Features
# MAGIC [Link to Post-Pipeline Features Notebook in DataBricks](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1215577238261059/command/1215577238261067)
# MAGIC 
# MAGIC [Link to Post-Pipeline Features Notebook in Github](https://github.com/ColStaR/sparksandstripesforever/blob/main/Phase4/CleanPipeline_FeatureEngineering.py)

# COMMAND ----------


