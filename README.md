![Allay Airway Delays Title Image](https://raw.githubusercontent.com/ColStaR/sparksandstripesforever/main/images/AllayAirwayDelaysTitle.PNG)

# Introduction

This data science and data engineering project, named "Allay Airway Delays", was undertaken by a project team named, "Sparks and Stripes Forever". This project was done over a 7 weeek period as the final project for W261, "Machine Learning At Scale" at UC Berkeley during the Summer 2022 term. In order to accomplish the sophisticated machine learning and modelling tasks required on such a large data set, the team utilized a number of tools, including Apache Spark, DataBricks, Azure blob storage, ML Lib, and Pandas, while also implementing custom solutions such as Blocking Time Series Cross Validation. The project was one of the few in the class to have received a perfect 100/100 score and direct acclaim from the instructor, Professor De Sola.

# Project Team Members:
- Nashat Cabral
- Deanna Emery
- Nina Huang
- Ryan S. Wong

# Project Abstract

Flight delays: the bane of airline travelers everywhere. But can they be predicted and avoided? This project aims to help travelers by creating a machine learning model that predicts whether a flight will be delayed 2 hours before its departure time. We incorporated flight, weather, and weather station data across 27 features, including 10 newly created features such as a highly predictive previously-delayed flight tracker. The F0.5 metric was chosen as our primary metric in order to minimize false positives while balancing recall; precision is our secondary metric, as it focuses solely on minizing false positives.  Our baseline logistic regression model returned a test evaluation F0.5 score of 0.197 and precision of 0.328. Five models were trained with blocking time series cross validation: Logistic Regression, Gradient Boosted Tree, and Multilayer Perceptron Classifier Neural Network.  Gradient Boosted Trees demonstrated significant improvement over the baseline, having a test evaluation F0.5 score of 0.526 and a precision of 0.623. The most important features for this top-performing model were indicators for whether the flight was previously delayed or not, the hour of the flight's scheduled departure time, and the flight's airline carrier.

# Link to Final Report and Presentation:

- [Final Report](https://github.com/ColStaR/sparksandstripesforever/blob/main/final_report/Allay_Airway_Delays_Final_Research_Paper.html)

Note: the Final Report was exported into HTML format, which does not render in GitHub's preview window. As such, it is advised to download and read the file in a web browser.

- [Final Presentation](https://github.com/ColStaR/sparksandstripesforever/blob/main/final_report/Allay_Airway_Delays_Final_Presentation.pdf)

For further information regarding the data pipeline, the models trained, or the results, please refer to the notebooks that were used in this project, which are stored in the "src" folders.