# Hotel Booking Cancellation Prediction


## Overview
Predicts the probability that a hotel booking will be cancelled using:
- Logistic Regression
- XGBoost
- Random Forest

Plus descriptive statistics, Chi-square tests, and Welch t-tests.

Dataset: `hotel_bookings.csv` (119,390 bookings, City Hotel & Resort Hotel, Jul 2015 – Aug 2017) is included in this repository.

## How to run
1. The dataset `hotel_bookings.csv` is already in the repository.
2. Install packages: `install.packages(c("data.table", "caTools", "xgboost", "randomForest", "ggplot2"))`
3. Run `Hotel_Cancellation_Analysis.R` in RStudio (set working directory to source file location).

Outputs charts and result tables to `./outputs/`.
