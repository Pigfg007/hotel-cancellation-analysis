# Hotel Booking Cancellation Prediction


## Overview
Predicts the probability that a hotel booking will be cancelled using:
- Logistic Regression
- XGBoost
- Random Forest

Plus descriptive statistics, Chi-square tests, and Welch t-tests.

Dataset: `hotel_bookings.csv` (119,390 bookings, City Hotel & Resort Hotel, Jul 2015 – Aug 2017). Not included here due to size; download from public sources.

## How to run
1. Place `hotel_bookings.csv` in the working directory.
2. Install packages: `install.packages(c("data.table", "caTools", "xgboost", "randomForest", "ggplot2"))`
3. Run `Hotel_Cancellation_Analysis.R` in RStudio (set working directory to source file location).

Outputs charts and result tables to `./outputs/`.
