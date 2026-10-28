# =============================================================================
# Hotel Booking Cancellation Analysis
# Analysis Script: Predicting Hotel Booking Cancellations
# Dataset       : hotel_bookings.csv (119,390 bookings, City Hotel & Resort Hotel,
#                 arrivals Jul-2015 to Aug-2017)
# Techniques    : Category A - (1) Logistic Regression, (2) XGBoost, (3) Random Forest
#                 Category B - Descriptive statistics, Chi-square test, Welch t-test
# =============================================================================
#
# -----------------------------------------------------------------------------
# BUSINESS PROBLEM STATEMENT
# -----------------------------------------------------------------------------
# Situation : About 37% of all bookings in the dataset are cancelled (or no-show).
#             A hotel room is a perishable product - a room-night left empty
#             because of a late cancellation can never be sold again.
# Complication: Today the hotels treat every booking the same. They either
#             (a) hold the room and lose revenue when the guest cancels, or
#             (b) apply blanket overbooking / non-refundable deposits that annoy
#             good customers and risk "walking" guests when everyone shows up.
#             Management cannot tell, at the moment a booking is made, WHICH
#             bookings are likely to cancel.
# Key question: "Can we predict, at the time of booking, the probability that
#             a reservation will be cancelled, and which booking characteristics
#             drive cancellation risk?"
# Opportunity: A cancellation-risk score enables
#             1. Smarter, risk-based overbooking (overbook only against
#                high-risk bookings) -> recover lost room revenue;
#             2. Targeted deposit / pre-payment policies for high-risk profiles
#                instead of penalising every guest;
#             3. Proactive retention (reminder e-mails, upgrade offers, flexible
#                date change) for high-value bookings flagged as at-risk;
#             4. More accurate demand forecasting for staffing & F&B planning.
# Success measure: Out-of-sample accuracy, recall (share of real cancellations
#             caught), precision (share of flagged bookings that really cancel),
#             AUC, and estimated room revenue at risk that is correctly flagged.
#
# -----------------------------------------------------------------------------
# HOW TO RUN
# -----------------------------------------------------------------------------
# 1. Set the working directory to the "Team Project" folder (Session > Set
#    Working Directory > To Source File Location in RStudio).
# 2. Install packages once if missing:
#    install.packages(c("data.table","caTools","xgboost",
#                       "randomForest","ggplot2"))
# 3. Run the whole script. Charts and result tables are written to ./outputs/
#    XGBoost cross-validation and Random Forest training each take a few
#    minutes on a typical laptop.
# =============================================================================


# ---- 0. Libraries & global settings -----------------------------------------
library(data.table)    # fast data import & manipulation
library(caTools)       # sample.split() for stratified train-test split; colAUC()
library(xgboost)       # XGBoost (gradient boosted trees)
library(randomForest)  # Random Forest
library(ggplot2)       # charts for EDA

set.seed(2026)         # fix the random seed so every result is reproducible


# Folder to store all charts and tables produced by this script
dir.create("outputs", showWarnings = FALSE)


# ---- 1. Data import ----------------------------------------------------------
# "NULL" (used in agent/company) and "NA" (used in children/country) are both
# treated as missing values.
hotel.dt <- fread("G:/我的云端硬盘/NTU Masters/AN6003 Analytics Strategy/AN6003 Course Materials/Team Project/hotel_bookings.csv/hotel_bookings.csv", na.strings = c("NA", "NULL", ""))

dim(hotel.dt)          # 119,390 rows x 32 columns
str(hotel.dt)          # check data type of each column
summary(hotel.dt)      # quick numeric summary - spot outliers & missing values

# Count missing values per column
sapply(hotel.dt, function(x) sum(is.na(x)))
# -> children: 4 NA; country: 488 NA; agent: 16,340 NA; company: 112,593 NA


# ---- 2. Data cleaning --------------------------------------------------------
n.raw <- nrow(hotel.dt)

# 2.0 Safety check: make sure guest counts are numeric. If the CSV was ever
#     opened/re-saved in Excel, "NA" in children may be read as text, which
#     would break the calculations below. suppressWarnings() hides the
#     "NAs introduced by coercion" message for the text "NA" values.
num.cols <- c("adults", "children", "babies")
hotel.dt[, (num.cols) := lapply(.SD, function(x) suppressWarnings(as.numeric(x))),
         .SDcols = num.cols]
# Same idea for agent / company: treat any leftover "NULL" text as missing
hotel.dt[, agent   := ifelse(agent   %in% c("NULL", ""), NA, agent)]
hotel.dt[, company := ifelse(company %in% c("NULL", ""), NA, company)]

# 2.1 Children has only 4 NAs -> assume no children (most common value)
hotel.dt[is.na(children), children := 0]

# 2.2 Remove impossible bookings: no adults, no children and no babies
hotel.dt <- hotel.dt[!(adults == 0 & children == 0 & babies == 0)]

# 2.3 Remove invalid / extreme prices
#     - adr < 0 is a data-entry error
#     - one booking has adr = 5,400 (vs. median ~95) - clear outlier
hotel.dt <- hotel.dt[adr >= 0 & adr < 1000]

# 2.4 Remove rows with "Undefined" market segment / distribution channel
#     (only a handful of records, no business meaning)
hotel.dt <- hotel.dt[market_segment != "Undefined" & distribution_channel != "Undefined"]

# 2.5 Per the data dictionary, meal "Undefined" is the same as "SC" (no meal)
hotel.dt[meal == "Undefined", meal := "SC"]

cat("Rows removed during cleaning:", n.raw - nrow(hotel.dt),
    "| Rows remaining:", nrow(hotel.dt), "\n")


# ---- 3. Feature engineering --------------------------------------------------
# IMPORTANT - avoid DATA LEAKAGE. The model must only use information that is
# known AT THE TIME OF BOOKING. The following columns are dropped:
#   - reservation_status      : literally tells us Canceled / Check-Out / No-Show
#   - reservation_status_date : date of the final status (after the outcome)
#   - assigned_room_type      : only decided at check-in (after the outcome)
#   - arrival_date_year       : a calendar year cannot be re-used for future
#                               bookings (we keep month for seasonality)
#   - arrival_date_week_number / day_of_month : redundant with month

# 3.1 Total length of stay and total guests
hotel.dt[, total_nights := stays_in_weekend_nights + stays_in_week_nights]
hotel.dt[, kids := children + babies]

# 3.2 Agent / company IDs are hundreds of codes -> convert to Yes/No flags
#     (booked through a travel agent? booked by a company?)
hotel.dt[, has_agent   := factor(ifelse(is.na(agent),   "No", "Yes"))]
hotel.dt[, has_company := factor(ifelse(is.na(company), "No", "Yes"))]

# 3.3 Country has 177 levels -> keep the 10 largest source markets and group
#     the rest into "Other" (randomForest also cannot handle > 53 levels)
hotel.dt[is.na(country), country := "Unknown"]
top.countries <- names(sort(table(hotel.dt$country), decreasing = TRUE))[1:10]
hotel.dt[, country_grp := ifelse(country %in% top.countries, country, "Other")]

# 3.4 Room type was reserved in a rare category (P, L) -> group into "Other"
hotel.dt[, reserved_room := ifelse(reserved_room_type %in% c("P", "L"),
                                   "Other", reserved_room_type)]

# 3.5 Was the guest's room type changed by the hotel? -> NOT used as predictor
#     (assigned_room_type is only known at check-in), kept only for EDA below.
hotel.dt[, room_changed := as.integer(reserved_room_type != assigned_room_type)]

# 3.6 Estimated booking value = nightly rate x number of nights.
#     Used later to translate model results into revenue (dollar) terms.
hotel.dt[, booking_value := adr * total_nights]

# 3.7 Convert categorical columns to factors; month in calendar order
hotel.dt[, arrival_date_month := factor(arrival_date_month, levels = month.name)]
factor.cols <- c("hotel", "meal", "country_grp", "market_segment",
                 "distribution_channel", "reserved_room", "deposit_type",
                 "customer_type")
hotel.dt[, (factor.cols) := lapply(.SD, factor), .SDcols = factor.cols]

# Target variable: 1 = cancelled, 0 = not cancelled (as factor for
# classification). Labels "No"/"Yes" make outputs easier to read.
hotel.dt[, is_canceled := factor(is_canceled, levels = c(0, 1), labels = c("No", "Yes"))]
hotel.dt[, is_repeated_guest := factor(is_repeated_guest, levels = c(0, 1), labels = c("No", "Yes"))]

# 3.8 Final modelling dataset - only "known at booking time" variables
model.vars <- c("is_canceled",                       # target
                "hotel", "lead_time", "arrival_date_month",
                "stays_in_weekend_nights", "stays_in_week_nights",
                "adults", "kids", "meal", "country_grp",
                "market_segment", "distribution_channel",
                "is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled", "reserved_room",
                "booking_changes", "deposit_type", "has_agent", "has_company",
                "days_in_waiting_list", "customer_type", "adr",
                "required_car_parking_spaces", "total_of_special_requests")
model.dt <- hotel.dt[, ..model.vars]
str(model.dt)


# ---- 4. Exploratory data analysis (Category B: descriptive statistics) ------
# 4.1 Overall cancellation rate - the size of the business problem
prop.table(table(model.dt$is_canceled))      # ~37% of bookings are cancelled

# Room revenue lost to cancellations (approximate, using adr x nights)
revenue.summary <- hotel.dt[, .(bookings = .N,
                                room_revenue = sum(booking_value)),
                            by = is_canceled]
revenue.summary[, pct_revenue := round(100 * room_revenue / sum(room_revenue), 1)]
revenue.summary
fwrite(revenue.summary, "outputs/01_revenue_at_risk.csv")

# 4.2 Helper: cancellation rate by any categorical variable
cancel.rate.by <- function(var) {
  out <- hotel.dt[, .(bookings = .N,
                      cancel_rate = round(mean(is_canceled == "Yes") * 100, 1)),
                  by = var][order(-cancel_rate)]
  print(out)
  invisible(out)
}
cancel.rate.by("hotel")             # City hotel cancels much more than Resort
cancel.rate.by("deposit_type")      # Non-Refund ~99% cancel (group block bookings)
cancel.rate.by("market_segment")    # Groups & Online TA highest
cancel.rate.by("customer_type")
cancel.rate.by("country_grp")       # Portugal (home market) very high
cancel.rate.by("arrival_date_month")
cancel.rate.by("is_repeated_guest") # loyal guests rarely cancel

# 4.3 Charts (saved to ./outputs)
# Cancellation rate by month and hotel - seasonality
p1 <- ggplot(hotel.dt[, .(cancel_rate = mean(is_canceled == "Yes")),
                      by = .(hotel, arrival_date_month)],
             aes(x = arrival_date_month, y = cancel_rate, group = hotel, colour = hotel)) +
  geom_line(linewidth = 1) + geom_point() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Cancellation rate by arrival month",
       x = "Arrival month", y = "Cancellation rate", colour = "Hotel") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("outputs/02_cancel_rate_by_month.png", p1, width = 8, height = 4.5)

# Lead time distribution - cancelled bookings are made much further in advance
p2 <- ggplot(hotel.dt, aes(x = is_canceled, y = lead_time, fill = is_canceled)) +
  geom_boxplot(outlier.alpha = 0.1) +
  labs(title = "Lead time (days between booking and arrival)",
       x = "Cancelled?", y = "Lead time (days)") +
  theme_minimal() + theme(legend.position = "none")
ggsave("outputs/03_lead_time_boxplot.png", p2, width = 6, height = 4.5)

# Cancellation rate by deposit type
p3 <- ggplot(hotel.dt[, .(cancel_rate = mean(is_canceled == "Yes")), by = deposit_type],
             aes(x = deposit_type, y = cancel_rate)) +
  geom_col(fill = "steelblue") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Cancellation rate by deposit type", x = "Deposit type",
       y = "Cancellation rate") +
  theme_minimal()
ggsave("outputs/04_cancel_rate_by_deposit.png", p3, width = 6, height = 4.5)

# 4.4 Simple inferential tests (Category B: inferential statistics)
# H0: lead time is the same for cancelled vs. not-cancelled bookings
t.test(lead_time ~ is_canceled, data = hotel.dt)          # Welch two-sample t-test
# H0: cancellation is independent of deposit type / hotel type
chisq.test(table(hotel.dt$deposit_type, hotel.dt$is_canceled))
chisq.test(table(hotel.dt$hotel, hotel.dt$is_canceled))


# ---- 5. Train-test split -----------------------------------------------------
# 70% train / 30% test. sample.split() keeps the same proportion of
# cancellations (~37%) in both sets (stratified split).
split <- sample.split(model.dt$is_canceled, SplitRatio = 0.7)
trainset <- model.dt[split == TRUE]
testset  <- model.dt[split == FALSE]

# Check that both sets have a similar cancellation rate
prop.table(table(trainset$is_canceled))
prop.table(table(testset$is_canceled))

# Booking value of test-set bookings (for the revenue-impact section later)
test.value <- hotel.dt$booking_value[split == FALSE]

# ---- Helper: evaluate any model's predictions on the test set ---------------
# prob      : predicted probability of cancellation (class "Yes")
# threshold : cut-off above which a booking is flagged as "will cancel"
evaluate.model <- function(model.name, prob, actual, threshold = 0.5) {
  pred <- factor(ifelse(prob > threshold, "Yes", "No"), levels = c("No", "Yes"))
  cm   <- table(Actual = actual, Predicted = pred)   # confusion matrix
  TP <- cm["Yes", "Yes"]; TN <- cm["No", "No"]
  FP <- cm["No", "Yes"];  FN <- cm["Yes", "No"]
  accuracy  <- (TP + TN) / sum(cm)
  precision <- TP / (TP + FP)       # of the bookings flagged, how many cancel
  recall    <- TP / (TP + FN)       # of the real cancellations, how many caught
  f1        <- 2 * precision * recall / (precision + recall)
  auc       <- as.numeric(colAUC(prob, actual))      # area under ROC curve
  cat("\n=====", model.name, "=====\n"); print(cm)
  data.table(Model = model.name,
             Accuracy  = round(accuracy, 4),
             Precision = round(precision, 4),
             Recall    = round(recall, 4),
             F1        = round(f1, 4),
             AUC       = round(auc, 4))
}


# =============================================================================
# ---- 6. MODEL 1: LOGISTIC REGRESSION ----------------------------------------
# =============================================================================
# Why: produces an easy-to-explain model - each coefficient shows how a booking
# attribute raises/lowers the odds of cancellation (via odds ratios).

# 6.1 Fit the full model using all predictors on the trainset
#     NOTE: R will warn "fitted probabilities numerically 0 or 1 occurred".
#     This is expected - 99% of Non-Refund deposit bookings are cancelled, so
#     that variable almost perfectly separates the classes. Predictions are
#     still valid; the very large odds ratio simply confirms its strength.
log.full <- glm(is_canceled ~ ., family = binomial, data = trainset)
summary(log.full)
formula(log.full)

# 6.3 Odds ratios with 95% confidence intervals (Wald)
#     OR > 1 -> increases odds of cancellation; OR < 1 -> reduces the odds
log.coef <- summary(log.full)$coefficients
log.or <- data.table(Variable = rownames(log.coef),
                     OddsRatio = round(exp(log.coef[, "Estimate"]), 3),
                     OR_Lower95 = round(exp(log.coef[, "Estimate"] - 1.96 * log.coef[, "Std. Error"]), 3),
                     OR_Upper95 = round(exp(log.coef[, "Estimate"] + 1.96 * log.coef[, "Std. Error"]), 3),
                     p_value = signif(log.coef[, "Pr(>|z|)"], 3))
log.or[order(-OddsRatio)]
fwrite(log.or, "outputs/05_logistic_odds_ratios.csv")

# Interpretation examples:
#  - deposit_typeNon Refund: very large OR -> non-refundable group blocks are
#    almost always cancelled (typical of tour-operator allotments)
#  - previous_cancellations: OR > 1 -> guests who cancelled before do it again
#  - total_of_special_requests / required_car_parking_spaces: OR < 1 -> guests
#    who personalise their stay are committed and rarely cancel
#  - lead_time: OR slightly > 1 per day -> adds up over months of lead time

# 6.4 Predict on the testset and evaluate
log.prob <- predict(log.full, newdata = testset, type = "response")
res.log  <- evaluate.model("Logistic Regression", log.prob, testset$is_canceled)
res.log


# =============================================================================
# ---- 7. MODEL 2: XGBOOST (Extreme Gradient Boosting) ------------------------
# =============================================================================
# Why: builds hundreds of small decision trees one after another, where each
# new tree corrects the errors of the trees before it. Captures non-linear
# effects and interactions (e.g. long lead time AND no deposit) and is usually
# among the most accurate methods for tabular data. Trade-off: a "black box",
# so we rely on variable importance to interpret it.

# 7.1 XGBoost needs a purely numeric matrix -> one-hot encode the factors.
#     "- 1" removes the intercept column. Train and test share the same factor
#     levels, so both matrices have identical columns.
x.train <- model.matrix(is_canceled ~ . - 1, data = trainset)
x.test  <- model.matrix(is_canceled ~ . - 1, data = testset)
y.train <- as.integer(trainset$is_canceled == "Yes")   # 1 = cancelled
y.test  <- as.integer(testset$is_canceled == "Yes")
dtrain  <- xgb.DMatrix(data = x.train, label = y.train)
dtest   <- xgb.DMatrix(data = x.test,  label = y.test)

# 7.2 Hyperparameters (standard, moderately conservative settings)
#     eta              : learning rate - small steps reduce over-fitting
#     max_depth        : depth of each tree (allows interactions up to 6 vars)
#     subsample        : each tree sees a random 80% of the bookings
#     colsample_bytree : each tree sees a random 80% of the predictors
xgb.params <- list(objective        = "binary:logistic",  # outputs P(cancel)
                   eval_metric      = "auc",
                   eta              = 0.1,
                   max_depth        = 6,
                   min_child_weight = 1,
                   subsample        = 0.8,
                   colsample_bytree = 0.8)

# 7.3 Choose the number of trees with 5-fold cross-validation on the trainset.
#     Training stops once CV AUC has not improved for 30 rounds (early
#     stopping), so the testset is never used for tuning. 1000 is an upper
#     cap: if CV AUC is still creeping up it may hit the cap, but the curve
#     (06_xgb_cv_auc.png) is flat by ~600 trees, so extra trees change little.
xgb.cv.res <- xgb.cv(params = xgb.params, data = dtrain, nrounds = 1000,
                     nfold = 5, early_stopping_rounds = 30, verbose = 0)
cv.log  <- as.data.table(xgb.cv.res$evaluation_log)
best.nrounds <- which.max(cv.log$test_auc_mean)
cat("XGBoost best number of trees (5-fold CV):", best.nrounds,
    "| CV AUC:", round(max(cv.log$test_auc_mean), 4), "\n")

png("outputs/06_xgb_cv_auc.png", width = 900, height = 600, res = 120)
plot(cv.log$iter, cv.log$test_auc_mean, type = "l", lwd = 2, col = "steelblue",
     xlab = "Number of trees", ylab = "Cross-validated AUC",
     main = "XGBoost: 5-fold CV AUC vs. number of trees")
abline(v = best.nrounds, lty = 2, col = "red")
dev.off()

# 7.4 Fit the final model on the full trainset with the chosen number of trees
xgb.model <- xgb.train(params = xgb.params, data = dtrain, nrounds = best.nrounds)
xgb.model
# 7.5 Variable importance. Gain = share of the model's total improvement in
#     accuracy contributed by splits on that variable (sums to 100%).
#     Factor variables appear as their dummy columns, e.g. "deposit_typeNon Refund".
xgb.imp <- as.data.table(xgb.importance(model = xgb.model))
xgb.imp[, Gain_pct := round(100 * Gain, 1)]
xgb.imp[1:15, .(Feature, Gain_pct)]
fwrite(xgb.imp, "outputs/07_xgb_variable_importance.csv")

p4 <- ggplot(xgb.imp[1:15], aes(x = reorder(Feature, Gain_pct), y = Gain_pct)) +
  geom_col(fill = "steelblue") + coord_flip() +
  labs(title = "XGBoost variable importance (top 15, by Gain)",
       x = NULL, y = "Gain (% of total)") +
  theme_minimal()
ggsave("outputs/06b_xgb_variable_importance.png", p4, width = 7, height = 5)

# 7.6 Predict on the testset and evaluate
xgb.prob <- predict(xgb.model, dtest)
res.xgb  <- evaluate.model("XGBoost", xgb.prob, testset$is_canceled)
res.xgb


# =============================================================================
# ---- 8. MODEL 3: RANDOM FOREST ----------------------------------------------
# =============================================================================
# Why: an ensemble of many decision trees built on bootstrap samples with
# random subsets of variables at each split. Usually the most accurate model,
# less prone to over-fitting than a single tree. Trade-off: harder to explain,
# so we rely on variable importance to interpret it.

# 8.1 Train with default settings: 500 trees, mtry = floor(sqrt(#predictors))
#     importance = TRUE stores Mean Decrease Accuracy & Gini for each variable.
rf.model <- randomForest(is_canceled ~ ., data = trainset,
                         ntree = 500, importance = TRUE)
rf.model                  # Out-Of-Bag (OOB) error = built-in test error estimate

# 8.2 Check that 500 trees are enough - OOB error should be flat by the end
png("outputs/08_rf_oob_error.png", width = 900, height = 600, res = 120)
plot(rf.model, main = "Random Forest: OOB error vs. number of trees")
legend("topright", colnames(rf.model$err.rate), col = 1:3, lty = 1:3)
dev.off()

# 8.3 Variable importance
rf.imp <- importance(rf.model)
rf.imp[order(-rf.imp[, "MeanDecreaseAccuracy"]), ]
png("outputs/09_rf_variable_importance.png", width = 1200, height = 700, res = 120)
varImpPlot(rf.model, main = "Random Forest variable importance")
dev.off()
fwrite(data.table(Variable = rownames(rf.imp), rf.imp)[order(-MeanDecreaseAccuracy)],
       "outputs/10_rf_variable_importance.csv")

# 8.4 Predict on the testset and evaluate
rf.prob <- predict(rf.model, newdata = testset, type = "prob")[, "Yes"]
res.rf  <- evaluate.model("Random Forest", rf.prob, testset$is_canceled)
res.rf


# =============================================================================
# ---- 9. MODEL COMPARISON -----------------------------------------------------
# =============================================================================
model.comparison <- rbindlist(list(res.log, res.xgb, res.rf))
model.comparison
fwrite(model.comparison, "outputs/11_model_comparison.csv")

# ROC curves of all 3 models on one chart (higher & further left = better)
png("outputs/12_roc_curves.png", width = 900, height = 800, res = 120)
colAUC(cbind(Logistic = log.prob, XGBoost = xgb.prob, RandomForest = rf.prob),
       testset$is_canceled, plotROC = TRUE)
title("ROC curves on test set")
dev.off()


# =============================================================================
# ---- 10. BUSINESS IMPACT - translating accuracy into revenue -----------------
# =============================================================================
# Question for management: "Of the room revenue that is actually lost to
# cancellations in the test period, how much would each model have warned us
# about in advance?" Bookings flagged early can be targeted with deposits,
# reminders or covered by risk-based overbooking.
actual.cancel <- testset$is_canceled == "Yes"
revenue.lost  <- sum(test.value[actual.cancel])

revenue.flagged <- function(prob, threshold = 0.5) {
  flagged <- prob > threshold
  data.table(
    Revenue_Lost_to_Cancellations = round(revenue.lost),
    Revenue_Flagged_in_Advance    = round(sum(test.value[flagged & actual.cancel])),
    Pct_Lost_Revenue_Flagged      = round(100 * sum(test.value[flagged & actual.cancel]) / revenue.lost, 1),
    # Bookings wrongly flagged (guests who actually showed up) - cost of
    # false alarms, e.g. risk of walking a guest if we overbook on them
    False_Alarm_Bookings          = sum(flagged & !actual.cancel)
  )
}
business.impact <- rbindlist(list(
  cbind(Model = "Logistic Regression", revenue.flagged(log.prob)),
  cbind(Model = "XGBoost",             revenue.flagged(xgb.prob)),
  cbind(Model = "Random Forest",       revenue.flagged(rf.prob))))
business.impact
fwrite(business.impact, "outputs/13_business_impact.csv")

# 10.1 Risk tiers for the operational playbook (using the best model by test
#      AUC - XGBoost or Random Forest):
#      Low (<30%), Medium (30-70%), High (>70%) - shows that the score
#      separates bookings into groups with very different real cancel rates.
best.model <- model.comparison[Model != "XGBoost"][which.max(AUC), Model]
best.prob  <- list("Logistic Regression" = log.prob,
                   "Random Forest" = rf.prob)[[best.model]]
cat("Risk tiers based on:", best.model, "\n")
risk.tier <- cut(best.prob, breaks = c(-Inf, 0.3, 0.7, Inf),
                 labels = c("Low (<30%)", "Medium (30-70%)", "High (>70%)"))
tier.summary <- data.table(Tier = risk.tier, Cancelled = actual.cancel, Value = test.value)[
  , .(Bookings = .N,
      Pct_of_Bookings = NA_real_,
      Actual_Cancel_Rate = round(100 * mean(Cancelled), 1),
      Room_Revenue = round(sum(Value))), by = Tier][order(Tier)]
tier.summary[, Pct_of_Bookings := round(100 * Bookings / sum(Bookings), 1)]
tier.summary
fwrite(tier.summary, "outputs/14_risk_tiers.csv")
