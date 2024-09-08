# Group No: 2
# Project name: Bike Sharing Demand
# Phase: 3
# Group Members: vishnu kariyattu, Harsha Bharati & Krithika Kaushik 

# Code Start

# STEPS
# LOADING LIBRARIES

library(ggplot2)      # Package for Data visualization
library(caret)        # Package for Data Splitting, Pre-Processing, Feature Selection, Model Training, Resampling,Performance Measurement, Ensembling etc.  
library(rpart)        # Package for Decision Tree Building, Handling Missing Values, Pruning Trees, Cross-Validation
library(rpart.plot)   # Package for Plotting Decision Trees
library(corrplot)     # For Heat map
library(randomForest) # For Random Forest Model
library(gbm)          # For Gradient Boosting Model

#LOAD
source("F:/2. Babson/1. Curriculam + Studies/3. Sem 3/1. Course Work/Machine Learning/Complete data Set/Data/BabsonAnalytics.R")
df = read.csv("F:/2. Babson/1. Curriculam + Studies/3. Sem 3/1. Course Work/Machine Learning/Complete data Set/Data/train.csv")

# Data is loaded from the source for further analysis

# DATA VISUALISATION
# View (df) # This function can be used to understand the how the data is and what is the nature of the data.

# RESULT
# 10866 observations of 12 variables

# Understanding Data Variable type 

str(df) # Shows the data type of the 12 variables. This helps in feature Engineering

# MANAGE THE DATA [Feature Engineering](Count is our target)
# The columns "season","holiday","workingday" and "weather" are not "int" data types and should be converted to "categorical" data type.

df$season = as.factor(df$season)
df$holiday = as.factor(df$holiday)
df$workingday = as.factor(df$workingday)
df$weather = as.factor(df$weather)

# From the "datetime' coloumn, we need to create "date,"hour","weekDay", and "month"
# Since 'datetime' is a character, we need to convert it to POSIXct to extract components

df$datetime = as.POSIXct(df$datetime, format = "%m/%d/%Y %H:%M")

# Now we can extract the components from the 'datetime' column

df$date = as.Date(df$datetime)  # Extracts date
df$hour = format(df$datetime, "%H")  # Extracts hour as a character string
df$weekday = weekdays(df$date)  # Extracts the day of the week
df$month = months(df$date)  # Extracts the name of the month

# The columns "hour", "weekday", and "month" are created as character data types by default.
# We need to convert "hour" to numeric, and "weekday" and "month" to factor (categorical) data types.

df$hour = as.numeric(df$hour)
df$weekday = factor(df$weekday)
df$month = factor(df$month)

# After extracting the relevant data, let us remove the "datetime" column

df$datetime = NULL

# View (df) # This function can be used to check whether the data is accurate and to our liking

# CLEANING THE DATA

# Missing Values Analysis 
total_missing_values = sum(is.na(df)) # To check missing values for the entire data set 

#RESULT
# Zero missing values observed

# OUTLIER ANALYSIS 
# Coded with the help of Chatgpt

# For all numerical variables in the dataframe
boxplot(df[,sapply(df, is.numeric)], main = "Boxplots for all numeric variables")

# Box Plot 1: Count
p1 <- ggplot(df, aes(y = count)) + geom_boxplot() + labs(title = "Box Plot On Count", y = "Count")
# plot(p1)

# Box Plot 2: Count across Season
p2 <- ggplot(df, aes(x = season, y = count)) + geom_boxplot() + labs(title = "Box Plot On Count Across Season", x = "Season", y = "Count")
plot(p2)
# Spring season has the lowest dip in count. 

# Box Plot 3: Count across Hour of the Day
p3 <- ggplot(df, aes(x = factor(hour), y = count)) + geom_boxplot() + labs(title = "Box Plot On Count Across Hour Of The Day", x = "Hour Of The Day", y = "Count")
# plot(p3)
# The median value are relatively higher from 7AM - 8AM and from 5PM - 6PM, and this can be attributed to regular school and office users at that time.

# Box Plot 4: Count across Working Day
p4 <- ggplot(df, aes(x = factor(workingday), y = count)) + geom_boxplot() + labs(title = "Box Plot On Count Across Working Day", x = "Working Day", y = "Count")
# plot(p4)

# Most of the outlier points are contributed from "Working Day" than from "Non Working Day".  

# CORRELATION ANALYSIS
# GENERATE THE HEATMAP TO CALCULATE THE CORRELATION MATRIX

corrMatt <- cor(df[, c("temp", "atemp", "casual", "registered", "humidity", "windspeed", "count")], use = "complete.obs")

# Create a heatmap using corrplot
corrplot(corrMatt, method = "color", type = "upper", order = "hclust", tl.col = "black", tl.srt = 45, addCoef.col = "black", diag = FALSE) 

#RESULT
# "humidity" has negative correlation with "count"
# "temp" and "atemp" have positive correlation with "count"
# "windspeed" is not going be of much use as its correlation count denotes. 

# At this point we the value of windspeed for 26 obs is 0. any value below 6 is inputted as 0. 
# There is an option of building a Random Forest Model to predict the 0's in "Windspeed", however, we are ignoring this step as the impact of "windspeed" on count, as stated above, is very minimal. 


# PARTITION THE DATA 
set.seed(1234)
N = nrow(df) #counting the total number of rows
trainingSize = round(N*0.6) # decide how many rows go to training, here 60% 
trainingCases = sample(N, trainingSize) # simple random sample the rows for training
training = df[trainingCases, ] #slice training from df
test = df[-trainingCases, ] #slice test from df 

#BUILD THE MODEL
# MODEL 1 - LINEAR REGRESSION MODEL
lm_model = lm(count ~ ., data = training)
lm_model = step(lm_model)
summary(lm_model)

### PREDICTIONS ###
predictions = predict(lm_model, test)

# EVALUATE
observations = test$count   #to compare with the true prices
errors_lm = observations - predictions  # negative is overestimating 

### PERFORMANCE EVALUATION ###

#RMSE - root mean squared error
rmse_lm = sqrt(mean(errors_lm^2))  #number measured in units of the target. On average 1341 euros off. It doesn't matter if + or -

#MAPE - mean average percentage error
mape_lm = mean(abs(errors_lm/observations))  #this number is measured in %. 9% off in this case

# BENCHMARKING AGAINST LINEAR REGRESSION MODEL

predictions_bench = mean(training$count)
errors_bench  = observations - predictions_bench
rmse_bench = sqrt(mean(errors_bench^2))
mape_bench = mean(abs(errors_bench/observations))


# MODEL 2 - RANDOM FOREST MODEL
### BAGGING ###
# Multiple decision trees are built and their results are averaged.

rf_model = randomForest(count ~., data = training, ntree = 500)
pred_rf = predict(rf_model, test)
predTF_rf = (pred_rf > 0.5)

#EVALUATE 
error_rf = sum(observations != predTF_rf)/nrow(test)

### PERFORMANCE EVALUATION ###

#RMSE - root mean squared error
rmse_rf = sqrt(mean(error_rf^2))

#MAPE - mean average percentage error
mape_rf = mean(abs(error_rf/observations))


# MODEL 3 - GRADIENT BOOSTING MODEL
### BOOSTING IN R ###
set.seed(1234)
gbm_model = gbm(count ~ ., data = training, distribution = "gaussian", n.trees = 5000, interaction.depth = 4, cv.folds = 5)
best_trees = gbm.perf(gbm_model, method = "cv")
predictions_gbm = predict(gbm_model, test, n.trees = best_trees)

# Evaluate
errors_gbm = test$count - predictions_gbm

### PERFORMANCE EVALUATION ###
rmse_gbm = sqrt(mean(errors_gbm^2))
mape_gbm <- mean(abs(errors_gbm/test$count))

# MANAGER MODEL

# Predictions on training data for stacking
pred_lm_train = predict(model, training)
pred_rf_train = predict(rf, training)
pred_gbm_train = predict(gbm_model, training, n.trees = best_trees)

# Combine predictions
combined_pred_train = data.frame(pred_lm_train, pred_rf_train, pred_gbm_train)

# Train a meta-model (e.g., linear regression) on combined predictions
meta_model = lm(count ~ ., data = cbind(training, combined_pred_train))

# Predictions on test data
pred_lm_test = predict(model, test)
pred_rf_test = predict(rf, test)
pred_gbm_test = predict(gbm_model, test, n.trees = best_trees)

# Combine test predictions
combined_pred_test = data.frame(pred_lm_test, pred_rf_test, pred_gbm_test)

# Final predictions using meta-model
final_pred = predict(meta_model, newdata = cbind(test, combined_pred_test))

# Evaluate
errors_stack = test$count - final_pred
rmse_stack = sqrt(mean(errors_stack^2))
mape_stack = mean(abs(errors_stack/test$count))













