
# Load necessary libraries
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(psych)
library(corrplot)
library(ResourceSelection)

################################################################################

# Importing dataset
df <- read.csv("/users/harinim/downloads/Diabetes Dataset.csv")
head(df)

################################################################################

#Dataset description
# displaying the column name
colnames(df)
# Displaying structure of data
str(df)
# Description of dataset
summary(df)
# Display unique value of target variable and their propotion
unique(df$CLASS)
table(df$CLASS)

################################################################################

# Splitting the values of P from rest for prediction
# Storing only the values with P
df_predict <- subset(df, CLASS == 'P')
head(df_predict)
# Dropping the rows with P values
df_model <- subset(df, CLASS != 'P')

################################################################################

#Data validation
# checking for NAs and null values
sum(is.na(df_model))
na_values <- data.frame(colSums(is.na(df_model)))
head(na_values, 25)


################################################################################

# Data Visualisation

# Select numeric variables in first step
numeric_vars <- c("AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI")

# Correlation plot
# Create correlation matrix after selecting and storing the numeric variables
cor_matrix <- cor(df_model[numeric_vars])
cor_matrix
corrplot(cor_matrix, method = "square", type = "upper", tl.col = "black", tl.srt = 45)

#Scatter plot
pairs(df_model[, c("AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI")])

################################################################################

# Data Pre processing

#Removing unnecessary columns:
#since ID and no_of_patient make no difference to diabetic
df_model <- df_model %>% select(-c(ID, No_Pation))
head(df_model)

# Converting categorical data into binary data
df_model <- df_model %>% 
  mutate(Gender = case_when(Gender == "M" ~ 1, Gender == "F" ~ 0),
         CLASS = case_when(CLASS == "Y" ~ 1, CLASS == "N" ~ 0))
head(df_model)

################################################################################

#Split the data-set into train and test sets
set.seed(123)
split <- createDataPartition(y = df_model$CLASS, p = 0.7, list = FALSE)
train1 <- df_model[split, ]
test1 <- df_model[-split, ]

################################################################################

#Function to evaluate the model
modelOutput <- function(test,pred){
  # Calculate accuracy and F1 score
  conf_matrix <- table(test$CLASS, pred)
  #rmse <- sqrt(mean((pred - test)^2))
  accuracy <- mean(pred == test$CLASS)
  precision <- sum(pred[test$CLASS == 1] == 1) / sum(pred == 1)
  recall <- conf_matrix[2,2]/sum(conf_matrix[2,])
  f1 <- 2 * precision * recall / (precision + recall)
  
  metrics_df <- data.frame(
    Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
    Value = c(round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3))
  )
  return(head(metrics_df))
}

################################################################################

# MODEL 1

#Logistic regression model fitting to the training set using every available inputs
model_1 <- glm(CLASS ~ ., data = train1, family = binomial)
summary(model_1)
par(mfrow=c(2,2))
plot(model_1)
# Make predictions on the test set
pred <- predict(model_1, newdata = test1, type = "response")
pred_class <- ifelse(pred > 0.5, 1, 0)
modelOutput(test1,pred_class)

################################################################################

# MODEL 2

#Creating a logistic regression model using training data after adjusting for irrelevant factors.
model_2 <- glm(CLASS ~ HbA1c + Chol + TG + BMI, family = binomial, data = train1)
summary(model_2)
par(mfrow=c(2,2))
plot(model_2)
# Make predictions on the test set
pred <- predict(model_2, newdata = test1, type = "response")
pred_class <- ifelse(pred > 0.5, 1, 0)
modelOutput(test1,pred_class)

################################################################################

#visualizing the data to know more about the data set
histogram <- function(x)
{
  title <- paste(deparse(substitute(x), 500), collapse="\n")
  sdx <- sd(x) 
  mx <- mean(x)
  hist(x, prob=TRUE,main=paste("Histogram of ",title), 
       xlim=c(mx-3*sdx, mx+3*sdx), ylim=c(0, 0.5/sdx)) 
  curve(dnorm(x, mean=mx, sd=sdx), col='red', lwd=3, add=TRUE)
}

par(mfrow=c(2,2))
histogram(log(df_model$HbA1c))
histogram(df_model$HbA1c)
histogram(df_model$HbA1c)

histogram(df_model$Chol)
histogram(sqrt(df_model$Chol))

histogram(log(df_model$TG)) #
histogram(df_model$TG)
histogram(sqrt(df_model$TG))

histogram(log(df_model$BMI))
histogram(df_model$BMI)
histogram(sqrt(df_model$BMI)) #

################################################################################

#MODEL 3
 
#applying model by considering above distribution of the data 
model_3 <- glm(CLASS ~ log(HbA1c) + log(Chol) +log(TG) + log(BMI), family = binomial, data = train1)
summary(model_3)
par(mfrow=c(2,2))
plot(model_3)
# Make predictions on the test set
pred <- predict(model_3, newdata = test1, type = "response")
pred_class <- ifelse(pred > 0.5, 1, 0)
modelOutput(test1,pred_class)

################################################################################

# Good fit test

# Perform Hosmer-Lemeshow test
hl_test <- hoslem.test(train1$CLASS, predict(model_2, type = "response"))
hl_test


################################################################################

# Predict for model 2
#testing the final model on the 'Diabetes==P' cases 
predicted_values_p <- predict(model_2, newdata = df_predict, type = "response")
prob_diabetic <- sum(predicted_values_p)/length(predicted_values_p)
cat("Probability of diagnosing the 'P' class cases as diabetic: ", prob_diabetic, "\n")

################################################################################




