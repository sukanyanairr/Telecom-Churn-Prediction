library(readr)
if (!require(readr)) {
  install.packages("readr")
  library(readr)
}
library(dplyr)
library(ggplot2)
library(summarytools)
if (!require(summarytools)) {
  install.packages("summarytools")
  library(summarytools)
}
summary(telco_data)

# Missing Value Analysis and Imputation
telco_data$TotalCharges <- as.numeric(telco_data$TotalCharges)
telco_data$TotalCharges[is.na(telco_data$TotalCharges)] <- mean(telco_data$TotalCharges, na.rm = TRUE)

# Standardizing numerical variables
telco_data$MonthlyCharges <- scale(telco_data$MonthlyCharges)
telco_data$TotalCharges <- scale(telco_data$TotalCharges)

# Feature selection/engineering
# Creating tenure groups
telco_data <- telco_data %>%
  mutate(tenure_group = case_when(
    tenure <= 12 ~ '0-12 Months',
    tenure > 12 & tenure <= 24 ~ '12-24 Months',
    tenure > 24 & tenure <= 48 ~ '24-48 Months',
    tenure > 48 & tenure <= 60 ~ '48-60 Months',
    tenure > 60 ~ '60+ Months'
  ))

# Outlier Detection and Treatment
# Detecting outliers using IQR
Q1 <- quantile(telco_data$MonthlyCharges, 0.25)
Q3 <- quantile(telco_data$MonthlyCharges, 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

telco_data <- telco_data %>%
  filter(MonthlyCharges >= lower_bound & MonthlyCharges <= upper_bound)


# Visualizations
# Histogram for tenure
ggplot(telco_data, aes(x=tenure)) + 
  geom_histogram(binwidth=1) + 
  labs(title="Distribution of Tenure")

# Box plot for MonthlyCharges
ggplot(telco_data, aes(y=MonthlyCharges)) + 
  geom_boxplot() + 
  labs(title="Boxplot of Monthly Charges")
# Scatter plot for tenure vs MonthlyCharges colored by Churn
ggplot(telco_data, aes(x=tenure, y=MonthlyCharges, color=Churn)) +
  geom_point(alpha=0.6, size=3) +
  scale_color_manual(values=c("No"="#1f77b4", "Yes"="#d62728")) +
  labs(title="Scatter Plot of Tenure vs Monthly Charges",
       x="Tenure (months)",
       y="Monthly Charges",
       color="Churn") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size=16, face="bold"),
        axis.title = element_text(size=14),
        legend.title = element_text(size=14),
        legend.text = element_text(size=12))

# Install and load necessary libraries
if (!require(psych)) install.packages("psych")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(DataExplorer)) install.packages("DataExplorer")
if (!require(car)) install.packages("car")
if (!require(lmtest)) install.packages("lmtest")
if (!require(Metrics)) install.packages("Metrics")
if (!require(MASS)) install.packages("MASS")

library(psych)
library(ggplot2)
library(DataExplorer)
library(car)
library(lmtest)
library(Metrics)
library(MASS)



str(telco_data)

# Missingness analysis
summary(telco_data)
plot_missing(telco_data)


# Convert categorical variables to factors
categorical_vars <- c("gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                      "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
                      "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "Churn")

telco_data[categorical_vars] <- lapply(telco_data[categorical_vars], as.factor)

str(telco_data)

# Convert 'Churn' to numeric for regression analysis
telco_data$Churn <- as.numeric(telco_data$Churn) - 1  # Yes=1, No=0

# Handle missing values
telco_data <- na.omit(telco_data)

# Split the data into training and testing sets
set.seed(1234)
sample <- sample.int(n = nrow(telco_data), size = floor(.7*nrow(telco_data)), replace = F)
train <- telco_data[sample, ]
test <- telco_data[-sample, ]

# EDA (Optional)
plot_histogram(train)
plot_density(train)
plot_correlation(train, maxcat = 5)

# Model 1: Full model using all relevant predictors
full_model <- lm(Churn ~ tenure + MonthlyCharges + TotalCharges + Contract + PaperlessBilling + PaymentMethod, data=train)
summary(full_model)

# Model 2: Using a subset of predictors based on domain knowledge
subset_model <- lm(Churn ~ tenure + MonthlyCharges + Contract + InternetService, data=train)
summary(subset_model)

# Model 3: Stepwise selection based on AIC
stepwise_model <- stepAIC(lm(Churn ~ tenure + MonthlyCharges + TotalCharges + Contract + PaperlessBilling + PaymentMethod + 
                               InternetService + OnlineSecurity + TechSupport + StreamingTV + StreamingMovies, data=train), 
                          direction="both")
summary(stepwise_model)

# Predict and evaluate on test data
full_model_pred <- predict(full_model, newdata=test)
subset_model_pred <- predict(subset_model, newdata=test)
stepwise_model_pred <- predict(stepwise_model, newdata=test)

# Calculate performance metrics
full_model_r2 <- summary(full_model)$r.squared
full_model_test_r2 <- cor(test$Churn, full_model_pred)^2

subset_model_r2 <- summary(subset_model)$r.squared
subset_model_test_r2 <- cor(test$Churn, subset_model_pred)^2

stepwise_model_r2 <- summary(stepwise_model)$r.squared
stepwise_model_test_r2 <- cor(test$Churn, stepwise_model_pred)^2

# Compare R-squared values
cat("Full Model - Train R2:", full_model_r2, "Test R2:", full_model_test_r2, "\n")
cat("Subset Model - Train R2:", subset_model_r2, "Test R2:", subset_model_test_r2, "\n")
cat("Stepwise Model - Train R2:", stepwise_model_r2, "Test R2:", stepwise_model_test_r2, "\n")

