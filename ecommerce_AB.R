# E-commerce A/B Testing Analysis Project
# Repository: ecommerce-ab-testing-r

library(tidyverse)
library(readr)
library(car)
library(randomForest)
library(glmnet)
library(pROC)
library(corrplot)
library(ggplot2)
library(dplyr)
library(caret)
library(MASS)



# Create realistic A/B test scenario for e-commerce conversion
set.seed(42)
n_users <- 10000

ab_data <- data.frame(
  user_id = 1:n_users,
  variant = sample(c("A", "B"), n_users, replace = TRUE),
  age = sample(18:65, n_users, replace = TRUE),
  session_duration = round(rnorm(n_users, mean = 300, sd = 120), 0),
  pages_viewed = rpois(n_users, lambda = 5),
  previous_purchases = rpois(n_users, lambda = 2),
  device_type = sample(c("mobile", "desktop", "tablet"), n_users, 
                       replace = TRUE, prob = c(0.6, 0.3, 0.1)),
  traffic_source = sample(c("organic", "paid", "social", "direct"), n_users,
                          replace = TRUE, prob = c(0.4, 0.3, 0.2, 0.1)),
  country = sample(c("US", "UK", "CA", "DE", "FR"), n_users, replace = TRUE),
  day_of_week = sample(1:7, n_users, replace = TRUE)
)

# Create conversion probability based on variant and features
conversion_prob <- ifelse(ab_data$variant == "B", 0.08, 0.06) +
  (ab_data$session_duration / 1000) * 0.02 +
  (ab_data$pages_viewed / 10) * 0.01 +
  (ab_data$previous_purchases / 5) * 0.03 +
  ifelse(ab_data$device_type == "desktop", 0.01, 0) +
  ifelse(ab_data$traffic_source == "paid", 0.015, 0)

conversion_prob <- pmax(pmin(conversion_prob, 0.95), 0.01)
ab_data$converted <- rbinom(n_users, 1, conversion_prob)

# Add revenue data
ab_data$revenue <- ifelse(ab_data$converted == 1, 
                          round(rnorm(n_users, mean = 50, sd = 20), 2), 0)
ab_data$revenue <- pmax(ab_data$revenue, 0)

# ===== EXPLORATORY DATA ANALYSIS =====
summary(ab_data)
table(ab_data$variant, ab_data$converted)

conversion_by_variant <- aggregate(cbind(users = user_id, conversions = converted, 
                                         revenue = revenue) ~ variant, 
                                   data = ab_data, 
                                   FUN = function(x) c(length(x), sum(x), mean(x)))

conversion_by_variant$users <- conversion_by_variant$users[,1]
conversion_by_variant$conversions <- conversion_by_variant$conversions[,2] 
conversion_by_variant$conversion_rate <- conversion_by_variant$conversions / conversion_by_variant$users
conversion_by_variant$avg_revenue <- conversion_by_variant$revenue[,3]
conversion_by_variant$total_revenue <- aggregate(revenue ~ variant, data = ab_data, sum)$revenue
conversion_by_variant <- conversion_by_variant[, !names(conversion_by_variant) %in% "revenue"]

print(conversion_by_variant)

# ===== STATISTICAL TESTING METHODS =====

# 1. Chi-Square Test
chi_test <- chisq.test(table(ab_data$variant, ab_data$converted))
print(chi_test)

# 2. Two-Sample Proportion Test
prop_test <- prop.test(
  x = c(sum(ab_data$converted[ab_data$variant == "A"]),
        sum(ab_data$converted[ab_data$variant == "B"])),
  n = c(sum(ab_data$variant == "A"), sum(ab_data$variant == "B"))
)
print(prop_test)

# 3. T-Test for Revenue
t_test_revenue <- t.test(revenue ~ variant, data = ab_data)
print(t_test_revenue)

# 4. ANOVA for multiple factors
ab_data$variant_num <- as.numeric(as.factor(ab_data$variant))
ab_data$device_num <- as.numeric(as.factor(ab_data$device_type))
ab_data$traffic_num <- as.numeric(as.factor(ab_data$traffic_source))

anova_model <- aov(converted ~ variant + device_type + traffic_source + 
                     age + session_duration + pages_viewed, data = ab_data)
summary(anova_model)

# 5. Two-Way ANOVA
anova_interaction <- aov(converted ~ variant * device_type, data = ab_data)
summary(anova_interaction)

# 6. Levene's Test for Homogeneity of Variances
levene_test <- leveneTest(converted ~ variant, data = ab_data)
print(levene_test)

# ===== MACHINE LEARNING MODELS =====

# Prepare data for ML
ab_ml <- ab_data[, !names(ab_data) %in% c("user_id", "revenue")]
ab_ml$variant <- as.factor(ab_ml$variant)
ab_ml$device_type <- as.factor(ab_ml$device_type)
ab_ml$traffic_source <- as.factor(ab_ml$traffic_source)
ab_ml$country <- as.factor(ab_ml$country)
ab_ml$converted <- as.factor(ab_ml$converted)

# Create dummy variables
ab_ml_dummy <- model.matrix(converted ~ . - 1, data = ab_ml)
ab_ml_dummy <- data.frame(ab_ml_dummy)
ab_ml_dummy$converted <- ab_ml$converted

# Train-test split
train_idx <- createDataPartition(ab_ml_dummy$converted, p = 0.8, list = FALSE)
train_data <- ab_ml_dummy[train_idx, ]
test_data <- ab_ml_dummy[-train_idx, ]

# 1. Logistic Regression
logistic_model <- glm(converted ~ ., data = train_data, family = binomial)
summary(logistic_model)

logistic_pred <- predict(logistic_model, test_data, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, 1, 0)

# 2. Ridge Regression
x_train <- model.matrix(converted ~ . - 1, data = train_data)
x_train <- x_train[, !colnames(x_train) %in% "converted"]
y_train <- as.numeric(train_data$converted) - 1
x_test <- model.matrix(converted ~ . - 1, data = test_data)
x_test <- x_test[, !colnames(x_test) %in% "converted"]
y_test <- as.numeric(test_data$converted) - 1

ridge_model <- cv.glmnet(x_train, y_train, alpha = 0, family = "binomial")
ridge_pred <- predict(ridge_model, x_test, type = "response", s = "lambda.min")
ridge_pred_class <- ifelse(ridge_pred > 0.5, 1, 0)

# 3. Lasso Regression
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")
lasso_pred <- predict(lasso_model, x_test, type = "response", s = "lambda.min")
lasso_pred_class <- ifelse(lasso_pred > 0.5, 1, 0)

# 4. Random Forest
rf_model <- randomForest(converted ~ ., data = train_data, ntree = 500)
rf_pred <- predict(rf_model, test_data, type = "prob")[,2]
rf_pred_class <- ifelse(rf_pred > 0.5, 1, 0)

# 5. Stepwise Logistic Regression
stepwise_model <- stepAIC(logistic_model, direction = "both", trace = FALSE)
stepwise_pred <- predict(stepwise_model, test_data, type = "response")
stepwise_pred_class <- ifelse(stepwise_pred > 0.5, 1, 0)

# ===== MODEL EVALUATION =====

# Calculate metrics for all models
calculate_metrics <- function(actual, predicted_prob, predicted_class) {
  roc_obj <- roc(actual, predicted_prob)
  auc_score <- auc(roc_obj)
  
  confusion_matrix <- table(actual, predicted_class)
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  if (length(unique(predicted_class)) > 1) {
    precision <- confusion_matrix[2,2] / sum(confusion_matrix[,2])
    recall <- confusion_matrix[2,2] / sum(confusion_matrix[2,])
    f1_score <- 2 * (precision * recall) / (precision + recall)
  } else {
    precision <- recall <- f1_score <- 0
  }
  
  return(list(
    auc = auc_score,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score
  ))
}

# Evaluate all models
models_performance <- data.frame(
  Model = c("Logistic", "Ridge", "Lasso", "Random Forest", "Stepwise"),
  AUC = c(
    calculate_metrics(y_test, logistic_pred, logistic_pred_class)$auc,
    calculate_metrics(y_test, ridge_pred, ridge_pred_class)$auc,
    calculate_metrics(y_test, lasso_pred, lasso_pred_class)$auc,
    calculate_metrics(y_test, rf_pred, rf_pred_class)$auc,
    calculate_metrics(y_test, stepwise_pred, stepwise_pred_class)$auc
  ),
  Accuracy = c(
    calculate_metrics(y_test, logistic_pred, logistic_pred_class)$accuracy,
    calculate_metrics(y_test, ridge_pred, ridge_pred_class)$accuracy,
    calculate_metrics(y_test, lasso_pred, lasso_pred_class)$accuracy,
    calculate_metrics(y_test, rf_pred, rf_pred_class)$accuracy,
    calculate_metrics(y_test, stepwise_pred, stepwise_pred_class)$accuracy
  ),
  Precision = c(
    calculate_metrics(y_test, logistic_pred, logistic_pred_class)$precision,
    calculate_metrics(y_test, ridge_pred, ridge_pred_class)$precision,
    calculate_metrics(y_test, lasso_pred, lasso_pred_class)$precision,
    calculate_metrics(y_test, rf_pred, rf_pred_class)$precision,
    calculate_metrics(y_test, stepwise_pred, stepwise_pred_class)$precision
  ),
  Recall = c(
    calculate_metrics(y_test, logistic_pred, logistic_pred_class)$recall,
    calculate_metrics(y_test, ridge_pred, ridge_pred_class)$recall,
    calculate_metrics(y_test, lasso_pred, lasso_pred_class)$recall,
    calculate_metrics(y_test, rf_pred, rf_pred_class)$recall,
    calculate_metrics(y_test, stepwise_pred, stepwise_pred_class)$recall
  )
)

print(models_performance)

# ===== FEATURE IMPORTANCE ANALYSIS =====

# Logistic Regression Coefficients
logistic_coefs <- summary(logistic_model)$coefficients
significant_vars <- logistic_coefs[logistic_coefs[,4] < 0.05, ]
print("Significant Variables in Logistic Regression:")
print(significant_vars)

# Random Forest Feature Importance
rf_importance <- importance(rf_model)
print("Random Forest Feature Importance:")
print(rf_importance[order(rf_importance, decreasing = TRUE), ])

# Lasso Coefficients
lasso_coefs <- coef(lasso_model, s = "lambda.min")
print("Lasso Regression Coefficients:")
print(lasso_coefs[lasso_coefs[,1] != 0, ])

# ===== BUSINESS IMPACT ANALYSIS =====

# Calculate lift and statistical significance
variant_a_conv <- mean(ab_data$converted[ab_data$variant == "A"])
variant_b_conv <- mean(ab_data$converted[ab_data$variant == "B"])
lift <- (variant_b_conv - variant_a_conv) / variant_a_conv * 100

# Revenue impact
variant_a_revenue <- mean(ab_data$revenue[ab_data$variant == "A"])
variant_b_revenue <- mean(ab_data$revenue[ab_data$variant == "B"])
revenue_lift <- (variant_b_revenue - variant_a_revenue) / variant_a_revenue * 100

# Statistical power analysis
n_a <- sum(ab_data$variant == "A")
n_b <- sum(ab_data$variant == "B")
pooled_conv <- (sum(ab_data$converted[ab_data$variant == "A"]) + 
                  sum(ab_data$converted[ab_data$variant == "B"])) / (n_a + n_b)
se <- sqrt(pooled_conv * (1 - pooled_conv) * (1/n_a + 1/n_b))
z_score <- (variant_b_conv - variant_a_conv) / se

business_results <- data.frame(
  Metric = c("Conversion Rate A", "Conversion Rate B", "Lift %", 
             "Revenue per User A", "Revenue per User B", "Revenue Lift %",
             "Z-Score", "P-Value", "Statistical Significance"),
  Value = c(
    round(variant_a_conv, 4),
    round(variant_b_conv, 4),
    round(lift, 2),
    round(variant_a_revenue, 2),
    round(variant_b_revenue, 2),
    round(revenue_lift, 2),
    round(z_score, 3),
    round(2 * (1 - pnorm(abs(z_score))), 4),
    ifelse(abs(z_score) > 1.96, "Yes", "No")
  )
)

print("Business Impact Analysis:")
print(business_results)

# ===== SEGMENTATION ANALYSIS =====

# Conversion by segments
segment_analysis <- aggregate(cbind(users = user_id, conversions = converted, revenue = revenue) ~ 
                                variant + device_type + traffic_source, 
                              data = ab_data, 
                              FUN = function(x) c(length(x), sum(x), mean(x)))

segment_analysis$users <- segment_analysis$users[,1]
segment_analysis$conversions <- segment_analysis$conversions[,2]
segment_analysis$conversion_rate <- segment_analysis$conversions / segment_analysis$users
segment_analysis$avg_revenue <- segment_analysis$revenue[,3]
segment_analysis <- segment_analysis[, !names(segment_analysis) %in% "revenue"]
segment_analysis <- segment_analysis[order(segment_analysis$conversion_rate, decreasing = TRUE), ]

print("Segment Analysis:")
print(segment_analysis)

# ===== CONFIDENCE INTERVALS =====

# Bootstrap confidence intervals for conversion rates
bootstrap_ci <- function(data, variant_name, n_bootstrap = 1000) {
  variant_data <- data[data$variant == variant_name, ]
  bootstrap_rates <- replicate(n_bootstrap, {
    sample_data <- sample(variant_data$converted, replace = TRUE)
    mean(sample_data)
  })
  
  ci_lower <- quantile(bootstrap_rates, 0.025)
  ci_upper <- quantile(bootstrap_rates, 0.975)
  
  return(c(ci_lower, ci_upper))
}

ci_a <- bootstrap_ci(ab_data, "A")
ci_b <- bootstrap_ci(ab_data, "B")

print(paste("Variant A 95% CI:", round(ci_a[1], 4), "-", round(ci_a[2], 4)))
print(paste("Variant B 95% CI:", round(ci_b[1], 4), "-", round(ci_b[2], 4)))

# ===== VISUALIZATION =====

# Conversion rate comparison
ggplot(conversion_by_variant, aes(x = variant, y = conversion_rate, fill = variant)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(conversion_rate * 100, 2), "%")), 
            vjust = -0.5) +
  labs(title = "Conversion Rate by Variant",
       x = "Variant", y = "Conversion Rate") +
  theme_minimal()

# ROC curves comparison
roc_logistic <- roc(y_test, logistic_pred)
roc_rf <- roc(y_test, rf_pred)
roc_lasso <- roc(y_test, lasso_pred)

plot(roc_logistic, col = "blue", main = "ROC Curves Comparison")
lines(roc_rf, col = "red")
lines(roc_lasso, col = "green")
legend("bottomright", legend = c("Logistic", "Random Forest", "Lasso"),
       col = c("blue", "red", "green"), lwd = 2)

# Feature importance plot
varImpPlot(rf_model, main = "Random Forest Feature Importance")

# ===== FINAL RECOMMENDATIONS =====

cat("\n=== FINAL RECOMMENDATIONS ===\n")
cat("1. Variant B shows", round(lift, 2), "% lift in conversion rate\n")
cat("2. Statistical significance:", ifelse(abs(z_score) > 1.96, "YES", "NO"), "\n")
cat("3. Revenue impact:", round(revenue_lift, 2), "% increase per user\n")
cat("4. Best performing model:", models_performance$Model[which.max(models_performance$AUC)], "\n")
cat("5. Key drivers: session_duration, pages_viewed, previous_purchases\n")

# Save results
write.csv(models_performance, "model_performance.csv", row.names = FALSE)
write.csv(business_results, "business_impact.csv", row.names = FALSE)
write.csv(segment_analysis, "segment_analysis.csv", row.names = FALSE)

cat("\nAnalysis complete. Results saved to CSV files.\n")