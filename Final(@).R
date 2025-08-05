# Load necessary libraries
library(AER)  # For the CreditCard dataset
library(ggplot2)  # For data visualization
library(caTools)  # For data splitting
library(rpart)  # For decision trees
library(rpart.plot)  # For decision tree visualization
library(randomForest)  # For random forests
library(caret)  # For model tuning and evaluation
library(pROC)

# Load the CreditCard dataset
data("CreditCard")
dim(CreditCard)
summary(CreditCard)

# Data cleaning: remove rows with missing values
creditcard_clean <- na.omit(CreditCard)

# Data exploration
# Distribution of target variable (card status)
ggplot(creditcard_clean, aes(x = card)) +
  geom_bar(fill = "blue", alpha = 0.7) +
  labs(title = "Distribution of Credit Card Status",
       x = "Card Status (0 = No Default, 1 = Default)",
       y = "Count") +
  theme_minimal()

# Distribution of age
ggplot(creditcard_clean, aes(x = age)) +
  geom_histogram(binwidth = 2, fill = "orange", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Age",
       x = "Age",
       y = "Count") +
  theme_minimal()

# Distribution of income
ggplot(creditcard_clean, aes(x = income)) +
  geom_histogram(binwidth = 1, fill = "green", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Income",
       x = "Income",
       y = "Count") +
  theme_minimal()

# Distribution of reports
ggplot(creditcard_clean, aes(x = reports)) +
  geom_histogram(binwidth = 1, fill = "purple", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Reports",
       x = "Number of Reports",
       y = "Count") +
  theme_minimal()

# Split data into training and testing sets (70% training, 30% testing)
set.seed(123)  # For reproducibility
split <- sample.split(creditcard_clean$card, SplitRatio = 0.7)
training <- subset(creditcard_clean, split == TRUE)
testing <- subset(creditcard_clean, split == FALSE)

# Define control parameters for model tuning
ctrl <- trainControl(method = "repeatedcv", number = 10)  # 10-fold repeated cross-validation

# Define a grid of hyperparameters to search for decision tree
param_grid_tree <- expand.grid(.cp = c(0.001, 0.01, 0.1))

# Perform grid search to find the best hyperparameters for decision tree
best_tree_params <- train(card ~ age + income + reports + dependents + months + owner + selfemp,
                          data = training, method = "rpart",
                          trControl = ctrl, tuneGrid = param_grid_tree,
                          metric = "Accuracy")


# Extract the best hyperparameters
best_cp <- best_tree_params$bestTune$cp
best_maxdepth <- best_tree_params$bestTune$maxdepth

# Train a decision tree model with the best hyperparameters
model_tree <- rpart(card ~ age + income + reports + dependents + months + owner + selfemp,
                    data = training, method = "class",
                    control = rpart.control(cp = best_cp))

# Visualize the decision tree
rpart.plot(model_tree, box.palette = "Blues", shadow.col = "gray", main = "Decision Tree")

# Predict on testing data using the decision tree model
predictions_tree <- predict(model_tree, testing, type = "class")

# Calculate and print confusion matrix for decision tree model on testing data
conf_matrix_tree <- table(predictions_tree, testing$card)
print("Decision Tree Model Confusion Matrix (Testing Data):")
print(conf_matrix_tree)

# Calculate evaluation metrics for decision tree model
accuracy_tree <- sum(diag(conf_matrix_tree)) / sum(conf_matrix_tree)
precision_tree <- sum(conf_matrix_tree[1, 1]) / sum(conf_matrix_tree[, 1])
recall_tree <- sum(conf_matrix_tree[1, 1]) / sum(conf_matrix_tree[1, ])
f1_tree <- 2 * ((precision_tree * recall_tree) / (precision_tree + recall_tree))
print("Decision Tree Model Evaluation Metrics:")
print(paste("Accuracy:", round(accuracy_tree * 100, 2), "%"))
print(paste("Precision:", round(precision_tree * 100, 2), "%"))
print(paste("Recall:", round(recall_tree * 100, 2), "%"))
print(paste("F1 Score:", round(f1_tree * 100, 2), "%"))

# Define a grid of hyperparameters to search for random forest
param_grid_rf <- expand.grid(mtry = c(1, 3, 5, 7))

# Perform grid search to find the best hyperparameters for random forest
best_rf_params <- train(card ~ age + income + reports + dependents + months + owner + selfemp,
                        data = training, method = "rf",
                        trControl = ctrl, tuneGrid = param_grid_rf,
                        metric = "Accuracy")

# Extract the best hyperparameters
best_mtry <- best_rf_params$bestTune$mtry

# Train a random forest model with the best hyperparameters
model_rf <- randomForest(card ~ age + income + reports + dependents + months + owner + selfemp,
                         data = training, ntree = 500, mtry = best_mtry)

# Predict on testing data using the random forest model
predictions_rf <- predict(model_rf, testing)

# Calculate and print confusion matrix for random forest model on testing data
conf_matrix_rf <- table(predictions_rf, testing$card)
print("Random Forest Model Confusion Matrix (Testing Data):")
print(conf_matrix_rf)

# Calculate evaluation metrics for random forest model
accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
precision_rf <- sum(conf_matrix_rf[1, 1]) / sum(conf_matrix_rf[, 1])
recall_rf <- sum(conf_matrix_rf[1, 1]) / sum(conf_matrix_rf[1, ])
f1_rf <- 2 * ((precision_rf * recall_rf) / (precision_rf + recall_rf))
print("Random Forest Model Evaluation Metrics:")
print(paste("Accuracy:", round(accuracy_rf * 100, 2), "%"))
print(paste("Precision:", round(precision_rf * 100, 2), "%"))
print(paste("Recall:", round(recall_rf * 100, 2), "%"))
print(paste("F1 Score:", round(f1_rf * 100, 2), "%"))


# Optional: Plot feature importance for random forest model
varImpPlot(model_rf, main = "Feature Importance for Random Forest")

print("Analysis completed.")

