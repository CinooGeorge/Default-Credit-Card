# Default-Credit-Card
Credit Card Default Prediction in R

## 📌 Project Overview
This project predicts whether a credit card holder is likely to default on their payments using machine learning techniques.  
We applied **Decision Tree** and **Random Forest** models to the dataset and compared their performance.

## 👥 Authors
- Vaibhavi Panchal  
- Millicent N.O Boadu  
- Cinoo Bosco Thomas  

## 📊 Dataset
The dataset comes from the **AER** package in R and includes 1,319 observations with 12 variables such as age, income, reports, and more.

## 🛠️ Packages Used
- `AER`
- `ggplot2`
- `caTools`
- `rpart`
- `rpart.plot`
- `randomForest`
- `caret`

## 🔍 Exploratory Data Analysis
The analysis includes:
- Distribution of credit card status  
- Distribution of age  
- Distribution of income  
- Distribution of reports  

## 🤖 Models Used
1. **Decision Tree**  
   - Accuracy: 84.34%  
   - Precision: 35.96%  
   - Recall: 86.49%  
   - F1 Score: 50.79%  

2. **Random Forest**  
   - Accuracy: 82.83%  
   - Precision: 44.94%  
   - Recall: 67.8%  
   - F1 Score: 54.05%  

## ✅ Results
- Decision Tree: Higher recall → better at catching defaulters  
- Random Forest: Higher precision → fewer false positives  
- Model selection depends on whether minimizing false positives or false negatives is the priority.

## 📌 Conclusion
The choice of model depends on business goals. Decision Trees are easier to interpret, while Random Forests give better balance between precision and recall.

## 📝 References
1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.  
2. Bierman, L. (2001). *Random forests*. Machine learning, 45(1), 5-32.  
3. Kuhn, M. (2008). *Building predictive models in R using the caret package*. Journal of Statistical Software.

---

## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/CinooGeorge/Default-Credit-Card.git
