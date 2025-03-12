
# Loan Approval Prediction using Machine Learning

## 1. Introduction

### Problem Statement
The company seeks to automate the loan approval process by leveraging machine learning models that predict loan eligibility based on customer data submitted through online applications. The objective is to accelerate decision-making and improve efficiency in loan approvals.

### Objectives of the Project
- Perform Exploratory Data Analysis (EDA) on customer data.
- Build various machine learning models to predict loan approval.
- Compare model performance and determine the best approach for loan approval prediction.

## 2. Data Description

### Dataset Overview
The dataset consists of 13 variables:
- **8 categorical variables** (e.g., Gender, Married, Education, Self_Employed, etc.).
- **4 continuous variables** (e.g., ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term).
- **1 unique identifier** (Loan_ID - removed for model training).

### Sample Data Structure
| Variable Name       | Description                                 | Sample Data |
|---------------------|---------------------------------------------|-------------|
| Loan_ID            | Unique Loan reference number                | LP001002    |
| Gender             | Applicant gender (Male/Female)              | Male        |
| Married            | Applicant marital status                    | Yes         |
| Dependents         | Number of dependents                        | 0,1,2,3+    |
| Education          | Applicant education level                   | Graduate    |
| Self_Employed      | Employment status (Yes/No)                  | No          |
| ApplicantIncome    | Applicant's monthly income                  | 5849        |
| CoapplicantIncome | Co-applicant's monthly income               | 1508        |
| LoanAmount        | Loan amount requested                        | 128         |
| Loan_Amount_Term  | Loan repayment period (in days)             | 360         |
| Credit_History    | Record of previous credit history (0/1)      | 1           |
| Property_Area     | Property location (Rural/Semiurban/Urban)    | Urban       |
| Loan_Status       | Loan Approval Status (Y/N)                   | Y           |

## 3. Exploratory Data Analysis (EDA)

### Key Insights from EDA
- The dataset contains missing values in Gender, Married, Dependents, Self_Employed, Credit_History, and LoanAmount.
- Majority of applicants are male (~79.64%).
- Most applicants are married (~64.82%).
- Graduates form the majority of applicants (~78.18%).
- Around 81.43% of applicants are not self-employed.
- Good credit history applicants have a higher probability of loan approval.
- Semiurban areas have the highest number of approved loans.
- ApplicantIncome, CoapplicantIncome, and LoanAmount distributions are positively skewed.

### Heatmap Analysis
A heatmap was generated to visualize feature correlations. A moderate positive correlation was observed between LoanAmount and ApplicantIncome.

### Handling Missing Values
- Categorical variables were imputed using mode.
- Numerical variables were imputed using mean.

### Data Preprocessing
- One-hot encoding was applied to categorical variables.
- Outliers were removed using the Interquartile Range (IQR) method.
- Skewed distributions were normalized using square root transformation.
- Oversampling was performed using the SMOTE technique to balance the dataset.
- Data was normalized using MinMaxScaler.
- Data was split into 80% training and 20% testing sets.

## 4. Machine Learning Models

### 4.1 Logistic Regression
- Achieved an accuracy of **82.70%**.

### 4.2 K-Nearest Neighbors (KNN)
- Achieved a maximum accuracy of **80.54%** with optimized k-values.

### 4.3 Support Vector Machine (SVM)
- Achieved an accuracy of **82.70%**.

### 4.4 Naive Bayes
- Categorical Naive Bayes accuracy: **82.70%**
- Gaussian Naive Bayes accuracy: **83.24%**

### 4.5 Decision Tree
- Achieved an accuracy of **82.70%** with optimized max_leaf_nodes.

### 4.6 Random Forest
- Achieved an accuracy of **82.70%**.

### 4.7 Gradient Boosting
- Achieved an accuracy of **80.54%** after hyperparameter tuning.

## 5. Model Comparison

| Model               | Accuracy (%) |
|---------------------|--------------|
| Gaussian Naive Bayes | 83.24       |
| Logistic Regression | 82.70       |
| Support Vector Machine | 82.70     |
| Categorical Naive Bayes | 82.70    |
| Decision Tree      | 82.70        |
| Random Forest      | 82.70        |
| K-Nearest Neighbors | 80.54       |
| Gradient Boosting  | 80.54        |

## 6. Conclusion
- The **Gaussian Naive Bayes classifier** achieved the highest accuracy (**83.24%**) and is the best model for loan approval prediction in this scenario.
- Feature engineering, handling missing values, and balancing the dataset using SMOTE improved model performance.
- Future improvements could involve hyperparameter tuning, feature selection, and alternative ML models such as deep learning.

## 7. Recommendations
- The trained Gaussian Naive Bayes model can be deployed as an API for real-time loan approval predictions.
- Further exploration with ensemble techniques or deep learning could improve performance.
- The impact of additional financial parameters should be evaluated to enhance predictive capability.

---

This project successfully demonstrates the use of machine learning models in automating the loan approval process, helping financial institutions make faster and data-driven decisions.

