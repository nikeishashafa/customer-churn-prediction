# Customer Churn Prediction

A machine learning project to analyze customer churn patterns and predict customers who are at risk of churning.

## Project Overview

Customer churn is an important business problem because losing customers can directly affect company revenue and long-term growth.

This project analyzes customer behavior and builds machine learning models to identify factors associated with customer churn and predict whether a customer is likely to churn.

The project covers the complete workflow from data exploration and analysis to machine learning modeling and customer risk profiling.

## Dataset

The dataset contains **440,833 customer records** with customer demographic, behavioral, subscription, and churn information.

### Features

- Age
- Gender
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Subscription Type
- Contract Length
- Total Spend
- Last Interaction

### Target

- `Churn` - indicates whether a customer churned:
  - `0` = No Churn
  - `1` = Churn

## Exploratory Data Analysis

The analysis investigates several patterns related to customer churn, including:

- Churn distribution
- Churn rate by contract length
- Churn rate by support calls
- Churn rate by payment delay
- Churn rate by usage frequency
- Total spending differences between churned and non-churned customers
- Last interaction patterns
- Subscription type and contract length
- Correlations between numerical features and churn

One important finding is that **monthly contract customers have substantially higher churn rates** compared with annual and quarterly contract customers.

The analysis also shows that churn increases considerably as the number of **support calls** increases.

## Hidden Churn Analysis

A rule-based analysis was also performed using several high-risk conditions:

- Monthly contract
- Support Calls >= 6
- Payment Delay >= 21 days

The analysis identified **53,400 churned customers** who were not captured by these main risk rules, representing approximately **21.36% of all churned customers**.

These hidden churners showed several notable characteristics, including:

- Higher average age
- More support calls
- Lower total spending
- Lower usage frequency
- Longer time since the last interaction

This suggests that churn cannot be explained by a small number of simple rules alone.

## Machine Learning

Two classification models were developed:

1. Logistic Regression
2. Random Forest

The dataset was split into:

- **80% training data**
- **20% testing data**

Stratified sampling was used to maintain a similar churn distribution between the training and testing sets.

### Preprocessing

Numerical features were standardized using `StandardScaler`.

Categorical features were transformed using `OneHotEncoder`.

The preprocessing steps were combined with the machine learning models using Scikit-learn pipelines.

## Logistic Regression Performance

The baseline Logistic Regression model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 89.34% |
| Precision | 92.34% |
| Recall | 88.55% |
| F1-Score | 90.40% |
| ROC-AUC | 95.90% |

These results indicate that the Logistic Regression model can effectively distinguish between churned and non-churned customers.

## Random Forest

A Random Forest classifier was also developed to capture potentially non-linear relationships between customer characteristics and churn.

The Random Forest pipeline combines:

- Numerical feature scaling using `StandardScaler`
- Categorical feature encoding using `OneHotEncoder`
- Random Forest classification

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- 5-Fold Cross-Validation

Feature importance was also analyzed to identify factors that contribute most strongly to churn prediction.

The trained Random Forest pipeline is saved in:

`models/customer_churn_random_forest.pkl`

## Customer Risk Profiling

The final Random Forest model generates a churn probability for each customer.

Customers are categorized into three risk levels:

- **Low** - churn probability below 30%
- **Medium** - churn probability between 30% and 70%
- **High** - churn probability above 70%

This risk segmentation can help businesses prioritize retention efforts.

## Business Recommendations

Based on the analysis, several retention strategies can be considered:

- Provide proactive support for customers with frequent support calls.
- Offer payment reminders and assistance to customers experiencing payment delays.
- Provide personalized retention incentives for customers with lower spending.
- Increase engagement for customers with low usage frequency.
- Re-engage customers who have not interacted with the service recently.
- Encourage customers on monthly contracts to consider longer-term plans.

## Project Structure

```text
customer-churn-prediction/
|
|-- data/
|   `-- customer_churn_dataset-training-master.csv
|
|-- models/
|   `-- customer_churn_random_forest.pkl
|
|-- notebooks/
|   `-- customer_churn_analysis.ipynb
|
|-- outputs/
|   `-- customer_churn_predictions.csv
|
|-- .gitignore
|-- README.md
`-- requirements.txt

Technologies
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Jupyter Notebook

How to Run

Clone the repository and install the required dependencies:

pip install -r requirements.txt

Then open the notebook:

jupyter notebook

Open:

notebooks/customer_churn_analysis.ipynb

Run the notebook cells sequentially.

Key Takeaway

Customer churn is influenced by multiple behavioral and subscription-related factors.

The analysis highlights support call frequency, contract length, total spending, usage frequency, and customer interaction patterns as important areas for customer retention.

The machine learning models provide a foundation for identifying high-risk customers and supporting data-driven retention strategies.