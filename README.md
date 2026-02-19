# Science-and-data-analytics

### -EDA_ACTIVITY – Exploratory Data Analysis (EDA) on Credit Risk Dataset
Course: Data Science and Analytics.
Overview:
This notebook performs a full Exploratory Data Analysis (EDA) on a credit risk dataset containing 32,581 records and 12 features related to loan applicants. The goal is to understand the structure of the data, detect anomalies, and uncover patterns that could help predict whether a borrower will default on a loan (loan_status).
Key steps covered:

Data loading and inspection – Loading the credit_risk_dataset.csv file and reviewing its structure, data types, and basic statistics.
Missing value analysis – Identifying and handling null values across the dataset.
Outlier detection – Analyzing numerical variables such as person_age and person_emp_length to flag unrealistic or extreme values.
Univariate analysis – Visualizing the distribution of individual variables using histograms and bar plots (e.g., loan intent, home ownership, loan grade).
Bivariate analysis – Exploring relationships between features and the target variable (loan_status) using grouped plots and cross-tabulations.
Correlation analysis – Computing and visualizing a correlation heatmap, finding that loan_int_rate (loan interest rate) has the strongest correlation with the likelihood of default.

Tools used: Python, Pandas, NumPy, Matplotlib, Seaborn.
Main finding: The variable most correlated with loan default is the loan interest rate, suggesting that higher-risk borrowers are assigned higher rates and are also more likely to default.

### -DATA ANALYSIS_UBER  – Data Analysis, Visualization & Transformation with Uber Dataset
Course: Data Science and Analytics
Overview:
This notebook focuses on data analysis, visualization, and transformation using a real-world Uber pickups dataset from New York City. The dataset contains 29,101 records covering the first half of 2015 (January–June), combining ride pickup information across NYC boroughs with hourly weather conditions.
Dataset features include: pickup datetime, borough (Bronx, Brooklyn, Manhattan, Queens, Staten Island, EWR), number of pickups per hour, wind speed, visibility, temperature, dew point, sea-level pressure, precipitation (1h, 6h, 24h), snow depth, and whether the day was a holiday.
Key topics covered:

Data loading and inspection – Loading uber.csv into a Pandas DataFrame and reviewing its structure and basic statistics.
Data cleaning and transformation – Handling missing values (e.g., null borough entries), parsing datetime columns, and preparing data for analysis.
Time-based feature engineering – Extracting temporal features such as hour, day of week, and month from the pickup_dt column to analyze ride demand patterns over time.
Data visualization – Creating charts and plots to explore pickup trends by borough, time of day, and weather conditions using Matplotlib and Seaborn.
Correlation analysis – Building a heatmap to examine relationships between weather variables and pickup counts, with key findings such as: dew point correlates strongly with temperature, visibility correlates negatively with precipitation, snow depth correlates negatively with temperature, and no strong relationship was found between weather conditions and the number of pickups.

Tools used: Python, Pandas, NumPy, Matplotlib, Seaborn.
Main takeaway: Weather conditions alone do not appear to be strong predictors of Uber demand in NYC. The analysis also reinforces the important statistical principle that correlation does not imply causation.
