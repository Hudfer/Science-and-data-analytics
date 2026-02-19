# Science-and-data-analytics

### -EDA_ACTIVITY – Exploratory Data Analysis (EDA) on Credit Risk Dataset
Course: Data Science and Analytics.
Overview:
This notebook performs a full Exploratory Data Analysis (EDA) on a credit risk dataset containing 32,581 records and 12 features related to loan applicants. The goal is to understand the structure of the data, detect anomalies, and uncover patterns that could help predict whether a borrower will default on a loan (loan_status).

Key steps covered:
- Data loading and inspection – Loading the credit_risk_dataset.csv file and reviewing its structure, data types, and basic statistics.
- Missing value analysis – Identifying and handling null values across the dataset.
- Outlier detection – Analyzing numerical variables such as person_age and person_emp_length to flag unrealistic or extreme values.
- Univariate analysis – Visualizing the distribution of individual variables using histograms and bar plots (e.g., loan intent, home ownership, loan grade).
- Bivariate analysis – Exploring relationships between features and the target variable (loan_status) using grouped plots and cross-tabulations.
- Correlation analysis – Computing and visualizing a correlation heatmap, finding that loan_int_rate (loan interest rate) has the strongest correlation with the likelihood of default.

Tools used: Python, Pandas, NumPy, Matplotlib, Seaborn.

Main finding: The variable most correlated with loan default is the loan interest rate, suggesting that higher-risk borrowers are assigned higher rates and are also more likely to default.

### -DATA ANALYSIS_UBER  – Data Analysis, Visualization & Transformation with Uber Dataset
Course: Data Science and Analytics
Overview:
This notebook focuses on data analysis, visualization, and transformation using a real-world Uber pickups dataset from New York City. The dataset contains 29,101 records covering the first half of 2015 (January–June), combining ride pickup information across NYC boroughs with hourly weather conditions.
Dataset features include: pickup datetime, borough (Bronx, Brooklyn, Manhattan, Queens, Staten Island, EWR), number of pickups per hour, wind speed, visibility, temperature, dew point, sea-level pressure, precipitation (1h, 6h, 24h), snow depth, and whether the day was a holiday.

Key topics covered:
- Data loading and inspection – Loading uber.csv into a Pandas DataFrame and reviewing its structure and basic statistics.
- Data cleaning and transformation – Handling missing values (e.g., null borough entries), parsing datetime columns, and preparing data for analysis.
- Time-based feature engineering – Extracting temporal features such as hour, day of week, and month from the pickup_dt column to analyze ride demand patterns over time.
- Data visualization – Creating charts and plots to explore pickup trends by borough, time of day, and weather conditions using Matplotlib and Seaborn.
- Correlation analysis – Building a heatmap to examine relationships between weather variables and pickup counts, with key findings such as: dew point correlates strongly with temperature, visibility correlates negatively with precipitation, snow depth correlates negatively with temperature, and no strong relationship was found between weather conditions and the number of pickups.

Tools used: Python, Pandas, NumPy, Matplotlib, Seaborn.

Main takeaway: Weather conditions alone do not appear to be strong predictors of Uber demand in NYC. The analysis also reinforces the important statistical principle that correlation does not imply causation.

### -CAR_ACTIVITY – Feature Engineering on a Used Car Listings Dataset (CAR_ACTIVITY)
Course: Data Science and Analytics
Overview:
This notebook applies Feature Engineering (FE) techniques to a large used car listings dataset from Craigslist, containing over 426,000 raw records and 26 features (price, year, manufacturer, model, condition, mileage, fuel type, transmission, state, and more). The goal is to clean, transform, and encode the data to produce a machine-learning-ready dataset.

Key steps covered:
- Data loading and inspection – Loading the raw dataset and reviewing its structure, data types, and missing value distribution.
- Data cleaning – Dropping irrelevant columns (e.g., URLs, image links, description text), filtering rows with excessive missing values, and handling null entries to produce a cleaner working dataset.
- Outlier removal – Filtering extreme values in price and odometer to reduce noise and improve data quality, reducing the dataset to ~366,000 records.
- Feature creation – Engineering a new age variable calculated from the vehicle's year, capturing how old each car is at the time of listing.
- Numerical scaling – Applying Min-Max normalization to price, age, and odometer, storing the result in minmax_df so all values fall in the [0, 1] range.
- One-Hot Encoding – Applied to low/medium cardinality categorical variables (e.g., manufacturer, state, condition, fuel, type), stored in onehot_df.
- Binary Encoding – Applied to high-cardinality variables (model) using binary representation to reduce dimensionality, stored in binary_df.
- Final dataset assembly – Concatenating minmax_df, onehot_df, and binary_df into a single df_final with 148 columns, ready for downstream machine learning models.

Tools used: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.

Main takeaway: Proper feature engineering, including creating meaningful new variables, removing outliers, choosing the right encoding strategy per variable, and normalizing numerical features, is essential to prepare real-world messy data for effective machine learning.


### -LOGISTIC REGRESSION – Logistic Regression for Medical Diagnosis
Course: Data Science and Analytics 
Overview:
This notebook implements a Logistic Regression model applied to a breast cancer diagnosis dataset. The goal is to build a binary classification model capable of predicting whether a tumor is malignant (M) or benign (B) based on 30 numerical features derived from cell nucleus measurements.
Dataset: The Wisconsin Breast Cancer dataset, containing 569 patient records and features such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension — each measured as mean, standard error, and worst value.

Key steps covered:
- Data loading and inspection – Loading the dataset and reviewing its structure and class distribution between malignant and benign diagnoses.
- Preprocessing pipeline – Building a Scikit-learn pipeline that handles imputation of missing values, feature scaling (Min-Max and Standard Scaler), and column transformations using ColumnTransformer.
- Model training – Training a Logistic Regression classifier on the preprocessed dataset using Scikit-learn's LogisticRegression.
- Model evaluation – Assessing performance using a confusion matrix, accuracy, precision, and recall scores.
- Threshold analysis – Reflecting on which metric matters most in a medical diagnosis context. The team concluded that recall (sensitivity) is the most critical metric, since false negatives (predicting benign when actually malignant) are far more dangerous than false positives. They proposed lowering the classification threshold to reduce false negatives and catch more true positive cases.

Tools used: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.

Main takeaway: In medical diagnosis scenarios, minimizing false negatives is crucial. Adjusting the decision threshold of a logistic regression model is a practical strategy to prioritize recall and ensure high-risk cases are not missed.
