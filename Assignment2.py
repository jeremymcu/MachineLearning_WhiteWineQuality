# 10890402
# Jeremy (蕭智強)

########################################################################
# Part 1: Import libraries
########################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv

pd.set_option('display.max_columns', 100)

########################################################################
# Part 2: Load data into a dataframe and do some basic data exploration
########################################################################
print('Part 2: Load data into a dataframe and do some basic data exploration')
# 2.1. Load data into a dataframe and print the first 5 rows
print('2.1. Load data into a dataframe and print the first 5 rows')
df = pd.read_csv('winequality-white.csv', sep=';')
print(df.head(5))
print("-" * 100)

# 2.2. Check the data types of the columns
print('2.2. Check the data types of the columns')
print(df.dtypes)
print("-" * 100)

# 2.3. Check the number of rows and columns in the dataframe
print('2.3. Check the number of rows and columns in the dataframe')
print(df.shape)
print("-" * 100)

# 2.4. Check the descriptive statistics of the dataframe
print('2.4. Check the descriptive statistics of the dataframe')
print(df.describe())
print("-" * 100)

# 2.5. Remove outliers and unnecessary column
print('2.5. Remove outliers')
df_cleaned = pd.DataFrame
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
df_cleaned = df_cleaned.drop(['total sulfur dioxide'], axis=1)
print(df_cleaned.shape)


########################################################################
# Part 3: Build and train the machine learning models
########################################################################
print('Part 3: Build and train the machine learning models')
# 3.1. Split the data into training and testing sets
print('3.1. Split the data into training and testing sets')
X = df_cleaned.drop('quality', axis=1)
y = df_cleaned['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
print("x_train shape:", X_train.shape) # (3918, 10)
print("y_train shape:", y_train.shape) # (3918,)
print("x_test shape:", X_test.shape) # (980, 10)
print("y_test shape:", y_test.shape) # (980,)
print("-" * 100)

# 3.2. Build, train, and evaluate machine learning models
print('3.2. Build, train and evaluate machine learning models')
# 4.2.1. Decision Tree
print('4.2.1. Decision Tree')
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy score:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree F1 score:", f1_score(y_test, y_pred_dt, average='weighted'))
print("-" * 100)

# 4.2.2. Random Forest
print('4.2.2. Random Forest')
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy score:", accuracy_score(y_test, y_pred_rf))
print("Random Forest F1 score:", f1_score(y_test, y_pred_rf, average='weighted'))
print("-" * 100)

# 4.2.3. AdaBoost
print('4.2.3. AdaBoost')
AdaBoost = AdaBoostClassifier()
AdaBoost.fit(X_train, y_train)
y_pred_AdaBoost = AdaBoost.predict(X_test)
print("AdaBoost Accuracy score:", accuracy_score(y_test, y_pred_AdaBoost))
print("AdaBoost F1 score:", f1_score(y_test, y_pred_AdaBoost, average='weighted'))
print("-" * 100)

# 4.2.4. Gradient Boosting
print('4.2.4. Gradient Boosting')
GradientBoosting = GradientBoostingClassifier()
GradientBoosting.fit(X_train, y_train)
y_pred_GradientBoosting = GradientBoosting.predict(X_test)
print("Gradient Boosting Accuracy score:", accuracy_score(y_test, y_pred_GradientBoosting))
print("Gradient Boosting F1 score:", f1_score(y_test, y_pred_GradientBoosting, average='weighted'))
print("-" * 100)

# 4.2.5. CatBoost Classifier
print('4.2.5. CatBoost Classifier')
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.9, random_state=10)
CatBoostModel = CatBoostClassifier(custom_loss=['Accuracy'], random_seed=77, logging_level='Silent')
CatBoostModel.fit(X_train, y_train, eval_set=(X_validation, y_validation))
y_pred_CatBoost = CatBoostModel.predict(X_validation)
print("CatBoost Accuracy score:", accuracy_score(y_validation, y_pred_CatBoost))
print("CatBoost F1 score:", f1_score(y_validation, y_pred_CatBoost, average='weighted'))
print("-" * 100)

########################################################################
# Part 5: Predict the quality of wine
########################################################################
print('Part 5: Predict the quality of wine')
new_data = [[7.5, 0.27, 0.36, 20.7, 0.045, 45, 1.001, 3, 0.45, 8.8]]
prediction_result = [
        ['Decision Tree', dt.predict(new_data)[0]],
        ['Random Forest', rf.predict(new_data)[0]],
        ['AdaBoost', AdaBoost.predict(new_data)[0]],
        ['Gradient Boosting', GradientBoosting.predict(new_data)[0]],
        ['CatBoost', CatBoostModel.predict(new_data)[0][0]]
    ]
print("Predicted wine quality:", prediction_result)
print("-" * 100)

########################################################################
# Part 6: Build the web application using Streamlit
########################################################################
print('Part 6: Build the web application using Streamlit')

st.title('Wine Quality Prediction')
st.write('This is a web application for predicting the quality of white wine.')
st.sidebar.write('Please enter the following data and click the "Predict" button to predict the quality of the white wine.')
fixed_acidity = st.sidebar.text_input("fixed acidity", "7.5")
volatile_acidity = st.sidebar.text_input("volatile acidity", "0.27")
citric_acid = st.sidebar.text_input("citric acid", "0.36")
residual_sugar = st.sidebar.text_input("residual sugar", "20.7")
chlorides = st.sidebar.text_input("chlorides", "0.045")
free_sulfur_dioxide = st.sidebar.text_input("free sulfur dioxide", "45")
density = st.sidebar.text_input("density", "1.001")
pH = st.sidebar.text_input("pH", "3")
sulphates = st.sidebar.text_input("sulphates", "0.45")
alcohol = st.sidebar.text_input("alcohol", "8.8")
new_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, density, pH, sulphates, alcohol]]
if st.button('Predict'):
    st.write('The data you entered:')
    st.write(pd.DataFrame(new_data, columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'], index=["Value"]))
    prediction_result = {
        "Decision Tree" : dt.predict(new_data)[0],
        "Random Forest" : rf.predict(new_data)[0],
        "AdaBoost" : AdaBoost.predict(new_data)[0],
        "Gradient Boosting" : GradientBoosting.predict(new_data)[0],
        "CatBoost" : CatBoostModel.predict(new_data)[0][0]
    }
    st.write("Predicted wine quality:", pd.DataFrame(data=prediction_result, index=["Quality"]))
    st.write('The quality of wine is between 0 and 10.')
    st.write('0: Very Bad')
    st.write('1: Very Bad')
    st.write('2: Bad')
    st.write('3: Bad')
    st.write('4: Medium')
    st.write('5: Medium')
    st.write('6: Good')
    st.write('7: Very Good')
    st.write('8: Very Good')
    st.write('9: Excellent')
    st.write('10: Excellent')
print("-" * 100)

