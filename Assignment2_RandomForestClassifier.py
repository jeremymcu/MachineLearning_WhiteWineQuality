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
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

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

########################################################################
# Part 3: Train a random forest classifier
########################################################################
print('Part 3: Train a random forest classifier')
# 3.1. Split the dataset into training set and testing set
print('3.1. Split the dataset into training set and testing set')
X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=2)
print("-" * 100)

# 3.2. Train a random forest classifier
print('3.2. Train a random forest classifier')
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# 3.3. evaluate the model
print('3.3. evaluate the model')
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("-" * 100)

########################################################################
# Part 4: Save the model
########################################################################
print('Part 4: Save the model')
# 4.1. Save the model
print('4.1. Save the model')
joblib.dump(rf_model, 'rf_model.pkl')
print("-" * 100)

# 4.2. Load the model
print('4.2. Load the model')
rf_model = joblib.load('rf_model.pkl')
print("-" * 100)

########################################################################
# Part 7: Build the web application using Streamlit
########################################################################
print('Part 7: Build the web application using Streamlit')
# 7.1. Build the web application using Streamlit
print('7.1. Build the web application using Streamlit')
st.title('Wine Quality Prediction')
st.write('This is a web application for predicting the quality of white wine.')
st.write('Please enter the following data and click the "Predict" button to predict the quality of the white wine.')
fixed_acidity = st.text_input("fixed acidity", "7.5")
volatile_acidity = st.text_input("volatile acidity", "0.27")
citric_acid = st.text_input("citric acid", "0.36")
residual_sugar = st.text_input("residual sugar", "20.7")
chlorides = st.text_input("chlorides", "0.045")
free_sulfur_dioxide = st.text_input("free sulfur dioxide", "45")
total_sulfur_dioxide = st.text_input("total sulfur dioxide", "170")
density = st.text_input("density", "1.001")
pH = st.text_input("pH", "3")
sulphates = st.text_input("sulphates", "0.45")
alcohol = st.text_input("alcohol", "8.8")
new_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
if st.button('Predict'):
    st.write("Predicted wine quality:", rf_model.predict(new_data))
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

# 7.2. Run the web application
print('7.2. Run the web application')
# streamlit run Assignment2_KNN_NaiveBayes_SVM.py
print("-" * 100)

