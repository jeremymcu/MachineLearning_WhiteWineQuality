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

#
# ########################################################################
# # Part 3: Exploratory Data Analysis (EDA) of the data
# ########################################################################
# print('Part 3: Exploratory Data Analysis (EDA) of the data')
# # 3.1. Bar charts
# print('3.1. Bar charts')
# plt.bar(df['pH'], df['quality'])
# plt.xlabel('pH')
# plt.ylabel('quality')
# plt.title('pH to Quality of Wine')
# plt.tight_layout()
# plt.savefig('pH_to_Quality_of_Wine.png')
# plt.show()
# print("-" * 100)
#
# # 3.2. Histograms
# print('3.2. Histograms')
# sns.displot(df['quality'])
# plt.title("Distribution of wine quality")
# plt.tight_layout()
# plt.savefig('Distribution_of_wine_quality.png')
# plt.show()
# print("-" * 100)
#
# # 3.3. Box plots
# print('3.3. Box plots')
# sns.boxplot(data=df.drop(columns=['quality']))
# plt.xlabel('Features')
# plt.ylabel('Values')
# plt.title('IQR (interquartile range) of features and class attributes')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('IQR_of_features_and_class_attributes.png')
# plt.show()
# print("-" * 100)
#
# # 3.4. Scatter plots
# print('3.4. Scatter plots')
# # 3.4.1. Scatter plot of alcohol vs. quality
# print('3.4.1. Scatter plot of alcohol vs. quality')
# plt.scatter(df['alcohol'], df['quality'])
# plt.xlabel('alcohol')
# plt.ylabel('quality')
# plt.title('alcohol to Quality of Wine')
# plt.tight_layout()
# plt.savefig('alcohol_to_Quality_of_Wine.png')
# plt.show()
# print("-" * 100)
#
# # 3.4.2. Scatter plot of density vs. quality
# print('3.4.2. Scatter plot of density vs. quality')
# plt.scatter(df['density'], df['quality'])
# plt.xlabel('density')
# plt.ylabel('quality')
# plt.title('density to Quality of Wine')
# plt.tight_layout()
# plt.savefig('density_to_Quality_of_Wine.png')
# plt.show()
# print("-" * 100)
#
# # 3.4.3. Scatter plot of pH vs. quality
# print('3.4.3. Scatter plot of pH vs. quality')
# plt.scatter(df['pH'], df['quality'])
# plt.xlabel('pH')
# plt.ylabel('quality')
# plt.title('pH to Quality of Wine')
# plt.tight_layout()
# plt.savefig('pH_to_Quality_of_Wine.png')
# plt.show()
# print("-" * 100)
#
# # 3.4.4. Scatter plot of fixed acidity vs. quality
# print('3.4.4. Scatter plot of fixed acidity vs. quality')
# plt.scatter(df['fixed acidity'], df['quality'])
# plt.xlabel('fixed acidity')
# plt.ylabel('quality')
# plt.title('fixed acidity to Quality of Wine')
# plt.tight_layout()
# plt.savefig('fixed_acidity_to_Quality_of_Wine.png')
# plt.show()
# print("-" * 100)
#
# # 3.5. Correlation matrix
# print('3.5. Correlation matrix')
# corr = df.corr()
# print(corr)
# print("-" * 100)
#
# # 3.6. Heatmap
# print('3.6. Heatmap')
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# plt.tight_layout()
# plt.savefig('Heatmap.png')
# plt.show()
# print("-" * 100)


########################################################################
# Part 4: Build and train the machine learning models
########################################################################
print('Part 4: Build and train the machine learning models')
# 4.1. Split the data into training and testing sets
print('4.1. Split the data into training and testing sets')
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['quality']), df['quality'], test_size=0.2, random_state=8)
print("x_train shape:", X_train.shape) # (3918, 11)
print("y_train shape:", y_train.shape) # (3918,)
print("x_test shape:", X_test.shape) # (980, 11)
print("y_test shape:", y_test.shape) # (980,)
print("-" * 100)

# 4.2. Build and train KNN, Naive Bayes, SVM and Decision Tree models
print('4.2. Build and train KNN, Naive Bayes, SVM and Decision Tree models')
# # 4.2.1. KNN
# print('4.2.1. KNN')
# knn = KNeighborsClassifier(n_neighbors=5) # Accuracy score: 0.49, F1 score: 0.48
# knn.fit(X_train, y_train)
# print("-" * 100)
#
# # 4.2.2. Naive Bayes
# print('4.2.2. Naive Bayes')
# nb = GaussianNB() # Accuracy score: 0.44, F1 score: 0.43
# nb.fit(X_train, y_train)
# print("-" * 100)
#
# # 4.2.3. SVM
# print('4.2.3. SVM')
# svm = svm.SVC(kernel='linear') # Accuracy score: 0.54, F1 score: 0.46
# svm.fit(X_train, y_train)
# print("-" * 100)

# 4.2.4. Decision Tree
print('4.2.4. Decision Tree')
dt = tree.DecisionTreeClassifier() # Accuracy score: 0.59, F1 score: 0.60
dt.fit(X_train, y_train)
print("-" * 100)

#
# ########################################################################
# # Part 5: Evaluate the models
# ########################################################################
# print('Part 5: Evaluate the models')
# # 5.1. Evaluate the models using the test data
# print('5.1. Evaluate the models using the test data')
# # 5.1.1. KNN
# print('5.1.1. KNN')
# knn_pred = knn.predict(X_test)
# print(classification_report(y_test, knn_pred))
# print("MCC:", matthews_corrcoef(y_test, knn_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, knn_pred))
# print("-" * 100)
# """
# 5.1.1. KNN
#               precision    recall  f1-score   support
#
#            3       0.00      0.00      0.00         3
#            4       0.12      0.10      0.11        31
#            5       0.46      0.48      0.47       295
#            6       0.54      0.61      0.57       445
#            7       0.42      0.33      0.37       168
#            8       0.50      0.14      0.22        36
#            9       0.00      0.00      0.00         2
#
#     accuracy                           0.49       980
#    macro avg       0.29      0.24      0.25       980
# weighted avg       0.48      0.49      0.48       980
#
# MCC: 0.20764512657420628
# Confusion Matrix:
# [[  0   0   0   2   1   0   0]
#  [  1   3  13  12   2   0   0]
#  [  0  14 143 122  15   1   0]
#  [  0   7 114 270  51   3   0]
#  [  0   2  31  79  55   1   0]
#  [  0   0   8  15   8   5   0]
#  [  0   0   0   2   0   0   0]]
#
# """
#
# # 5.1.2. Naive Bayes
# print('5.1.2. Naive Bayes')
# nb_pred = nb.predict(X_test)
# print(classification_report(y_test, nb_pred))
# print("MCC:", matthews_corrcoef(y_test, nb_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, nb_pred))
# print("-" * 100)
# """
# 5.1.2. Naive Bayes
#               precision    recall  f1-score   support
#
#            3       0.00      0.00      0.00         3
#            4       0.40      0.32      0.36        31
#            5       0.50      0.54      0.52       295
#            6       0.51      0.31      0.39       445
#            7       0.35      0.69      0.46       168
#            8       0.25      0.08      0.12        36
#            9       0.00      0.00      0.00         2
#
#     accuracy                           0.44       980
#    macro avg       0.29      0.28      0.26       980
# weighted avg       0.46      0.44      0.43       980
#
# MCC: 0.22093402352689492
# Confusion Matrix:
# [[  0   0   1   1   1   0   0]
#  [  1  10   9   7   3   1   0]
#  [  7  10 159  86  33   0   0]
#  [  3   5 132 140 161   3   1]
#  [  0   0  15  31 116   5   1]
#  [  0   0   4   9  20   3   0]
#  [  0   0   0   1   1   0   0]]
# """
#
# # 5.1.3. SVM
# print('5.1.3. SVM')
# svm_pred = svm.predict(X_test)
# print(classification_report(y_test, svm_pred))
# print("MCC:", matthews_corrcoef(y_test, svm_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, svm_pred))
# print("-" * 100)
# """
# 5.1.3. SVM
#               precision    recall  f1-score   support
#
#            3       0.00      0.00      0.00         3
#            4       0.00      0.00      0.00        31
#            5       0.59      0.54      0.56       295
#            6       0.52      0.82      0.64       445
#            7       0.00      0.00      0.00       168
#            8       0.00      0.00      0.00        36
#            9       0.00      0.00      0.00         2
#
#     accuracy                           0.54       980
#    macro avg       0.16      0.20      0.17       980
# weighted avg       0.41      0.54      0.46       980
#
# MCC: 0.2431150325549826
# Confusion Matrix:
# [[  0   0   0   3   0   0   0]
#  [  0   0  20  11   0   0   0]
#  [  0   0 160 135   0   0   0]
#  [  0   0  78 367   0   0   0]
#  [  0   0  10 158   0   0   0]
#  [  0   0   4  32   0   0   0]
#  [  0   0   0   2   0   0   0]]
# """
#
# # 5.1.4. Decision Tree
# print('5.1.4. Decision Tree')
# dt_pred = dt.predict(X_test)
# print(classification_report(y_test, dt_pred))
# print("MCC:", matthews_corrcoef(y_test, dt_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, dt_pred))
# print("-" * 100)
# """
# 5.1.4. Decision Tree
#               precision    recall  f1-score   support
#
#            3       0.00      0.00      0.00         3
#            4       0.29      0.32      0.31        31
#            5       0.66      0.61      0.64       295
#            6       0.64      0.65      0.64       445
#            7       0.49      0.53      0.51       168
#            8       0.41      0.42      0.41        36
#            9       0.00      0.00      0.00         2
#
#     accuracy                           0.59       980
#    macro avg       0.36      0.36      0.36       980
# weighted avg       0.60      0.59      0.60       980
#
# MCC: 0.3974811013007629
# Confusion Matrix:
# [[  0   0   0   3   0   0   0]
#  [  1  10   7  12   0   1   0]
#  [  0  13 181  91   9   1   0]
#  [  0   9  72 288  69   6   1]
#  [  0   2  11  52  89  14   0]
#  [  0   0   2   6  13  15   0]
#  [  0   0   0   1   1   0   0]]
# """


# 5.2. Save the best model
print('5.2. Save the best model')
joblib.dump(dt, 'decisionTree_model.pkl')

########################################################################
# Part 6: Predict the quality of wine
########################################################################
print('Part 6: Predict the quality of wine')
# 6.1. Load the best model
print('6.1. Load the best model')
model = joblib.load('decisionTree_model.pkl')
print("-" * 100)

# 6.2. Predict the quality of wine
print('6.2. Predict the quality of wine')
# 6.2.1. Predict the quality of wine with a new data
print('6.2.1. Predict the quality of wine with a new data')
new_data = [[7.5, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8]]
print("Predicted wine quality:", model.predict(new_data))
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
    st.write("Predicted wine quality:", model.predict(new_data))
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

