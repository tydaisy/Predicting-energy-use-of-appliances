import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error,median_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import PolynomialFeatures

#user interface
print("How many features would you like to select")
print("A.all features")
print("B.filtered features")
response1 = input("please input 'A' or 'B': ")

print("What percentage dataset you prefer to use as training set?")
print("A.60%")
print("B.70%")
response2 = input("please input 'A' or 'B': ")

print("What kind of regression do you prefer?")
print("A.Linear Regression")
print("B.Polynomial Regression ")
response3 = input("please input 'A' or 'B': ")

#Load a matrix from csv_file
dataset= pd.read_csv('energydata_complete.csv')
dataset = dataset.values

# Extract inputs and outputs
y = dataset[:,0:1] # outputs:the column of Appliances
if response1=='A':
    x = dataset[:,1:30] # inputs: other features
if response1=='B':
    x = dataset[:,1:26] # exclude rv1,rv2,RH_5 and Visibility

# Data normalisation(0-1)
scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x)

# Split the data into training/testing sets
if response2 == 'A':
    percentage = 0.6
if response2 == 'B':
    percentage = 0.7

# Train linear Regression model
if response3=='A':
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = percentage, random_state=0)
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(x_train, y_train)
    y_pred = linear_regression.predict(x_test) # Make predictions through using the model

# Train polynomial Regression model
if response3=='B':
    polynomial_regression = PolynomialFeatures(degree=3)
    x_poly = polynomial_regression.fit_transform(x)
    x_poly_train,x_poly_test,y_train,y_test = train_test_split(x_poly, y, train_size = percentage, random_state=0)
    linearrreg_2 = LinearRegression()
    linearrreg_2.fit(x_poly_train,y_train)
    y_pred = linearrreg_2.predict(x_poly_test) # Make predictions through using the model

# The mean squared error
print("Root mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_test, y_pred)))

# Explained variance score
print("Explained variance score: %.2f"
      % (explained_variance_score(y_test, y_pred)))

# Mean absolute error
print("Mean absolute error: %.2f"
      % (mean_absolute_error(y_test, y_pred)))

# Median absolute error
print("Median absolute error: %.2f"
      % (median_absolute_error(y_test, y_pred)))

# r2_score
print("r2_score: %.2f"
      % (r2_score(y_test, y_pred)))

# Plot outputs
fig, ax = plt.subplots()
fontsize = '12'
params = {'axes.labelsize': fontsize,
'axes.titlesize': fontsize}
plt.rcParams.update(params)
ax.set_xlabel('samples number')
ax.set_ylabel('energy consumption')
ax.set_title('predected energy consumption(green line)\n VS.\n actual energy consumpiton(blue line)')
ax.grid(color='lightgray', linestyle='-', linewidth='1')
plt.plot(y_pred[:150], 'g')
plt.plot(y_test[:150], 'b')
plt.show()
