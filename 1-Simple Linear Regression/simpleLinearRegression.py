#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:22:30 2020

@author: abdurrahim
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values #matrix
y = dataset.iloc[:, 1].values #vector

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#simple linear regresyon için bunu ayriyeten yapmamıza gerek yok çünkü kütüphane bizim yerimize kendisi yapıyor
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting slr to the set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #düz çizgi yapcağımız için parametre girmemize gerek yok
regressor.fit(x_train, y_train)
#en basit ml modeli

#predicting the test set results
y_pred = regressor.predict(x_test) #y_test ile karşılaştırılır

#visualizing results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color ='blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#visualizing test results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color ='blue')
plt.title('salary vs experience for test data')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()