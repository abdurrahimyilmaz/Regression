#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:44:10 2020

@author: abdurrahim
"""

#ensemble learningin bir türü yani toplu soru soruyoruz genelin görüşünü öğreniyoruz buna ensemble deniyor
#ormandaki tek bir ağacın önemi yok ama hepsini değerlendirdiğimizde bi önem oluyor ortalamasını alıp sonucu görmeye rfr deniyor
#ayrıca decision treeninde bir parçası yani non cont

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values #matrix
y = dataset.iloc[:, 2].values #vector

""""# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""
"""
# Feature Scaling - svrda otomatik yapmıyor manuel yapmamız gerekiyor
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = y.reshape(-1,1) #1d array = vector to 2d array yaptık
y = sc_y.fit_transform(y)"""

#fitting the regression model to the dataset
#create the regressor here
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0) #default kriter mse en iyi çalışan - ne kadar ağaç o kadar köfte
regressor.fit(x,y)

#predict [[scalar]] or np.array([[scalar]]) bi fark yok aslında
y_pred = regressor.predict([[6.5]])

"""#visualizing results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color ='blue')
plt.title('regression model')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()"""

#visualizing results for high precision for dtr that it is the most convenient way
#information entropy mevzusundan dolayı böyle yapılmalı
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color ='blue')
plt.title('regression model')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()