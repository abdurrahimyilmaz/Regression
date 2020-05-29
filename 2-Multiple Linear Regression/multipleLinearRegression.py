#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:32:58 2020

@author: abdurrahim
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values #matrix - independent variables kapsar sondaki dependent olduğu için -1 ile dışarıda bıraktık
y = dataset.iloc[:, 4].values #vector - dependent variable vector olur

#encoding categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

#encoding dependent variables
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])],remainder = 'passthrough')
x = np.array(ct.fit_transform(x),dtype=np.float)

#y değerlerini de bu komutla encode ediyoruz ama bunda ihtiyacımız yok categorizeler
#y = LabelEncoder().fit_transform(y)

#avoiding the dummy variable trap = gölge değişken tuzağı
#burada çoklu bağlantı ihtimalini önlemek için bi sütunu düşürüyoruz 3 sütunumuz var mutlaka 1i 1 olmak zorunda onun için 1 tanesini silmek sorun olmuyor
x = x[:,1:] #normalde otomatik oluyor ama bazı kütüphaneler manuel yapmamızı istiyor

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#fitting mlr to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test results
y_pred = regressor.predict(x_test)

#building the optimal model using backward elimination
import statsmodels.api as sm
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
x_opt = x[:, [0,1,2,3,5]]  #öncekinde en yüksek p değerine sahip olan siliniir backward elimination ana kuralı bu zaten
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
x_opt = x[:, [0,1,2,5]]  #öncekinde en yüksek p değerine sahip olan siliniir backward elimination ana kuralı bu zaten
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
#şuan herhangi bi p değeri bizim başta belirlediğimiz sl=0.05 değerinden büyük olmadığı için burada durabiliriz
#buradan anlıyoruzki geri kalan tek değer r&d değeri bu bizim için iyi bir gösterge diğerlerine göre



#automation
"""Hi guys,

if you are also interested in some automatic implementations of Backward Elimination in Python, please find two of them below:

Backward Elimination with p-values only:

        import statsmodels.formula.api as sm
        def backwardElimination(x, sl):
            numVars = len(x[0])
            for i in range(0, numVars):
                regressor_OLS = sm.OLS(y, x).fit()
                maxVar = max(regressor_OLS.pvalues).astype(float)
                if maxVar > sl:
                    for j in range(0, numVars - i):
                        if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                            x = np.delete(x, j, 1)
            regressor_OLS.summary()
            return x
         
        SL = 0.05
        X_opt = X[:, [0, 1, 2, 3, 4, 5]]
        X_Modeled = backwardElimination(X_opt, SL)

Backward Elimination with p-values and Adjusted R Squared:

        import statsmodels.formula.api as sm
        def backwardElimination(x, SL):
            numVars = len(x[0])
            temp = np.zeros((50,6)).astype(int)
            for i in range(0, numVars):
                regressor_OLS = sm.OLS(y, x).fit()
                maxVar = max(regressor_OLS.pvalues).astype(float)
                adjR_before = regressor_OLS.rsquared_adj.astype(float)
                if maxVar > SL:
                    for j in range(0, numVars - i):
                        if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                            temp[:,j] = x[:, j]
                            x = np.delete(x, j, 1)
                            tmp_regressor = sm.OLS(y, x).fit()
                            adjR_after = tmp_regressor.rsquared_adj.astype(float)
                            if (adjR_before >= adjR_after):
                                x_rollback = np.hstack((x, temp[:,[0,j]]))
                                x_rollback = np.delete(x_rollback, j, 1)
                                print (regressor_OLS.summary())
                                return x_rollback
                            else:
                                continue
            regressor_OLS.summary()
            return x
         
        SL = 0.05
        X_opt = X[:, [0, 1, 2, 3, 4, 5]]
        X_Modeled = backwardElimination(X_opt, SL)

Kind regards,

Hadelin"""