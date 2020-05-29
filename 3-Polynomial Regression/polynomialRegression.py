# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values #matrix
y = dataset.iloc[:, 2].values #vector

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""
#simple linear regresyon için bunu ayriyeten yapmamıza gerek yok çünkü kütüphane bizim yerimize kendisi yapıyor
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#for lr
from sklearn.linear_model import LinearRegression
linReg = LinearRegression() 
linReg.fit(x, y)

#for pl
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 4) #derece ne kadar artarsa o kadar regresyondan interpolasyona(fitting) geçiyoruz
x_poly = polyReg.fit_transform(x)
polyReg.fit(x_poly, y)
linReg2 = LinearRegression()
linReg2.fit(x_poly, y)

#visualising lr
plt.scatter(x,y,color ='red')
plt.plot(x, linReg.predict(x), color ='blue')
plt.show()

#visualizing pr
plt.scatter(x,y,color ='red')
#polyregi de 2 nokta arası linear reg ile çizdirebiliyoruz
plt.plot(x, linReg2.predict(polyReg.fit_transform(x)), color ='blue')
plt.show()

#visualizing pr precisely
x_grid = np.arange(min(x), max(x), 0.1) #0.1 aralıklarla x aralğında diziyi oluşturduk
x_grid = x_grid.reshape((len(x_grid),1)) #nx1 boyutlu bir matrise dönüştürdük vektör halinden
plt.scatter(x,y,color ='red')
#polyregi de 2 nokta arası linear reg ile çizdirebiliyoruz
plt.plot(x_grid, linReg2.predict(polyReg.fit_transform(x_grid)), color ='blue')
plt.show()

#predicting for any value 
#6.5 =scalar [6.5] = 1d array [[6.5]] = 2d array
linReg.predict([[6.5]]) 
linReg2.predict(polyReg.fit_transform([[6.5]]))



