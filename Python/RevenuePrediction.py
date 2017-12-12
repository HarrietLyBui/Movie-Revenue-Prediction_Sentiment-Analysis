import os
import csv
import numpy as np
from numpy import genfromtxt
import pandas as pd
import math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from __future__ import division  # this line is important to avoid unexpected behavior from division

%matplotlib inline
plt.rcParams['figure.figsize'] = (5, 4) # set default size of plots

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

PATH = "C:\CMPSCI585\Project\OurProject\Data\RottenTomatoesData.csv"   #Change the path accordingly

df=pd.read_csv(PATH, sep=',')
data = df.values
#print data[:2][0][0:5]
len_data = data.shape[0]
print 'Total length of data',len_data
train_X = df.drop('revenue', axis=1).values[1:int(0.80*len_data)]
train_y = df.iloc[:,-1].values[1:int(0.80*len_data)]
test_X = df.drop('revenue', axis=1).values[int(0.80*len_data):]
test_y = df.iloc[:,-1].values[int(0.80*len_data):]
print train_X.shape
print train_y.shape
print test_X.shape
print test_y.shape


# model = LinearRegression()
# model.fit(train_X,train_y)
# predictions = model.predict(test_X)
# print predictions[:5]
# print test_y[:5]

# MSE = mean_squared_error(test_y, predictions)
# print math.sqrt(MSE)

selector = SelectKBest(f_regression, k=15)
selector.fit(train_X,train_y)
X_train = train_X[:, selector.get_support()]
X_test = test_X[:, selector.get_support()]
print selector.get_support()

model = LinearRegression()
model.fit(X_train,train_y)
predictions = model.predict(X_test)
print predictions[:5]
print test_y[:5]

MSE = mean_squared_error(test_y, predictions)
print math.sqrt(MSE)