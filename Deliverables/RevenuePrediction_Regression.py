## LinearRegression to predict movie revenues
from __future__ import division  # this line is important to avoid unexpected behavior from division
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet


def linear_regression(train_X, train_y, test_X, test_y):
    selector = SelectKBest(f_regression, k=15)
    selector.fit(train_X,train_y)
    X_train = train_X[:, selector.get_support()]
    X_test = test_X[:, selector.get_support()]
    #print selector.get_support()

    model = LinearRegression()
    model.fit(X_train,train_y)
    predictions = model.predict(X_test)
    # print predictions[:5]
    # print test_y[:5]

    MSE = mean_squared_error(test_y, predictions)
    print 'Linear Regression RMSE : ',math.sqrt(MSE)/1000000,     #Diving the RMSE by 1000000 because the values are in millions

def DT_regression(train_X, train_y, test_X, test_y):
    selector = SelectKBest(f_regression)
    selector.fit(train_X, train_y)
    X_train = train_X[:, selector.get_support()]
    X_test = test_X[:, selector.get_support()]
    clf = DecisionTreeRegressor(max_depth = 5, )
    clf.fit(X_train,train_y)
    predictions = clf.predict(X_test)
    MSE = mean_squared_error(test_y,predictions)
    RMSE = math.sqrt(MSE)
    print 'Decision Tree Regression RMSE',RMSE/1000000

def elastic_net(train_X, train_y, test_X, test_y):
    selector = SelectKBest(f_regression)
    selector.fit(train_X, train_y)
    X_train = train_X[:, selector.get_support()]
    X_test = test_X[:, selector.get_support()]
    clf = ElasticNet()
    clf.fit(X_train, train_y)
    predictions = clf.predict(X_test)
    print 'Elastic Net Predictions',predictions
    MSE = mean_squared_error(test_y, predictions)
    RMSE = math.sqrt(MSE)
    print 'Elastic Net RMSE : ',RMSE/1000000

def ridge_regression(train_X, train_y, test_X, test_y):
    selector = SelectKBest(f_regression, k=17)
    selector.fit(train_X, train_y)
    X_train = train_X[:, selector.get_support()]
    X_test = test_X[:, selector.get_support()]
    clf = ElasticNet()
    clf.fit(X_train, train_y)
    predictions = clf.predict(X_test)
    MSE = mean_squared_error(test_y, predictions)
    RMSE = math.sqrt(MSE)
    print 'Elastic Net RMSE : ',RMSE/1000000

def main():
    ## Loading the data into python data structures
    PATH = "C:\CMPSCI585\Project\Movie-Revenue-Prediction_Sentiment-Analysis\Data\CSV file\dataset5.csv"  # Change the path accordingly

    df = pd.read_csv(PATH, sep=',')
    data = df.values
    # print data[:2][0][0:5]
    len_data = data.shape[0]
    print 'Total length of data', len_data
    train_X = df.drop('revenue', axis=1).values[0:int(0.80 * len_data)]
    train_y = df.iloc[:, -1].values[0:int(0.80 * len_data)]
    test_X = df.drop('revenue', axis=1).values[int(0.80 * len_data):]
    test_y = df.iloc[:, -1].values[int(0.80 * len_data):]
    #====================Dividing Budget and revenue by million (1000000)===================
    budget_axis = 7
    train_X[:,budget_axis] = train_X[:,budget_axis]/1000000
    test_X[:,budget_axis] = test_X[:,budget_axis]/1000000

    mean_train_budget = np.mean(train_X[:,budget_axis])
    mean_test_budget = np.mean(test_X[:,budget_axis])
    train_X[:, budget_axis] = train_X[:, budget_axis] - mean_train_budget
    test_X[:, budget_axis] = test_X[:, budget_axis] - mean_test_budget
    #
    train_y = train_y
    train_y_mean = np.mean(train_y)
    train_y = np.abs(train_y - train_y_mean)
    print 'train_y',train_y
    test_y = test_y
    test_y_mean = np.mean(test_y)
    test_y = np.abs(test_y - test_y_mean)
    print 'test_y',test_y

    print train_X.shape
    print train_X.shape
    print train_y.shape
    print test_X.shape
    print test_y.shape

    #linear_regression(train_X, train_y, test_X, test_y)
    DT_regression(train_X, train_y, test_X, test_y)
    elastic_net(train_X, train_y, test_X, test_y)


if __name__ == "__main__":
    main()
