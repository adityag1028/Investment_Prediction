import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'/Users/ayjeeg/Downloads/All ML assignments/Project-4.Investment Prediction/Investment.csv')

x = df.iloc[:,:-1]
y = df.iloc[:,4]

# lets convert catagory data to numerical
x = pd.get_dummies(x)

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(xtrain, ytrain)

y_pred = regressor.predict(xtest)

slope = regressor.coef_
constant = regressor.intercept_

bias = regressor.score(xtrain, ytrain)
varience = regressor.score(xtest, ytest)

import statsmodels.formula as sm

x = np.append(arr = np.full((50,1), constant).astype(int), values = x, axis = 1)

import statsmodels.api as sm

X_opt = x[:,[0,1,2,3,4,5]].astype(int)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

import statsmodels.api as sm

X_opt = x[:,[0,1,2,3,5]].astype(int)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

import statsmodels.api as sm

X_opt = x[:,[0,1,2,3]].astype(int)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

X_opt = x[:, [0,1,3]].astype(int)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = x[:, [0,1]].astype(int)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
