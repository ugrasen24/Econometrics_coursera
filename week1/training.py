# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 00:23:35 2021

@author: bob
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

t1 = pd.read_csv("e1.csv")
x = t1.Age.values
y = t1.Expenditures.values

x = x.reshape((-1, 1))

model = LinearRegression()

model.fit(x,y)
intercept = model.intercept_
coefficient = model.coef_
print(model.intercept_)
print(model.coef_)

x = t1.Age.values
plt.scatter(x,y)
plt.plot([15,57],[intercept+15*coefficient,intercept+57*coefficient])
plt.title("Regression with Age as explanatory and Expenditure as explained variable")
plt.xlabel("Age (Years)")
plt.ylabel("Expenditure")
plt.show()

Age = sm.add_constant(x)
Expenditure = y
model_2 = sm.OLS(Expenditure,Age)

results_2 = model_2.fit()

print(results_2.summary())

young = t1[t1.Age<40]
old = t1[t1.Age>=40]

# Regression for the young 
x = young.Age.values
Age = sm.add_constant(x)
Expenditure = young.Expenditures
model_2 = sm.OLS(Expenditure,Age)

results_2 = model_2.fit()

print(results_2.summary())

# Regression for the young 
x = old.Age.values
Age = sm.add_constant(x)
Expenditure = old.Expenditures
model_2 = sm.OLS(Expenditure,Age)

results_2 = model_2.fit()

print(results_2.summary())