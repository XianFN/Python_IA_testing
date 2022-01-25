# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
#import io
#import requests
#import re
#import warnings
#import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

#pio.templates

import matplotlib.pyplot as plt
#% matplotlib
#inline
plt.style.use('seaborn-notebook')
from matplotlib.ticker import StrMethodFormatter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

import os
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


#Data = pd.read_csv('train.csv', index_col='PassengerId')
#Data_test = pd.read_csv('test.csv', index_col='PassengerId')
Data = pd.read_csv('CarPrice_Assignment.csv')

print(Data.head())

print(Data.shape)
print(Data.columns)

print(Data.isnull().sum())



Data['symboling'].unique()
Data[Data['doornumber'] == 'two']
Data[Data['carbody'] == "hatchback"]
Data[Data['enginelocation'] == "rear"]


cat_cols = [x for x in Data.columns if Data[x].dtypes == "object"]
cat_cols

num_cols = [x for x in Data.columns if Data[x].dtypes != "object"]
num_cols

print(num_cols)

print(cat_cols)

Data = pd.get_dummies(Data, drop_first = True)
print(Data.head())

X = Data.drop('price', axis=1)
y = Data['price']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)

model.fit(X_train, y_train)

model_pred = model.predict(X_test)


print(model.score(X_test, y_test))

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, model_pred)
print(mse)





