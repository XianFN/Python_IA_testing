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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(X_train.head())

input_shape = [190]
model = keras.Sequential([
    layers.Dense(300, input_shape=[190]),
    layers.Activation('relu'),
    layers.Dense(300),
    layers.Activation('relu'),
    layers.Dense(1),
])

activation_layer = layers.Activation('selu')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()

model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=256,
    epochs=200,
)


history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']].plot();

print(("Best Validation Loss: {:0.4f}" )\
      .format(history_df['loss'].min()))

#other loss 13247.33962488121





