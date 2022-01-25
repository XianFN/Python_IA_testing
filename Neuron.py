# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

import tensorflow as tf
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


def parameters_test1():
    param_test1 = {
        'n_estimators': [100, 200, 500, 750, 1000],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'gamma': [i / 10.0 for i in range(0, 5)],
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1],
        'learning_rate': [0.01, 0.02, 0.05, 0.1]
    }

def train1(X,X_test):
    parameters_test1()

    y = X.Survived

    print(X.head(10))

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(X[features])
    X_test = pd.get_dummies(X_test[features])


    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestClassifier(n_estimators=200, random_state=2))
                                  ])

    my_pipeline.fit(X, y)
    preds = my_pipeline.predict(X_test)

    print(y_test)
    print(preds)
    # Evaluate the model

    score = y_test == preds
    print('total:', score.count)


#Open files

#X = pd.read_csv('train.csv', index_col='PassengerId')
#X_test = pd.read_csv('test.csv', index_col='PassengerId')
X = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
gender_data = pd.read_csv("gender_submission.csv")


print(X.head())

#Usefull data

print(X["Sex"].value_counts())
print(X.nunique())


print(X.columns)

women = X.loc[X.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = X.loc[X.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of women survived:", rate_women)
print("% of men survived:", rate_men)



#Read and prepare data adding more usefull columns
print("\n_______________________________________")
print("_____________PREPARE DATA_____________")
print("_______________________________________\n")

data = [X, X_test]
for dataset in data:
    mean = X["Age"].mean()
    std = X_test["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = X["Age"].astype(int)


embarked_mode = X['Embarked'].mode()
data = [X, X_test]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(embarked_mode)


data = [X, X_test]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'
    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'

X["Embarked"] = X["Embarked"].astype(str)
print(X["Embarked"].dtypes)
print(X[X["Embarked"] == "C"])

y_test= gender_data['Survived']


#Train model
print("\n_______________________________________")
print("_____________TRAIN MODEL _____________")
print("_______________________________________\n")

from tensorflow import keras
from tensorflow.keras import layers

X.shape # (rows, columns)

parameters_test1()

y = X.Survived

print(X.head(10))

#features = ["Pclass", "Sex", "SibSp", "Parch"]
#X = pd.get_dummies(X[features])
#X_test = pd.get_dummies(X_test[features])

dataTypeSeries = X.dtypes
print('Data type of each column of Dataframe :')
print(dataTypeSeries)

X['Name'] = X['Name'].astype("string")
X['Sex'] = X['Sex'].astype("string")
X['Ticket'] = X['Ticket'].astype("string")
X['Cabin'] = X['Cabin'].astype("string")
X['Embarked'] = X['Embarked'].astype("string")
X['travelled_alone'] = X['travelled_alone'].astype("string")

X['PassengerId'] = X['PassengerId'].astype(np.float32)
X['Survived'] = X['Survived'].astype(np.float32)
X['Pclass'] = X['Pclass'].astype(np.float32)
X['Age'] = X['Age'].astype(np.float32)
X['SibSp'] = X['SibSp'].astype(np.float32)
X['Parch'] = X['Parch'].astype(np.float32)
X['Fare'] = X['Fare'].astype(np.float32)
X['relatives'] = X['relatives'].astype(np.float32)

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(X[features])
X_test = pd.get_dummies(X_test[features])

dataTypeSeries = X.dtypes
print('Data type of each column of Dataframe :')
print(dataTypeSeries)

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1])
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)


history = model.fit(
    X, y,
    validation_data=(X_test, y_test),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0,  # hide the output because we have so many epochs
)


history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(),
              history_df['val_binary_accuracy'].max()))