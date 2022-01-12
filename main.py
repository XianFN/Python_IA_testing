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

def plot_SurvivalRate():
    survived = 'survived'
    not_survived = 'not survived'
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    women = X[X['Sex'] == 'female']
    men = X[X['Sex'] == 'male']
    ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[0], kde=False,
                      color="green")
    ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[0], kde=False,
                      color="red")
    ax.legend()
    ax.set_title('Survival Rate for Female')
    ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False,
                      color="green")
    ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False,
                      color="red")
    ax.legend()
    _ = ax.set_title('Survival Rate for Male');
    ax.figure.show()


def plot_TravelCompanions():
    fig = px.scatter_3d(X, x='Name', y='SibSp', z='Age', color='Age')
    fig.show()
def plot_AgeSurvived():
    for template in ["plotly"]:
        fig = px.scatter(X,
                         x="PassengerId", y="Age", color="Survived",
                         log_x=True, size_max=20,
                         template=template, title="Which Age Survived?")
        fig.show()

def plot_EmbarkedDiferences():
    FacetGrid = sns.FacetGrid(X, row='Embarked', height=4.5, aspect=1.6)
    FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None, hue_order=None)
    FacetGrid.add_legend();
    plt.show()


def plot_NumberOfRelatives():
    axes = sns.catplot('relatives', 'Survived', data=X, aspect=2.5, );
    plt.show()

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

    score = mean_absolute_error(y_test, preds)
    print('MAE:', score)

def train2(X,X_test):
    parameters_test1()

    # Break off validation set from training data
    y = X.Survived

    print(X.head(10))

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(X[features])
    X_test = pd.get_dummies(X_test[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2)
    model.fit(X, y)

    # VALIDATE

    predictions = model.predict(X_test)
    print(y_test)
    print(predictions)
    score = mean_absolute_error(y_test, predictions)
    print('MAE:', score)


def train3(X,X_test):
    from sklearn.model_selection import cross_val_score
    parameters_test1()

    # Break off validation set from training data
    y = X.Survived

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(X[features])
    X_test = pd.get_dummies(X_test[features])

    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestClassifier(n_estimators=100, random_state=2))
                                  ])

    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')

    print("Average MAE score:", scores.mean())

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

#PLOTS
print("\n_______________________________________")
print("_____________PLOTS_____________")
print("_______________________________________\n")

#plot_SurvivalRate()
#plot_TravelCompanions()
#plot_AgeSurvived()
#plot_EmbarkedDiferences()
#plot_NumberOfRelatives()

#Train model
print("\n_______________________________________")
print("_____________TRAIN MODEL _____________")
print("_______________________________________\n")

train1(X, X_test)
#train2(X, X_test)
#train3(X, X_test)

#Train model
print("\n_______________________________________")
print("_____________TRAIN MODEL_____________")
print("_______________________________________\n")

