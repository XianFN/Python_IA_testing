# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import Normalizer
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



#Open files

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender_data = pd.read_csv("gender_submission.csv")


train.dropna(subset=['Embarked'], how='all', inplace=True)
train = train.drop(['PassengerId'], axis=1)

for x in train.columns:
    print(f"{x}\n{train[x].unique()[:10]}")

train.drop(train[train["Fare"]>500].index, inplace=True)
train["Fare"].describe()

cols = ["Fare", "Cabin", "Ticket", "Name", "Age"]
train[cols][:5]




print(train[train["Cabin"]!=np.nan].values[:200])
train["Ticket"].value_counts() #tickets with most family members, largest family being of 7 members


train[train["Ticket"] == "CA. 2343"] # The CA. 2343 family, no one survived.
#Read and prepare data adding more usefull columns


def nameExtract(x):
    x = x.lower().split(",")[1].split(".")[0].replace(" ", "")
    return x


train["Title"] = train["Name"].apply(lambda x: nameExtract(x))
test["Title"] = test["Name"].apply(lambda x: nameExtract(x))



f, axes = plt.subplots(2, 1, figsize=(25, 10), sharex=True)
sns.histplot(x="Title", hue="Survived", data=train, ax=axes[0]);
sns.barplot(x="Title", y="Age", data=train, ax=axes[1]);


train["Title"].unique()

for x in train["Title"].unique():
    train[train["Title"] == x] = train[train["Title"] == x].fillna(train[train["Title"] == x].Age.mean())

for x in test["Title"].unique():
    test[test["Title"] == x] = test[test["Title"] == x].fillna(test[test["Title"] == x].Age.mean())

f, axes = plt.subplots(1, 1, figsize=(25, 8), sharex=True)
sns.barplot(x="Title", y="Age", data=train, ax=axes);

test = test.fillna(40)

def cabinExtract(x):
    try:
        x = [n.lower() for n in x if n.isalpha()][0]
    except:
        return np.nan
    return x

train["CabinLetter"] = train["Cabin"].apply(lambda x:cabinExtract(x))


train["FareBin10"] = train["Fare"].apply(lambda x:round(x/10)*10) # creating bins of 10 for fare
train["AgeBin5"] = train["Age"].apply(lambda x:round(x/5)*5) # creating bings of 5 for age
test["FareBin10"] = test["Fare"].apply(lambda x:round(x/10)*10)
test["AgeBin5"] = test["Age"].apply(lambda x:round(x/5)*5)

train["CabinLetter"].value_counts()

cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Age', 'Title'] # 'Title'
cat_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title'] # 'Title'
num_cols= ['Fare', 'Age']
training = train[cols]#.astype("Float16")
testing = test[cols]
label = train["Survived"].values.astype("float16")

combined = pd.concat([training, testing]).astype("object")
combined.info()

transformer = Normalizer()
transformed = pd.DataFrame()
transformed[["Age", "Fare"]]=transformer.fit_transform(combined[["Age", "Fare"]])

combined = pd.get_dummies(combined[cat_cols])
combined[["Age", "Fare"]]=transformed[["Age", "Fare"]].astype("float16")
print(combined.shape)
combined.info()

print(training.shape)
print(testing.shape)

training = combined[:886]
testing = combined[886:]
print(training.shape)
print(testing.shape)

print(train.head())

model = keras.Sequential([
    layers.Dense(50, input_shape=[43]),
    layers.Activation('relu'),
    layers.Dense(50),
    layers.Activation('relu'),
    layers.Dense(1),


])



model.compile(
    optimizer=tf.keras.optimizers.Adamax(
        learning_rate=0.0005),
    loss=tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        label_smoothing=0.0,
        reduction="auto"),
    metrics=['accuracy'])



history = model.fit(training, label, batch_size=128, epochs=200, verbose=False)
plt.plot(history.history['accuracy'])

score = model.evaluate(training, label, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

preds = model.predict(testing)

out=[]
for idx, x in enumerate(preds):
    out.append(round(preds[idx][0]))



y_test = gender_data['Survived']

Correct = (y_test == out)

print("TOTAL")
print(Correct)

countAll = len(y_test)
countCorrect = np.count_nonzero(Correct)
print('Print count of True elements in array: ', countCorrect)
print('Print count of ALL elements in array: ', countAll)

print("TOTAL ACCURATE: ", countCorrect/countAll)




