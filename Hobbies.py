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
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



#Open files

train = pd.read_csv('Hobby_Data.csv')



train.dropna( how='all', inplace=True)


for x in train.columns:
    print(f"{x}\n{train[x].unique()[:10]}")


train['Olympiad_Participation'] = train['Olympiad_Participation'].map({'Yes': 1, 'No': 0})
train['Scholarship'] = train['Scholarship'].map({'Yes': 1, 'No': 0})
train['School'] = train['School'].map({'Yes': 1, 'No': 0})
train['Projects'] = train['Projects'].map({'Yes': 1, 'No': 0})
train['Medals'] = train['Medals'].map({'Yes': 1, 'No': 0})
train['Career_sprt'] = train['Career_sprt'].map({'Yes': 1, 'No': 0})
train['Act_sprt'] = train['Act_sprt'].map({'Yes': 1, 'No': 0})
train['Fant_arts'] = train['Fant_arts'].map({'Yes': 1, 'No': 0})
train['Won_arts'] = train['Won_arts'].map({'Yes': 2, 'Maybe': 1, 'No': 0})



Fav_subMathematics = []
Fav_subScience = []
Fav_subAny_language = []
Fav_subHistory_Geography = []

for x in train.Fav_sub:
    if x == "Mathematics":
        Fav_subMathematics.append(1)
        Fav_subScience.append(0)
        Fav_subAny_language.append(0)
        Fav_subHistory_Geography.append(0)

    elif x == "Science":
        Fav_subMathematics.append(0)
        Fav_subScience.append(1)
        Fav_subAny_language.append(0)
        Fav_subHistory_Geography.append(0)
    elif x == "Any language":
        Fav_subMathematics.append(0)
        Fav_subScience.append(0)
        Fav_subAny_language.append(1)
        Fav_subHistory_Geography.append(0)
    elif x == "History/Geography":
        Fav_subMathematics.append(0)
        Fav_subScience.append(0)
        Fav_subAny_language.append(0)
        Fav_subHistory_Geography.append(1)




train["Fav_subMathematics"] = Fav_subMathematics
train["Fav_subScience"] = Fav_subScience
train["Fav_subAny_language"] = Fav_subAny_language
train["Fav_subHistory_Geography"] = Fav_subHistory_Geography
train.drop('Fav_sub', inplace=True, axis=1)




print(train.dtypes)

train['Olympiad_Participation'] = train['Olympiad_Participation'].astype(np.float32)
train['Scholarship'] = train['Scholarship'].astype(np.float32)
train['School'] = train['School'].astype(np.float32)
train['Projects'] = train['Projects'].astype(np.float32)
train['Grasp_pow'] = train['Grasp_pow'].astype(np.float32)
train['Time_sprt'] = train['Time_sprt'].astype(np.float32)
train['Medals'] = train['Medals'].astype(np.float32)
train['Career_sprt'] = train['Career_sprt'].astype(np.float32)
train['Act_sprt'] = train['Act_sprt'].astype(np.float32)
train['Fant_arts'] = train['Fant_arts'].astype(np.float32)
train['Won_arts'] = train['Won_arts'].astype(np.float32)
train['Time_art'] = train['Time_art'].astype(np.float32)
train['Fav_subMathematics'] = train['Fav_subMathematics'].astype(np.float32)
train['Fav_subScience'] = train['Fav_subScience'].astype(np.float32)
train['Fav_subAny_language'] = train['Fav_subAny_language'].astype(np.float32)
train['Fav_subHistory_Geography'] = train['Fav_subHistory_Geography'].astype(np.float32)

train['Predicted Hobby'] = train['Predicted Hobby'].astype("string")

print(train.dtypes)


###########
'''
y = pd.DataFrame([])
y["Predicted Hobby"] = train["Predicted Hobby"]
y_training = y[:1200]
y_testing = y[1200:]


OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(y_training["Predicted Hobby"]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(y_testing["Predicted Hobby"]))

# One-hot encoding removed index; put it back
OH_cols_train.index = y_training.index
OH_cols_valid.index = y_testing.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = y_training.drop('Predicted Hobby', axis=1)
num_X_valid = y_testing.drop('Predicted Hobby', axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print(y.head())
'''
#'''
y = pd.DataFrame([])

Predicted_Academics = []
Predicted_Arts = []
Predicted_Sports = []

for x in train['Predicted Hobby']:
    if x == "Academics":
        Predicted_Academics.append(1)
        Predicted_Arts.append(0)
        Predicted_Sports.append(0)
    elif x == "Arts":
        Predicted_Academics.append(0)
        Predicted_Arts.append(1)
        Predicted_Sports.append(0)
    elif x == "Sports":
        Predicted_Academics.append(0)
        Predicted_Arts.append(0)
        Predicted_Sports.append(1)

y["Predicted_Academics"] = Predicted_Academics
y["Predicted_Arts"] = Predicted_Arts
y["Predicted_Sports"] = Predicted_Sports

y['Predicted_Academics'] = y['Predicted_Academics'].astype(np.float32)
y['Predicted_Arts'] = y['Predicted_Arts'].astype(np.float32)
y['Predicted_Sports'] = y['Predicted_Sports'].astype(np.float32)
print(y.head())

train.drop('Predicted Hobby', inplace=True, axis=1)

#'''

'''
y = train["Predicted Hobby"]
train.drop('Predicted Hobby', inplace=True, axis=1)
'''
print(train.describe())
print(train.head())
print(y.head())

print(train.shape)
X_training = train[:1200]
X_testing = train[1200:]
y_training = y[:1200]
y_testing = y[1200:]

model = keras.Sequential([
    layers.Dense(30, input_shape=[16]),
    layers.Activation('relu'),
    layers.Dense(30),
    layers.Activation('relu'),
    layers.Dense(3, activation='softmax'),
])


print(y.dtypes)
early_stopping =callbacks.EarlyStopping(
    monitor='accuracy',
    min_delta=0.0005,
    patience=15,
    restore_best_weights=True,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01),
    loss='categorical_crossentropy',
    #sparse_categorical_crossentropy
    metrics=['accuracy']
)



history = model.fit(
    X_training, y_training,
    validation_data=(X_testing, y_testing),
    batch_size=64,
    epochs=300,
    callbacks=[early_stopping],
)


plt.plot(history.history['accuracy'])
plt.show()
plt.plot(history.history['loss'])
plt.show()

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']].plot();

print(("Best accuracy {:0.4f}" )\
      .format(history_df['accuracy'].max()))

score = model.evaluate(X_testing, y_testing, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

preds = model.predict(X_testing)

print(y_testing)
print(y_testing.to_numpy().argmax(axis=1))

print(preds.argmax(axis=1))

'''
out=[]
for idx, x in enumerate(preds):
    out.append(round(preds[idx][0]))


Correct = (y_testing == out)

print("TOTAL")
print(Correct)

countAll = len(y_testing)
countCorrect = np.count_nonzero(Correct)
print('Print count of True elements in array: ', countCorrect)
print('Print count of ALL elements in array: ', countAll)

print("TOTAL ACCURATE: ", countCorrect/countAll)
'''



