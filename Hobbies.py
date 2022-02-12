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

Data = pd.read_csv('Hobby_Data.csv')



Data.dropna( how='all', inplace=True)


for x in Data.columns:
    print(f"{x}\n{Data[x].unique()[:10]}")


Data['Olympiad_Participation'] = Data['Olympiad_Participation'].map({'Yes': 1, 'No': 0})
Data['Scholarship'] = Data['Scholarship'].map({'Yes': 1, 'No': 0})
Data['School'] = Data['School'].map({'Yes': 1, 'No': 0})
Data['Projects'] = Data['Projects'].map({'Yes': 1, 'No': 0})
Data['Medals'] = Data['Medals'].map({'Yes': 1, 'No': 0})
Data['Career_sprt'] = Data['Career_sprt'].map({'Yes': 1, 'No': 0})
Data['Act_sprt'] = Data['Act_sprt'].map({'Yes': 1, 'No': 0})
Data['Fant_arts'] = Data['Fant_arts'].map({'Yes': 1, 'No': 0})
Data['Won_arts'] = Data['Won_arts'].map({'Yes': 2, 'Maybe': 1, 'No': 0})



Fav_subMathematics = []
Fav_subScience = []
Fav_subAny_language = []
Fav_subHistory_Geography = []

for x in Data.Fav_sub:
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




Data["Fav_subMathematics"] = Fav_subMathematics
Data["Fav_subScience"] = Fav_subScience
Data["Fav_subAny_language"] = Fav_subAny_language
Data["Fav_subHistory_Geography"] = Fav_subHistory_Geography
Data.drop('Fav_sub', inplace=True, axis=1)




print(Data.dtypes)

Data['Olympiad_Participation'] = Data['Olympiad_Participation'].astype(np.float32)
Data['Scholarship'] = Data['Scholarship'].astype(np.float32)
Data['School'] = Data['School'].astype(np.float32)
Data['Projects'] = Data['Projects'].astype(np.float32)
Data['Grasp_pow'] = Data['Grasp_pow'].astype(np.float32)
Data['Time_sprt'] = Data['Time_sprt'].astype(np.float32)
Data['Medals'] = Data['Medals'].astype(np.float32)
Data['Career_sprt'] = Data['Career_sprt'].astype(np.float32)
Data['Act_sprt'] = Data['Act_sprt'].astype(np.float32)
Data['Fant_arts'] = Data['Fant_arts'].astype(np.float32)
Data['Won_arts'] = Data['Won_arts'].astype(np.float32)
Data['Time_art'] = Data['Time_art'].astype(np.float32)
Data['Fav_subMathematics'] = Data['Fav_subMathematics'].astype(np.float32)
Data['Fav_subScience'] = Data['Fav_subScience'].astype(np.float32)
Data['Fav_subAny_language'] = Data['Fav_subAny_language'].astype(np.float32)
Data['Fav_subHistory_Geography'] = Data['Fav_subHistory_Geography'].astype(np.float32)

Data['Predicted Hobby'] = Data['Predicted Hobby'].astype("string")

print(Data.dtypes)


###########
'''
y = pd.DataFrame([])
y["Predicted Hobby"] = Data["Predicted Hobby"]
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


train = Data.sample(frac=0.75, random_state=200)


test = Data.drop(train.index)


y_training = pd.DataFrame([])

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

y_training["Predicted_Academics"] = Predicted_Academics
y_training["Predicted_Arts"] = Predicted_Arts
y_training["Predicted_Sports"] = Predicted_Sports

y_training['Predicted_Academics'] = y_training['Predicted_Academics'].astype(np.float32)
y_training['Predicted_Arts'] = y_training['Predicted_Arts'].astype(np.float32)
y_training['Predicted_Sports'] = y_training['Predicted_Sports'].astype(np.float32)


train.drop('Predicted Hobby', inplace=True, axis=1)
print(y_training.head())
#'''
########################################## TEST DATA ###################
y_testing = pd.DataFrame([])

Predicted_Academics = []
Predicted_Arts = []
Predicted_Sports = []

for x in test['Predicted Hobby']:
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

y_testing["Predicted_Academics"] = Predicted_Academics
y_testing["Predicted_Arts"] = Predicted_Arts
y_testing["Predicted_Sports"] = Predicted_Sports

y_testing['Predicted_Academics'] = y_testing['Predicted_Academics'].astype(np.float32)
y_testing['Predicted_Arts'] = y_testing['Predicted_Arts'].astype(np.float32)
y_testing['Predicted_Sports'] = y_testing['Predicted_Sports'].astype(np.float32)


test.drop('Predicted Hobby', inplace=True, axis=1)
print(y_testing.head())


'''
y = Data["Predicted Hobby"]
Data.drop('Predicted Hobby', inplace=True, axis=1)
'''


print(train.shape)
print(test.shape)

X_training =train
X_testing = test

model = keras.Sequential([
    #layers.Dropout(0.2, input_shape=(16,)),
    layers.Dense(30, input_shape=[16]),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(30),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax'),
])

early_stopping =callbacks.EarlyStopping(
    monitor='accuracy',
    min_delta=0.001,
    patience=20,
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
    # verbose=0,
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

'''
preds = model.predict(X_testing)

print(y_testing)
print(y_testing.to_numpy().argmax(axis=1))

print(preds.argmax(axis=1))






Correct = (y_testing.to_numpy().argmax(axis=1) == preds.argmax(axis=1))

print("TOTAL")
print(Correct)

countAll = len(y_testing)
countCorrect = np.count_nonzero(Correct)
print('Print count of True elements in array: ', countCorrect)
print('Print count of ALL elements in array: ', countAll)

print("TOTAL ACCURATE: ", countCorrect/countAll)



'''
