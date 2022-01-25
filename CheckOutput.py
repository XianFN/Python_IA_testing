# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


gender_dataOTHER = pd.read_csv("gender_submissionOTHERPERSON.csv")
gender_dataOTHER2 = pd.read_csv("gender_submissionOTHERPERSON2.csv")


gender_data = pd.read_csv("gender_submission.csv")


y_test = gender_data['Survived']
y_testOther= gender_dataOTHER['Survived']

Correct = (y_test == y_testOther)

print("TOTAL")
print(Correct)

countAll = len(y_test)
countCorrect = np.count_nonzero(Correct)
print('Print count of True elements in array: ', countCorrect)
print('Print count of ALL elements in array: ', countAll)

print("TOTAL ACCURATE: ", countCorrect/countAll)


y_testOther2= gender_dataOTHER2['Survived']

Correct2 = (y_test == y_testOther2)

countCorrect2 = np.count_nonzero(Correct2)
print('Print count of True elements in array2: ', countCorrect2)
print('Print count of ALL elements in array2: ', countAll)

print("TOTAL ACCURATE: ", countCorrect2/countAll)





