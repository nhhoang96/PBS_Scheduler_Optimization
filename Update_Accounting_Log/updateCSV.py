from sklearn.ensemble import RandomForestClassifier
from sys import argv
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import time


# Update the Resource_List.walltime field based on the model predictions

# The argument order is as follows:
#   1. CSV file name of parsed accounting logs
#   2. Model file name ending with .pkl (i.e. hello.pkl)
#   3. Modified CSV file name


df2 = pd.read_csv(argv[1])
feature_array = np.arange(1, len(df2.columns), 1)

userPredictions = df2['Resource_List.walltime'].values
userPredictions = userPredictions/ 3600.0
print (userPredictions)

features = df2.iloc[:,np.arange(1,len(df2.columns),1)]
labels = df2.iloc[:,[0]]

clf = joblib.load(argv[2])

features = np.array(features)
labels = np.array(labels)
labels = labels.reshape((-1,1))

julPred = clf.predict(features)

newArray = []
for x in range (julPred.size):
    # print julPred[x]
    if (julPred[x] == 0):
        newPrediction = 1 + userPredictions[x]
        newPrediction = int(newPrediction * 3600)
        newArray.append(newPrediction)
    elif (julPred[x] == 1):
        newArray.append(int(userPredictions[x]*3600))
    elif (julPred[x] == 2):
        newArray.append(int(userPredictions[x] - 1.0) * 3600)
    elif (julPred[x] == 3):
        newArray.append(int(userPredictions[x] - 2.0) * 3600)
    elif (julPred[x] == 4):
        newArray.append(int(userPredictions[x] - 2.5) * 3600)
    elif (julPred[x] == 5):
        newArray.append(int(userPredictions[x] - 3.0) * 3600)
    else:
        newArray.append(int(userPredictions[x] - 3.5) * 3600)

#print (newArray)
df2['Modified Resource_List.walltime'] = newArray

df2.to_csv(argv[3])