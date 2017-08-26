from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sys import argv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import time
import csv
import math

start_time = time.time()
df = pd.read_csv(argv[1])

feature_array = np.arange(1, len(df.columns), 1)

features = df.iloc[:,np.arange(1,len(df.columns),1)]
labels = df.iloc[:,[0]]
#print ("Features")
#print (features)

#print ("Labels")
#print (labels)

features = np.array(features)
labels = np.array(labels)
labels = labels.reshape((-1,1))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=0)

clf = RandomForestClassifier(n_jobs = 6)
clf.fit(X_train, y_train)
joblib.dump(clf, 'multiple_simulatenous_test.pkl')
duration = time.time() - start_time

score = clf.score(X_test, y_test)
print (score)

feature_name = df.columns[1:len(df.columns)]
feature_import = list(zip(feature_name, clf.feature_importances_))
#print (feature_import)


