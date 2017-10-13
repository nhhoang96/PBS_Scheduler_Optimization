from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sys import argv
import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split


# The argument has the following order:
#   1. Full path CSV file name for training model
#   2. Output directory

df = pd.read_csv(argv[1])

feature_array = np.arange(1, len(df.columns), 1)

features = df.iloc[:,np.arange(1,len(df.columns),1)]
labels = df.iloc[:,[0]]

features = np.array(features)
labels = np.array(labels)
labels = labels.reshape((-1,1))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=0)

clf = RandomForestClassifier(n_jobs = 4)
clf.fit(X_train, y_train)
joblib.dump(clf, argv[2] + 'multiple_simulatenous_test.pkl')

score = clf.score(X_test, y_test)
print (score)

feature_name = df.columns[1:len(df.columns)]
feature_import = list(zip(feature_name, clf.feature_importances_)) #Identify feature importances (if necessary)


