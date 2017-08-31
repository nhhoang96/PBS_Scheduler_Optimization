# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import csv
from sys import argv

# Argument order is as follows:
#   1. CSV file name containing data from accounting logs
#       needing Feature Importance Analysis
#   2. CSV output file name containing Feature Importance Result

df = read_csv(argv[1])
column_names = df
array = df.values
features = df.iloc[:, np.arange(1, len(df.columns), 1)]
labels = df.iloc[:, [0]]
# feature extraction
model = ExtraTreesClassifier()
model.fit(features, labels)
print(model.feature_importances_)

feature_name = df.columns[1:len(df.columns)]
outputFile = open(argv[2], 'w')
output_writer = csv.writer(outputFile)
for i,j in zip(feature_name, model.feature_importances_):
    output_writer.writerow([i, j])
