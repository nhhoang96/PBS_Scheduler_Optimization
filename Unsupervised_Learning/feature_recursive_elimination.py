from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv
from sys import argv

# Argument order is as follows:
#   1. CSV file name containing data from accounting logs
#       needing Feature Recursive Elimination Analysis
#   2. CSV output file name containing Feature_Recursive_Elimination Result

df = read_csv(argv[1])
array = df.values
features = df.iloc[:, np.arange(1, len(df.columns), 1)]
labels = df.iloc[:, [0]]
feature_name = df.columns[1:len(df.columns)]

# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(features, labels)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_

outputFile = open(argv[2], 'w')
output_writer = csv.writer(outputFile)
for i,j in zip(feature_name, fit.ranking_):
    output_writer.writerow([i, j])
