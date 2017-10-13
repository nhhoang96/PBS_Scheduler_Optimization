from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import csv
from sys import argv

# Argument order is as follows:
#   1. Full path CSV file name containing data from accounting logs
#       needing Principal Component Analysis
#   2. Full path CSV output file name containing PCA Result

def main():
    df = pd.read_csv(argv[1])
    df.drop(df.columns[[0]], 1, inplace = True)

    X = df.iloc[:, np.arange(1, len(df.columns), 1)]
    Y = df.iloc[:, [0]]
    X = StandardScaler().fit_transform(X)

    X_pca = PCA()
    Y_sklearn = X_pca.fit_transform(X)

    percentVariance = []

    for i in X_pca.explained_variance_ratio_:
        percentVariance.append(i)

    cor_mat1 = np.corrcoef(X.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)


    eig_val_list = []

    for i in eig_vals:
        eig_val_list.append(i)

    headings = []
    with open (argv[1], 'rb') as f:
        reader = csv.reader(f, delimiter="\t")

        for line in reader:
            elements = line[0].split(",")
            elements.pop(0)

            for i in elements:
                headings.append(i)
            break

    with open(argv[2], 'wb') as f:
        writer = csv.writer(f)
        for i in range (len(headings) - 1):
            writer.writerow([headings[i], eig_val_list[i],percentVariance[i]])

if __name__ == "__main__":
	main()
