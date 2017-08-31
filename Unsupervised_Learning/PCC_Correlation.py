from sklearn.preprocessing import StandardScaler
from scipy.stats.stats import pearsonr
import itertools
from sys import argv

import numpy as np
import pandas as pd
from pandas import DataFrame
import time
import csv

# Argument order is as follows:
#   1. CSV file name containing data from accounting logs
#       needing Pearson Correlation Coefficient Correlation Analysis
#   2. CSV output file name containing PCC Result

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df


def main():
    correlations = {}

    df = pd.read_csv(argv[1])
    df = handle_non_numerical_data(df)

    columns = df.columns.tolist()

    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[col_a + ' ___ ' + col_b] = pearsonr(df.loc[:, col_a], df.loc[:, col_b])

    result = DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']

    #print(result.sort_index())
    result.to_csv(argv[2])

if __name__ == "__main__":
	main()
