import pandas as pd
from sys import argv
import numpy as np

# Argument order is as follows:
#   1. Original full path csv file name from previous parsing stage
#   2. Output full path csv file name all the fields (including labels,
#           estimation error, resources_used.walltime)
#   3. Output full path name of csv file name without estimation error
# and resources_used.walltime (used for training models)


# Convert all non-numerical-data into random values
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
    if (len(argv) < 3):
        print("Missing arguments! Please double-check")
    else:
        df = pd.read_csv(argv[1])

        userPredicted = np.array(df['Resource_List.walltime'].values)
        actualRun = np.array(df['resources_used.walltime'].values)

        df.convert_objects(convert_numeric=True)
        df.fillna(0, inplace=True)
        handle_non_numerical_data(df)

        # Focus on ompthreads, host, hyperthreads
        if not ('ompthreads' in df.columns):
            df.insert(len(df.columns), 'ompthreads', np.zeros(len(df.values)))

        if not ('host' in df.columns):
            df.insert(len(df.columns), 'host', np.zeros(len(df.values)))

        if not ('hyperthreads' in df.columns):
            df.insert(len(df.columns), 'hyperthreads', np.zeros(len(df.values)))

        difference = np.subtract(userPredicted, actualRun)
        difference = difference/ 3600.0

        label = []

        for i in np.nditer(difference):
            if ((i>=0) and (i <=2.0)):
                label.append(1)
            elif ((i>2.0) and (i <= 4.0)):
                label.append(2)
            elif ((i > 4.0) and (i <= 6.0)):
                label.append(3)
            elif ((i > 6.0) and (i <= 8.0)):
                label.append(4)
            elif ((i > 8.0) and (i <= 10.0)):
                label.append(5)
            elif (i > 10.0):
                label.append(5)
            else:
                label.append(0)

        myLabel = np.array(label)
        df.insert(0, 'Labels', myLabel)
        print (myLabel)
        df.to_csv(argv[2])

        # Drop any columns representing the information given after jobs are done executing
        df = df[df.columns.drop(list(df.filter(regex='resources_used')))]
        df = df[df.columns.drop('Estimation Error')]
        df.to_csv(argv[3])

if __name__ == "__main__":
	main()