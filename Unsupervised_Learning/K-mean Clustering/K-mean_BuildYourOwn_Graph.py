import matplotlib.pyplot as plt
from matplotlib import style
from sys import argv

style.use('ggplot')
import numpy as np
import pandas as pd
import time

# Argument order is as follows:
#   1. Full path CSV file name containing data from accounting logs
#       needing Pearson Correlation Coefficient Correlation Analysis
#   2. Y value (vertical value for K-mean visualization)
#	3. X value (horizontal value for K-mean visualization)
#	4. Minimum number of clusters
#	5. Maximum number of clusters

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

class K_Means:
	def __init__(self, k, tol = 0.0001, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		self.centroids = {}

		for i in range(self.k):
			self.centroids[i] = data[i]

		# Optimization process
		for i in range(self.max_iter):
			#Key: centroid, values: dataset
			self.classifications = {}

			for i in range (self.k):
				self.classifications[i] = []

			for featureset in data:
				distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset) #

			# Compare the two centroids to see how much it changes
			prev_centroids = dict(self.centroids)


			for classification in self.classifications:
				# Find the mean to caclulate the centroid. Take the array value and get average of all
				# classifications that we have.  (The mean of all features)
				self.centroids[classification] = np.average(self.classifications[classification], axis = 0)

			optimized = True
			count = 0
			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]

				# If any of the centroids move more than the tolerance, we're not optimized
				if np.sum((current_centroid - original_centroid)/original_centroid * 100.0) > self.tol:
					#How many iterations, how big the change is (in percent)
					count += 1
					print "Counter" + str(count)
					print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
					optimized = False

			#When we are optimized, break out of the loop => don't go unnecessary
			if (optimized):
				break

	def predict(self, data):
		distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

def main():
	start_time = time.time()

	fields = [argv[2],argv[3]]
	df = pd.read_csv(argv[1], usecols = fields)
	df = handle_non_numerical_data(df)
	df.convert_objects(convert_numeric=True)
	df.fillna(0, inplace=True)

	X = np.array(df.astype(float))
	#X = preprocessing.scale(X)

	for n in range(argv[4],argv[5], 1):
		cluster = K_Means(k = n)
		cluster.fit(X)

		correct = 0
		colors = 10 * ["g", "r", "c", "b", "k"]

		for centroid in cluster.centroids:
			if (cluster.centroids[centroid].size == 1):
				pass
			else:
				plt.scatter(cluster.centroids[centroid][0], cluster.centroids[centroid][1], marker ='o', color ='k', s=150,linewidth = 5)

		for classification in cluster.classifications:
			color = colors[classification]
			for featureset in cluster.classifications[classification]:
				plt.scatter(featureset[0], featureset[1], marker='x', color= color, s= 150, linewidths=2)

		plt.xlabel(fields[1])
		plt.ylabel(fields[0])
		plt.title("Clustering plot of "  + str(n) + " clusters " + fields[1] + " vs " + fields[0])
		plt.savefig("Clustering plot of " +str (n) + " clusters " + fields[0] + " vs " + fields[1] + ".png")

	print("The time it takes is ")
	print(time.time() - start_time)


if __name__ == "__main__":
	main()