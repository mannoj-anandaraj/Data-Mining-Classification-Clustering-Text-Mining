# Part 2: Cluster Analysis

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from itertools import combinations

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	df = pd.read_csv(data_file)
	df = df.drop(columns=['Channel', 'Region'])
	return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	stats = pd.DataFrame({
		'mean': df.mean().round(0).astype(int),
		'std': df.std().round(0).astype(int),
		'min': df.min(),
		'max': df.max()
	})
	return stats

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	df_copy = df.copy()
	df_standardized = (df_copy - df_copy.mean()) / df_copy.std()
	return df_standardized

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
	km = KMeans(n_clusters=k, n_init=1, random_state=None)
	labels = km.fit_predict(df)
	return pd.Series(labels)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	km = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=None)
	labels = km.fit_predict(df)
	return pd.Series(labels)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	agg = AgglomerativeClustering(n_clusters=k)
	labels = agg.fit_predict(df)
	return pd.Series(labels)

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X, y):
	return silhouette_score(X, y)

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	df_std = standardize(df)
	k_values = [3, 5, 10]
	results = []

	for k in k_values:
		# Kmeans on original data - 10 runs
		for _ in range(10):
			y = kmeans(df, k)
			score = clustering_score(df, y)
			results.append({
				'Algorithm': 'Kmeans',
				'data': 'Original',
				'k': k,
				'Silhouette Score': score
			})

		# Kmeans on standardized data - 10 runs
		for _ in range(10):
			y = kmeans(df_std, k)
			score = clustering_score(df_std, y)
			results.append({
				'Algorithm': 'Kmeans',
				'data': 'Standardized',
				'k': k,
				'Silhouette Score': score
			})

		# Agglomerative on original data - 1 run (deterministic)
		y = agglomerative(df, k)
		score = clustering_score(df, y)
		results.append({
			'Algorithm': 'Agglomerative',
			'data': 'Original',
			'k': k,
			'Silhouette Score': score
		})

		# Agglomerative on standardized data - 1 run (deterministic)
		y = agglomerative(df_std, k)
		score = clustering_score(df_std, y)
		results.append({
			'Algorithm': 'Agglomerative',
			'data': 'Standardized',
			'k': k,
			'Silhouette Score': score
		})

	return pd.DataFrame(results)

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return rdf['Silhouette Score'].max()

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	df_std = standardize(df)
	y = kmeans(df_std, 3)
	cols = df_std.columns.tolist()
	pairs = list(combinations(cols, 2))

	fig, axes = plt.subplots(3, 5, figsize=(20, 12))
	axes = axes.flatten()

	colors = ['red', 'green', 'blue']
	cluster_colors = [colors[label] for label in y]

	for idx, (col1, col2) in enumerate(pairs):
		axes[idx].scatter(df_std[col1], df_std[col2], c=cluster_colors, alpha=0.5, s=10)
		axes[idx].set_xlabel(col1)
		axes[idx].set_ylabel(col2)
		axes[idx].set_title(f'{col1} vs {col2}')

	plt.tight_layout()
	plt.savefig('scatter_plots.png')
	plt.close()
