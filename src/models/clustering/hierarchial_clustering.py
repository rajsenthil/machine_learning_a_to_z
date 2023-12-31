import os
import configparser

ml_conf_loc = os.getenv('ML_CONF')
config_parser=configparser.RawConfigParser()
config_parser.read(ml_conf_loc)
dataset_loc = config_parser.get('DATASETS', 'MALL_CUSTOMERS')

import pandas as pd

dataset = pd.read_csv(dataset_loc)

X = dataset.iloc[:, [3,4]].values

print(X)

import scipy.cluster.hierarchy as sch

clusterer = sch.dendrogram(sch.linkage(X, method='ward'))

import matplotlib.pyplot as plt

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()


from sklearn.cluster import AgglomerativeClustering
clusterer = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = clusterer.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()