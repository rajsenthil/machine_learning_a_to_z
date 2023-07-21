import os
import configparser

ml_conf_loc = os.getenv('ML_CONF')

config_parser = configparser.RawConfigParser()
config_parser.read(ml_conf_loc)
dataset_loc = config_parser.get('DATASETS', 'MALL_CUSTOMERS')

import pandas as pd
dataset = pd.read_csv(dataset_loc)
print(dataset)

X = dataset.iloc[:, [3, 4]].values
y=dataset.iloc[:,-1]

print(X)
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
y_pred_list = []
ac_scores = []
inertias=[]
for i in range(1,11):
    clusterer = KMeans(n_clusters=i,random_state=42, n_init='auto').fit(X)
    # y_pred = clusterer.predict(X_test)    
    # y_pred_list.append (y_pred)
    # ac_score = accuracy_score(y_pred=y_pred, y_true=y_test)
    # ac_scores.append(ac_score)
    inertias.append(clusterer.inertia_)
# print(ac_scores)
print(inertias)

import matplotlib.pyplot as plt
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.plot(range(1,11), inertias)
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()