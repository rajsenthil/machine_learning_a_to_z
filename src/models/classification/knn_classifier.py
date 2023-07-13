from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import configparser

ml_conf_loc = os.getenv('ML_CONF')
config_parser = configparser.RawConfigParser()
config_parser.read(ml_conf_loc)
dataset_loc = config_parser.get('DATASETS', 'SOCIAL_NETWORK_ADS2')


dataset = pd.read_csv(dataset_loc)
print(dataset)

X = dataset.iloc[:, 0:-1]
y = dataset.iloc[:, -1]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, train_size=0.8)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(y_train)


classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X=X_train, y=y_train)

y_pred = classifier.predict(X_test)
y_pred = y_pred.reshape(len(y_pred), 1)
print(y_pred)
y_test = np.array(y_test).reshape(len(y_test), 1)
print(y_test)

# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

print(np.concatenate((y_pred, y_test), 1))

print(confusion_matrix(y_true=y_test, y_pred=y_pred))

ac_score = accuracy_score(y_test, y_pred)
print(ac_score)
