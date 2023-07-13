import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.svm import SVC
import pandas as pd
import os
import configparser

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ml_conf_loc = os.getenv('ML_CONF')

config_parser = configparser.RawConfigParser()
config_parser.read(ml_conf_loc)
dataset_loc = config_parser.get('DATASETS', 'SOCIAL_NETWORK_ADS2')

print(dataset_loc)
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


classifier = SVC(kernel='rbf', degree=3, gamma='auto')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

y_pred = np.array(y_pred).reshape(len(y_pred), 1)
y_test = np.array(y_test).reshape(len(y_test), 1)
print(y_pred)
print(y_test)

print(np.concatenate((y_pred, y_test), 1))

cf = confusion_matrix(y_pred=y_pred, y_true=y_test)
print(cf)

ac_score = accuracy_score(y_pred=y_pred, y_true=y_test)
print(ac_score)

# Visualising the Training set results
X_set, y_set = scaler.inverse_transform(X_train), y_train
print(X_set)
print(y_set)
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=1),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=1))
plt.contourf(X1, X2, classifier.predict(scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
X_set, y_set = scaler.inverse_transform(X_test), y_test.ravel()
print(X_set)
print(y_set)
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=1),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=1))
plt.contourf(X1, X2, classifier.predict(scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
