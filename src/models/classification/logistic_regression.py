import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import configparser

ml_conf_loc = os.getenv("ML_CONF")

config_parser = configparser.RawConfigParser()
config_parser.read(ml_conf_loc)
dataset_loc = config_parser.get("DATASETS", "SOCIAL_NETWORK_ADS2")


dataset = pd.read_csv(dataset_loc)
print(dataset)

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('X_train:', X_train)
print('X_test: ', X_test)

classifer = LogisticRegression()
classifer.fit(X_train, y_train)

y_test = np.array(y_test)
y_pred = classifer.predict(X_test)
print(y_test)
print(y_pred)

y_pred_age_30_sal_87k = classifer.predict(scaler.transform([[30, 87000]]))

print('For age: 30 and Salary: 87000, the result: ', y_pred_age_30_sal_87k)

np.printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

print(confusion_matrix(y_true=y_test, y_pred=y_pred))

ac_score = accuracy_score(y_test, y_pred)
print(ac_score)

X_set, y_set = scaler.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max()+10, step=0.25),
                     np.arange(start=X_set[:, 1].min()-1000, stop=X_set[:, 1].max()+1000, step=0.25))


y_pred_x_set = np.array([X1.ravel(), X2.ravel()]).T

y_pred_x_set = classifer.predict(y_pred_x_set)

y_pred_x_set = y_pred_x_set.reshape(X1.shape)

plt.contourf(X1, X2, y_pred_x_set, alpha=0.75,
             cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i))

plt.title('Logistci Regression Model')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
