import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import configparser

# Step - 1: Read the environemt for config file location and then get the startups dataset location
conf_loc = os.getenv("ML_CONF")
print(conf_loc)

config_parser = configparser.RawConfigParser()
config_parser.read(conf_loc)
dataset_path = config_parser.get("DATASETS", "STARTUPS_50")
print("Dataset Path: ", dataset_path)

# Step - 2: Read the datasets

dataset = pd.read_csv(dataset_path).values
# print(dataset)

# Thoughts:
# Datasets has column of wide variety of ranges, needs to be scaled
# State has texts and need mapping to numeric values
# Has mulitple columns as feature data a) R&D b) Spend c) Administration d) Marketing Spend e) State with dependent column Profit

# Step - 3: get the X_data and y_data

X = dataset[:, :-1]
y = dataset[:, -1]

# print(X)

# Step - 4: Convert the State column with text value to numeric by mapping them to unique numeric key
col_trans = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = col_trans.fit_transform(X)
print(X)

# Step - 5: Split the  data into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=1)

print(X_train)
# Step - 6
#   i) scalar the columns R&D Spend,Administration,Marketing Spend
#  ii) map the State column to numeric vector values

# NOT REQUIRED

# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:, 2:-1])
# X[:, 2:-1] = imputer.transform(X[:, 2:-1])

# Step - 7
# Apply scaling for columns a) R&D Spend b) Administration b) Marketing Spend
# Note: The Column transformer encoded the State column and inserted at the first 3 cols.
# So R&D Spend col is positioned at the index 3, Administration at 4 and Marketting Spend at 5

# NOT REQUIRED
# scaler = StandardScaler()
# X_train[:, 3:] = scaler.fit_transform(X_train[:, 3:])
# print(X_train)
# print(X_test)
# X_test[:, 3:] = scaler.transform(X_test[:, 3:])
# print(X_test)

# Step - 8
# Training the model

linear_regressor = LinearRegression()
linear_regressor.fit(X=X_train, y=y_train)

# Step - 9: Predicting the test set results
y_pred = linear_regressor.predict(X=X_test)
print('y_pred: ', y_pred)
print('y_test: ', y_test)

# Step - 10: Visualizing the training set results
np.set_printoptions(precision=2)
result = np.concatenate((y_pred.reshape(len(y_pred), 1),
                         np.array(y_test).reshape(len(y_test), 1)), axis=1)
# plt.title("Expeceted vs Actual")
# plt.xlabel("Expected")
# plt.ylabel("Actual")
# plt.scatter(X_test[3], y_test, c="green")
# plt.scatter(X_test[3], y_pred, c="purple")
# plt.show()

print(result)
