# Step - 0: Read config property file and the dataset location
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import os
import configparser

conf_loc = os.environ.get('ML_CONF')
print('Conf location: ', conf_loc)
conf = configparser.RawConfigParser()
conf.read(conf_loc)
dataset_loc = conf.get(
    'DATASETS', 'YEARSOFEXP_SALARY')

print('Dataset location: ', dataset_loc)

# Step - 1: Read the datasets
dataset = pd.read_csv(dataset_loc)
print(dataset)

# Step - 2: Clean up and drop any unused data
# This datasets has USERID column which is not required and needs to be dropped

# NOT REQUIRED

# Step - 3: Divide the data by features and dependent values
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
print(X)
print(y)

# Step - 4: Any missing data, then take care of it

# NOT REQUIRED

# Step - 5: Encode the string data into numeric.
# GENDER column contains string MALE/FEMALE. This text MALE/FEMALE needs to be mapped to numeric value

# NOT REQUIRED

# Step - 6: Split the dataset into training and test data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=1)
print(X_train)

# Step - 7: Feature scaling.
# Scaling is important as to keep the feature columns having close range of values.
# If a column contains a high range of numbers and another column contains only fractions,
# then this would create unnecessary bias.
# There are two columns AGE and AnnualSalary, where AGE ranges in two digits and AnnualSalary range in five figures.
# These two columns are applied scaling to keep in close range.

# Note: Also the scaling is done only to TRAINing datasets and not to the test datasets.
# This is very important so that there is no dataleak happens
scaler = StandardScaler()
X_train = np.array(X_train).reshape(-1, 1)
X_train = scaler.fit_transform(X_train)

print('X_train', X_train)

# Step - 8: Now use the same scaler to scale it for TEST data as well
X_test = np.array(X_test).reshape(-1, 1)
X_test = scaler.transform(X_test)
print('X_test:', X_test)

# Step - 9: Training the model
linear_regressor = LinearRegression()
linear_regressor.fit(X=X_train, y=y_train)

# Step - 10: Predicting the test set results
y_pred = linear_regressor.predict(X=X_test)
print('y_pred: ', y_pred)
print('y_test: ', y_test)

# Step - 11: Visualizing the training set results

plt.title("Years of Experience vs Salary Training sets")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.scatter(X_train, y_train, c="red")
plt.plot(X_train, linear_regressor.predict(X_train), c="lightblue")
plt.scatter(X_test, y_test, c="green")
plt.scatter(X_test, y_pred, c="purple")
plt.show()
