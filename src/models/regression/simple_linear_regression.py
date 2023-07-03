# Step - 0: Read config property file and the dataset location
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
    'DATASETS', 'USERID_GENDER_AGE_SALARY_CAR_PURCHASE_DATASET')

print('Dataset location: ', dataset_loc)

# Step - 1: Read the datasets
dataset = pd.read_csv(dataset_loc)
print(dataset)

# Step - 2: Clean up and drop any unused data
# This datasets has USERID column which is not required and needs to be dropped
dataset = dataset.iloc[:, 1:]
print(dataset)

# Step - 3: Divide the data by features and dependent values
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Step - 4: Any missing data, then take care of it
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# consider the numeric data columns Gender (after converting to numeric), Age, AnnualSalary and salary only
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

# Step - 5: Encode the string data into numeric.
# GENDER column contains string MALE/FEMALE. This text MALE/FEMALE needs to be mapped to numeric value
col_transformer = ColumnTransformer(transformers=[
                                    ('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = col_transformer.fit_transform(X)
print(X)


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
X_train[:, 2:] = scaler.fit_transform(X_train[:, 2:])

print('X_train', X_train)

# Step - 8: Now use the same scaler to scale it for TEST data as well
X_test[:, 2:] = scaler.transform(X_test[:, 2:])
print('X_test:', X_test)

# Step - 9: Training the model
