from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

fields = ["Age", "AnnualSalary", "Purchased"]

# df = pd.read_csv('/home/senthil/git3/machine_learning_a_to_z/data/raw/car_data.csv',  skipinitialspace=True, usecols=fields)

# STEP - 1: Read dataset
# Dataset columns => User ID, Gender, Age, AnnualSalary, Purchased
dataset = pd.read_csv(
    '/home/senthil/git3/machine_learning_a_to_z/data/raw/car_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# STEP - 2: Clean up or drop unnecessary data
# Note, here the userid is unique to an user and it needs to be dropped as it is not a feature data

X = X[:, 1:]
print(X)

# STEP - 3: Take care of missing data. Note this needs some manual parsing throu the data

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# consider the numeric data columns Gender (after converting to numeric), Age, AnnualSalary and salary only
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

# STEP - 4: Encode categorical data
# Gender values contains text as 'Male', 'Female'. Encode those text into a numeric value say - 'Male' value as 0 and 'Female' value as 1

col_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(col_transformer.fit_transform(X))
print(X)

# STEP - 5: Split the dataset into the Training and Test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

print(X_train)

# STEP - 6: Feature scaling
# Note feature scaling is done after train/test split. Otherwise, the feature scaling may leak data in the test.
# To avoid this leak, the feature scaling is done only after the train/test splits

#########################################################################################
# Apply the feature scaling only to train data.                                         #
# for standardization feature scaling, the formula is                                   #
# x_train_feature_scaling = ( x - Mean of x_train ) / ( standard deviation of x_train ) #
# What ever the Mean of x_train and standard deviation of x_train calculated,           #
# the same values are used for x_test as well                                           #
#########################################################################################

scaler = StandardScaler()
X_train[:, 2:] = scaler.fit_transform(X_train[:, 2:])

X_test[:, 2:] = scaler.transform(X_test[:, 2:])

print(X_train)
print(X_test)
