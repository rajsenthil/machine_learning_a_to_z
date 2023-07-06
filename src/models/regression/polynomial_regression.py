import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import configparser
import os

# Step - 1: Read the environemt for config file location and then get the startups dataset location
conf_loc = os.getenv("ML_CONF")
print(conf_loc)

config_parser = configparser.RawConfigParser()
config_parser.read(conf_loc)
dataset_path = config_parser.get("DATASETS", "POSTION_LEVEL_SALARY")
print("Dataset Path: ", dataset_path)

# Step - 2: Read the datasets
dataset = pd.read_csv(dataset_path)
print(dataset)

# Step - 3: Create feature columns and dependent column  (X, y)

X = dataset.iloc[:, 1:-1]  # skip the position title
y = dataset.iloc[:, -1]

print(X)
print(y)

# Step 4: Any empty values
# None observed and skipping

# Step 5: Change the position title from text to number using ColumnTransformer

# Step 6: Create training and test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0)

print(X_train)

# Step 7: Train with Linear Regression
lin_reg = LinearRegression()
X_lin = lin_reg.fit(X=X, y=y)


print('Linear Regression: ', X_lin)

# Step 8: Train with Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X=X)

# Note: Refer to teh polynomial equation y = b0 + b1*x^1 + b2*x^2 + ..... + bnxn
# With degree of 2, this will create features b1x1 and b2x2 feature matrix.
# Use this feature matrix and apply it to the linear regression model

print('Polynomial Regression: ', X_poly)

# Applying the x_poly features for b1*x^1 and b2*x^2 to the linear model
lin_reg2 = LinearRegression()
X_lin2 = lin_reg2.fit(X_poly, y)

# Plot the given data as it is first X, y
plt.scatter(X, y, c="red")
plt.plot(X, lin_reg.predict(X), c="blue")
plt.title('Polynomial Lienar Regression model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.xlim = 15
plt.ylim = 2000000
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), c="purple")
plt.show()


print(lin_reg.predict([[6.5]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))
