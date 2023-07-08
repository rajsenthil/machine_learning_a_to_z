import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
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

X = dataset.iloc[:, 1:-1].values  # skip the position title
y = dataset.iloc[:, -1].values

print(X)
print(y)

# Step 4: Apply feature scaling
# Note: For SVR, a difference, the feature scaling is required for both feature/independent variables and dependent variable.
#
x_scaler = StandardScaler()
y_scaler = StandardScaler()
y = y.reshape(len(y), 1)
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

print(X)
print(y)

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.scatter(X, y, c='red')
# plt.show()

# Step 5: Create Model
regressor = SVR(kernel='rbf')

regressor.fit(X, y)

# y_pred = regressor.predict(6.5)
# print("Y Predict for value 6.5: ", y_pred)

ax.plot(X, regressor.predict(X), c='blue')
_var1 = x_scaler.transform([[6.5]]).reshape(-1, 1)
_var2 = regressor.predict(_var1)
ax.scatter(_var1, _var2, c='yellow')
plt.show()

print("For 6.5 years of experience: ", y_scaler.inverse_transform([_var2]))


# y = y.reshape(len(y), 1)
# print(y)

# # Feature Scaling
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X = sc_X.fit_transform(X)
# y = sc_y.fit_transform(y)
# print(X)
# print(y)

# # Training the SVR model on the whole dataset
# regressor = SVR(kernel='rbf')
# regressor.fit(X, y)

# # Predicting a new result
# y_pred = sc_y.inverse_transform(regressor.predict(
#     sc_X.transform([[6.5]])).reshape(-1, 1))

# print('Predicted salary for 6.5', y_pred)

# # Visualising the SVR results
# plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
# plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(
#     regressor.predict(X).reshape(-1, 1)), color='blue')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# # Visualising the SVR results (for higher resolution and smoother curve)
# X_grid = np.arange(min(sc_X.inverse_transform(X)),
#                    max(sc_X.inverse_transform(X)), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
# plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(
#     sc_X.transform(X_grid)).reshape(-1, 1)), color='blue')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()
