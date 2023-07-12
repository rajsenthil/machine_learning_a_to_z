from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import os
import configparser

ml_conf_loc = os.getenv("ML_CONF")
config_parser = configparser.RawConfigParser()
config_parser.read(ml_conf_loc)
dataset_loc = config_parser.get("DATASETS", "POSTION_LEVEL_SALARY")

print("Datasdets used: ", dataset_loc)

dataset = pd.read_csv(dataset_loc)
print(dataset)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

regressor = DecisionTreeRegressor(random_state=0)
y = y.reshape(len(y), 1)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])
print("The predicted salary for 6.5 years of experience: ", y_pred)
