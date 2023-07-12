from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
import configparser

ml_conf_loc = os.getenv("ML_CONF")
config_parser = configparser.RawConfigParser()

config_parser.read(ml_conf_loc)
dataset_loc = config_parser.get("DATASETS", "POSTION_LEVEL_SALARY")


dataset = pd.read_csv(dataset_loc)
print(dataset)

X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)
print(X)
print(y)


regressor = RandomForestRegressor()
regressor.fit(X, y)

y_pred_6point5 = regressor.predict([[6.5]])

print(y_pred_6point5)
