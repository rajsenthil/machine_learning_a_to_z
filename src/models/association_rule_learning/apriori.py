import os
import configparser

ml_conf_loc = os.getenv('ML_CONF')
config_parser=configparser.RawConfigParser()
config_parser.read(ml_conf_loc)
dataset_loc = config_parser.get('DATASETS', 'MARKET_BASKET')


import pandas as pd

dataset = pd.read_csv(dataset_loc, header=None)
record_len = len(dataset)
print('total number of records in dataset: ', record_len)
transactions = []
for i in range (0, record_len):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

support_min = 3/len(transactions) # at least 3 transactions
confidence_min = 20/100 # 20%
lift_min = 3

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = support_min, min_confidence = confidence_min, min_lift = lift_min, min_length = 2, max_length = 2)

# Visualising the results
results = list(rules)
print(results)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs,rhs,supports, confidence, lifts))

resultsInDataframe = pd.DataFrame(data=inspect(results), columns=['Left Hand', 'Right Hand', 'Supports', 'Confidence', 'Lifts'])

print(resultsInDataframe)
