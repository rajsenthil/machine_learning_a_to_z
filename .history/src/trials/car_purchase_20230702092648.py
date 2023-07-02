import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

fields = ["Age", "AnnualSalary", "Purchased"]

df = pd.read_csv('/home/senthil/git3/machine_learning_a_to_z/data/raw/car_data.csv',  skipinitialspace=True, usecols=fields)

#print (df[["Age", "AnnualSalary", "Purchased"]])

# make data
x = df[["Age"]]
print(x)
y = df[["AnnualSalary"]]
b = df[["Purchased"]]

# plot
plt.title('Age with Annualsalary')


for i in range(len(b)):    
    if b.iloc[i]["Purchased"] == 1:
        plt.scatter(x.iloc[i]["Age"], y.iloc[i]["AnnualSalary"], c="red")
    else:
        plt.scatter(x.iloc[i]["Age"], y.iloc[i]["AnnualSalary"], c="blue")        

plt.show()

# Feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_sc = sc.fit_transform(x)

# Train logistic model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_sc, y)

# Visualizing the model results
import numpy as np
X_set, y_set = sc.inverse_transform(x_sc), y

print(X_set)
print(y_set)
# for i in range(len(b)):    
#     if b.iloc[i]["Purchased"] == 1:
#         plt.scatter(X_set.iloc[i]["Age"], y_set.iloc[i]["AnnualSalary"], c="red")
#     else:
#         plt.scatter(X_set.iloc[i]["Age"], y_set.iloc[i]["AnnualSalary"], c="blue")        

# plt.title('Age with Annualsalary V2')

# plt.show()        