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
    print(b.iloc[i]["Purchased"])
    print(x.iloc[i]["Age"])
    if b.iloc[i]["Purchased"] == 1:
        plt.scatter(x.iloc[i]["Age"], y.iloc[i]["AnnualSalary"], c="red")
    else:
        plt.scatter(x.iloc[i]["Age"], y.iloc[i]["AnnualSalary"], c="blue")        

#plt.scatter(x, y, color="red")

plt.show()

