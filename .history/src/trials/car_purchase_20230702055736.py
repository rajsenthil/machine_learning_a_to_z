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


for i in range(b):
    plt.plot(x[i], y[i], c=ListedColormap(("salmon", "dodgeblue"))(i))

#plt.scatter(x, y, color="red")

plt.show()

