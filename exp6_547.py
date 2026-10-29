import pandas as pd                               # tables
from sklearn.cluster import AgglomerativeClustering  # the tree-based grouping model
data = pd.read_csv("Mall_Customers.csv")          # load the CSV
X = data[["Annual Income (k$)","Spending Score (1-100)"]]  # the 2 columns used for grouping
model = AgglomerativeClustering(n_clusters=5)     # ask for 5 groups from the tree
data["Group"] = model.fit_predict(X)              # find groups and save each row's group
print(data.groupby("Group")[X.columns].mean())    # average of each group

new_customer = [80, 15]                           # one new row: income 80, spending 15
averages = data.groupby("Group")[X.columns].mean()  # centre (average) of each group
distances = ((averages - new_customer) ** 2).sum(axis=1)  # squared distance to each centre
print("Closest group:", distances.idxmin())       # the nearest group wins
