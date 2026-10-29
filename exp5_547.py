import pandas as pd                               # tables
from sklearn.cluster import KMeans                # the grouping model
data = pd.read_csv("Mall_Customers.csv")          # load the CSV
X = data[["Annual Income (k$)","Spending Score (1-100)"]]  # the 2 columns used for grouping
model = KMeans(n_clusters=5, n_init=10, random_state=1)  # ask for 5 groups
model.fit(X)                                      # find the groups
data["Group"] = model.labels_                     # save each row's group number
print(data["Group"].value_counts())               # how many rows in each group

new_customer = pd.DataFrame([[75, 85]], columns=X.columns)  # FIX: same column names as X - no warning
group = model.predict(new_customer)[0]            # which group is it nearest to?
print("This customer belongs to group:", group)   # show the group number

import matplotlib.pyplot as plt                   # the charting library
x = data["Annual Income (k$)"]                    # x-axis column
y = data["Spending Score (1-100)"]                # y-axis column
plt.scatter(x, y, c=model.labels_, cmap="tab10", s=30)  # colour each dot by its group
cen = model.cluster_centers_                      # the 5 group centres
plt.scatter(cen[:,0], cen[:,1], marker="X", s=200, c="black")  # mark the centres
plt.xlabel("income (k$)"); plt.ylabel("spending score")  # axis labels
plt.title("5 customer groups"); plt.show()        # title and display
