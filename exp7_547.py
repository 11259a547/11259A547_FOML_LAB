from sklearn.datasets import load_breast_cancer   # the dataset loader
from sklearn.model_selection import train_test_split  # splits data
from sklearn.tree import DecisionTreeClassifier   # one tree
from sklearn.ensemble import RandomForestClassifier  # a forest of trees
data = load_breast_cancer()                       # load the dataset
X_train, X_test, y_train, y_test = train_test_split(  # split learn + exam
    data.data, data.target, test_size=0.2, random_state=1)
tree = DecisionTreeClassifier(random_state=1)     # make one tree
tree.fit(X_train, y_train)                        # train the single tree
forest = RandomForestClassifier(n_estimators=100, random_state=1)  # make 100 trees
forest.fit(X_train, y_train)                      # train the forest
print("One tree accuracy:", round(tree.score(X_test, y_test), 3))  # single tree score
print("Forest accuracy  :", round(forest.score(X_test, y_test), 3))  # forest score

names = data.target_names               # class names stored inside the dataset
new_tumour = [X_test[0]]                # one new example from the hidden part
result = forest.predict(new_tumour)[0]  # the forest votes
print("Diagnosis:", names[result])      # show the class name

import matplotlib.pyplot as plt                   # the charting library
importances = forest.feature_importances_         # FIX: the manual said model
names = data.feature_names                        # feature names stored inside the dataset
top = sorted(zip(importances, names), reverse=True)[:5]  # keep the top 5
vals = [t[0] for t in top]; labs = [t[1] for t in top]  # split into values and labels
plt.barh(labs[::-1], vals[::-1], color="#2F49D1")  # horizontal bars
plt.xlabel("importance"); plt.title("Top 5 features")  # labels
plt.show()                                        # display
