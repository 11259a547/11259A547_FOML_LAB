import pandas as pd                               # tables
from sklearn.model_selection import train_test_split  # splits data
from sklearn.ensemble import GradientBoostingClassifier  # the boosting model
data = pd.read_csv("heart.csv")                   # load the CSV
X = data.drop("target", axis=1)                   # inputs: every column EXCEPT the answer
y = data["target"]                                # the answer column
X_train, X_test, y_train, y_test = train_test_split(  # split learn + exam
    X, y, test_size=0.2, random_state=1)
model = GradientBoostingClassifier()              # make the booster
model.fit(X_train, y_train)                       # train it
print("Accuracy:", round(model.score(X_test, y_test), 3))  # score on hidden rows

new_patient = X_test.iloc[[0]]                  # FIX: double brackets [[0]] keep it as a 1-row TABLE with column names - no warning
result = model.predict(new_patient)[0]          # predict
print("Heart disease? (1=yes, 0=no):", result)  # show the answer
