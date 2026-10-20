import pandas as pd                               # FIX: added - needed to give the new row column names
import seaborn as sns                             # gives us the dataset
from sklearn.model_selection import train_test_split  # splits data
from sklearn.linear_model import LinearRegression  # the straight-line model
data = sns.load_dataset("mpg").dropna()           # load the table and drop rows with blanks
X = data[["cylinders","displacement","horsepower","weight","acceleration","model_year"]]  # inputs (features)
y = data["mpg"]                                   # the number we want to predict
X_train, X_test, y_train, y_test = train_test_split(  # split learn + exam
    X, y, test_size=0.2, random_state=1)
model = LinearRegression()                        # create the model
model.fit(X_train, y_train)                       # learn the rule from training rows
print("R2 score:", round(model.score(X_test, y_test), 3))  # 1.0 = perfect fit

new_car = pd.DataFrame([[4, 120, 90, 2500, 16, 80]], columns=X.columns)  # FIX: a table with the SAME column names as X - no feature-name warning
mpg = model.predict(new_car)[0]                   # predict the number
print("Predicted mileage:", round(mpg, 1), "mpg")  # show the answer

import matplotlib.pyplot as plt                   # the charting library
import numpy as np                                # for the trend line
w = data["weight"]; m = data["mpg"]               # x-axis column and y-axis column
plt.scatter(w, m, alpha=0.5, color="#2F49D1", label="cars")  # each dot = one row
a, b = np.polyfit(w, m, 1)                        # best-fit straight line
plt.plot(w, a*w + b, color="red", linewidth=2, label="trend line")  # draw the line
plt.xlabel("weight (lbs)"); plt.ylabel("mpg")     # axis labels
plt.title("Heavier cars get fewer mpg"); plt.legend()  # title and legend
plt.show()                                        # display the graph
