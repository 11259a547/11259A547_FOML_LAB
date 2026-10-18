from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))


species = ["setosa", "versicolor", "virginica"]  # names for class numbers 0, 1, 2
new_flower = [[5.1, 3.5, 1.4, 0.2]]              # one new example: 4 measurements
answer = model.predict(new_flower)[0]            # ask the model
print("This flower is:", species[answer]) 
