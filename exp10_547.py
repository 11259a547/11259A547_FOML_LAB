import numpy as np                                # fast maths
import tensorflow as tf                           # neural network toolbox
from sklearn.datasets import load_digits          # the dataset loader
from sklearn.model_selection import train_test_split  # splits data
digits = load_digits()                            # load the dataset
X_train, X_test, y_train, y_test = train_test_split(  # split learn + exam
    digits.data, digits.target, test_size=0.2, random_state=1)
model = tf.keras.Sequential([                     # build the network layer by layer
    tf.keras.Input(shape=(64,)),                  # input layer: 64 features
    tf.keras.layers.Dense(64, activation="relu"),  # hidden layer: 64 units
    tf.keras.layers.Dense(10, activation="softmax")  # output layer: 10 classes
])                                                # end of the layer list
model.compile(optimizer="adam",                   # how it learns
    loss="sparse_categorical_crossentropy",       # how it measures mistakes
    metrics=["accuracy"])                         # track accuracy
model.fit(X_train, y_train, epochs=10, verbose=0)  # train: 10 passes over the data
loss, acc = model.evaluate(X_test, y_test, verbose=0)  # test on hidden rows
print("Test accuracy:", round(acc, 3))            # show the score

new_image = X_test[0:1]                           # FIX: X_test[0:1] is a proper 2-D array with 1 row
prediction = model.predict(new_image, verbose=0)  # one probability per class
digit = np.argmax(prediction)                     # pick the most likely class number
print("Predicted digit:", digit)                  # show it
