import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class P2:
    data = pd.read_csv('irisdata.csv')
    X = data.loc[:, 'sepal_length':'petal_width']
    y = data.loc[:, 'species']

    X = X.to_numpy()
    label = LabelEncoder()
    y = label.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 50)

    x_test_values = X_test

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    mlp1 = MLPClassifier(hidden_layer_sizes=(1,), activation='logistic', solver='lbfgs', 
                                    alpha=0.0001, batch_size = 'auto', learning_rate_init = 0.001,
                                    max_iter = 1000, shuffle = True)


    mlp2 = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='lbfgs', 
                                    alpha=0.0001, batch_size = 'auto', learning_rate_init = 0.001,
                                    max_iter = 1000, shuffle = True)
    
    mlp1.fit(X_train, y_train)
    mlp2.fit(X_train, y_train)

    y_prediction = mlp1.predict(X_test)
    y_prediction2 = mlp2.predict(X_test)

    print("1 hidden layer: \n")
    print("setosa - 0\nversicolor - 1\nvirginica - 2\n")

    print("\npredicted iris types:")
    print(y_prediction)

    print("\nactual iris types:")
    print(y_test)

    print("\nresults\n")
    target_names = ['setosa', 'versicolor', 'virginica']
    results = classification_report(y_test, y_prediction, target_names = target_names)
    print(results)

    print("\n\n10 hidden layers: \n")

    print("\npredicted iris types:")
    print(y_prediction2)

    print("\nactual iris types:")
    print(y_test)


    print("\nresults\n")
    target_names = ['setosa', 'versicolor', 'virginica']
    results = classification_report(y_test, y_prediction2, target_names = target_names)
    print(results)

