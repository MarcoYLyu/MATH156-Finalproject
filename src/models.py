"""
================================================
== 	Filename: models.py
== 	Author: Yi Lyu
==	Status: Complete
================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

## just in case someone wants to implement them instead of using sklearn

def rmse(X_test, Y_test, model):
    Y_pred = model.predict(X_test)
    return mean_squared_error(Y_test, Y_pred, squared=False)

def plot_knn(ns, rmses):
    plt.plot(ns, rmses, 'r*')
    plt.xlabel('# of neighbors')
    plt.ylabel('RMSE')
    
    plt.savefig('graphs/knn_choice_n.png', bbox_inches='tight')
    plt.clf()

def knn(xs, ys, n=10):
    #X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size= .1, random_state = 40)
    num_cols = len(xs.columns)
    i = 5

    best_index = 4
    best_score = 10000
    nums = [i for i in range(5, int(np.sqrt(num_cols)))]
    cvs = []

    for num in nums:
        model = KNeighborsRegressor(n_neighbors=num, algorithm='kd_tree', weights='distance')
        temp = cross_val_score(model, xs, ys, cv=5).mean()
        temp = np.sqrt(1 - temp)
        if temp < best_score:
            best_score = temp
            best_index = num
    print(best_index)
    return KNeighborsRegressor(n_neighbors=best_index, algorithm='kd_tree', weights='distance').fit(xs, ys)

    return best_model

def ann(xs, ys):
    n = len(xs.columns)
    ANN = Sequential()
    ANN.add(Dense(units = 6, activation = "relu", input_dim = n))
    ANN.add(Dense(units = 4, activation = "relu"))
    ANN.add(Dense(units = 1))

    ANN.compile(optimizer = "adam", loss = "mean_squared_error")
    ANN.fit(xs, ys, batch_size = 2, epochs = 100)
    return ANN

def gamma_model(xs, ys):
    model = GammaRegressor().fit(xs, ys)
    return model

def linear_model(xs, ys, m):
    model = make_pipeline(PolynomialFeatures(m), Ridge(normalize=True)).fit(xs, ys)
    return model

def random_forest(xs, ys):
    model = RandomForestRegressor(criterion='mse').fit(xs, ys)
    return model