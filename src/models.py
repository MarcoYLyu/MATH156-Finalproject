import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
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
    X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size= .1, random_state = 40)
    num_cols = len(X_train.columns)
    i = 5

    best_model = KNeighborsRegressor(n_neighbors=i, algorithm='kd_tree').fit(X_train, Y_train)
    best_rmse = rmse(X_test, Y_test, best_model)

    ### Cross Validation
    ns = [n]
    rmses = [best_rmse]
    ### You can change 5 to * 2 or * 3 here for a better result, but slower.
    for n in range(i, int(np.sqrt(num_cols)) + 5):
        model = KNeighborsRegressor(n_neighbors=n, algorithm='kd_tree').fit(X_train, Y_train)
        temp = rmse(X_test, Y_test, model)
        ns.append(n)
        rmses.append(temp)
        if temp < best_rmse:
            best_model = model
            best_rmse = temp
    plot_knn(ns, rmses)

    return best_model

def ann(xs, ys):
    n = len(xs.columns)
    ANN = Sequential()
    ANN.add(Dense(units = 6, activation = "elu", input_dim = n))
    ANN.add(Dense(units = 4, activation = "elu"))
    ANN.add(Dense(units = 1))

    ANN.compile(optimizer = "rmsprop", loss = "mean_squared_error")
    ANN.fit(xs, ys, batch_size = 1, epochs = 100)
    return ANN

def linear_model(xs, ys, m):
    model = make_pipeline(PolynomialFeatures(m), Ridge(normalize=True)).fit(xs, ys)
    return model

def random_forest(xs, ys):
    model = RandomForestRegressor(criterion='mse').fit(xs, ys)
    return model