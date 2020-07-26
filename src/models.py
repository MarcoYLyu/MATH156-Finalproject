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

def gamma_model(xs, ys):
    model = GammaRegressor().fit(xs, ys)
    return model

def linear_model(xs, ys, m):
    model = make_pipeline(PolynomialFeatures(m), Ridge(normalize=True)).fit(xs, ys)
    return model

def random_forest(xs, ys):
    model = RandomForestRegressor(criterion='mse').fit(xs, ys)
    return model


"""
## Modified KNeighborsRegressor so that it uses median rather than mean
from sklearn.neighbors.regression import check_array, _get_weights

class MedianKNeighborsRegressor(KNeighborsRegressor):
    def predict(self, X):
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.median(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

def medianknn(xs, ys, n):
    model = MedianKNeighborsRegressor(n_neighbors=n).fit(xs, ys)
    return model
"""