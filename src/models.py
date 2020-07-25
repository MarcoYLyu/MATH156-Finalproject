import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors.regression import check_array, _get_weights

## just in case someone wants to implement them instead of using sklearn

## Modified KNeighborsRegressor so that it uses median rather than mean
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

def knn(xs, ys, n):
    model = KNeighborsRegressor(n_neighbors=n, algorithm='kd_tree').fit(xs, ys)
    return model

def gamma_model(xs, ys):
    model = GammaRegressor().fit(xs, ys)
    return model

def linear_model(xs, ys, m):
    model = make_pipeline(PolynomialFeatures(m), Ridge(normalize=True)).fit(xs, ys)
    return model

def random_forest(xs, ys):
    model = RandomForestRegressor().fit(xs, ys)
    return model