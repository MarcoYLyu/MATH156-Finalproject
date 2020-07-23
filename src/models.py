import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

## just in case someone wants to implement them instead of using sklearn

def knn(xs, ys, n):
    model = KNeighborsRegressor(n_neighbors=n).fit(xs, ys)
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