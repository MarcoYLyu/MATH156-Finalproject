import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def pca(data, n):
    """Returns the principal components of the data

    Args:
        data (np.array): data stored in rows
    """
    mu = np.mean(data, axis=0)
    adjusted_data = data - mu

    u, s, vh = np.linalg.svd(adjusted_data)
    u = u[:, :n]
    newdata = u * s[:n]
    return pd.DataFrame(newdata)

def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # load dataset into Pandas DataFrame
    df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width', 'target'])
    std_data = standardize(df.loc[:, :'petal width'])
    principal_components = pca(std_data, 2)
    finaldf = pd.concat([principal_components, df[['target']]], axis=1)
    print(finaldf)

    