import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from helper import Videogames, getWorkDir
from models import *


def read_data():
    videogames = Videogames(os.path.join(getWorkDir() ,"data/math156.db"))
    videogames.read_data_in(os.path.join(getWorkDir() ,"data/videogames.csv"), "VIDEOGAMES", True)
    res = np.array(videogames.execute('''
        SELECT name, g_total, cscore, uscore, genre, publisher FROM (
            SELECT name AS name,
                   SUM(Global_Sales) AS g_total,
                   critic_score AS cscore,
                   user_score AS uscore,
                   genre AS genre,
                   publisher AS publisher
            FROM VIDEOGAMES 
            WHERE user_score != -1 and critic_score != -1 and year_of_release >= 2010
            GROUP BY name) AS VideogameSummary
        WHERE g_total != 0.0 and cscore >= 1 and uscore >= 1
        ORDER BY g_total DESC;
        '''))
    return res

def feature_score(X):
    #X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=4).fit(X)
    return pca.explained_variance_ratio_

def get_freqs(data):
    """Get the frequency table for categorical data

    Args:
        data (array): the categorical data
    
    Returns:
        a dictionary mapping categorical data to their frequency
    """
    table = {}
    for datum in data:
        if datum not in table:
            table[datum] = 1
        else:
            table[datum] += 1
    return table

def transform_categorical(data):
    """Transform the categorical data to frequencies

    Introduced as an alternative for dummy variables.
    Seems to work better for random forest and knn than
    the dummy ones.

    Args:
        data (array): the categorical data

    Returns:
        (np.array): n * 1 array where each row 
        corresponds to the frequency of the category
    """
    freqtable = get_freqs(data)
    res = []
    for datum in data:
        res.append([freqtable[datum]])
    return np.array(res)
    

if __name__ == "__main__":
    res = read_data()
    ## the critical scores and user scores
    scores = np.array(res[:, 2:4], dtype=np.float64)

    gtotal = np.array(res[:, 1], dtype=np.float64)
    gtotal.shape = (len(gtotal), 1)

    ## transform categorical data into dummy variables
    ## avoids inter-correlation by dropping a col
    genre = np.array(pd.get_dummies(data=res[:, 4], drop_first=True))
    publisher = np.array(pd.get_dummies(data=res[:, 5], drop_first=True))

    ## principal component analysis
    X = np.concatenate((scores, genre, publisher), axis=1)
    print("The score for principal component analysis: ", feature_score(X))

    ## The above tells us that critical score can basically represent all other factors
    X = np.array(res[:, 2:3], dtype=np.float64)
    Y = gtotal
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .20, random_state = 40)

    ## Linear Regression
    lregr = linear_model(X_train, Y_train.ravel(), 7)

    ## Random Forest
    rfregr = random_forest(X_train, Y_train.ravel())

    ## KNN
    knnregr = knn(X_train, Y_train.ravel(), 6)

    r2_template = "R^2\t{name:20}{value:18}"
    print(r2_template.format(name="random forest", value=rfregr.score(X_test, Y_test)))
    print(r2_template.format(name="Knn", value=knnregr.score(X_test, Y_test)))
    print(r2_template.format(name="linear regression", value=lregr.score(X_test, Y_test)))


    """
    ## Plot linear regression
    newxs = np.linspace(0, 100, 10000)
    plt.plot(X, Y, 'r+', newxs, lregr.predict(newxs.reshape(-1, 1)), 'b-')
    plt.savefig('global.png', bbox_inches='tight')
    """