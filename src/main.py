import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

from helper import Videogames, getWorkDir, get_dir
from models import *
from plotting import *

def read_data():
    videogames = Videogames(get_dir("data/math156.db"))
    videogames.read_data_in(get_dir("data/videogames.csv"), "VIDEOGAMES", True)
    res = np.array(videogames.execute('''
        SELECT name, g_total, cscore, uscore, genre, publisher FROM (
            SELECT name AS name,
                   SUM(global_sales) AS g_total,
                   critic_score AS cscore,
                   user_score AS uscore,
                   genre AS genre,
                   publisher AS publisher
            FROM VIDEOGAMES 
            WHERE year_of_release >= 2004 and uscore != 0 and cscore != 0
            GROUP BY name) AS VideogameSummary
        WHERE g_total != 0 and g_total <= 35
        ORDER BY g_total DESC;
        '''))
    return res

if __name__ == "__main__":
    ## the critic scores and user scores
    columns = ['name', 'gtotal', 'cscore', 'uscore', 'genre', 'publisher']
    res = pd.DataFrame(read_data(), columns=columns)

    n = len(res)
    factor = 0.1
    quantile1 = round(n * factor)
    quantile2 = n - round(n * factor)
    res = res.loc[quantile1:quantile2 + 1, :]

    ## Transform data into appropriate form for regression
    scores = res[['cscore', 'uscore']]
    genre = pd.get_dummies(res['genre'], drop_first=True)
    publisher = pd.get_dummies(res['publisher'], drop_first=True)

    X = pd.concat((scores, genre, publisher), axis=1)
    Y = res['gtotal'].astype('float64')


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .20, train_size=.80, random_state = 40)

    rfregr = random_forest(X_train, Y_train.ravel())   ## Random Forest
    knnregr = knn(X_train, Y_train.ravel())            ## KNN

    print("The mean is", np.mean(Y))
    ## RMSE
    rmse_template='RMSE\t{name:25}{value:18}'
    print(rmse_template.format(name='random forest', value=rmse(X_test, Y_test, rfregr)))
    print(rmse_template.format(name='Knn', value=rmse(X_test, Y_test, knnregr)))

    print('===================')

    ## R2
    r2_template = 'R^2\t{name:25}{value:18}'
    print(r2_template.format(name='random forest', value=rfregr.score(X_test, Y_test)))
    print(r2_template.format(name='Knn', value=knnregr.score(X_test, Y_test)))

    plot_predictions(X_test, Y_test, rfregr, knnregr)