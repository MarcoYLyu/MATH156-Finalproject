import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from helper import Videogames
from models import *


def read_data():
    videogames = Videogames("./data/math156.db")
    videogames.read_data_in("./data/videogames.csv", "VIDEOGAMES", True)
    res = np.array(videogames.execute('''
        SELECT name, jp_total, cscore, uscore, genre, publisher FROM (
            SELECT name AS name,
                   SUM(JP_sales) AS jp_total,
                   critic_score AS cscore,
                   user_score AS uscore,
                   genre AS genre,
                   publisher AS publisher
            FROM VIDEOGAMES 
            WHERE user_score != -1 and critic_score != -1
            GROUP BY name) AS VideogameSummary
        WHERE jp_total != 0.0 and cscore >= 1 and uscore >= 1
        ORDER BY jp_total DESC;
        '''))
    return res

if __name__ == "__main__":
    res = read_data()
    ## the critical scores and user scores
    scores = np.array(res[:, 2:4], dtype=np.float64)

    jptotal = np.array(res[:, 1], dtype=np.float64)
    jptotal.shape = (len(jptotal), 1)

    ## transform categorical data into dummy variables
    genre = res[:, 4]
    ## avoids inter-correlation by dropping a col
    genre = np.array(pd.get_dummies(data=genre, drop_first=True))
    publisher = res[:, 5]
    publisher = np.array(pd.get_dummies(data=publisher, drop_first=True))

    X = np.concatenate((scores, genre, publisher), axis=1)
    Y = jptotal
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .20, random_state = 40)

    ## Linear Regression
    lregr = linear_model(X_train, Y_train.ravel(), 2)

    ## Random Forest
    rfregr = random_forest(X_train, Y_train.ravel())

    ## KNN
    knnregr = knn(X_train, Y_train, 5)

    r2_template = "R^2\t{name:20}{value:18}"
    print(r2_template.format(name="random forest", value=rfregr.score(X_test, Y_test)))
    print(r2_template.format(name="Knn", value=knnregr.score(X_test, Y_test)))
    print(r2_template.format(name="linear regression", value=lregr.score(X_test, Y_test)))
    
    