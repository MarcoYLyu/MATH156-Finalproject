import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from helper import Videogames, getWorkDir
from models import *


def read_data():
    videogames = Videogames(os.path.join(getWorkDir() ,"data/math156.db"))
    videogames.read_data_in(os.path.join(getWorkDir() ,"data/videogames.csv"), "VIDEOGAMES", True)
    res = np.array(videogames.execute('''
        SELECT name, g_total, cscore, uscore, genre, publisher FROM (
            SELECT name AS name,
                   SUM(global_sales) AS g_total,
                   critic_score AS cscore,
                   user_score AS uscore,
                   genre AS genre,
                   publisher AS publisher
            FROM VIDEOGAMES 
            WHERE user_score != -1 and critic_score != -1 and year_of_release >= 2010
            GROUP BY name) AS VideogameSummary
        WHERE g_total != 0 and cscore >= 1 and uscore >= 1
        ORDER BY g_total DESC;
        '''))
    return res

def predict(X_test, Y_test, model):
    """Predict the sales based on the dataset

    Args:
        X_test (DataFrame): Data
        Y_test (Series): Actual Sales
        model (object): Model we are using

    Returns:
        Series: predicted scales
    """
    temp = model.predict(X_test)
    return pd.Series(model.predict(X_test))

def plot_helper(xs, data_ys, predict_ys, model_name='Unknown'):
    """Plot the predicted sales

    Args:
        xs (Series): the x values
        data_ys (Series): the actual sales
        predict_ys (Series): the predicted sales
        model_name (str, optional): the name of the model. Defaults to 'Unknown'.
    """
    xs = xs.astype(np.float64)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.xticks(np.linspace(0, 100, 11))

    fig.suptitle(model_name)

    ax1.plot(xs, data_ys, 'k+', label='data')
    ax1.set_title('Actual Sales')
    ax1.set(xlabel='Critical Score', ylabel='Global Sales (million)')
    
    ax2.plot(xs, predict_ys, 'y*', label='prediction')
    ax2.set_title('Predicted Sales')
    ax2.set(xlabel='Critical Score', ylabel='Global Sales (million)')

    plt.savefig('{0}.png'.format(model_name.replace(' ', '_').lower()), bbox_inches='tight')
    plt.clf()

def plot_predictions(X_test, Y_test, gregr, rfregr, knnregr):
    """Plot the Predicted sales of each model

    Args:
        X_test (DataFrame): data
        Y_test (Series): actual sales
        gregr (GammaRegressor): Gamma Regressor
        rfregr (RandomForestRegressor): Random Forest Regressor
        knnregr (KNNRegressor): KNN Regressor
    """
    cscores = X_test['cscore']
    ## Get predicted sales
    gres = predict(X_test, Y_test, gregr)
    rfres = predict(X_test, Y_test, rfregr)
    knnres = predict(X_test, Y_test, knnregr)

    ## Correct the indices in case
    temp = pd.DataFrame(pd.concat([cscores, Y_test], axis=1).to_numpy(),
                        columns=['cscore', 'gtotal'],
                        index=np.arange(0, len(cscores), 1))

    ## Create a pandas DataFrame sorted by Critical Score
    df = pd.concat([temp, gres, rfres, knnres], axis=1)
    df = pd.DataFrame(df.sort_values(by='cscore', ascending=True).to_numpy(),
                columns=['cscore', 'gtotal', 'gres', 'rfres', 'knnres'])

    plot_helper(df['cscore'], df['gtotal'], df['rfres'], 'Random Forest')
    plot_helper(df['cscore'], df['gtotal'], df['knnres'], 'KNN')
    plot_helper(df['cscore'], df['gtotal'], df['gres'], 'Gamma Regression')

def rmse(X_test, Y_test, model):
    Y_pred = model.predict(X_test)
    return mean_squared_error(Y_test, Y_pred, squared=False)

if __name__ == "__main__":
    ## the critical scores and user scores
    columns = ['name', 'gtotal', 'cscore', 'uscore', 'genre', 'publisher']
    res = pd.DataFrame(read_data(), columns=columns)

    scores = res[['cscore', 'uscore']]
    genre = pd.get_dummies(res['genre'], drop_first=True)
    publisher = pd.get_dummies(res['publisher'], drop_first=True)

    X = pd.concat((scores, genre, publisher), axis=1)
    
    Y = res['gtotal'].astype('float64')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .20, random_state = 40)

    ## Gamma Regression
    gregr = gamma_model(X_train, Y_train.ravel())

    ## Random Forest
    rfregr = random_forest(X_train, Y_train.ravel())

    ## KNN
    knnregr = knn(X_train, Y_train.ravel(), 6)

    plot_predictions(X_test, Y_test, gregr, rfregr, knnregr)

    rmse_template='RMSE\t{name:25}{value:18}'
    print(rmse_template.format(name='random forest', value=rmse(X_test, Y_test, rfregr)))
    print(rmse_template.format(name='Knn', value=rmse(X_test, Y_test, knnregr)))
    print(rmse_template.format(name='Gamma regression', value=rmse(X_test, Y_test, gregr)))

    print('===================')

    r2_template = 'R^2\t{name:25}{value:18}'
    print(r2_template.format(name='random forest', value=rfregr.score(X_test, Y_test)))
    print(r2_template.format(name='Knn', value=knnregr.score(X_test, Y_test)))
    print(r2_template.format(name='Gamma regression', value=gregr.score(X_test, Y_test)))