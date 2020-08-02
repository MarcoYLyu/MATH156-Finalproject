"""
================================================
== 	Filename: plotting.py
== 	Author: Yi Lyu
==	Status: Complete
================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from helper import getWorkDir, get_dir

def predict(X_test, Y_test, model):
    """Predict the sales based on the dataset

    Args:
        X_test (DataFrame): Data
        Y_test (Series): Actual Sales
        model (object): Model we are using

    Returns:
        DataFrame: predicted scales
    """
    return pd.DataFrame(model.predict(X_test))

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
    ax1.set(xlabel='Critic Score', ylabel='Global Sales (million)')
    
    ax2.plot(xs, predict_ys, 'y*', label='prediction')
    ax2.set_title('Predicted Sales')
    ax2.set(xlabel='Critic Score', ylabel='Global Sales (million)')

    pic_path = 'graphs/{0}.png'.format(model_name.replace(' ', '_').lower())

    pic_dir = get_dir(pic_path)

    plt.savefig(pic_dir, bbox_inches='tight')
    plt.clf()

def plot_helper2(data_ys, predicted_ys, model_name='Unknown'):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle(model_name)

    bins = np.arange(0, 6, 0.1)
    sns.distplot(data_ys, bins=bins, hist=True, kde=True, ax=ax1, color='r', axlabel='Sales')
    ax1.set_title('Actual Sales -- Density Plot')
    ax1.set_xlim(0, 2)
    sns.distplot(predicted_ys, bins=bins, hist=True, kde=True, ax=ax2, color='b', axlabel='Sales')
    ax2.set_title('Predicted Sales -- Density Plot')
    ax2.set_xlim(0, 2)

    pic_path = 'graphs/{0}_hist.png'.format(model_name.replace(' ', '_').lower())
    pic_dir = get_dir(pic_path)

    plt.savefig(pic_dir, bbox_inches='tight')
    plt.clf()

def plot_predictions(X_test, Y_test, rfregr, knnregr, annregr):
    """Plot the Predicted sales of each model

    Args:
        X_test (DataFrame): data
        Y_test (Series): actual sales
        rfregr (RandomForestRegressor): Random Forest Regressor
        knnregr (KNNRegressor): KNN Regressor
        annregr (ANNRegressor): Artificial Neural Network Regressor
    """
    cscores = X_test['cscore']
    ## Get predicted sales
    rfres = predict(X_test, Y_test, rfregr)
    knnres = predict(X_test, Y_test, knnregr)
    annres = predict(X_test, Y_test, annregr)

    ## Correct the indices in case
    temp = pd.DataFrame(pd.concat([cscores, Y_test], axis=1).to_numpy(),
                        columns=['cscore', 'gtotal'],
                        index=np.arange(0, len(cscores), 1))

    ## Create a pandas DataFrame sorted by Critic Score
    df = pd.concat([temp, rfres, knnres, annres], axis=1)
    df = pd.DataFrame(df.sort_values(by='cscore', ascending=True).to_numpy(),
                columns=['cscore', 'gtotal', 'rfres', 'knnres', 'annres'])

    plot_helper(df['cscore'], df['gtotal'], df['rfres'], 'Random Forest')
    plot_helper(df['cscore'], df['gtotal'], df['knnres'], 'KNN')
    plot_helper(df['cscore'], df['gtotal'], df['annres'], 'ANN')

    plot_helper2(df['gtotal'], df['rfres'], 'Random Forest')
    plot_helper2(df['gtotal'], df['knnres'], 'KNN')
    plot_helper2(df['gtotal'], df['annres'], 'ANN')