#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:32:35 2020

Pre analysis for model selection
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from helper import Videogames, getWorkDir
from models import *
from main import read_data


def extra_data():
    res = read_data()
    '''
    res[1]:
    name, g_total, cscore, uscore, genre, publisher, ccount, ucount, platform    
    ['Call of Duty: Modern Warfare 3' '30.59' '88' '3.4' 'Shooter' 'Activision' '81' '8713' 'X360']
    '''
    
def anova_test(res, classifier):
    '''
    classifier = classifier number , eg.  4 represents genre
    Null hypotheses: Groups means are equal (no variation in means of groups)
    Alternative hypotheses: At least, one group mean is different from other groups
    If Null is reject, include X into model
    '''
    classes = res[:, classifier]
    y = res[:, 1]
    table = {}
    for i in range(len(classes)):
        if classes[i] not in table:
            table[classes[i]] = [float(y[i]), 1]
        else:
            table[classes[i]][0] += float(y[i])
            table[classes[i]][1] += 1
    y_total = 0
    keys = table.keys()
    for key in table.keys():
        y_total += table[key][0]
        table[key] = [table[key][0] / table[key][1], 0]
    for i in range(len(classes)):
        table[classes[i]][1] += np.square((float(y[i]) - table[classes[i]][0]))
    sse = 0
    for key in keys:
        sse += table[key][1]
    y_bar = y_total / len(y)
    sst = 0
    for i in range(len(classes)):
        sst += np.square(float(y[i]) - y_bar)
    ssr = sst - sse
    F = (ssr / len(keys)) / (sse / (len(y)-len(keys)))
    return st.f.sf(F, len(keys), len(y)-len(keys))

if __name__=="__main__":
    print("p-value for Genre:     ", anova_test(read_data(), 4))
    print("p-value for Publisher: ", anova_test(read_data(), 5))
    print("p-value for platform:  ", anova_test(read_data(), 8))
    






 