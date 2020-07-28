#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np

def run():
	ANN = load_model("./ann2")
	df = pd.read_csv("CS_GL_FQ.csv")
	X = df.iloc[ : , 1:4]
	X['bias'] = np.repeat(1, 7516)
	Y = df['ds']
	Y_pred = pd.DataFrame(ANN.predict(X))
	out = pd.concat([X, Y_pred], axis=1)
	out.to_csv("./predictedData.csv")

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .70, random_state = 40)
	print("Evaluate on test data")
	results = ANN.evaluate(X_test, Y_test, batch_size=128)
	print("test loss, test acc:", results)

	print(ANN.summary())

if __name__ == '__main__':
	run()