#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Value, Lock
import time


def create(opt="Nadam", act="selu", fl=6, sl=4):
	with open('./mse.txt', 'r') as f:
		mse_so_far = f.read()
		if mse_so_far == '':
			mse_so_far = 1
		else:
			mse_so_far = float(mse_so_far)
		print(mse_so_far)
		

	df = pd.read_csv("CS_GL_FQ.csv")
	X = df.iloc[ : , 1:4]
	X['bias'] = np.repeat(1, 7516)
	Y = df['ds']
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .30, random_state = 40)

	ANN = Sequential()
	ANN.add(Dense(units = fl, activation = act, input_dim = 4))
	ANN.add(Dense(units = sl, activation = act))
	ANN.add(Dense(units = 1))

	ANN.compile(optimizer = opt, loss = "mean_squared_error")
	ANN.fit(X_train, Y_train, batch_size = 2, epochs = 20)
	Y_pred = ANN.predict(X_test)

	MSE = mean_squared_error(Y_test, Y_pred)
	print("MSE: ", MSE)

	if MSE < mse_so_far:
		ANN.save('./ann')
		print("\n\n\n------------\
			MSE is less than previous model: ", mse_so_far, "Updata Mode\n\n\n")
		with open('./mse.txt', 'w') as f:
			f.write(str(MSE))

	print("Evaluate on test data")
	results = ANN.evaluate(X_test, Y_test, batch_size=128)
	print("test loss, test acc:", results)
	return(MSE)

def createN(opt, act, fl, sl, val, lock):
	with open('./mse.txt', 'r') as f:
		mse_so_far = f.read()
		if mse_so_far == '':
			mse_so_far = 1
		else:
			mse_so_far = float(mse_so_far)
		print(mse_so_far)
		

	df = pd.read_csv("CS_GL_FQ.csv")
	X = df.iloc[ : , 1:4]
	X['bias'] = np.repeat(1, 7516)
	Y = df['ds']
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .30, random_state = 40)

	ANN = Sequential()
	ANN.add(Dense(units = fl, activation = act, input_dim = 4))
	ANN.add(Dense(units = sl, activation = act))
	ANN.add(Dense(units = 1))

	ANN.compile(optimizer = opt, loss = "mean_squared_error")
	ANN.fit(X_train, Y_train, batch_size = 2, epochs = 100)
	Y_pred = ANN.predict(X_test)

	MSE = mean_squared_error(Y_test, Y_pred)
	print("MSE: ", MSE)

	if MSE < mse_so_far:
		ANN.save('./ann')
		print("\n\n\n------------\
			MSE is less than previous model: ", mse_so_far, "Updata Mode\n\n\n")
		with open('./mse.txt', 'w') as f:
			f.write(str(MSE))
	with lock:
		if (MSE < val.value):
			ANN.save('./ann2')
			val.value = MSE

	print("Evaluate on test data")
	results = ANN.evaluate(X_test, Y_test, batch_size=128)
	print("test loss, test acc:", results)
	return(MSE)

def task(L):
	L += create()

if __name__ == '__main__':
	mse = Value('d', 1.0)
	lock = Lock()
	cores = int(mp.cpu_count())
	pool = mp.Pool(cores)
	start = time.time()
	#results = [pool.apply_async(create, args=("Nadam", "selu", 6, 4, mse, lock)) for i in range(5)]
	#results = [p.get() for p in results]
	procs = [Process(target=createN, args=("adam", "selu", 6, 4, mse, lock)) for i in range(8)]
	for p in procs: p.start()
	for p in procs: p.join()
	span = time.time() - start
	print(cores)
	print("Running time: ", span)
	print(mse.value)


