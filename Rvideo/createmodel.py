import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def create():
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
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .20, random_state = 40)

	ANN = Sequential()
	ANN.add(Dense(units = 6, activation = "elu", input_dim = 4))
	ANN.add(Dense(units = 4, activation = "elu"))
	ANN.add(Dense(units = 1))

	ANN.compile(optimizer = "rmsprop", loss = "mean_squared_error")
	ANN.fit(X_train, Y_train, batch_size = 1, epochs = 100)
	Y_pred = ANN.predict(X_test)

	MSE = mean_squared_error(Y_test, Y_pred)
	print("MSE: ", MSE)

	if MSE < mse_so_far:
		ANN.save('./ann')
		print("MSE is less than previous model: ", mse_so_far, "Updata Mode")
		with open('./mse.txt', 'w') as f:
			f.write(str(MSE))
	ANN.save('./ann2')




	print("Evaluate on test data")
	results = ANN.evaluate(X_test, Y_test, batch_size=128)
	print("test loss, test acc:", results)

	# for i in range(len(Y_test)):
	#	print(Y_test[i], Y_pred[i])



if __name__ == '__main__':
	create()

