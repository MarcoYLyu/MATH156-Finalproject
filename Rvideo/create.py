#!/usr/bin/env python3

import pandas as pd
import time
from createmodel import create

def main():
	f = open('./exceptions.txt', 'a')
	f.write("\n\n\n New Session Start at " + time.asctime(time.gmtime()))
	df=pd.DataFrame(columns=("Optimizer", "Activation", "MSE", "Status"))

	
	opts = ["sgd", "rmsprop", "Adam", "Adamax", "Nadam"]
	acts = ["elu", "softmax", "selu", "relu", "sigmoid"]
	count = 1
	table = {}
	table["sgd"] = []
	table["rmsprop"] = []
	table["Adam"] = []
	for opt in opts:
		for act in acts:
			print("\n\n\
				START:", opt, act, "\
				\n\n")
			for i in range(5):
				try:
					df.loc[count] = [opt, act, create(opt, act), "Normal"]
				except ValueError as ve:
					f.write(("ValueError: " + opt + " " + act))
					df.loc[count] = [opt, act, 0, "ValueError"]
				count += 1
	

	f.close()
	df.to_csv("./nnselection.csv")


			

if __name__ == '__main__':
	main()