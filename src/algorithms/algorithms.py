import numpy as np
import pandas as pd

def svd_impute(data, k = 0, num_iter = 5):
     svd_data = data.replace(np.nan, 0)
     if k == 0:
         k = min(svd_data.shape[0], svd_data.shape[1])
     miss_matrix = np.zeros(svd_data.shape)

     for index in range(len(svd_data)):
         row = svd_data.iloc[index, :]
         for i in range(len(row)):
             if row[i] == 0:
                 miss_matrix[index][i] = 1
     imputed_matrix = svd_data

     for i in range(num_iter):
         mean = np.array(np.mean(imputed_matrix, axis=0))
         temp = np.zeros(data.shape)
         for index, row in enumerate(miss_matrix):
             temp[index] = row * mean.T
         U, s, V = np.linalg.svd(temp, full_matrices=False)
         approx_matrix = U[:, :k].dot(np.diag(s[:k]).dot(V[:k, :]))
         imputed_matrix = svd_data + np.multiply(miss_matrix, approx_matrix)
     return imputed_matrix