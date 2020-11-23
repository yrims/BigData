import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import plot_importance
import matplotlib.pyplot as plt


Y = np.load('./np_array/Y_6_0618.npy')
Y = Y.astype(float)
cpk_mean = np.mean(Y, axis=0)

print(cpk_mean)

np.savetxt('cpk.txt', Y)