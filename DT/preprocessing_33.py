import numpy as np
import pandas as pd
import math

DATA = 'D:/users/hwx107m/Desktop/BigData/DT/data/ALL_1.csv'
        
""" with open(DATA, newline='')as csvfile:
    all_data = csv.reader(csvfile)
    column = [row[10] for row in all_data] """

all_data = pd.read_csv(DATA, encoding='Big5')
data_9 = all_data.iloc[:,[9]]

x_temp = pd.DataFrame()
y_temp = pd.DataFrame()
X = pd.DataFrame()
Y = pd.DataFrame()

for num in range(0, 366, 1):
    x_temp = data_9.iloc[295+319*num:298+319*num]
    x_temp = x_temp.reset_index(drop=True)
    x_temp = x_temp.append([data_9.iloc[320+319*num:350+319*num]], ignore_index=True)
    """ print(x_temp)
    print("----------") """
    X = X.append([x_temp.unstack()], ignore_index=True)

    y_temp = data_9.iloc[204+319*num:210+319*num]
    y_temp = y_temp.reset_index(drop=True)
    """ print(y_temp)
    print("----------") """
    Y = Y.append([y_temp.unstack()], ignore_index=True)

""" print(X)
print("--------")
print(Y)
print("--------") """


x_count = 0
y_count = 0

for i in range(365, -1, -1):
    if X.iloc[i,:].isnull().any() or Y.iloc[i,:].isnull().any():
        X = X.drop(index=[i])
        Y = Y.drop(index=[i])

X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)
print(X)
print("--------")
print(Y)
print("--------")

""" for i in range(0, 331):
    if X.iloc[i,:].isnull().any():
        x_count += 1

    if Y.iloc[i,:].isnull().any():
        y_count += 1

print("X含空值:", x_count)
print("Y含空值:", y_count) """

X = X.values
Y = Y.values

X = np.concatenate((X[:68,:],X[69:,:]))
Y = np.concatenate((Y[:68,:],Y[69:,:]))

print(X)
print(X.shape)
print("--------")
print(Y)
print(Y.shape)
print("--------")

np.save('./np_array/X_33.npy',X)
np.save('./np_array/Y_33.npy',Y)