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
    x_temp = data_9.iloc[0+319*num:204+319*num]
    x_temp = x_temp.reset_index(drop=True)
    x_temp = x_temp.append([data_9.iloc[210+319*num:319+319*num]], ignore_index=True)
    """ print(x_temp)
    print("----------") """
    X = X.append([x_temp.unstack()], ignore_index=True)

    y_temp = data_9.iloc[204+319*num:210+319*num]
    y_temp = y_temp.reset_index(drop=True)
    """ print(y_temp)
    print("----------") """
    Y = Y.append([y_temp.unstack()], ignore_index=True)

x_count = 0
y_count = 0

for i in range(365, -1, -1):
    if X.iloc[i,:].isnull().all() or Y.iloc[i,:].isnull().all():
            X = X.drop(index=[i])
            Y = Y.drop(index=[i])
X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)

for i in range(312, -1, -1):  
    if X.iloc[:,i].isnull().any():
            X = X.drop(X.columns[i], axis=1)
print(X)
print("--------")
print(Y)
print("--------")

""" for i in range(0, 332):
    if X.iloc[i,:].isnull().any():
        x_count += 1

    if Y.iloc[i,:].isnull().any():
        y_count += 1

print("X含空值:", x_count)
print("Y含空值:", y_count) """

X = X.values
Y = Y.values

#delete string
for i in range(88, 78, -1):
    X = np.delete(X, i, axis=1)

X = np.concatenate((X[:68,:],X[69:,:]))
Y = np.concatenate((Y[:68,:],Y[69:,:]))

print(X)
print(X.shape)
print("--------")
print(Y)
print(Y.shape)
print("--------")

np.save('./np_array/X_all.npy',X)
np.save('./np_array/Y_all.npy',Y)