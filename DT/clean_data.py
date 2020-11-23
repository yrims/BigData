import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

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
    # 把CPK取出到Y
    # 其他資料為X
    x_temp = data_9.iloc[0+319*num:204+319*num]
    x_temp = x_temp.reset_index(drop=True)
    x_temp = x_temp.append([data_9.iloc[210+319*num:319+319*num]], ignore_index=True)
    """ print(x_temp)
    print("----------") """
    X = X.append([x_temp.unstack()], ignore_index=True) #

    y_temp = data_9.iloc[204+319*num:210+319*num]
    y_temp = y_temp.reset_index(drop=True)
    """ print(y_temp)
    print("----------") """
    Y = Y.append([y_temp.unstack()], ignore_index=True)

x_count = 0
y_count = 0

for i in range(365, -1, -1):
    # 把全空、0、false的機台刪除
    if X.iloc[i,:].isnull().all() or Y.iloc[i,:].isnull().all():
            X = X.drop(index=[i])
            Y = Y.drop(index=[i])
            # print('delete data:', i)
            
X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)

for i in range(312, -1, -1):
    # 把全空、0、false的參數欄位刪除
    print('i:',i)
    if X.iloc[:,i].isnull().all():
            X = X.drop(X.columns[i], axis=1)
            print('delete param:', i)

X = X.reset_index(drop=True)

""" for i in range(0, 332):
    if X.iloc[i,:].isnull().any():
        x_count += 1

    if Y.iloc[i,:].isnull().any():
        y_count += 1

print("X含空值:", x_count)
print("Y含空值:", y_count) """

X = X.values
Y = Y.values

train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0)
train = np.concatenate((train_X,train_y), axis=1)
test = np.concatenate((test_X,test_y), axis=1)
train_s = train_X.shape
test_s = test_X.shape


row = []
for i in range(0,train_s[1]+1):
    if i==0:
        row.append('Number')
    else:
        row.append('Input%03d' % i)
for i in range(0,6):
    row.append('Output%03d' %(i+1))
row = np.array(row)
print(row.shape)

col_tr = []
for i in range(0,train_s[0]):
    col_tr.append('Train%03d' %(i+1))
col_tr = np.array(col_tr)
print(col_tr.shape)

col_te = []
for i in range(0,test_s[0]):
    col_te.append('Test%03d' %(i+1))
col_te = np.array(col_te)
print(col_te.shape)

""" print(train.shape)
print(train.T.shape)
print(test.shape)
print(test.T.shape)

train = np.concatenate((col_tr,train.T), axis=0)
train = np.concatenate((row,train.T), axis=0)
test = np.concatenate((col_te,test.T), axis=0)
test = np.concatenate((row,test.T), axis=0) """
# list cant concat with np.array

col_tr = col_tr.reshape(col_tr.shape[0],1)
col_te = col_te.reshape(col_te.shape[0],1)
row = row.reshape(row.shape[0],1)

train = np.concatenate((col_tr,train), axis=1)
train = np.concatenate((row,train.T), axis=1).T
test = np.concatenate((col_te,test), axis=1)
test = np.concatenate((row,test.T), axis=1).T
print(train)
print("------------")
print(test)

trnpy = train[1:,1:]
tenpy = test[1:,1:]
print(trnpy)
print(tenpy)
np.save('./clean_data/train.npy', trnpy)
np.save('./clean_data/test.npy', tenpy)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

train.to_csv('./clean_data/train.csv',index=False,header=False)
test.to_csv('./clean_data/test.csv',index=False,header=False)