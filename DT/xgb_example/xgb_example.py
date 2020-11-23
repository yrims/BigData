import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

def modelfit(alg, x_train, y_train, useTrainCV=True, cv_fold=5, early_stopping_rounds=50):
    alg.fit(x_train, y_train)
    train_predictions = alg.predict(x_train)
    
    print("MAE score : ", metrics.mean_absolute_error(y_train, train_predictions))
    print("MSE score : ", metrics.mean_squared_error(y_train, train_predictions))
    print("RMSE score : ", metrics.mean_squared_error(y_train, train_predictions)**0.5)

X = np.load('./data/X_33.npy')
Y = np.load('./data/Y_33.npy')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print("train:", x_train.shape)
print("test:", x_test.shape)

# 建立模型
model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100, objective='reg:linear')

# 訓練
modelfit(model, x_test, y_test[:,0])

