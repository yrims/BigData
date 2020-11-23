import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics
from xgboost import plot_importance
import matplotlib.pyplot as plt

def modelfit(alg, n_model, x_train, y_train, x_test, y_test, useTrainCV=True, cv_fold=5, early_stopping_rounds=50):
    alg.fit(x_train, y_train)
    train_predictions = alg.predict(x_train)
    test_predictions = alg.predict(x_test)
    
    print("Model : %d" % n_model)
    print("------train-------")
    print("MAE score : %.5f" % metrics.mean_absolute_error(y_train, train_predictions))
    print("MSE score : %.5f" % metrics.mean_squared_error(y_train, train_predictions))
    print("RMSE score : %.5f" % metrics.mean_squared_error(y_train, train_predictions)**0.5)
    print("------------------")

    print("------test-------")
    print("MAE score : %.5f" % metrics.mean_absolute_error(y_test, test_predictions))
    print("MSE score : %.5f" % metrics.mean_squared_error(y_test, test_predictions))
    print("RMSE score : %.5f" % metrics.mean_squared_error(y_test, test_predictions)**0.5)
    print("------------------")
    
def feature_validation(model, n_model, K, X, Y, imp):
    a = imp
    a = np.sort(a, axis=None)
    a = a[::-1]
    # a = a[0:90]
    a = np.array(a)
    s = 0
    b = model.estimators_[0].feature_importances_
    ind = []
    for i in range(len(b)):
        s = s + a[i]
        ind.append(np.where(b == a[i]))
        if s > K:
            print("i:", i+1)
            print("s: %.5f" % s)
            break 
    ind = np.squeeze(np.array(ind))
    X = X[:,ind]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    model_ = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100, objective='reg:linear')
    modelfit(model_, n_model, x_train, y_train[:,5], x_test, y_test[:,5]) 

def	plot(model):
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.xticks(np.arange(0, len(model.feature_importances_), 5.0))
    plt.axis('auto')
    plt.show()

def main():
    X = np.load('./DT/np_array/X_all.npy')
    Y = np.load('./DT/np_array/Y_all.npy')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
 
    model = MultiOutputRegressor(xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100, objective='reg:linear'))
    modelfit(model, 1, x_train, y_train, x_test, y_test)

    print(model.estimators_[1])
    # K = 0.90
    # imp = model.estimators_[0].feature_importances_
    # feature_validation(model, 1, K, X, Y, imp)

    print("train:", x_train.shape)
    print("test:", x_test.shape)

if  __name__ == '__main__':
    main()


