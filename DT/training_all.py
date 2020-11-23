import xgboost as xgb
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
import matplotlib.pyplot as plt

def modelfit(alg, n_model, train_x, train_y, test_x, test_y, useTrainCV=True, cv_fold=5, early_stopping_rounds=50):
    print("Model : %d" % n_model)

    alg.fit(train_x, train_y, eval_metric='rmse', verbose=0, eval_set=[(test_x, test_y)], early_stopping_rounds=30)
    train_predictions = alg.predict(train_x, ntree_limit=alg.best_ntree_limit)
    test_predictions = alg.predict(test_x, ntree_limit=alg.best_ntree_limit)
    # train_mae = metrics.mean_absolute_error(train_y, train_predictions)
    train_mse = metrics.mean_squared_error(train_y, train_predictions)
    # test_mae = metrics.mean_absolute_error(test_y, test_predictions)
    test_mse = metrics.mean_squared_error(test_y, test_predictions)
    evalu = [train_mse, test_mse]
    return evalu

def feature_validation(model, n_model, K, X, Y, imp):
    # 用百分比驗證
    a = model.feature_importances_
    b = np.sort(a, axis=None) # 由小到大
    # b = a[::-1] # 由大到小
    a = np.array(a)
    # print('a,',a.shape)
    s = 0
    ind = []
    for i in range(a.shape[0]):
        s = s + b[i]
        c = np.where(a == b[i])[0].tolist()
        ind.append(c[0])
        if s > K:
            print("i:", i+1)
            print("s: %.5f" % s)
            break 
    # ind = np.array(ind)
    # ind = ind.tolist()
    X = X[:,ind]
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    # print('xtest',test_x.shape)
    # print('ytest', test_y.shape)
    model_ = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100, objective='reg:linear')
    evalu = modelfit(model_, n_model, train_x, train_y[:, n_model-1], test_x, test_y[:, n_model-1])
    evalu = np.array(evalu)
    return evalu 

def feature_validation_(model, n_model, K, X, Y, imp):
    # 用維度數量驗證
    a = imp
    b = np.sort(a, axis=None) # 由小到大
    # b = a[::-1] # 由大到小
    b = np.array(b)
    # print('a,',a.shape)
    s = 0
    ind = []
    for i in range(a.shape[0]):
        s = s + b[i]
        c = np.where(a == b[i])[0].tolist()
        ind.append(c[0])
        if s > K:
            print("i:", i+1)
            print("s: %.5f" % s)
            break 
    # ind = np.array(ind)
    # ind = ind.tolist()
    X = X[:,ind]
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    # print('xtest',test_x.shape)
    # print('ytest', test_y.shape)
    model_ = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100, objective='reg:linear')
    evalu = modelfit(model_, n_model, train_x, train_y[:, n_model-1], test_x, test_y[:, n_model-1])
    evalu = np.array(evalu)
    return evalu 

def cus_feature_validation(model, n_model, K, X, Y, imp):
    # 用維度數量驗證
    a = imp
    b = np.sort(a, axis=None) # 由小到大
    b = b[::-1] # 由大到小
    b = np.array(b)
    # print('a,',a.shape)
    s = 0
    ind = []
    for i in range(K):
        s = s + b[i]
        c = np.where(a == b[i])[0].tolist()
        ind.append(c[0])
        
    X = X[:,ind]
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    # print('xtest',test_x.shape)
    # print('ytest', test_y.shape)
    model_ = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100, objective='reg:linear')
    evalu = modelfit(model_, n_model, train_x, train_y[:, n_model-1], test_x, test_y[:, n_model-1])
    evalu = np.array(evalu)
    return evalu 

def each_feature_validation(model, n_model, K, X, Y, imp):
    # 全部特徵扣除每5個特徵數量驗證，重要性由大到小
    a = imp
    b = np.sort(a, axis=None) # 由小到大
    b = b[::-1] # 由大到小
    b = np.array(b)
    # print('a,',a.shape)
    s = 0
    ind = []
    for i in range(K, K+5):
        c = np.where(a == b[i])[0].tolist()
        ind.append(c[0])
    X_ = []
    for i in range(115):
        if i not in ind:
            X_.append(X[:,i])
    X_ = np.array(X_)
    X_ = np.transpose(X_)
    train_x, test_x, train_y, test_y = train_test_split(X_, Y, test_size=0.2, random_state=0)
    # print('xtest',test_x.shape)
    # print('ytest', test_y.shape)
    # model_ = xgb.XGBRegressor()
    evalu = modelfit(model, n_model, train_x, train_y[:, n_model-1], test_x, test_y[:, n_model-1])
    evalu = np.array(evalu)
    return evalu 

def	plot(model):
    # plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.plot()
    plt.xticks(np.arange(0, len(model.feature_importances_), 5.0))
    plt.axis('auto')
    plt.show()

def correlation(a, b):
    corr = np.corrcoef(a, b)
    corr = corr[0,1]
    return corr

def covariance(a, b):
    cov = np.cov()

def save_imp(fn, imp):
    ind = []
    with open('DT/' + fn, 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['Dimention', 'importance'])
        count = 0
        for num, i in enumerate(imp):
            if i > 0:
                csvWriter.writerow([num, i])
                count = count + 1
                ind.append(num)
        print(count)
    return ind

def find_common(imp):
    tmp = []
    for i in range(len(imp[0])):
        if imp[0][i] > 0 and imp[1][i] > 0 and imp[2][i] > 0 and imp[3][i] > 0 and imp[4][i] > 0 and imp[5][i] > 0:
            tmp.append([i, imp[0][i], imp[1][i], imp[2][i], imp[3][i], imp[4][i], imp[5][i]])
    tmp = np.array(tmp)
    with open('DT/imp/common.csv', 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        for i in range(tmp.shape[0]):
            csvWriter.writerow(tmp[i, :])
    com = tmp[:, 0]
    return com

def save_corr(fn, corr):
    with open(fn, 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        for i in range(281):
            csvWriter.writerow(corr[:, i])

def process_string(data):
    tmp = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            tmp.append(data[i,j])
    tmp = np.array(tmp)
    tmp = np.reshape(tmp, (i+1, j+1))

    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            try:
                tmp[i,j] = tmp[i,j].astype(np.float16)
            except:
                tmp[i,j] = 77777
    data = tmp.astype(np.float)
    return data

def process_empty(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                data[i, j] = 99999
    return data

def split_label(data):
    x = data[:, :-6]
    y = data[:, -6:]
    return x, y

def select_corr(corr):
    all_corr = []
    tmp = []
    count = 0
    for i in range(6):
        for j in range(281):
            if abs(corr[i, j]) > 0.05:
                tmp.append(j)
                count += 1
        all_corr.append(tmp)
        tmp = []
        print(count)
        count = 0
    return all_corr
    
def main():
    train_data = np.load('./DT/clean_data/train.npy')
    test_data= np.load('./DT/clean_data/test.npy')
    
    # pre-processing
    train_data = process_string(train_data)
    test_data = process_string(test_data)
    train_data = process_empty(train_data)
    test_data = process_empty(test_data)
    
    # save pre-processing data
    with open('./DT/clean_data/pre_train.csv', 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        for row in train_data:
            csvWriter.writerow(row)
    with open('./DT/clean_data/pre_test.csv', 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        for row in test_data:
            csvWriter.writerow(row)
    
    train_x, train_y = split_label(train_data)
    test_x, test_y = split_label(test_data)
    
    corr = []
    tmp = []
    fn = 'DT/corr/corr.csv'
    for i in range(6):
        for j in range(train_x.shape[1]):
            tmp.append(correlation(train_x[:, j], train_y[:, i]))
        corr.append(tmp)
        tmp = []
    corr = np.array(corr)
    save_corr(fn, corr)
    print(corr)
    # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=49)
    param = {
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 50,
            'min_child_weight': 7,
            'max_delta_step': 0,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 1,
            'reg_lambda': 0.4,
            'scale_pos_weight': 0.8,
            'silent': 0,
            'objective': 'reg:squarederror',
            'missing': None,
            'eval_metric': 'rmse',
            'seed': 5269,
            'gamma': 0.2
    }

    model1 = xgb.XGBRegressor(**param)
    model2 = xgb.XGBRegressor(**param)
    model3 = xgb.XGBRegressor(**param)
    model4 = xgb.XGBRegressor(**param)
    model5 = xgb.XGBRegressor(**param)
    model6 = xgb.XGBRegressor(**param)
    
    eval_1 = modelfit(model1, 1, train_x, train_y[:,0], test_x, test_y[:,0])
    eval_2 = modelfit(model2, 2, train_x, train_y[:,1], test_x, test_y[:,1])
    eval_3 = modelfit(model3, 3, train_x, train_y[:,2], test_x, test_y[:,2])
    eval_4 = modelfit(model4, 4, train_x, train_y[:,3], test_x, test_y[:,3])
    eval_5 = modelfit(model5, 5, train_x, train_y[:,4], test_x, test_y[:,4])
    eval_6 = modelfit(model6, 6, train_x, train_y[:,5], test_x, test_y[:,5])
    
    # plot(model1)
    # plot(model2)
    # plot(model3)
    # plot(model4)
    # plot(model5)
    # plot(model6)

    """
    驗證 feature importances 的正確性
    """
    
    imp_1 = model1.feature_importances_
    imp_2 = model2.feature_importances_
    imp_3 = model3.feature_importances_
    imp_4 = model4.feature_importances_
    imp_5 = model5.feature_importances_
    imp_6 = model6.feature_importances_
    
    imp = [imp_1, imp_2, imp_3, imp_4, imp_5, imp_6]
    
    # train for common features
    common = list(find_common(imp))
    common = [int(i) for i in common]

    model1 = xgb.XGBRegressor(**param)
    model2 = xgb.XGBRegressor(**param)
    model3 = xgb.XGBRegressor(**param)
    model4 = xgb.XGBRegressor(**param)
    model5 = xgb.XGBRegressor(**param)
    model6 = xgb.XGBRegressor(**param)
    
    eval_1_ = modelfit(model1, 1, train_x[:, common], train_y[:,0], test_x[:, common], test_y[:,0])
    eval_2_ = modelfit(model2, 2, train_x[:, common], train_y[:,1], test_x[:, common], test_y[:,1])
    eval_3_ = modelfit(model3, 3, train_x[:, common], train_y[:,2], test_x[:, common], test_y[:,2])
    eval_4_ = modelfit(model4, 4, train_x[:, common], train_y[:,3], test_x[:, common], test_y[:,3])
    eval_5_ = modelfit(model5, 5, train_x[:, common], train_y[:,4], test_x[:, common], test_y[:,4])
    eval_6_ = modelfit(model6, 6, train_x[:, common], train_y[:,5], test_x[:, common], test_y[:,5])

    all_mae = np.array([eval_1, eval_2, eval_3, eval_4, eval_5, eval_6])
    all_mae_ = np.array([eval_1_, eval_2_, eval_3_, eval_4_, eval_5_, eval_6_])
    avg_mae = np.mean(all_mae, axis=0)
    avg_mae_ = np.mean(all_mae_, axis=0)

    ind_1 = save_imp('imp/f1.csv', imp_1)
    ind_2 = save_imp('imp/f2.csv', imp_2)
    ind_3 = save_imp('imp/f3.csv', imp_3)
    ind_4 = save_imp('imp/f4.csv', imp_4)
    ind_5 = save_imp('imp/f5.csv', imp_5)
    ind_6 = save_imp('imp/f6.csv', imp_6)    
    
    model1 = xgb.XGBRegressor(**param)
    model2 = xgb.XGBRegressor(**param)
    model3 = xgb.XGBRegressor(**param)
    model4 = xgb.XGBRegressor(**param)
    model5 = xgb.XGBRegressor(**param)
    model6 = xgb.XGBRegressor(**param)

    eval_1_ = modelfit(model1, 1, train_x[:, ind_1], train_y[:,0], test_x[:, ind_1], test_y[:,0])
    eval_2_ = modelfit(model2, 2, train_x[:, ind_2], train_y[:,1], test_x[:, ind_2], test_y[:,1])
    eval_3_ = modelfit(model3, 3, train_x[:, ind_3], train_y[:,2], test_x[:, ind_3], test_y[:,2])
    eval_4_ = modelfit(model4, 4, train_x[:, ind_4], train_y[:,3], test_x[:, ind_4], test_y[:,3])
    eval_5_ = modelfit(model5, 5, train_x[:, ind_5], train_y[:,4], test_x[:, ind_5], test_y[:,4])
    eval_6_ = modelfit(model6, 6, train_x[:, ind_6], train_y[:,5], test_x[:, ind_6], test_y[:,5])
    
    all_mae = np.array([eval_1, eval_2, eval_3, eval_4, eval_5, eval_6])
    all_mae_ = np.array([eval_1_, eval_2_, eval_3_, eval_4_, eval_5_, eval_6_])
    avg_mae = np.mean(all_mae, axis=0)
    avg_mae_ = np.mean(all_mae_, axis=0)

    high_corr = select_corr(corr)
    model1 = xgb.XGBRegressor(**param)
    model2 = xgb.XGBRegressor(**param)
    model3 = xgb.XGBRegressor(**param)
    model4 = xgb.XGBRegressor(**param)
    model5 = xgb.XGBRegressor(**param)
    model6 = xgb.XGBRegressor(**param)

    eval_1_ = modelfit(model1, 1, train_x[:, high_corr[0]], train_y[:,0], test_x[:, high_corr[0]], test_y[:,0])
    eval_2_ = modelfit(model2, 2, train_x[:, high_corr[1]], train_y[:,1], test_x[:, high_corr[1]], test_y[:,1])
    eval_3_ = modelfit(model3, 3, train_x[:, high_corr[2]], train_y[:,2], test_x[:, high_corr[2]], test_y[:,2])
    eval_4_ = modelfit(model4, 4, train_x[:, high_corr[3]], train_y[:,3], test_x[:, high_corr[3]], test_y[:,3])
    eval_5_ = modelfit(model5, 5, train_x[:, high_corr[4]], train_y[:,4], test_x[:, high_corr[4]], test_y[:,4])
    eval_6_ = modelfit(model6, 6, train_x[:, high_corr[5]], train_y[:,5], test_x[:, high_corr[5]], test_y[:,5])

    all_mae_ = np.array([eval_1_, eval_2_, eval_3_, eval_4_, eval_5_, eval_6_])
    avg_mae_ = np.mean(all_mae_, axis=0)

    K_list = [110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50,
              45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
    K_list = K_list[::-1]
    eval_avg = []

    # # 用後30%重要的特徵評估結果
    # eval_1 = feature_validation(model1, 1, 0.3, X, Y, imp_1)
    # eval_2 = feature_validation(model2, 2, 0.3, X, Y, imp_2)
    # eval_3 = feature_validation(model3, 3, 0.3, X, Y, imp_3)
    # eval_4 = feature_validation(model4, 4, 0.3, X, Y, imp_4)
    # eval_5 = feature_validation(model5, 5, 0.3, X, Y, imp_5)
    # eval_6 = feature_validation(model6, 6, 0.3, X, Y, imp_6)
    # eval_avg.append(np.mean([eval_1, eval_2, eval_3, eval_4, eval_5, eval_6], axis=0))
    # # 用前30%重要的特徵評估結果
    # eval_1 = cus_feature_validation(model1, 1, 0.3, X, Y, imp_1)
    # eval_2 = cus_feature_validation(model2, 2, 0.3, X, Y, imp_2)
    # eval_3 = cus_feature_validation(model3, 3, 0.3, X, Y, imp_3)
    # eval_4 = cus_feature_validation(model4, 4, 0.3, X, Y, imp_4)
    # eval_5 = cus_feature_validation(model5, 5, 0.3, X, Y, imp_5)
    # eval_6 = cus_feature_validation(model6, 6, 0.3, X, Y, imp_6)
    # eval_avg.append(np.mean([eval_1, eval_2, eval_3, eval_4, eval_5, eval_6], axis=0))
    for K in K_list:
        # imp = model.estimators_[0].feature_importances_
        # feature_validation(model, 1, K, X, Y, imp)
        print('K:',K)
        eval_1 = each_feature_validation(model1, 1, K, X, Y, imp_1)
        eval_2 = each_feature_validation(model2, 2, K, X, Y, imp_2)
        eval_3 = each_feature_validation(model3, 3, K, X, Y, imp_3)
        eval_4 = each_feature_validation(model4, 4, K, X, Y, imp_4)
        eval_5 = each_feature_validation(model5, 5, K, X, Y, imp_5)
        eval_6 = each_feature_validation(model6, 6, K, X, Y, imp_6)
        eval_avg.append(np.mean([eval_1, eval_2, eval_3, eval_4, eval_5, eval_6], axis=0))
    eval_avg = np.array(eval_avg)
    # print('eval_avg:', eval_avg)
    
    print("train:", train_x.shape)
    print("test:", test_x.shape)

    train_mae = plt.plot(K_list, eval_avg[:,0], label='train_mae')
    plt.legend( loc='best')
    plt.title("train evaluation")
    plt.xlabel("# of Features")
    plt.ylabel("Error")
    plt.xticks(np.arange(5, 115, 10))
    plt.axis('auto')
    plt.show()

    test_mae = plt.plot(K_list, eval_avg[:,1], label='test_mae')
    plt.legend( loc='best')
    plt.title("test evaluation")
    plt.xlabel("# of Features")
    plt.ylabel("Error")
    plt.xticks(np.arange(5, 115, 10))
    plt.axis('auto')
    plt.show()
    

if  __name__ == '__main__':
    main()


