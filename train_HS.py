
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

CLASSES = ['G8', 'G11', 'G15', 'G17', 'G32', 'G34', 'G48']
DATASIZE = 500

if __name__ == '__main__':
    dataset_x = []
    dataset_y = []
    train_x = []
    train_y = []
    
    print('loading data...')
    for i, c in enumerate(CLASSES):
        ROOT = 'data/HS/preprocess/%s' % c
        files = os.listdir(ROOT)
        for f in files:
            if f[-4:] == '.txt':
                path = ROOT + '/' + f
                # print(path)
                temp = np.loadtxt(path, dtype=float)
                dataset_x.append(temp)
                dataset_y.append(i)
    print('loading completed.')

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)
    # dataset_x = np.reshape(dataset_x, (-1, DATASIZE))
    print('dataset_x shape:', dataset_x.shape)
    print('dataset_y shape:', dataset_y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.25, random_state=50)
    print('train data shape:', X_train.shape)
    print('test data shape:', X_test.shape)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    print('acc of Random Forest Classifier on testing set', rfc.score(X_test, y_test))

    xgbc = XGBClassifier()
    xgbc.fit(X_train, y_train)
    print('acc of eXtreme Gradient Boosting Classifier on testing set', xgbc.score(X_test, y_test))            
                
                