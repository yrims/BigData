# pre-processing the csv data

import os
import csv
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

CLASSES = ['G8', 'G11', 'G15', 'G17', 'G32', 'G34', 'G48', 'G49']

# PTC Positive Temperature Coefficient, I guess

class Data():

    def __init__(self, path, class_name, file_name):
        self.path = path
        self.data = {'time':[], 'data':[]}
        self.class_name = class_name
        self.file_name = file_name
        self.data_size = 0
        self.save_data_size = 500
        
    def preprocess(self):
        with open(self.path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for i, row in enumerate(csv_reader):
                
                if i > 1 and len(row[0]) == 8:
                    # time = int(row[0][0:2]) * 3600 + int(row[0][3:5]) * 60 + int(row[0][6:8])
                    time = row[0]
                    # print('time:', time)
                    # self.data['time'].append(i-1)
                    row = np.array(row[1:]).astype(float)
                    tc_mean = np.mean(row[1:])
                    self.data['data'].append(tc_mean)
                    # print('time:', time, 'mean:', tc_mean)
        self.data_size = len(self.data['data'])
                
    def sample_(self):
        sp = self.data['data'][:]
        sp_temp = [] 
        while(len(sp) < self.save_data_size):
            for i in range(1, len(sp)):
                sp_temp.append(sp[i-1])
                sp_temp.append((sp[i-1]+sp[i])/2)
                
            if len(sp_temp >= self.save_data_size):
                break

        sp = sp[0:self.save_data_size]

    def sample(self):
        x = np.linspace(0, self.data_size, num=self.data_size)
        fl = interp1d(x, self.data['data'], kind='linear')
        xint = np.linspace(x.min(), x.max(), self.save_data_size)
        yintl = fl(xint)
        self.data['data'] = yintl
        # print('hi')

    def save(self):
        
        # self.save_data = np.chararray((self.data_size, 2), itemsize=10, unicode=True)
        self.save_data = np.zeros((self.save_data_size))
        for j in range(self.save_data_size):
            # self.save_data[j, 0] = self.data['time'][j]
            self.save_data[j] = self.data['data'][j]

        save_path = 'data/HS/preprocess/%s' % self.class_name + '/' + self.file_name[:-4] + '.txt'
        if not os.path.exists('data/HS/preprocess/%s' % self.class_name):
            os.mkdir('data/HS/preprocess/%s' % self.class_name)

        # print('save_path:', save_path)
        np.savetxt(save_path, self.save_data, fmt='%5.10f', delimiter=',')

    def plot(self):
        fig = plt.figure()
        # plt.plot(self.data['time'], self.data['data'])
        plt.plot(self.data['data'])
        plt.ylabel('temporature')
        plt.xlabel('time')
        fig.savefig('data/HS/preprocess/%s' % self.class_name + '/' + self.file_name[:-4] + '.png')
        # plt.show()
        plt.close(fig)
        

if __name__ == '__main__':

    for c in CLASSES:
        ROOT = 'data/HS/%s' % c
        files = os.listdir(ROOT)
        for f in files:
            if f[-4:] == '.csv':
                path = ROOT + '/' + f
                print(path)
                data = Data(path, c, f)
                data.preprocess()
                data.sample()
                data.save()
                data.plot()
                