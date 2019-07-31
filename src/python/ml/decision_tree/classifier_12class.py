# -*- coding: UTF-8 -*-
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import time

def loadData(filename):
    with open(filename,'r') as file:
        lines = []
        for line in file.readlines():
            if line != '\n':
                lines.append(line)
        dataset = [ [] for i in range(len(lines)-1)]
        x = []
        y = []
        category = []
        #print(lines[0])
        random.shuffle(lines)
        #print(lines[0])
        for i in range(len(dataset)):
            dataset[i][:] = (item for item in lines[i].strip().split('='))   # 逐行读取数据
            x.append(dataset[i][0])
            y.append(float(dataset[i][1]))
            cls = dataset[i][2].strip().split('-')
            category.append(int(cls[0])*2 + int(cls[1]))
        #print("dateset:",dataset)
        #y = [[] for i in range(len(lines)-1)]
        #print('x',type(x[0]))
        #print('y',y)
      # print('category', category)
        #print('cls1', cls1)
        print('gemfield category', category)
        x_new = []

        for i in range(len(x)):
            x_temp = []
            tmp = eval(x[i])
            #print('tmp', tmp[0][0])
            for j in tmp:
                x_temp.append(j[0]*j[1])
            x_new.append(x_temp)
        #print(x_new)
    return x_new, y, category

def classifier(X, y, cls, clf):
    X = np.array(X)
    y = np.asarray(y)
    #X = preprocessing.normalize(X, norm='l2')
    #min_max_scaler = preprocessing.MinMaxScaler()
    #X = min_max_scaler.fit_transform(np.transpose(X))
    #print(X)
    # y_nor = []
    # for yi in y:
    #     yi = float(yi - y.mean())/y.std()
    #     y_nor.append(yi)
    # print(np.shape(y_nor))
    #y = y_nor
    #y = min_max_scaler.fit_transform(np.transpose(y))
    #y = preprocessing.normalize(np.transpose(y), norm='l2')

    # 划分训练集和测试
    X_train = X[:-100]
    X_test = X[-100:]
    cls_train = cls[:-100]
    cls_test = cls[-100:]

    # 从仍然需要对训练和测试的特征数据进行标准化
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    clf = OneVsRestClassifier(clf)
    # 训练模型
    clf.fit(X_train, cls_train)
    # 预测样本
    cls_pred = clf.predict(X_test) 
    print('gemfield cls prediction->right:%d, wrong:%d' %((cls_test==cls_pred).sum(),(cls_test!=cls_pred).sum()))
    return cls_test, cls_pred

if __name__ == '__main__':

    start = time.time()
    [X, y, cls] = loadData('test.dataset')

    #模型选择
    #clf = svm.LinearSVC()
    #clf = OneVsRestClassifier(svm.LinearSVC())
    clf = tree.DecisionTreeClassifier()  #88-90%
    #clf = AdaBoostClassifier(n_estimators=50) #87%
    #clf = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0) #87%

    [cls_test, cls_pred] = classifier(X, y, cls, clf)

    # 时间
    print('========================================================')
    end = time.time()
    print('gemfield phase:', end - start)
