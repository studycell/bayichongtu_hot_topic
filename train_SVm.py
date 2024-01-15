#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/12/8 18:42
# @Author   :studycell
# @File     : train_SVm.py
# @Description:
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle


def train_svm():
    #  load iris dataset


    iris = np.load("DocumentTopicMatrix.npy")
    n_samples, n_features = iris.shape
    target = np.load("julei.npy")


    # split train test

    train_data, test_date = train_test_split(iris, random_state=1, train_size=0.7, test_size=0.3)
    train_label, test_label = train_test_split(target, random_state=1, train_size=0.7, test_size=0.3)

    # create SVM

    classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')
    classifier.fit(train_data, train_label.ravel())
    f = open('svm.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

    # predict

    pre_train = classifier.predict(train_data)
    pre_test = classifier.predict(test_date)

    print("train:", accuracy_score(train_label, pre_train))
    print("test:", accuracy_score(test_label, pre_test))

    # confusion matrix

    confusion = confusion_matrix(test_label, pre_test)

    plt.imshow(confusion, cmap=plt.cm.Blues)

    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confusion))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    plt.xticks(indices, [0, 1, 2, 3, 4, 5, 6])
    plt.yticks(indices, [0, 1, 2, 3, 4, 5, 6])


    plt.colorbar()

    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('SVM Accuracy')

    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for first_index in range(len(confusion)):  # 第几行
        for second_index in range(len(confusion[first_index])):  # 第几列
            plt.text(first_index, second_index, confusion[first_index][second_index])
    # 在matlab里面可以对矩阵直接imagesc(confusion)

    # 显示
    plt.show()

train_svm()
