#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/12/7 14:36
# @Author   :studycell
# @File     : predict_SVM.py
# @Description:


from public_opinion_analysis import infile
from gensim import corpora
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
import pickle
import jieba.posseg as pseg
import numpy as np
from ast import literal_eval


def fenci(inputs):
    # 分词
    words = pseg.cut(str(inputs))
    # 取名词
    # print(words)
    noun_words = []
    for word, flag in words:
        if flag == 'n':
            noun_words.append(word)

    return noun_words


def predict_svm():
    f1 = open('svm.pickle', 'rb')  # 注意此处model是rb
    s1 = f1.read()
    svmmodel = pickle.loads(s1)
    f2 = open('ldamodel.pickle', 'rb')
    s2 = f2.read()
    ldamodel = pickle.loads(s2)
    # 之后就可以使用model了

    # 输入文档
    print("请输入文档：")
    text = input().split('\n')


    #将文档转换成文档-主题分布矩阵
    words = fenci(text)

    f = open("input_words.txt", 'w')
    outstr = ""
    #list1 = literal_eval(words)
    words = " ".join(words)
    f.write(words + '\n')
    f.close()


    data_set = infile("words.txt")
    dictionary = corpora.Dictionary(data_set)  # 构建词典
    data_set2 = infile("input_words.txt")
    corpus = [dictionary.doc2bow(text) for text in data_set2]

    num_topics = 15
    DocumentTopicVector = np.zeros([1, num_topics])

    for id in range(len(corpus)):
        doc_top = ldamodel.get_document_topics(corpus[id], per_word_topics=False)
        m = 0
        for x in doc_top:
            DocumentTopicVector[0][m] = x[1]
            m += 1

    #载入SVM预测
    result = svmmodel.predict(DocumentTopicVector)
    hot_topics = np.load("hot_topics.npy")
    print(hot_topics[result])


predict_svm()

