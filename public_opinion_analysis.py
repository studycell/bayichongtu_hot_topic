#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/12/4 12:16
# @Author   :studycell
# @File     : public_opinion_analysis.py
# @Description:

import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
import pyLDAvis.gensim
from gensim import models

warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pandas as pd
from ast import literal_eval
# 分别用PCA和t-SNE降维看效果
from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib.pyplot as plt
from adjustText import adjust_text
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
from collections import defaultdict
from sklearn.cluster import KMeans
import pickle
from sklearn.decomposition import PCA
#PATH = "words.txt"

#file_object2 = open(PATH, errors='ignore').read().split('\n')  # 一行行的读取内容

def infile(fliepath):
    #输入分词好的TXT，返回train
    train = []
    fp = open(fliepath,'r')
    #fp = open(fliepath,'r', encoding='utf-8')
    for line in fp:
        new_line=[]
        if len(line)>1:
            line = line.strip().split(' ')
            for w in line:
                w.encode(encoding='utf-8')
                new_line.append(w)
        if len(new_line)>1:
            train.append(new_line)
    return train


num_topics = 15

#data_set = infile("words.txt")
data_set = infile("tokenized_text.txt")
dictionary = corpora.Dictionary(data_set)  # 构建词典
corpus = [dictionary.doc2bow(text) for text in data_set]  #表示为第几个单词出现了几次
#print(corpus)
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30,random_state = 1)   #分为10个主题

#file = open('ldamodel.pickle', 'wb')
file = open('ldamodel.pickle', 'wb')
pickle.dump(ldamodel, file)
file.close()


def kmeans(DocumentTopicMatrix, n_clusters):
    X = DocumentTopicMatrix
    #print(type(X1))
    # 正式定义模型
    model1 = KMeans(n_clusters=n_clusters)
    # 跑模型
    model1.fit(X)
    # 需要知道每个类别有哪些参数
    C_i = model1.predict(X)
    #print(C_i)
    np.save("julei.npy", C_i)
    f = open('rfc.pickle', 'wb')
    pickle.dump(model1, f)
    f.close()
    # 还需要知道聚类中心的坐标
    Muk = model1.cluster_centers_
    # 画图
    plt.scatter(X[:, 0], X[:, 1], c=C_i, cmap=plt.cm.Paired)
    # 画聚类中心
    plt.scatter(Muk[:, 0], Muk[:, 1], marker='*', s=60)
    for i in range(n_clusters):
        plt.annotate('中心' + str(i + 1), (Muk[i, 0], Muk[i, 1]))
    plt.show()


def get_document_topics(number_of_doc):


    """start：文档-主题分布矩阵，使用DocumentTopicMatrix[][]存储"""
    DocumentTopicMatrix = np.zeros([number_of_doc, num_topics])  # 存储文档-主题分布,len(doc)就是文档的总个数

    # 存储文档-主题分布矩阵
    for id in range(len(corpus)):
        doc_top = ldamodel.get_document_topics(corpus[id], per_word_topics=False)
        m = 0
        for x in doc_top:
            DocumentTopicMatrix[id][m] = x[1]
            m += 1

    #kmeans(DocumentTopicMatrix)
    # 保存文档-主题分布矩阵
    np.save("DocumentTopicMatrix.npy", DocumentTopicMatrix)
    np.savetxt("DocumentTopicMatrix.txt", DocumentTopicMatrix)
    #print("文档-主题分布矩阵保存成功")
    """end：文档-主题分布矩阵，使用DocumentTopicMatrix[][]存储"""

def pca_for_matrix(DocumentTopicMatrix):
    pca = PCA(n_components=2)
    pca.fit(DocumentTopicMatrix)
    np.save("DocumentTopicMatrix2d.npy", pca.transform(DocumentTopicMatrix))

#计算困惑度
def perplexity(num_topics):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=15))
    print(ldamodel.log_perplexity(corpus))
    return ldamodel.log_perplexity(corpus)
#计算coherence
def coherence(num_topics):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30,random_state = 1)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
    ldacm = CoherenceModel(model=ldamodel, texts=data_set, dictionary=dictionary, coherence='c_v')
    print(ldacm.get_coherence())
    return ldacm.get_coherence()
if __name__ == "__main__":

    #d = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    # df = pd.read_csv("data5.csv")
    #df = pd.read_csv("data5.csv")
    get_document_topics(4299)
    pca_for_matrix(np.load("DocumentTopicMatrix.npy"))
    kmeans(np.load("DocumentTopicMatrix2d.npy"), 7)
    #pyLDAvis.show(d)

'''
    x = range(1, 20)
    z = [pow(2, perplexity(i)) for i in x]  # 如果想用困惑度就选这个
    y = [coherence(i) for i in x]
    plt.plot(x, z)
    plt.xlabel('number of topics')
    plt.ylabel('perplexity')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.show()
'''





