#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/12/12 16:18
# @Author   :studycell
# @File     : get_hot_topic.py
# @Description: get hot topics from kmeans

import pandas as pd
import numpy as np
import pickle
import re
import heapq

clusters_num = 7
topic_nums = 15
num_words = 10


def extract_keywords(input_string):
    keyword_weight_list = re.findall(r'([\d.]+)\*"(.*?)"', input_string)
    return [(float(weight), keyword) for weight, keyword in keyword_weight_list]

def get_wordsweight_and_words(topic_words):
    words_weight = np.empty(shape=(topic_nums, num_words))
    #words = np.empty(shape=(topic_nums, num_words))
    #words = np.array([topic_nums, num_words], dtype=str)
    words = []
    words_total = []
    cnt2 = 0
    for topic in topic_words:
        input_str = topic[1]
        keywords = extract_keywords(input_str)
        cnt = 0
        tmp_words = []
        for weight, keyword in keywords:
            words_weight[cnt2, cnt] = weight
            tmp_words.append(keyword)
            words_total.append(keyword)
            cnt += 1
        cnt2 += 1
        words.append(tmp_words)
    return words_weight, words, words_total


f = open('ldamodel.pickle', 'rb')
s = f.read()
ldamodel = pickle.loads(s)
julei_result = np.load("julei.npy")  #[8 3 0 ... 1 8 0]
#print(ans)
topic_words = ldamodel.print_topics(num_words = 10) #每个主题以10个单词表示
#[(0, '0.031*"计划" + 0.020*"会晤" + 0.013*"事情" + 0.013*"2020" + 0.012*"世面" + 0.011*"能否" + 0.009*"哭泣" + 0.009*"大多数" + 0.008*"关键时刻" + 0.007*"美好"'),
DocumentTopicMatrix = np.load("DocumentTopicMatrix.npy")
#print(topic_words)
words_weight, words, words_total = get_wordsweight_and_words(topic_words)
words_total = list(set(words_total))    #去重
'''
目的：生成kmeans聚类结果的每个簇对应的词汇
每个簇由若干个文档组成
每个文档由15个LDA主题词带权表示
每个主题词由10个词汇带权表示

算法：
    定义每个簇对应的词汇矩阵<type=float> 拥有所有词汇 7*(15*10)
    遍历每篇文档 算出每篇文档对应的每个主题词对应的每个词汇权重
    词汇权重=LDA的概率分布矩阵中每个主题权重*对应矩阵的词汇权重
'''



#cluster_matrix = np.zeros((clusters_num, topic_nums, num_words))
cluster_matrix = np.zeros((clusters_num, len(words_total)))
for index in range(len(julei_result)):      #24928
    for topic_index in range(topic_nums):
        for word_index in range(num_words):
            #cluster_matrix[julei_result[index], topic_index, word_index] += \
            #DocumentTopicMatrix[index, topic_index] * \
            #words_weight[topic_index,word_index]
            pos = words_total.index(words[topic_index][word_index])
            cluster_matrix[julei_result[index], pos] += \
            DocumentTopicMatrix[index, topic_index] * \
            words_weight[topic_index, word_index]

hot_topics = []
for index in range(clusters_num):
    list1 = cluster_matrix[index]
    pos = heapq.nlargest(10, range(len(list1)), list1.__getitem__)
    tmp = []
    for i in pos:
        tmp.append(words_total[i])
    hot_topics.append(tmp)

print(hot_topics)
np.save("hot_topics.npy", hot_topics)

