import jieba
import pandas as pd
import jieba.posseg as pseg
from wordcloud import WordCloud
import jieba.analyse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import nltk
from nltk import word_tokenize
import re
from ast import literal_eval


#处理的文件
#df = pd.read_csv("data5.csv")

#df = pd.read_csv("data_less_11.csv")
df = pd.read_csv("data_11_15_31.csv")
#fenci_PATH = 'words_date_less_11.csv'
fenci_PATH = 'words_data_11_15_31.csv'
#wordcloud_PATH = 'tfidf中文词云图早于11月份.jpg'
wordcloud_PATH = 'tfidf中文词云图11月份15_31.jpg'
#words_PATH = 'words_less_11.txt'
words_PATH = 'words_11_15_31.txt'
#data_PATH = 'data_less_11.csv'
data_PATH = 'data_11_15_31.csv'
#data4_PATH = 'data4_less_11.csv'
data4_PATH = 'data4_11_15_31.csv'



#创建停用词list
#INPUT:
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8', errors='ignore').readlines()]
    return stopwords

stopwords = stopwordslist("stop_words_ch.txt")  # 这里加载停用词的路径



#对句子进行分词
#INPUT:     string
#OUTPUT:    string
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                #outstr += " "
    return outstr

#分词
#INPUT:     pd.DataFrame(columns = ["帖子内容", "发布时间"])
#OUTPUT:    words.csv pd.DataFrame(columns = ["date", "words"])
def fenci():
    df2 = pd.DataFrame(columns=['date', 'words'])
    newtxt = []
    for index in range(len(df)):
        #inputs = df[index]['帖子内容']
        inputs = df.loc[index, '帖子内容']
        results = re.compile(r'##', re.S)
        dd = results.sub("", inputs)
        inputs = dd
        inputs = seg_sentence(inputs)
        '''
        for line in inputs:
            line_seg = seg_sentence(line)
            df2 = df2.append([line_seg, df.loc[index, '帖子内容']])
        '''
        #分词
        words = pseg.cut(inputs)
        #取名词
        #print(words)
        noun_words = []
        for word, flag in words:
            if flag == 'n':
                noun_words.append(word)

        #print(noun_words)
        new_row = pd.Series([df.loc[index, '发布时间'], noun_words], index=['date', 'words'])
        df2 = df2.append(new_row, ignore_index=True)
        #df2 = df2.append([df.loc[index, '发布时间'], noun_words])
        #print(df2)


    #df2.to_csv('words_date.csv')    #输出分词序列和对应内容发布时间
    #df2.to_csv('words_date_less_11.csv')  # 输出分词序列和对应内容发布时间
    df2.to_csv(fenci_PATH)

def generate_wordcloud():

    #df = pd.read_csv("data4.csv")
    df = pd.read_csv(data4_PATH)
    newtxt = ""
    words = []
    for index in range(len(df)):
        #list = df.loc[index, "words"]
        list = df.loc[index, "keys"]
        for word in list:
            words.append(word)

    newtxt = '/'.join(words)
    wordcloud = WordCloud(font_path="msyh.ttc", width=800, height=400).generate(newtxt)
    #wordcloud.to_file('中文词云图.jpg')
    #wordcloud.to_file('tfidf中文词云图早于11月份.jpg')
    wordcloud.to_file(wordcloud_PATH)

def generate_wordcloud_en():

    #df = pd.read_csv("data4.csv")
    f = open("tokenized_text.txt", 'r')
    newtxt = ""
    words = []
    for index in range(len(df)):
        #list = df.loc[index, "words"]
        list = f.readline()
        list = list.split(' ')
        for word in list:
            words.append(word)

    newtxt = '/'.join(words)
    wordcloud = WordCloud(font_path="msyh.ttc", width=800, height=400).generate(newtxt)
    #wordcloud.to_file('中文词云图.jpg')
    #wordcloud.to_file('tfidf中文词云图早于11月份.jpg')
    wordcloud.to_file("英文词云图.jpg")

def delete_http():
    df = pd.read_csv("data3.csv")
    for index in range(len(df)):
        contents = df.loc[index, "帖子内容"]
        results = re.compile(r'#', re.S)
        dd = results.sub("", contents)
        df["帖子内容"].loc[index] = dd
    df.to_csv("data3.csv")
#tf-idf
def extract_key_words():
    #df = pd.read_csv("data5.csv")
    df = pd.read_csv(data_PATH)
    df["keys"] = np.nan
    #print(df)
    for index in range(len(df)):
        contents = df.loc[index, "帖子内容"]
        df["keys"].loc[index] = jieba.analyse.extract_tags(contents, topK=30)
    #df.to_csv("data4.csv")
    df.to_csv(data4_PATH)

def fenci_words_to_txt():
    #df = pd.read_csv("data4.csv")
    df = pd.read_csv(data4_PATH)
    #f = open("words.txt", 'w')
    #f = open("words2.txt", 'w', encoding='utf-8')
    f = open(words_PATH, 'w')
    for index in range(len(df)):
        outstr = ""
        #words = df.loc[index, "words"]
        words = df.loc[index, "keys"]
        #list1 = literal_eval(df.loc[index, "words"])
        list1 = literal_eval(df.loc[index, "keys"])
        words = " ".join(list1)
        f.write(words + '\n')

    f.close()



if __name__ == "__main__":
    #print(stopwords)
    #print(seg_sentence("我在北京上学啊"))
    #delete_http()

    fenci()
    extract_key_words()
    generate_wordcloud()
    fenci_words_to_txt()
    generate_wordcloud_en()
