import pandas as pd
import numpy as np

def fun1():
    df1 = pd.read_csv("../weibo_keyword_crawl-main/weibo_keyword_crawl-main/data.csv",encoding='gb18030')
    df2 = pd.DataFrame([df1['帖子内容'], df1['发布时间']]).T
    df2.to_csv('data.csv')


#统一格式 保存字数大于100内容
def deal_bayi_weibo():
    df1 = pd.read_csv("巴以冲突_weibo.csv")
    df2 = pd.DataFrame(columns=['帖子内容', '发布时间'])
    for index in range(len(df1)):
        contents = df1.loc[index, '微博正文']
        #print(len(contents))
        try:
            if len(contents) >= 100:
                new_row = pd.Series([df1.loc[index, '微博正文'], df1.loc[index, '发布时间']], index=['帖子内容', '发布时间'])
                df2 = df2.append(new_row, ignore_index=True)
        except:
            print("TypeError: object of type 'float' has no len()")
    df2.to_csv("data5.csv")

def deal_bayi_weibo1():
    df1 = pd.read_csv("巴以冲突_weibo.csv")
    df2 = pd.DataFrame(columns=['帖子内容', '发布时间'])
    for index in range(len(df1)):
        contents = df1.loc[index, '微博正文']
        #print(len(contents))
        try:
            if len(contents) >= 10 and df1.loc[index, '发布时间'] <= "2023-11-31" and df1.loc[index, '发布时间'] >= "2023-11-15":
                new_row = pd.Series([df1.loc[index, '微博正文'], df1.loc[index, '发布时间']], index=['帖子内容', '发布时间'])
                df2 = df2.append(new_row, ignore_index=True)
        except:
            print("TypeError: object of type 'float' has no len()")
    df2.to_csv("data_11_15_31.csv")

def readcsv_and_count_length(filepath):
    df = pd.read_csv(filepath)
    print(len(df))

deal_bayi_weibo1()
#readcsv_and_count_length("巴以冲突_weibo.csv")
