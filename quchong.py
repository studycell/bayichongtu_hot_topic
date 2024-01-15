import pandas as pd

def paixu():
    df = pd.read_csv('data.csv')
    res = df.sort_values(by='发布时间', ascending=True)
    res.reset_index(drop=True, inplace=True)
    res = res.drop(columns='Unnamed: 0')
    res.to_csv('data2.csv')

def quchong():
    df = pd.read_csv('data2.csv')
    df.drop_duplicates(subset=['帖子内容', '发布时间'] ,keep = 'first', inplace = True)
    df = df.drop(columns='Unnamed: 0')
    df.to_csv('data3.csv')

quchong()



