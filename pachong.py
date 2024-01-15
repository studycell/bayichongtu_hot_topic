import os
import re  # 正则表达式提取文本
import time

from jsonpath import jsonpath  # 解析json数据
import requests  # 发送请求
import pandas as pd  # 存取csv文件
import datetime  # 转换时间用
def trans_time(v_str):
	"""转换GMT时间为标准格式"""
	GMT_FORMAT = '%a %b %d %H:%M:%S +0800 %Y'
	timeArray = datetime.datetime.strptime(v_str, GMT_FORMAT)
	ret_time = timeArray.strftime("%Y-%m-%d %H:%M:%S")
	return ret_time

# 请求头
headers = {
    "user-agent":'''Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36''',
    "cookie": '''SINAGLOBAL=5269827683029.511.1616922078251; XSRF-TOKEN=rgKotRXhifDzF-Ub4kzLLeIl; SUB=_2AkMSPkvQf8NxqwFRmfoUy2jkb4l3zADEieKkYroLJRMxHRl-yT9kqkMStRB6Ob5lP3Zy4UlkWejTc3OkJlyOrZXt66OC; SUBP=0033WrSXqPxfM72-Ws9jqgMF55529P9D9W5h9sviI8Jkvbc6YROF6Kyo; WBPSESS=V0zdZ7jH8_6F0CA8c_ussTY6i8IeXMpqf9IuglJlPfwDS0OSDNl91pkKLJb0GdrNBwTNP4LHXwuQVGWmZ3AzQkGjLd7HN4ExGqxRm_jSbY4fOJfrFZ8aIMaxt6DsVDjvxZUXIs5dVzGNkUBE-8HV-XM4Kuy7UqZjt4KEn1sBeZQ=; _s_tentry=weibo.com; Apache=6207771035379.399.1700971761907; ULV=1700971761924:1:1:1:6207771035379.399.1700971761907:''',
    "accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-encoding":"gzip, deflate, br",
    "accept-language":"zh-CN,zh;q=0.9,ko;q=0.8,en;q=0.7",
    "cache-control":"max-age=0",
    "sec-ch-ua":'''"Google Chrome";v="107", "Chromium";v="107", "Not=A?Brand";v="24"''',
    "sec-fetch-dest" :"document",
    "sec-fetch-site": "none",
    "sec-fetch-user":"?1",
    "upgrade-insecure-requests": "1",
    "sec-fetch-mode":"navigate",
}

# 转发数

reposts_count_list = []

# 评论数
comments_count_list = []
# 点赞数
attitudes_count_list = []
time_list = []
text2_list = []
id_list = []
bid_list = []
author_list = []


def pachong(date_list):

	for date1 in date_list:
		time.sleep(1)
		for page in range(1, 50):
			time.sleep(1)
			# 请求地址
			url = 'https://m.weibo.cn/api/container/getIndex'
			# 请求参数
			params = {
				"containerid": "100103type=1&q=巴以冲突",
				"page_type": "searchall",
				"page": page
			}
			url1 = 'https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3D%E5%B7%B4%E4%BB%A5%E5%86%B2%E7%AA%81:' + date1 + '&page_type=searchall'
			# 发送请求
			#r = requests.get(url, headers=headers, params=params)
			r = requests.get(url1, headers=headers)
			# 解析json数据
			try:
				cards = r.json()["data"]["cards"]
			except:
				print(page)
			# 转发数

			reposts_count_list = jsonpath(cards, '$..mblog.reposts_count')

			# 评论数
			comments_count_list = jsonpath(cards, '$..mblog.comments_count')
			# 点赞数
			attitudes_count_list = jsonpath(cards, '$..mblog.attitudes_count')
			time_list = jsonpath(cards, '$..mblog.created_at')
			text2_list = jsonpath(cards, '$..mblog.text')
			id_list = jsonpath(cards, '$..mblog.id')
			bid_list = jsonpath(cards, '$..mblog.bid')
			author_list = jsonpath(cards, '$..mblog.author')

			# 把列表数据保存成DataFrame数据
			try:

				df = pd.DataFrame(
				{

					'微博id': id_list,
					'微博bid': bid_list,
					'微博作者': author_list,
					'发布时间': time_list,
					'微博内容': text2_list,
					'转发数': reposts_count_list,
					'评论数': comments_count_list,
					'点赞数': attitudes_count_list,
					}
				)

			except:
				print(123)
				# 删除重复数据
			df.drop_duplicates(subset=['微博bid'], inplace=True, keep='first')
			# 再次保存csv文件
			df.to_csv('test.csv', index=False, encoding='utf_8_sig', mode='a', header=False)

if __name__ == "__main__":
	date_list = []
	for j in range(10, 12):
		for i in range(1, 29):
			if j != 10:
				date_list.append('2023-' + str(j) + '-' + str(i))
			elif i > 8:
				date_list.append('2023-' + str(j) + '-' + str(j))

	pachong(date_list)
