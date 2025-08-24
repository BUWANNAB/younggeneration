# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:44:14 2025

@author: I is God
"""

import nltk
from nltk.book import *


#（2）访问在线古腾堡语料库，获取《伤寒杂病论》《孔雀东南飞》等网络数据资源。
import nltk
nltk.corpus.gutenberg.fileids()  # 获取古腾堡语料库的所有文本查找某个文件
emma = nltk.corpus.gutenberg.words('伤寒杂病论.txt') 
emma = nltk.corpus.gutenberg.words('孔雀东南飞.txt') 


#（3）构建一个本地语料库，并做分析。
#1. 构建作品集语料库
import nltk
from nltk.book import *
from nltk.corpus import PlaintextCorpusReader
corpus_root = 'D:\project\data2'  # 本地存放金庸先生部分作品集文本的目录
wordlists = PlaintextCorpusReader(corpus_root, '.*')  # 获取语料库中的文本标识列表
print(wordlists.fileids())  # 获取文件列表


#2. 读取本地语料
#统计《神雕侠侣》语料中总共用词量和平均每个词的使用次数
import nltk
from nltk.book import *
with open('E:/APP/QQNT/金庸-神雕侠侣.txt', 'r') as f:  # 打开文本
    str = f.read()  # 读取文本
    len(set(str))  # 统计用词量
    len(str)/len(set(str))  # 平均每个词的使用次数
print(len(set(str)))  # 统计用词量
print(len(str)/len(set(str)))  # 平均每个词的使用次数
#查看《神雕侠侣》文本中的“小龙女”“杨过”“雕”“侠”字的使用次数
print(str.count('小龙女'))  # 使用次数
print(str.count('杨过'))  # 使用次数
print(str.count('雕'))  # 使用次数
print(str.count('侠'))  # 使用次数
#查看《神雕侠侣》部分文本
str[5394:6008]  # 查看《神雕侠侣》部分文本
