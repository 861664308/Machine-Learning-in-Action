# -*- coding: utf-8 -*-
'''
利用sklearn库的朴素贝叶斯算法做新闻分类,一共是18846条新闻，标签是0-19个数字，一共20类
'''
from sklearn.datasets import fetch_20newsgroups #从skleran自带的datasets导入新闻数据抓取器
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer #导入文本特征向量化模块
from sklearn.naive_bayes import MultinomialNB #导入朴素贝叶斯模型
from sklearn.metrics import classification_report
#获取数据
news = fetch_20newsgroups(subset='all')
print('样本数='+str(len(news.data)))
#数据预处理：训练集和测试集分割，文本特征向量化
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target,
                test_size=0.25, random_state=33) #随机采样25%的数据作为测试集

#文本特征向量化
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
#多项式朴素贝叶斯分类器
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_predict = mnb.predict(X_test)
#获取结果报告
print('The Accuracy of Naive Bayes Classifier is: ' + str(mnb.score(X_test, y_test)))
print(classification_report(y_test, y_predict, target_names=news.target_names))
