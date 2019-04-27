# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:16:10 2019

@author: 鲁金川
"""
#计算给定数据集的香农熵

import numpy as np

def calShannonEnt(dataSet): #输入参数是数据集
    numEntries = len(dataSet) #获取样本数量
    labelCounts = {} #初始化标签字典
    for featVec in dataSet: #遍历数据集
        currentLabel = featVec[-1] #数据的最后一列是标签，所以取出来
        if currentLabel not in labelCounts.keys(): 
            #如果标签字典没有当前标签，把当前标签作为新的键并设置初始键值为0
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 #当前标签的键值加一
        shannonEnt = 0 #初始化香农熵为0
        for key in labelCounts: #遍历每一标签
            prob = float(labelCounts[key]) / numEntries #计算该标签的概率
            shannonEnt -= prob * np.math.log(prob, 2) #把该标签的信息加到香农熵里面
    return shannonEnt 


#建立createDataSet函数，加载书中的鱼鉴定数据
    
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


#按照给定特征划分数据集
    
def splitDataSet(dataSet, axis, value):
    #输入参数为:待划分的数据集，划分数据集的特征(比如说0代表第0个特征)，需要返回的特征的值
    #函数的功能是把dataSet中第aixs个特征为value的数据选出来
    retDataSet = [] #创建新的list
    for featVec in dataSet: #遍历数据集
        if featVec[axis] == value: #将符合特征的数据抽取出来
            '''
            下面两行代码的功能是把满足条件的数据去除掉aixs属性，
            因为该属性是相同的，所以再进一步时候，可以不予考虑
            '''
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1 :])
            retDataSet.append(reducedFeatVec)#把截取后的数据添加到retDataSet中
    return retDataSet

#选择最好的数据集划分方式
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 #获取特征数，第一行除去标签全是特征
    baseEntropy = calShannonEnt(dataSet) #计算整体数据集的香农熵
    bestInfoGain = 0 #初始化最优信息增益为0
    bestFeature = -1 #初始化最优特征为-1
    for i in range(numFeatures): #遍历所有的特征
        featList = [example[i] for example in dataSet] #取出每个样本的第i个特征
        uniqueVals = set(featList) #提取该特征的所有可能值，即把featList中重复的元素去掉
        newEntropy = 0 #初始化经验条件熵
        for value in uniqueVals: #遍历该特征所有的可能值
            subDataSet = splitDataSet(dataSet, i, value) #划分后的子集
            prob = len(subDataSet) / float(len(dataSet)) #计算概率
            newEntropy += prob * calShannonEnt(subDataSet) #根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy #计算信息增益
        #如果信息增益大于最优信息增益，更细最优信息增益和最优特征
        if (infoGain > bestInfoGain): 
            bestInfoGain = infoGain
            bestFeature = i 
    return bestFeature