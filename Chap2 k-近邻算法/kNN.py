# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:45:19 2019

@author: 鲁金川
"""

import numpy as np
import operator #导入运算符模块

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#k-近邻算法
    
def classify0(inX, dataSet, labels, k):
    '''
    输入参数
    inX 用于分类的向量
    dataSet 训练样本集
    labels 训练集标签向量
    k 最近邻数目
    '''
    dataSetSize = dataSet.shape[0] #获取训练样本集的数量
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #np.tile将矩阵横向或纵向复制，这里是将inX复制成dataSetSize行，再减去训练样本集
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    #计算目标inX到训练样本集的每一个样本的距离
    sortedDistIndices = distances.argsort() #argsort把距离从小到大排序
    classCount = {} #初始化字典
    #for循环取出最小的k个
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]] #取出第k近的训练样本的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        #计算类别次数
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True)
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    return sortedClassCount[0][0]
    #返回次数最多的类别,即所要分类的类别

#创建file2matrix函数，将TXT的约会数据文件输出为训练样本矩阵和类标签向量
def file2matrix(filename):
    #输入参数为文件名
    fr = open(filename)
    arrayOLine = fr.readlines()
    #打开文件并按行读取文件
    numberOfLines = len(arrayOLine)#获取文件行数
    returnMat = np.zeros((numberOfLines, 3)) #创建返回的Numpy矩阵
    classLabelVector = [] #创建返回的label向量
    index = 0
    for line in arrayOLine: #按行读取文件
        line = line.strip() #截取掉所有的回车字符
        listFromLine = line.split('\t') 
        #用tab字符将上一行得到的整行数据分割成一个元素列表
        returnMat[index, :] = listFromLine[0 : 3] #选取前3个元素存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1])) 
        #-1表示选取该行的最后一个元素，即标签
        index += 1
    return returnMat, classLabelVector

#归一化特征 （当前值 - 最小值）/（最大值 - 最小值）
def autoNorm(dataSet):
    #输入参数是样本集
    minVals = dataSet.min(0) #参数0表示选择列的最小值而不是行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

#分类器针对约会网站的测试代码
    
def datingClassTest():
    hoRatio = 0.1 #设置用于测试的数据比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') #读取文件
    normMat, ranges, minVals = autoNorm(datingDataMat) #数据归一化
    m = normMat.shape[0] #获取样本数目
    numTestVeces = int(m * hoRatio) #获取用于测试的样本数量
    errorCount = 0 #初始化错误分类的数目
    for i in range(numTestVeces):
        #使用3近邻分类，对最前面的样本进行测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVeces: m, :],
                                     datingLabels[numTestVeces : m], 3)
        #打印分类器结果和真实标签
        print('the classifier came back with: '+str(classifierResult)+'    '
              'the real answer is: ' + str(datingLabels[i]))
        if (classifierResult != datingLabels[i]): #如果分类错误，则errorCount加一
            errorCount += 1
    #打印错误率
    print('the total error rate is:' + str((errorCount / float(numTestVeces))))

#约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses'] #设置结果列表
    #从键盘依次输入三个属性
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') #读取样本数据
    normMat, ranges, minVals = autoNorm(datingDataMat) #数据归一化
    inArr = np.array([ffMiles, percentTats, iceCream]) #把特征组装成一个向量，作为输入
    #使用3近邻进行预测，注意也要把待预测的向量归一化
    classifierResult = classify0((inArr - minVals) / ranges, normMat,
                                 datingLabels, 3)
    print('You will probably like this person: ' + 
          str(resultList[classifierResult - 1]))
    
#b编写函数img2vector，将图像矩阵转化为一个numpy向量并返回
def img2vector(filename):
    #输入参数为文件名
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

#手写数字识别系统的测试代码
def handwritingClassTest():
    #从os导入listir，用于读取指定目录下的所有文件名称
    from os import listdir
    hwLabels = [] #初始化标签向量
    trainingFileList = listdir('trainingDigits') #读取训练文件夹下的所有文件名
    m = len(trainingFileList) #获取训练样本数量
    trainingMat = np.zeros((m, 1024)) #初始化训练样本矩阵
    #逐个对训练文件进行操作
    for i in range(m):
        fileNameStr = trainingFileList[i] #获取文件名
        #用.对文件名进行切片，第0个元素是A_B形式，如0_1, 其中0代表标签，1代表第一张
        fileStr = fileNameStr.split('.')[0] 
        #获取标签，并加入hwLabels
        classNumStr  = int(fileStr.split('_')[0]) 
        hwLabels.append(classNumStr)
        #将文件名保存在list中
        trainingMat[i, :] = img2vector('trainingDigits/' + fileNameStr) 
    testFileList = listdir('testDigits') #读取测试文件名
    errorCount = 0 #初始化分类错误数目
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/' + fileNameStr)
        #使用3近邻进行分类
        classifierResult = classify0(vectorUnderTest, 
                                     trainingMat, hwLabels, 3)
        print('the classifier came back with: ' + str(classifierResult) +
              ' the real answer is: ' + str(classNumStr)) #打印分类结果和真实标签
        if (classifierResult != classNumStr): #如果判断错误，errorCount加一
            errorCount += 1
    #打印错误数目和错误率
    print('\n the total number of error is: ' + str(errorCount))
    print('\n the toal error rate is: ' + str(errorCount / float(mTest)))