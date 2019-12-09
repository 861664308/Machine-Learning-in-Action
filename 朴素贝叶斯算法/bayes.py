import numpy as np
def loadDataSet():
    '''
    创建一些实验样本
    :return: postingList是进行词条切分后的文档集合，该文档来自斑点犬爱好者留言板
             classVec是类别标签的集合
    '''
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0, 1, 0, 1, 0, 1] #1代表侮辱性文字，0代表正常言论
    return postingList, classVec
def createVocabList(dataSet):
    '''
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet: 所有文档当成输入
    :return: 返回每篇文章出现的不重复单词列表
    '''
    vocabSet = set([]) #创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建两个集合的并集
    return list(vocabSet)
def setOfWords2Vec(vocabList, inputSet):
    '''
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: 文档向量，每一元素是1或者0，分别词汇表vocabList的每个元素
             是否在文档inputSet中出现
    '''
    returnVec = [0] * len(vocabList) #创建一个全零向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary' %word)
    return returnVec

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    '''
    :param trainMatrix: 训练文档转换后的矩阵，每一行代表一个样本的向量，行数=样本数，
           列数=单词个数
    :param trainCategory:类别标签的集合
    :return:
    '''
    numTrainDocs = len(trainMatrix) #获取文本数
    numWords = len(trainMatrix[0]) #获取单词个数
    #计算属于侮辱性文档的概率，由于是二分类，所以可以很容易得到属于非侮辱性的概率
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)
    '''初始化计算两个类别后验概率的分子分母，p0Num,p1Num是分子，p0Denom,p1Denom是分母,
    为了防止出现分母为0的情况，将分子初始化为1，分母初始化为2'''
    p0Num, p1Num, p0Denom, p1Denom = np.ones(numWords), np.ones(numWords), 2.0, 2.0
    for i in range(numTrainDocs):#遍历文档集合
        if trainCategory[i] == 1:#如果是侮辱性文档
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:#非侮辱性文档
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    #p1Vect,p0Vect分别返回侮辱性文档和非侮辱性文档下各个单词出现的概率
    p1Vect, p0Vect = np.log(p1Num / p1Denom), np.log(p0Num / p0Denom)#为防止数值下溢，使用对数
    return p0Vect, p1Vect, pAbusive
#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
#测试函数
def TestingNB():
    listOPosts, listClasses =  loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(str(testEntry) + 'classified as: ' + str(classifyNB(thisDoc, p0V, p1V, pAb)))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(str(testEntry) + 'classified as: ' + str(classifyNB(thisDoc, p0V, p1V, pAb)))
#朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
#文件解析和完整的垃圾邮件测试函数
def textParse(bigString):
    #接受一个字符串并将其解析为字符串列表，去掉少于两个字符的字符串，并将所有字符串转换成小写
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
#垃圾邮件分类器函数
def spamTest():
    docList, classList, fullText = [], [], []
    #导入并解析文本文件
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet, testingSet = list(range(50)), []
    #随机选取10个作为测试集，剩下的作为训练集
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testingSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    #取出训练集训练算法
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    #取出测试集测试算法
    errorCount = 0
    for docIndex in testingSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ' + str(float(errorCount) / len(testingSet)))

if __name__ == '__main__':
    #获取留言数据集listOPosts和类别listClasses（是否是侮辱性留言）
    listOPosts, listClasses = loadDataSet()
    #返回所有留言里面不重复单词的列表myVocabList
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    #判断某句留言是否有列表myVocabList里的单词
    print(setOfWords2Vec(myVocabList, listOPosts[0]))
    print(setOfWords2Vec(myVocabList, listOPosts[1]))
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    TestingNB()
    spamTest()