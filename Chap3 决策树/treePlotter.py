#使用文本注解绘制树节点
import matplotlib.pyplot as plt

dicisionNode = dict(boxstyle = 'sawtooth', fc = '0.8')
leafNode = dict(boxstyle = 'round4', fc = '0.8')
arrow_args = dict(arrowstyle = '<-') #设置箭头格式

def plotNode(nodeTex, centerPt, parentPt, nodeType):
    '''

    :param nodeTex: 要显示的文本
    :param centerPt: 文本的中心点
    :param parentPt: 指向文本的点
    :param nodeType: 箭头所在点
    :return:
    '''
    createPlot.ax1.annotate(nodeTex, xy = parentPt, xycoords = 'axes fraction',
                xytext = centerPt, textcoords = 'axes fraction', va = 'center',
                ha = 'center', bbox = nodeType, arrowprops = arrow_args)

def createPlot():
    #创建fig
    fig = plt.figure(1, facecolor='white')
    #清空fig
    fig.clf()
    #去掉x，y轴
    createPlot.ax1 = plt.subplot(111, frameon = False)
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), dicisionNode)
    plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
#获取树的节点
def getNumLeaf(myTree):
    '''
    :param myTree: 决策树
    :return: 决策树叶子节点的数目
    '''
    #初始化叶子
    numLeafs = 0
    '''
    python3中myTree.keys()返回的是dict_keys(),不是list，所以不能用
    myTree.keys()[0]获取叶节点属性，可以使用list(myTree.keys())[0]
    next() 返回迭代器的下一个项目 next(iterator[, default])
    '''
    firstStr = next(iter(myTree))
    #获取下一组字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #测试该节点是否为字典，如果不是字典，代表此节点为叶子节点
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeaf(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs
#获取树的层数
def getTreeDepth(myTree):
    # 初始化决策树深度
    maxDepth = 0
    # python3中myTree.keys()返回的是dict_keys,不是list,所以不能用
    # myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    # next() 返回迭代器的下一个项目 next(iterator[, default])
    firstStr = next(iter(myTree))
    # 获取下一个字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此节点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 更新最深层数
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    # 返回决策树的层数
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {
        'flippers':{0: 'no', 1 : 'yes'}}}},
        {'no surfacing': {0 : 'no', 1 : {'flippers':
        {0 : {'head' : {0 : 'no', 1 : 'yes'}}, 1 : 'no'
         }}}}]
    return listOfTrees[i]