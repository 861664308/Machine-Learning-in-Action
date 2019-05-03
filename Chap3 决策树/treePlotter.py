#使用文本注解绘制树节点
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
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
    # 定义箭头格式
    arrow_args = dict(arrowstyle="<-")
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    createPlot.ax1.annotate(nodeTex, xy = parentPt, xycoords = 'axes fraction',
                xytext = centerPt, textcoords = 'axes fraction', va = 'center',
                ha = 'center', bbox = nodeType, arrowprops = arrow_args)

def createPlot(inTree):
    #创建fig
    fig = plt.figure(1, facecolor='white')
    #清空fig
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    #去掉x，y轴
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    # 获取叶节点个数
    plotTree.totalW = float(getNumLeafs(inTree))
    # 获取层数
    plotTree.totalD = float(getTreeDepth(inTree))
    # x偏移与y偏移
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    #绘制决策树
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
#获取树的节点
def getNumLeafs(myTree):
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
            numLeafs += getNumLeafs(secondDict[key])
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

def plotMidText(cntrpt, parentpt, txtString):
    #计算标注位置
    xMid = (parentpt[0] - cntrpt[0]) / 2.0 +cntrpt[0]
    yMid = (parentpt[1] - cntrpt[1]) / 2.0 + cntrpt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    #获取叶节点的数目，决定树的宽度
    numLeafs = getNumLeafs(myTree)
    #获取决策树层数
    depth = getTreeDepth(myTree)
    #下个字典
    firstStr = next(iter(myTree))
    #中心位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0/plotTree.totalW, plotTree.yOff)
    # 标注有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制结点
    plotNode(firstStr, cntrPt, parentPt, dicisionNode)
    # 下一个字典，也就是继续绘制结点
    secondDict = myTree[firstStr]
    # y偏移
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            # 不是叶结点，递归调用继续绘制
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
        plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD