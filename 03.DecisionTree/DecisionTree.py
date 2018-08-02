# -*- coding: utf-8 -*-
"""

@author: Jerry
"""
from math import log
import operator

# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:                    ##遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #删掉axis的特征，保留剩下的特征并存到retDataSet
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 计算香农熵
def calcShannonEnt(dataSet):
    # 获取样本数目
    numEntries = len(dataSet)                           
    
    # 统计类别个数，通过字典存储（键：值）
    labelCounts = {}
    for featVec in dataSet:                             # 计算每个类别出现的次数的字典
        currentLabel = featVec[-1]                      # 取每个样本的类别，并计算出现的次数
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    # 计算熵
    shannonEnt = 0.0
    for key in labelCounts:                            
        prob = float(labelCounts[key])/numEntries      
        shannonEnt -= prob * log(prob,2)               
   
    return shannonEnt

# 选择最优特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #特征数目=总列数-标签列
    baseEntropy = calcShannonEnt(dataSet)  #计算香农熵
    
    
    bestInfoGain = 0.0;                 #初始化最大信息增益变量
    bestFeature = -1
    for i in range(numFeatures):        #遍历所有特征
        featList = [example[i] for example in dataSet]#取所有样本的第一个特征
        uniqueVals = set(featList)       #去重复值
        
        # 计算信息增益  g(D,A)=H(D)-H(D|A)
        newEntropy = 0.0
        for value in uniqueVals:        #按照第i个特征划分数据下的香农熵，信息增益
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy   
        
        if (infoGain > bestInfoGain):       #选择最大增益的特征索引
            bestInfoGain = infoGain         
            bestFeature = i
    return bestFeature                      #返回的是下标值

# 多数表决
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 使用ID3算法创建决策树，递归
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]  #取数据集的类别标签
    if classList.count(classList[0]) == len(classList): 
        return classList[0]   #递归停止条件一：如果类别完全相同则停止继续划分
    if len(dataSet[0]) == 1:   #递归停止条件二：遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择最优特征
#    print(bestFeat)

    bestFeatLabel = labels[bestFeat]             # 最优特征的标签
    myTree = {bestFeatLabel:{}}             #根据最优特征的标签生成树
    del(labels[bestFeat])
    
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #复制标签，递归创建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree   

# 使用决策树进行分类       
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]                                                      #获取决策树结点
    secondDict = inputTree[firstStr]                                                        #下一个字典
    featIndex = featLabels.index(firstStr)                                               
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel
