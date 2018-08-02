# -*- coding: utf-8 -*-
"""

分类回归树(CART)：简单数据集
@author: Jerry
"""
import numpy as np
#import matplotlib.pyplot as plt

class treeNode():
    def __init__(self, feat, val, right, left):
        self.featureToSplitOn = feat
        self.valueOfSplit = val
        self.rightBranch = right
        self.leftBranch = left

# 加载数据集
def loadDataSet(fileName):
    fr = open(fileName).readlines()
    data = [list(map(float,line.strip("\n").split("\t"))) for line in fr]
    return np.mat(data)

# 将数据集以feature=value划分为两部分
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1

# 生成叶结点
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

def modelLeaf(dataSet):
    w,x,y = linearSolve(dataSet)
    return w

# 计算回归误差
def regErr(dataSet):
    return np.var(dataSet[:,-1])* dataSet.shape[0]

def modelErr(dataSet):
    w,x,y = linearSolve(dataSet)
    y_hat = x * w
    return np.sum(np.power(y - y_hat,2))

# 找到最优划分并进行预剪枝
def chooseBestSplit(dataSet, leafType, errType, ops = (1,4)):
    tolS = ops[0] #容许的误差
    tolN = ops[1] #切分的最小样本数

    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #不进行划分
        return None,leafType(dataSet)

    S = errType(dataSet)
    bestS = np.inf
    bestFeature = 0
    bestValue = 0

    m,n = dataSet.shape
    for feature in range(n-1):
        for value in set(dataSet[:,feature].T.tolist()[0]):
            data0,data1 = binSplitDataSet(dataSet,feature,value)

            if data0.shape[0] < tolN or data1.shape[0] < tolN: continue #切分的样本数小于阀值，则不划分

            newS = errType(data0) + errType(data1)
            if newS < bestS:
                bestS = newS
                bestFeature = feature
                bestValue = value

    if (S - bestS) < tolS: #误差的减少不大，则不划分
        return None,leafType(dataSet)

    data0,data1 = binSplitDataSet(dataSet,bestFeature,bestValue)
    if data0.shape[0] < tolN or data1.shape[0] < tolN:
        return None,leafType(dataSet)
    return bestFeature,bestValue

# 创建树
def createTree(dataSet,leafType = regLeaf, errType = regErr, ops = (1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: 
        return val #创建叶子结点

    retTree = {}
    retTree["spInd"] = feat
    retTree["spVal"] = val
    data0,data1 = binSplitDataSet(dataSet,feat,val)
    retTree["left"] = createTree(data0,leafType,errType,ops)
    retTree["right"] = createTree(data1,leafType,errType,ops)
    
    return retTree

# 判断是否为叶子结点
def isTree(tree):
    return (type(tree).__name__ == "dict")

# 从上往下遍历树，直至遇到叶子结点，如果遇到两个叶子结点，返回叶子结点的平均值
def getMean(tree):
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    return (tree['left']+tree['right']/2)

# 剪枝函数
def prune(tree,testData):
    if testData.shape[0] == 0: 
        return getMean(tree) #无测试数据时，对树做塌陷处理
    
    if isTree(tree['left']) or isTree(tree['right']): #不是叶子结点
        data0,data1 = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    
    if isTree(tree['left']): 
        tree['left'] = prune(tree['left'],data0) #剪枝左子树
    
    if isTree(tree['right']): 
        tree['right'] = prune(tree['right'],data1) #剪枝右子树

    if not isTree(tree['left']) and not isTree(tree['right']): #左右子树都是叶子结点
        data0, data1 = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errnoMerge = np.sum(np.power(data0[:, -1] - tree['left'],2)) + np.sum(np.power(data0[:, -1] - tree['left'],2))
        treeMean = tree['left']+tree['right']/2
        errMerge = np.sum(np.power(testData[:, -1] - treeMean,2))
        
        if errMerge < errnoMerge:
            print("merging")
            return treeMean
        else: 
            return tree
    else: 
        return tree

#线性模型
def linearSolve(dataSet):
    m,n = dataSet.shape
    mat1 = np.ones((m,1))
    x = np.hstack((mat1,dataSet[:,0:n-1]))
    y = dataSet[:,n-1]
    if np.linalg.det(x.T * x) != 0.00:
        w = ((x.T * x).I) * (x.T * y)
        return w,x,y
    else: return False

