# -*- coding: utf-8 -*-
"""

@author: Jerry
"""

import numpy as np
from scipy import linalg as lg
import matplotlib.pyplot as plt

# 导入数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))-1  #默认最后一列为类别
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArray = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArray.append(float(curLine[i]))
        dataMat.append(lineArray)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 显示数据
def showData(x,y,y_hat):
    feature_x = [float(t[1]) for t in x]
    plt.figure()
    plt.scatter(feature_x,y) #散点图
    
    feature_mat = np.mat(feature_x).T
    com_x_yhat = np.hstack((feature_mat,y_hat)) #在行上合并矩阵
    com_sort = sorted(com_x_yhat.tolist(),key = lambda x:x[0]) #排序
    com_feature = [x[0] for x in com_sort]
    com_label = [x[1] for x in com_sort]
    
    plt.plot(com_feature,com_label,color = "r") #拟合线
    plt.show()

# 基于最小二乘法的线性回归(正规方程)
def StandLinearRegression(xArr, yArr):  
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    
    # 判断x是否可逆
    if lg.det(xTx) == 0.0: 
        print('this matrix is sigular,cannot do inverse')
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws

# 局部加权线性回归
def LocalWeightedLinearRegression(testPoint, xArr, yArr, k=1.0): #k由用户指定，控制衰减的速度
    xmat = np.mat(xArr)
    ymat = np.mat(yArr).T
    m = np.shape(xmat)[0] #训练样本的数目
    weight = np.mat(np.eye(m)) #每个样本对应一个权重，m阶单位矩阵
    for j in range(m): #遍历每条样本，得到权重矩阵w
        diffMat = testPoint - xmat[j,:]
        weight[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2)) #高斯核
    xTx = xmat.T*(weight*xmat) 
    if lg.det(xTx) == 0.0: #判断是否可逆
        print('this matrix is sigular,cannot do inverse')
        return
    ws = xTx.I*(xmat.T*(weight*ymat))  #求得ws
    return testPoint*ws 

def LocalWeightedTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    
    yMat = np.zeros(m)
    for i in range(m):
        yMat[i] = LocalWeightedLinearRegression(testArr[i],xArr,yArr,k) #每个样本集都需要计算一次数据集
    return np.mat(yMat).T

# 岭回归
def ridgeRegression(xMat,yMat,lam=0.2):
    xMat = np.mat(xMat)
    xTx = xMat.T*xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam  #岭回归的核心，加了个对角矩阵
    if lg.det(denom)==0.0:
        print("the matrix is singular, cannot do inverse")
        return
    ws = denom.I*(xMat.T*yMat) #根据权重公式计算
    return ws

def ridgeTest(xArr,yArr):
    xMat = np.mat(xArr) #测试数据
    yMat = np.mat(yArr).T #测试的值
    yMean = np.mean(yMat,0) #求均值
    yMat = yMat - yMean 
    xMeans = np.mean(xMat,0) #求均值
    xVar = np.var(xMat,0) #x的协方差
    xMat = (xMat-xMeans)/xVar #数据标准化
    numTestPts = 30 #测试的lamda的范围
    wMat = np.zeros((numTestPts, np.shape(xMat)[1])) #存储不同lamda得到的权重值
    for i in range(numTestPts):
        ws = ridgeRegression(xMat,yMat,np.exp(i-10)) #岭回归函数
        wMat[i,:] = ws.T #每一行对应一个lamda
    return wMat
  
# 正则化函数
def regularize(xMat):#按列正则化
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0)   #均值
    inVar = np.var(inMat,0)      #协方差
    inMat = (inMat - inMeans)/inVar #
    return inMat

# 残差平方和
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

# 前向逐步线性回归
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean  #数据标准化
    xMat = regularize(xMat)
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))
    ws = np.zeros((n,1)) #n个特征对应的权向量，初始化为1
    wsTest = ws.copy() 
    wsMax = ws.copy()
    for i in range(numIt):
#        print(ws.T)
        lowestError = np.inf
        for j in range(n): #对每个特征值都执行了两次for循环，分别计算增加或减少该特征对误差的影响
            for sign in [-1,1]: #做加1个eps或减1个eps处理
                wsTest =ws.copy() 
                wsTest[j] += eps*sign 
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A) #返回预测误差的平方和
                if rssE < lowestError: #如果rssE小于最小误差，则迭代
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat