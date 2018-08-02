# -*- coding: utf-8 -*-
"""

@author: Jerry
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName,'r')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    
    fr.close()
    return dataMat,labelMat

# Sigmoid函数  
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

# 梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()     
    m, n = np.shape(dataMatrix)                    
    alpha = 0.001                                   # 学习率
    maxCycles = 500                                # 最大迭代次数
    
    weights = np.ones((n,1))                       # 回归系数
    weights_array = np.array([])
    for k in range(maxCycles):                     
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array,weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA(),weights_array            # 将矩阵转化为数组，并返回权重数组

# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    dataArray = np.array(dataMatrix)
    m, n = np.shape(dataArray)
    alpha = 0.01
    weights = np.ones(n)
    weights_array = np.array([])
    for i in range(m):
        h = sigmoid(sum(dataArray[i] * weights))   # 选择随机选取的一个样本，记作h
        error = classLabels[i] - h                  # 计算误差
        weights = weights + alpha * error * dataArray[i]  # 更新回归系数
        weights_array = np.append(weights_array, weights, axis=0)  # 添加回归系数到数组中
    weights_array = weights_array.reshape(m, n)          # 改变维度
    return weights, weights_array


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    dataArray = np.array(dataMatrix)
    m, n = np.shape(dataArray)
    weights = np.ones(n)
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01          # 降低alpha的大小，每次减小1/(j+1)
            randIndex = int(random.uniform(0,len(dataIndex)))   # 随机选取样本
            h = sigmoid(sum(dataArray[randIndex] * weights))   # 选择随机选取的一个样本，记作h
            error = classLabels[randIndex] - h                  # 计算误差
            weights = weights + alpha * error * dataArray[randIndex]  # 更新回归系数
            weights_array = np.append(weights_array, weights, axis=0)  # 添加回归系数到数组中
            del(dataIndex[randIndex])                           # 删除已经使用的样本
    weights_array = weights_array.reshape(numIter*m, n)          # 改变维度
    return weights, weights_array

# 绘制基于回归系数的数据集及决策边界
def plotBestFit(dataMat, labelMat, weights,title):
    dataArray = np.array(dataMat)
    n = dataArray.shape[0]
    
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArray[i,1])
            ycord1.append(dataArray[i,2])
        else:
            xcord2.append(dataArray[i,1])
            ycord2.append(dataArray[i,2])
    
    fig = plt.figure()
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15) 
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)             # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='green', marker='o',alpha=.5)            # 绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    
    ax.plot(x, y)
    plt.title(title, fontproperties=font_set)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 利用LR生成的决策边界来分类
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

# 绘制回归系数与迭代次数的关系
def plotWeights(weights_array1, weights_array2):
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系', fontproperties=font_set)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0')
    plt.setp(axs0_title_text, size=14, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=14, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1')
    plt.setp(axs1_ylabel_text, size=14, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', fontproperties=font_set)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2')
    plt.setp(axs2_xlabel_text, size=14, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=14, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', fontproperties=font_set)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0')
    plt.setp(axs0_title_text, size=14, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=14, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1')
    plt.setp(axs1_ylabel_text, size=14, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', fontproperties=font_set)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W2')
    plt.setp(axs2_xlabel_text, size=14, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=14, weight='bold', color='black')

    plt.show()

