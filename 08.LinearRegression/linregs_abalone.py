# -*- coding: utf-8 -*-
"""

岭回归和前向逐步回归：鲍鱼数据集
@author: Jerry
"""

import LinearRegres
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 岭回归
    dataMat, labelMat = LinearRegres.loadDataSet('abalone.txt')
    ws = LinearRegres.ridgeTest(dataMat, labelMat)
    plt.plot(ws)
    plt.xlabel('log(lambda)')
    plt.ylabel('weight')
    plt.show()
    
    # 前向逐步回归
    eps = 0.001
    numIteration = 5000
    w = LinearRegres.stageWise(dataMat,labelMat,eps,numIteration)
    plt.plot(w)
    plt.xlabel('Number of Iteration')
    plt.ylabel('weight')
    plt.show()

    
