# -*- coding: utf-8 -*-
"""

标准线性回归与局部线性回归：简单数据集
@author: Jerry
"""
import numpy as np
import LinearRegres


if __name__ == '__main__':
    # 线性回归
    dataMat, labelMat = LinearRegres.loadDataSet('ex0.txt')
    ws = LinearRegres.StandLinearRegression(dataMat, labelMat)
    predictedLabelMat1 = dataMat * ws
    LinearRegres.showData(dataMat, labelMat, predictedLabelMat1)
    print('相关系数=',np.corrcoef(predictedLabelMat1.T,labelMat))
    
    # 局部线性回归
    predictedLabelMat2 = LinearRegres.LocalWeightedTest(dataMat,dataMat,labelMat,0.003)
    LinearRegres.showData(dataMat, labelMat, predictedLabelMat2)
    print('相关系数=',np.corrcoef(predictedLabelMat2.T,labelMat))

    
    
    
  
    
    
    
    