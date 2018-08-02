# -*- coding: utf-8 -*-
"""

逻辑斯蒂回归：简单数据集
@author: Jerry
"""
import numpy as np
import LogisticRegres



if __name__ == '__main__':
    dataMatrix, labelMatrix = LogisticRegres.loadDataSet('testSet.txt')
    
    #梯度上升算法
    weight1, weights_array1 = LogisticRegres.gradAscent(np.array(dataMatrix), labelMatrix)   
    LogisticRegres.plotBestFit(dataMatrix, labelMatrix, weight1, '梯度上升法')
    
    #随机梯度上升算法
    weight2, weights_array2 = LogisticRegres.stocGradAscent0(dataMatrix, labelMatrix)
    LogisticRegres.plotBestFit(dataMatrix, labelMatrix, weight2, '随机梯度上升法')
    
    #改进的随机梯度上升算法-迭代次数=200
    weight3, weights_array3 = LogisticRegres.stocGradAscent1(dataMatrix, labelMatrix,2)
    LogisticRegres.plotBestFit(dataMatrix, labelMatrix, weight3, '改进的随机梯度上升法-200次')
    
    #改进的随机梯度上升算法-迭代次数=1000
    weight4, weights_array4 = LogisticRegres.stocGradAscent1(dataMatrix, labelMatrix,10)
    LogisticRegres.plotBestFit(dataMatrix, labelMatrix, weight4, '改进的随机梯度上升法-1000次')
    
    LogisticRegres.plotWeights(weights_array3, weights_array4)