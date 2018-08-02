# -*- coding: utf-8 -*-
"""

AdaBoost：简单数据集
@author: Jerry
"""
import numpy as np
import AdaBoost

def loadDataSet():
    dataMat = np.matrix(([1.,2.1],
            [2.,1.1],
            [1.3,1.],
            [1.,1.],
            [2.,1.]))
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

if __name__ == '__main__':
    dataMat,classLabels = loadDataSet()
    
#    AdaBoost.adaBoostTrainDS(dataMat,classLabels, 9)

    classifierArray = AdaBoost.adaBoostTrainDS(dataMat,classLabels, 30)
    predictedLabel = AdaBoost.adaClassify([0,0], classifierArray)
    print(predictedLabel)