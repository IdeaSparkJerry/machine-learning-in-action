# -*- coding: utf-8 -*-
"""

@author: Jerry
"""

import numpy as np
import operator

# kNN算法 (测试样本，特征，类别， k)
def classify(input, features, labels, k):
    # 获取样本数
    dataSetSize = features.shape[0]
    
    # 计算欧氏距离
    diffMat = np.tile(input, (dataSetSize,1)) - features 
    sqDiffMat = diffMat**2 
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5  
    
    # 对所有的距离进行排序
    sortedDistIndicies = distances.argsort()   
    classCount={}       
    
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]                    # 取出前k个距离对应的标签
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1      # 计算每个类别的样本数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)    #1:按值排序， 0：按键排序

    return sortedClassCount[0][0] 
