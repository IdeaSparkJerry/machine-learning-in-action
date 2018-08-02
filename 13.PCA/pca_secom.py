# -*- coding: utf-8 -*-
"""

PCA：secom数据集
@author: Jerry
"""
import numpy as np
from scipy import linalg as lg
import matplotlib.pyplot as plt
import PCA

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [list(map(float,line)) for line in stringArr]
    return np.mat(dataArr)

    
if __name__ == "__main__":
    origDataMat = loadDataSet('secom.data.txt', ' ')
    dataMat = PCA.replaceNanWithMean(origDataMat)
    meanVals = np.mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals
    convMat = np.cov(meanRemoved, rowvar = 0)
    
    eigVals, eigVects = lg.eig(np.mat(convMat))
    print('eigVals=',eigVals)
    lowData, reconMat = PCA.pca(dataMat,1)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker="x",s = 90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker="o", s=50,c ="red")
    plt.show()