# -*- coding: utf-8 -*-
"""

PCA：简单数据集
@author: Jerry
"""

import numpy as np
import matplotlib.pyplot as plt
import PCA

def loadDataSet(fileName):
    with open(fileName) as fr:
        content = fr.readlines()
        lines = [list(map(float,line.strip("\n").split("\t"))) for line in content]
    return np.mat(lines)


if __name__ == "__main__":
    dataMat = loadDataSet("testSet.txt")
    lowData, reconMat = PCA.pca(dataMat,1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker="x", s = 50, c = "blue" )
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker="o", s = 50, c = "red")
    plt.show()