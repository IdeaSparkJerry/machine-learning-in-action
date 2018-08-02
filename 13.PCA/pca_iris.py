# -*- coding: utf-8 -*-
"""

PCA：iris数据集
@author: Jerry
"""

import matplotlib.pyplot as plt
import PCA
from sklearn.datasets import load_iris


if __name__ == "__main__":
    iris = load_iris()
    dataMat = iris.data
    lowData, reconMat = PCA.pca(dataMat,2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten(),dataMat[:,1].flatten(),marker="x",s = 90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker="o", s=50,c ="red")
    plt.show()