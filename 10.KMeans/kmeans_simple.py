# -*- coding: utf-8 -*-
"""

KMeans:简单数据集
@author: Jerry
"""
import numpy as np
import KMeans


if __name__ == '__main__':
    print('-------------KMeans-----------------------')
    dataMat1 = np.mat(KMeans.loadDataSet('testSet.txt'))
    centroids1, clusterAssment1 = KMeans.kMeans(dataMat1, 4)
    KMeans.dispCluster(dataMat1, 4, centroids1, clusterAssment1)
    print('centroids1=',centroids1)
    
    print('-------------biKMeans-----------------------')
    dataMat2 = np.mat(KMeans.loadDataSet('testSet2.txt'))
    centroids2, clusterAssment2 = KMeans.biKmeans(dataMat2, 3)
    KMeans.dispCluster(dataMat2, 3, centroids2, clusterAssment2)
    print('centroids2=',centroids2)

    