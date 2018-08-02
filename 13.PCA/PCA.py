# -*- coding: utf-8 -*-
"""

基于特征值分解的SVD
@author: Jerry
"""
import numpy as np

def pca(dataMat,topNfeat = 9999):
    # 去除平均值
    meanVal = np.mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVal
    
    # 计算协方差矩阵
    covMat = np.cov(meanRemoved,rowvar=0)
    
    # 计算特征值和特征向量
    eigVals,eigVects = np.linalg.eig(np.mat(covMat)) 
    
    # 将特征值从大到小排序
    eigInd = np.argsort(eigVals)[::-1][0:topNfeat] 
    
    # 保留最上面的N个特征向量
    redEigVects = eigVects[:,eigInd] 
    lowData = meanRemoved * redEigVects 
    
    # 将数据转换到上述N个特征向量构建的新空间中
    reconMat = (lowData * redEigVects.T) + meanVal #重构后的数据

    return lowData,reconMat

# 将NaN替换成平均值
def replaceNanWithMean(dataMat):
    #获取特征数目    
    numFeat = np.shape(dataMat)[1]
    for i in range(numFeat):
        # 计算所有非NaN特征的平均值
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:,i].A))[0],i]) 
        
        # 将NaN替换成均值
        dataMat[np.nonzero(np.isnan(dataMat[:,i].A))[0],i] = meanVal  
    return dataMat