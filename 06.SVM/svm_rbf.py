# -*- coding: utf-8 -*-
"""

SVM：RBF数据集
@author: Jerry
"""
import SVM
import numpy as np

if __name__ == '__main__':
    dataArr,labelArr = SVM.loadDataSet('testSetRBF.txt')						
    b,alphas = SVM.smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', 1.3))		
    
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    
    w = SVM.calcWeight(dataMat, labelMat, alphas)
#    SVM.showData(dataMat, labelMat, w, b, alphas)
    
    #获得支持向量
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]				
    labelSV = labelMat[svInd];
    print("支持向量个数:%d" % np.shape(sVs)[0])
	
    m,n = np.shape(dataMat)

    errorCount = 0
    for i in range(m):	
        kernelEval = SVM.kernelTrans(sVs,dataMat[i,:],('rbf', 1.3))			#计算各个点的核
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b 	#根据支持向量的点，计算超平面，返回预测结果
        if np.sign(predict) != np.sign(labelArr[i]): 
            errorCount += 1		
    print("训练集错误率: %.2f%%" % ((float(errorCount)/m)*100)) 			
	
    dataArr,labelArr = SVM.loadDataSet('testSetRBF2.txt') 						
    
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    
    m,n = np.shape(dataMat)
    
    errorCount = 0
    for i in range(m):
        kernelEval = SVM.kernelTrans(sVs,dataMat[i,:],('rbf', 1.3)) 			#计算各个点的核			
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b 		#根据支持向量的点，计算超平面，返回预测结果
        if np.sign(predict) != np.sign(labelArr[i]): 
            errorCount += 1    	
	
    print("测试集错误率: %.2f%%" % ((float(errorCount)/m)*100)) 			