# -*- coding: utf-8 -*-
"""

SVM：简单数据集
@author: Jerry
"""
import SVM

def loadDataSet(fileName):
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():                                     
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])      
		labelMat.append(float(lineArr[2]))                          
	return dataMat,labelMat



if __name__ == '__main__':
    dataMat,labelMat = loadDataSet('testSet.txt')
    b, alphas = SVM.smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    print('b=',b)
    print('alphas[alphas>0]=',alphas[alphas>0])
    
    w = SVM.calcWeight(dataMat, labelMat, alphas)
    SVM.showData(dataMat, labelMat, w, b, alphas)