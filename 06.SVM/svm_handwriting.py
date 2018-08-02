# -*- coding: utf-8 -*-
"""

SVM：MINIST数据集
@author: Jerry
"""

from os import listdir
import numpy as np
import SVM
from sklearn.svm import SVC

# 将二进制图像转1x1024向量
def img2vector(filename):
	returnVect = np.zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

# 加载图像
def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)           
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    
    for i in range(m):
        fileName = trainingFileList[i]   
        classNum = int(fileName.split('_')[0])
        hwLabels.append(classNum)        
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileName))
    
    return trainingMat,hwLabels

# 手写实现
def digitsTest1():
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = SVM.smoP(dataArr, labelArr, 200, 0.0001, 10, ('rbf',20))
	
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    
    svInd = np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print("支持向量个数:%d" % np.shape(sVs)[0])
	
    m,n = np.shape(datMat)
    errorCount = 0
	
    for i in range(m):
        kernelEval = SVM.kernelTrans(sVs, datMat[i,:], ('rbf',20))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): 
            errorCount += 1
    print("训练集错误率: %.2f%%" % (float(errorCount)/m))
	
    dataArr,labelArr = loadImages('testDigits')
    
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    
    m,n = np.shape(datMat)
    errorCount = 0
    
    for i in range(m):
        kernelEval = SVM.kernelTrans(sVs, datMat[i,:], ('rbf',20))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): 
            errorCount += 1    
    print("测试集错误率: %.2f%%" % (float(errorCount)/m))

# sklearn实现
def digitsTest2():
	trainingMat,hwLabels = loadImages('trainingDigits')
	clf = SVC(C=200, kernel='rbf')
	clf.fit(trainingMat,hwLabels)
	
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	
	for i in range(mTest):
		fileNameStr = testFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
		classifierResult = clf.predict(vectorUnderTest)
		print("预测类别：%d \t 真实类别：%d" % (classifierResult, classNumber))
		if(classifierResult != classNumber):
			errorCount += 1.0
	print("错误数: %d \n错误率: %f%%" % (errorCount, errorCount/mTest * 100))

if __name__ == '__main__':
#    digitsTest1()
    
    digitsTest2()