# -*- coding: utf-8 -*-
"""

kNN: 手写数字识别
@author: Jerry
"""
import numpy as np
from os import listdir
from time import clock
from sklearn.neighbors import KNeighborsClassifier
import kNN

# 将32x32的二进制图像转换为1x1024向量
def img2vector(filename):
    returnVect = np.zeros((1,1024))            #创建空numpy数组
    fr = open(filename)                         #打开文件
    for i in range(32):
        lineStr = fr.readline()                #读取每一行内容
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])#将每行前32个字符值存储在numpy数组中
    return returnVect

# classify实现 
def handwritingClassTest1():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')    #加载训练集
    m = len(trainingFileList)                     #计算当前文件夹下文件个数
    trainingMat = np.zeros((m,1024))            #初始化训练向量矩阵
    
    for i in range(m):
        fileNameStr = trainingFileList[i]        #获取文件名
        fileStr = fileNameStr.split('.')[0]     #从文件名中解析出分类的数字
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    
    testFileList = listdir('testDigits')        #加载测试集
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]          #从文件名中解析出测试样本的类别
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        
        classifierResult = kNN.classify(vectorUnderTest, trainingMat, hwLabels, 3) #开始分类
        print ('预测数字: %d, 真实数字: %d' % (classifierResult, classNumStr))
        
        if (classifierResult != classNumStr): 
            errorCount += 1.0            #计算分错的样本数
    print ('\n总错误样本数: %d' % errorCount)
    print ('\n错误率: %f' % (errorCount/float(mTest)))

# sklearn实现   
def handwritingClassTest2():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')    #加载训练集
    m = len(trainingFileList)                     #计算文件夹下文件的个数
    trainingMat = np.zeros((m,1024))            #初始化训练向量矩阵,，因为每一个文件是一个手写体数字
    
    for i in range(m):
        fileNameStr = trainingFileList[i]        #获取文件名
        fileStr = fileNameStr.split('.')[0]     #从文件名中解析出分类的数字
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    
    #构建kNN分类器
    neigh = KNeighborsClassifier(n_neighbors = 3, algorithm = 'auto')
    
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('testDigits')        #加载测试集
    mTest = len(testFileList)
    
    errorCount = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]          #从文件名中解析出测试样本的类别
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        
        classifierResult = neigh.predict(vectorUnderTest) #开始分类
        print ('预测数字: %d, 真实数字: %d' % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): 
            errorCount += 1.0            #计算分错的样本数
    print ('\n总错误样本数: %d' % errorCount)
    print ('\n错误率: %f' % (errorCount/float(mTest)))


if __name__ == '__main__':
    start = clock()
#    handwritingClassTest1()
    handwritingClassTest2()
    finish = clock()
    print('运行时间: %d 秒' % (finish - start))


