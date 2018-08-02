# -*- coding: utf-8 -*-
"""

逻辑斯蒂回归：疝气病症病马数据集
@author: Jerry
"""

import numpy as np
import LogisticRegres
from sklearn.linear_model import LogisticRegression

# 手写LR
def horseTest1():
    frTrain = open('horseColicTraining.txt','r')
    frTest = open('horseColicTest.txt','r')
    
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArray = []
        for i in range(len(currLine)-1):
            lineArray.append(float(currLine[i]))
        trainingSet.append(lineArray)
        trainingLabels.append(float(currLine[21]))

    trainWeights, trainWeightsArray = LogisticRegres.stocGradAscent1(np.array(trainingSet),trainingLabels,2)
    
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine = line.strip().split('\t')
        lineArray = []
        for i in range(len(currLine)-1):
            lineArray.append(float(currLine[i]))
        
        if int(LogisticRegres.classifyVector(np.array(lineArray), trainWeights)) != int(currLine[21]):
            errorCount+=1
    
    errorRate = errorCount/numTestVec
    return errorRate
    
def multiTest(numTests):
    errorSum = 0.0
    for k in range(numTests):
        errorSum += horseTest1()
    print('After %d iterations, the average error rate is: %f'% (numTests, errorSum/float(numTests)))

# sklearn实现
def horseTest2():
    frTrain = open('horseColicTraining.txt')                                        #打开训练集
    frTest = open('horseColicTest.txt')
    
    trainingSet = []
    trainingLables = []
    
    testSet = []
    testLabels = []
    
    for line in frTrain.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currentLine) - 1):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr)
        trainingLables.append(float(currentLine[-1]))
    
    for line in frTest.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currentLine) - 1):
            lineArr.append(float(currentLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currentLine[-1]))
    
    classifier = LogisticRegression(solver='liblinear', max_iter=10).fit(trainingSet, trainingLables)
    test_accurcy = classifier.score(testSet,testLabels) * 100
    print('正确率：%f%%' % test_accurcy)


if __name__ == '__main__':
#    horseTest1()
#    errorRate = horseTest1()
#    print('the error rate is: ', errorRate)    
#    multiTest(10)
    
    horseTest2()
    

    
    
    
    
    