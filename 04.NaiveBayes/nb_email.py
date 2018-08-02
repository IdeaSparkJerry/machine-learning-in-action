# -*- coding: utf-8 -*-
"""

朴素贝叶斯：邮件分类
@author: Jerry
"""

import numpy as np
import random
from sklearn.naive_bayes import MultinomialNB
import NaiveBayes


# 邮件分类 - classify实现
def emailTest1():
    #读取email文件
    docList = []
    classList = []
    fullText = []    
    for i in range(1, 26):
        wordList = NaiveBayes.textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)             #1-垃圾邮件
        
        wordList = NaiveBayes.textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)             #0-正常邮件
    
    vocabList = NaiveBayes.createVocabList(docList)
    trainingSet = list(range(50))    # 创建存储训练集的索引值的列表和测试集的索引值的列表
    testSet = []
         
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0,len(trainingSet)))#从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    testSampleText = []
    for i in range(10):
        testSampleText.append(fullText[testSet[i]])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(NaiveBayes.bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    p0V,p1V,pAb = NaiveBayes.trainingNB(np.array(trainMat),np.array(trainClasses))
    
    for testEntry in testSampleText:
        testDoc = np.array(NaiveBayes.bagOfWords2Vec(vocabList,testEntry))
        testResult = NaiveBayes.classifyNB(testDoc,p0V,p1V,pAb)
        
        print('The testSample is: ',testEntry,'\n')
        print('It is classified as : ', testResult,'\n')   
        print('------------------------------------------------')


# 邮件分类 - sklearn实现
def emailTest2():
    #读取email文件
    docList = []
    classList = []
    fullText = []    
    for i in range(1, 26):
        wordList = NaiveBayes.textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)             #1-垃圾邮件
        
        wordList = NaiveBayes.textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)             #0-正常邮件
    
    vocabList = NaiveBayes.createVocabList(docList)
    trainingSet = list(range(50))    # 创建存储训练集的索引值的列表和测试集的索引值的列表
    testSet = []
         
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0,len(trainingSet)))#从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    testSampleText = []
    for i in range(10):
        testSampleText.append(fullText[testSet[i]])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(NaiveBayes.bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    clf = MultinomialNB()
    model = clf.fit(np.array(trainMat), np.array(trainClasses))
    
    for testEntry in testSampleText:
        testDoc = np.array(NaiveBayes.bagOfWords2Vec(vocabList,testEntry))
        testResult = model.predict(np.array(testDoc).reshape(1, -1))[0]
        
        print('The testSample is: ',testEntry,'\n')
        print('It is classified as : ', testResult,'\n')   
        print('------------------------------------------------')

if __name__ == '__main__':
    emailTest1()
    
#    emailTest2()

