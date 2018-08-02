# -*- coding: utf-8 -*-
"""

朴素贝叶斯：使用朴素贝叶斯分类器从个人广告中获取区域倾向
这里原来网址不可用，就直接调用cnblogs和v2ex的rss源了。
@author: Jerry
"""
import NaiveBayes
import operator
import random
import numpy as np
import feedparser

def calcMostFreq(vocabList,fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    docList=[]
    classList = []
    fullText =[]
    
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = NaiveBayes.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = NaiveBayes.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = NaiveBayes.createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)   
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    
    trainingSet = list(range(2*minLen))
    testSet=[]           
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    
    trainMat=[]
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(NaiveBayes.bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = NaiveBayes.trainingNB(np.array(trainMat),np.array(trainClasses))
    
    errorCount = 0
    for docIndex in testSet:        
        wordVector = NaiveBayes.bagOfWords2Vec(vocabList, docList[docIndex])
        if NaiveBayes.classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(feed1,feed0):
    vocabList,p0V,p1V=localWords(feed1,feed0)
    
    feed1=[]
    feed0=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : 
            feed0.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : 
            feed1.append((vocabList[i],p1V[i]))
   
    sortedFeed0 = sorted(feed0, key=lambda pair: pair[1], reverse=True)
    print("----Feed0-----")
    for item in sortedFeed0:
        print(item[0])
    
    sortedFeed1 = sorted(feed1, key=lambda pair: pair[1], reverse=True)
    print("----Feed1-----")
    for item in sortedFeed1:
        print(item[0])

if __name__ == '__main__':
    v2ex=feedparser.parse('https://www.v2ex.com/index.xml')
    cnblogs=feedparser.parse('http://feed.cnblogs.com/blog/sitehome/rss')
    
    vocabList,p0V,p1V = localWords(v2ex,cnblogs)
    getTopWords(v2ex,cnblogs)
    
