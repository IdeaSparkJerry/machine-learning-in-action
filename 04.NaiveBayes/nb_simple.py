# -*- coding: utf-8 -*-
"""

朴素贝叶斯：简单文本分类
@author: Jerry
"""

import numpy as np
import NaiveBayes


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]        #类别标签向量，1代表侮辱性词汇，0代表不是
    
    return postingList,classVec


if __name__ == '__main__':
    listOfPosts,listClasses = loadDataSet()
    myVocabList = NaiveBayes.createVocabList(listOfPosts)
    
    trainMat = []
    for postInDoc in listOfPosts:
        trainMat.append(NaiveBayes.bagOfWords2Vec(myVocabList,postInDoc))

    p0V,p1V,pAb = NaiveBayes.trainingNB(np.array(trainMat),np.array(listClasses))
    
    print('p0V:',p0V)
    print('p1V:',p1V)
    print('listClasses:',listClasses)
    print('pAb:',pAb)
    
    testSample = [['love', 'my', 'dalmation'],
		            ['stupid', 'garbage']]
    
    for testEntry in testSample:
        testDoc = np.array(NaiveBayes.setOfWords2Vec(myVocabList,testEntry))
        testResult = NaiveBayes.classifyNB(testDoc,p0V,p1V,pAb)
        
        print(testEntry, 'is classified as :', testResult)






    

