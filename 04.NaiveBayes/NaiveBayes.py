# -*- coding: utf-8 -*-
"""

@author: Jerry
"""
import numpy as np
import re

# 文本解析
def textParse(bigString):
    listOfTokens = re.split(r'\W*',bigString) # \W 匹配任何非单词字符  * 匹配前面的子表达式零次或多次。
    return [tok.lower() for tok in listOfTokens if len(tok)>2]#除了单个字母，例如大写的I，其它单词变成小写

# 创建不重复的词汇表
def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet|set(document) #两个set之间取并集
    return list(vocabSet)

# 词袋模型:向量化词汇表
def bagOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 训练函数，原始版本
def trainingNB_orignal(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)       #文档数目
    numWords = len(trainMatrix[0])        #词条数目
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #文档属于侮辱类的概率
    
    
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 0
    p1Denom = 0
    
    
    for i in range(numTrainDocs):  
        if trainCategory[i] == 1:           #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)··
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                               #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)··
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    p1Vect = float(p1Num)/float(p1Denom)
    p0Vect = float(p0Num)/float(p0Denom)
    
    return  p0Vect,p1Vect,pAbusive          #p0V存放的是每个单词属于非侮辱类词汇的概率 p1V存放的就是各个单词属于侮辱类的条件概率。pAbusive就是先验概率。


# 训练函数，采用拉普拉斯平滑
def trainingNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)       #文档数目
    numWords = len(trainMatrix[0])        #词条数目
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #文档属于侮辱类的概率
    
    
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    
    # 拉普拉斯平滑
    p0Denom = 2
    p1Denom = 2
    
    for i in range(numTrainDocs):  
        if trainCategory[i] == 1:           #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)··
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                               #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)··
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    
    return  p0Vect,p1Vect,pAbusive          #p0V存放的是每个单词属于非侮辱类词汇的概率 p1V存放的就是各个单词属于侮辱类的条件概率。pAbusive就是先验概率。

# 分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) +np.log(pClass1)      #log p(w1|1)+log p(w2|1) +... = log  p(w1|1)* p(w2|1)*...
    p0 = sum(vec2Classify*p0Vec) +np.log(1.0-pClass1)
    
    if p1>p0:
        return 1        # 1 - 属于侮辱类
    else:
        return 0        # 0 - 属于非侮辱类