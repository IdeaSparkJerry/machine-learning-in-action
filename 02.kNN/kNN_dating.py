# -*- coding: utf-8 -*-
"""

kNN: 改进约会网站的配对效果
@author: Jerry
"""

import numpy as np
import kNN


# txt文件转换成矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)            
    featureMatrix = np.zeros((numberOfLines,3))     
    classLabelVector = []                        
    index = 0
    
    # 读取文件内容
    for line in arrayOfLines:
        line = line.strip()                   #截取掉所有的回车字符
        listFromLine = line.split('\t')       #使用tab字符\t将上一行得到的整行数据分割成一个元素列表
        featureMatrix[index,:] = listFromLine[0:3] 
        
        if listFromLine[-1] == 'largeDoses':  #极具魅力的人记为3
            classLabelVector.append(3)
        if listFromLine[-1] == 'smallDoses': #魅力一般的人记为2
            classLabelVector.append(2)
        if listFromLine[-1] == 'didntLike': #不喜欢的人记为1
            classLabelVector.append(1)
        index += 1
        
    return featureMatrix,classLabelVector

# 矩阵归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = (dataSet - np.tile(minVals, (m,1))) / np.tile(ranges, (m,1))  #归一化公式

    return normDataSet, ranges, minVals
       
# 分类
def classifyPerson(datingDataMat,datingLabels):
   
    #特征
    percentTats = float(input("玩视频游戏所耗时间百分比:"))  #控制台手动输入 0.96
    ffMiles = float(input("每年获得的飞行常客里程数:"))      #控制台手动输入 50000
    iceCream = float(input("每周消费的冰激淋公升数:"))    #控制台手动输入 1.55

    #标签
    resultList = ['不喜欢的人','魅力一般的人','极具魅力的人']
    
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    #测试集归一化
    inArray = np.array([percentTats, ffMiles, iceCream])
    normInArray = (inArray - minVals) / ranges
    
    #返回分类结果
    classifierResult = kNN.classify(normInArray, normMat, datingLabels, 3)
    print("他可能是你%s" % (resultList[classifierResult-1]))
    

def datingClassTest(datingDataMat,datingLabels):
    hoRatio = 0.10                     #将数据集且分为训练集和测试集的比例
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]             
    numTestVecs = int(m*hoRatio)      #测试集数目  
    
    errorCount = 0.0                 #分错样本数
    for i in range(numTestVecs):
        classifierResult = kNN.classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ('预测类别: %d, 真实类别: %d' % (classifierResult, datingLabels[i]))
        
        if (classifierResult != datingLabels[i]): 
            errorCount += 1.0    
    print ('错误率: %f' % (errorCount/float(numTestVecs)))


if __name__ =='__main__':
    #加载原始数据集
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    
    datingClassTest(datingDataMat,datingLabels)
#    classifyPerson(datingDataMat,datingLabels) 
    
    