# -*- coding: utf-8 -*-
"""

@author: Jerry
"""
import numpy as np

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) #获取特征数目
    dataMat = []
    labelMat = []
    fr = open(fileName) #打开文件
    for content in fr.readlines(): #遍历每一行
        lineArr = []
        curLine = content.strip().split('\t') #按tab键分割
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)  #特征集合
        labelMat.append(float(curLine[-1])) #类别标签
    return dataMat,labelMat #返回数据

def stumpClassify(dataMatrix,dimen, threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1)) #先将返回结果初始化为1的列向量
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal]= -1.0  #小于，其值赋为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0  #大于，其值赋为-1
    return retArray

def buildStump(dataArr, classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n): #遍历特征值
        rangeMin = dataMatrix[:,i].min() #特征值的最小值
        rangeMax = dataMatrix[:,i].max() #特征值的最大值
        stepSize = (rangeMax-rangeMin)/numSteps  #遍历特征值时的步长
        for j in range(-1,int(numSteps)+1): #根据步长遍历特征值
            for inequal in ['lt','gt']: #不等式的符号
                threshVal = (rangeMin+float(j)*stepSize) #每次分类的阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) #根据数据集，特征，阈值，不等号分类
                errArr =np.mat(np.ones((m,1))) #error初始化为1
                errArr[predictedVals == labelMat] = 0 #相等的err设为9
                weightedError = D.T*errArr #计算加权错误率
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i  #属性列标号，就是记录哪一列作为分类
                    bestStump['thresh'] = threshVal #阈值
                    bestStump['ineq'] = inequal #是大于还是小于
    return bestStump,minError,bestClassEst  #bestStump返回的分类器信息，minErr错误率，bestClassEst预测值

def adaBoostTrainDS(dataArr, classLabels, numIt = 40): #numIt迭代次数，需要用户唯一指定的参数
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
#        print("D:",D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16))) #计算alpha权重，分类器结果统计时的权重
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
#        print("classEst:",classEst.T) #预测信息
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) #这里只有1和-1，也就表示正确时为-alpha,错误时为alpha
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum() #求权重向量D
        aggClassEst += alpha*classEst #类别估计值
#        print("aggClassEst:",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m #总的错误率
        print("total error:",errorRate,"\n")
        if(errorRate == 0.0): #若错误率为0，直接跳出迭代
            break;
    #return weakClassArr #返回结果包含了每一次迭代的分类器信息
    #下面的return语句在测试plot roc时使用
    return weakClassArr,aggClassEst

# adaBoost分类函数
def adaClassify(dataToClass, classifierArr): #数据集，弱分类器信息
    dataMat = np.mat(dataToClass)
    m = np.shape(dataMat)[0] #有多少数据记录
    aggClassEst = np.mat(np.zeros((m,1))) #各记录预测值
    for i in range(len(classifierArr)): #遍历弱分类器
        #调用用弱分类器，得到预测值
        classEst = stumpClassify(dataMat,classifierArr[0][i]['dim'], \
            classifierArr[0][i]['thresh'],classifierArr[0][i]['ineq'])
        aggClassEst += classifierArr[0][i]['alpha']*classEst #加权求和
#        print (aggClassEst) #预测值
    return np.sign(aggClassEst) #得到最终的类别信息

