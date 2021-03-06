# -*- coding: utf-8 -*-
"""

预测乐高玩具套装的价格
@author: Jerry
"""

from time import sleep
import json
import urllib.request
import numpy as np
import LinearRegres

def searchForSet(retX, retY, setNum, yr, numPce, origPrc): 
    sleep(10)
    myAPIstr = ''
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: 
                newFlag = 0
            
            listOfInv = currItem['product']['inventories']
            
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: 
            print ('problem with item %d' % i)
    
def setDataCollect(retX, retY): 
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr,yArr,numVal=10): #交叉验证
    m = len(yArr)                           
    indexList = range(m)
    errorMat = np.zeros((numVal,30))#create error mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]  #分为训练集和测试集
        testX = []; testY = []
        np.random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]]) #训练集
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]]) #测试集
                testY.append(yArr[indexList[j]])
        wMat = LinearRegres.ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = np.mat(testX); matTrainX=np.mat(trainX) #数据标准化
            meanTrain = np.mean(matTrainX,0)
            varTrain = np.var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)#test ridge results and store
            errorMat[i,k]=LinearRegres.rssError(yEst.T.A,np.array(testY))
            #print errorMat[i,k]
    meanErrors = np.mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = np.mat(xArr); yMat=np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*sum(np.multiply(meanX,unReg)) + np.mean(yMat))


if __name__ == "__main__":
    lgX = []
    lgY = []
    
    #无法连接网络
    setDataCollect(lgX,lgY)
    
    lgX = np.mat(lgX)
    lgY = np.mat(lgY).T
    
    w,constant = crossValidation(lgX,lgY)
    
    
    