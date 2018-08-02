# -*- coding: utf-8 -*-
"""

AdaBoost：horseColic数据集
@author: Jerry
"""

import AdaBoost
import matplotlib.pyplot as plt
import numpy as np

def plotROC(predStrengths, classLabels):
    cur = (1.0,1.0) #保留绘制光标的位置
    ySum = 0.0 
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas); 
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#获取排序索引
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #画图
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; 
            delY = yStep;
        else:
            delX = xStep; 
            delY = 0;
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)



if __name__ == '__main__':
    trainingMat,trainingLabels = AdaBoost.loadDataSet('horseColicTraining2.txt')
    classifierArray = AdaBoost.adaBoostTrainDS(trainingMat,trainingLabels, 10)
    
    testMat,testLabels = AdaBoost.loadDataSet('horseColicTest2.txt')
    prediction10 = AdaBoost.adaClassify(testMat, classifierArray)
    print(prediction10)
    
    plotROC(prediction10, testLabels)