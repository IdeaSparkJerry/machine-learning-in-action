# -*- coding: utf-8 -*-
"""

CART：自行车速度与智力的关系
@author: Jerry
"""
import numpy as np
import CART
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  


def regTreeEval(model,inDat):
    return float(model)

def modelTreeEval(model,inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not CART.isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if CART.isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if CART.isTree(tree['right']):
             return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    m = len(testData)
    yMat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yMat[i,0] = treeForeCast(tree, np.mat(testData[i]),modelEval)
    return yMat

if __name__ == "__main__":
    trainMat = np.mat(CART.loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(CART.loadDataSet('bikeSpeedVsIq_test.txt'))
    
    myTree1 = CART.createTree(trainMat, ops = (1,20))
    y_hat1 = createForeCast(myTree1,testMat[:,0],modelEval=regTreeEval)
    corr1 = np.corrcoef(y_hat1,testMat[:,1],rowvar=0)[0,1]
    print("corr1: ",corr1)
    
    myTree2 = CART.createTree(trainMat,leafType=CART.modelLeaf,errType=CART.modelErr, ops = (1,20))
    y_hat2 = createForeCast(myTree2,testMat[:,0],modelEval=modelTreeEval)
    corr2 = np.corrcoef(y_hat2, testMat[:,1],rowvar=0)[0,1]
    print("corr2: ",corr2)
    
    w,x,y = CART.linearSolve(trainMat)
    print('w = ', w)
    
    
    fig = plt.figure()
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15) 
    ax = fig.add_subplot(111)
    ax.scatter(np.array(trainMat[:,0]), np.array(trainMat[:,1]), s=20, c='red', marker='o', alpha=0.5)  
    plt.xlabel('骑自行车的速度', fontproperties=font_set)
    plt.ylabel('智商（IQ）', fontproperties=font_set)
    plt.show()
    