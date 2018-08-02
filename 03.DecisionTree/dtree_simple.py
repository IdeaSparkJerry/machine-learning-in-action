# -*- coding: utf-8 -*-
"""

决策树：贷款申请
@author: Jerry
"""

'''
数据集信息： 
特征+取值：年龄： (1) 老年-2, (2) 中年-1, (3) 青年-0 
          有工作：(1) 是-1, (2) 否-0
          有房子：(1) 是-1, (2) 否-0 
          信贷情况：(1) 非常好-2, (2) 好-1，(3)一般-0 
类别：是否同意贷款：(1)是-'yes'，(2)否-'no'
'''

import DecisionTree


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['age', 'job', 'house', 'credit'] 

    return dataSet, labels


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    
    print('香农熵为：',DecisionTree.calcShannonEnt(dataSet))
    print("最优特征索引值:" + str(DecisionTree.chooseBestFeatureToSplit(dataSet)))
    
    myTree = DecisionTree.createTree(dataSet, labels)
    print (myTree)
    
    testVec = [0,0]       
    finalFeatLabels =['house', 'job']
    result = DecisionTree.classify(myTree, finalFeatLabels, testVec)
    print("是否同意贷款:" +result)
