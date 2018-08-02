# -*- coding: utf-8 -*-
"""

决策树：预测隐形眼镜类型-sklearn
@author: Jerry
"""

from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pandas as pd
import DecisionTree

'''
数据集信息： 
特征+取值：age（年龄）： (1) young=2, (2) pre-presbyopic=0, (3) presbyopic=1 
          prescript（症状）：(1) myope=1, (2) hypermetrope=0 
          astigmatic（是否散光）：(1) no=0, (2) yes=1 
          tearRate（眼泪数量）：(1) reduced=1, (2) normal=0 
类别：hard(硬材质)、soft(软材质)、no lenses(不适合佩戴隐形眼镜)
'''

# sklearn实现
def lensesTest1():
    fr = open('lenses.txt','r')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']#特征标签

    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append((each[lensesLabels.index(each_label)]))
        lenses_dict[each_label] = lenses_list
        lenses_list = []

    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    
    print('\n离散化后的结果：\n')

    le = LabelEncoder()                                     #创建LabelEncoder()对象          
    for col in lenses_pd.columns:                             #按列离散化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)
    
    clf = tree.DecisionTreeClassifier(max_depth = 4)               
    model = clf.fit(lenses_pd,lenses_target)

    #预测
    label = model.predict([[1,1,1,0]])
    print('预测结果为',label)

# 
def lensesTest2():
    file = open('lenses.txt','r')
    dataSet = [inst.strip().split('\t') for inst in file.readlines()]  
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']#特征标签
    
    print('香农熵为：',DecisionTree.calcShannonEnt(dataSet))
    print("最优特征索引值:" + str(DecisionTree.chooseBestFeatureToSplit(dataSet)))
    
    myTree = DecisionTree.createTree(dataSet, labels)
    print(myTree)
    
    testVec = ['normal','no','presbyopic','myope']       
    finalFeatLabels =['tearRate', 'astigmatic', 'age', 'prescript']
    result = DecisionTree.classify(myTree, finalFeatLabels, testVec)
    print("预测结果为" +result)


if __name__ == '__main__':
    lensesTest1()
    
#    lensesTest2()