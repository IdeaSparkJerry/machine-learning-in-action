# -*- coding: utf-8 -*-
"""

Apriori：简单数据集
@author: Jerry
"""

import Apriori

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]


if __name__ == "__main__":
    dataSet = loadDataSet()
    
    L,supportData = Apriori.apriori(dataSet,0.5)  
    print('L=',L)
    print('-----------------------------------------------------')

    rules=Apriori.generateRules(L,supportData,minConf=0.5)
    print('\nrules=',rules)