# -*- coding: utf-8 -*-
"""

Apriori算法：毒蘑菇数据集
@author: Jerry
"""

import Apriori

def loadDataSet(fileName):
    fr = open(fileName)
    dataSet = [line.split() for line in fr.readlines()]
    
    return dataSet


if __name__ == "__main__":
    mushroomDatSet = loadDataSet('mushroom.dat')
    
    L,supportData = Apriori.apriori(mushroomDatSet,0.3)  
    
    for item in L[1]:
        if item.intersection('2'):
            print(item)
