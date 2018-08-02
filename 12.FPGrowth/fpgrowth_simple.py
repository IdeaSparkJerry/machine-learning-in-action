# -*- coding: utf-8 -*-
"""

FPGrowth: 简单数据集
@author: Jerry
"""
import FPGrowth

# 创建数据集
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


if __name__ == '__main__':
    simpleData = loadSimpDat()
    initDict = FPGrowth.createInitSet(simpleData)
    
    myFPTree, myHeaderTab = FPGrowth.createTree(initDict,3)
    myFPTree.disp()
    
    print('-----------------------------------------------------------------------')
    print(FPGrowth.findPrefixPath('x', myHeaderTab['x'][1]))
    print(FPGrowth.findPrefixPath('z', myHeaderTab['z'][1]))
    print(FPGrowth.findPrefixPath('r', myHeaderTab['r'][1]))
    
    print('-----------------------------------------------------------------------')
    
    freqItems = []
    FPGrowth.mineTree(myFPTree, myHeaderTab, 3, set([]), freqItems)
    print('freqItems=',freqItems)