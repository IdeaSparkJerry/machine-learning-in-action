# -*- coding: utf-8 -*-
"""

FPGrowth：从新闻网站点击流中挖掘
@author: Jerry
"""
import FPGrowth

if __name__ == '__main__':
    fr = open('kosarak.dat','r')
    parsedData = [line.split() for line in fr.readlines()]
    
    initDict = FPGrowth.createInitSet(parsedData)
    
    newsFPTree, newsHeaderTab = FPGrowth.createTree(initDict,100000)
    newsFPTree.disp()
    
    newsFreqItems = []
    FPGrowth.mineTree(newsFPTree, newsHeaderTab, 100000, set([]), newsFreqItems)
    print('newsFreqItems=',newsFreqItems)