# -*- coding: utf-8 -*-
"""
CART：简单数据集
@author: Jerry
"""

import numpy as np
import CART

if __name__ == "__main__":
    dataMat00 = np.mat(CART.loadDataSet('ex00.txt'))
    myTree00 = CART.createTree(dataMat00)
    print('myTree00=',myTree00)
    
    print('---------------------------------------------------------------------------------------------')
    dataMat0 = np.mat(CART.loadDataSet('ex0.txt'))
    myTree0 = CART.createTree(dataMat0)
    print('myTree0=',myTree0)
    
    print('---------------------------------------------------------------------------------------------')
    dataMat2 = np.mat(CART.loadDataSet('ex2.txt'))
    myTree2 = CART.createTree(dataMat2)
    print('myTree2=',myTree2)

    print('---------------------------------------------------------------------------------------------')
    dataMatTest2 = np.mat(CART.loadDataSet('ex2test.txt'))
    myTreeTest2 = CART.createTree(dataMatTest2)
    myPrunedTree = CART.prune(myTreeTest2,dataMatTest2)
    print('myPrunedTree=',myPrunedTree)
    