# -*- coding: utf-8 -*-
"""

CART：模型树
@author: Jerry
"""
import numpy as np
import CART
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataMatExp2 = np.mat(CART.loadDataSet('exp2.txt'))
    myTreeExp2 = CART.createTree(dataMatExp2, CART.modelLeaf, CART.modelErr, ops = (1,10))
    print('myTreeExp2=',myTreeExp2)
    
    b1,w1 = myTreeExp2['left']
    b2,w2 = myTreeExp2['right']
    x = np.linspace(0,1.2,100)
    y1 = w1 * x + b1
    y2 = w2 * x + b2
    plt.plot(x,y1.T,'r.',x,y2.T,'b*')
    plt.show()
    