# -*- coding: utf-8 -*-
"""

kNN: 电影分类
@author: Jerry
"""

import numpy as np
import kNN

# 创建数据集
def createDataSet():
    #[笑脸镜头 高科技镜头 接吻镜头 打斗镜头]
    features = np.array([[5,10,32,114],
                      [2,5,23,150],
                      [1,9,8,154],
                      [121,10,12,11],
                      [98,2,20,5],
                      [4,97,14,10],
                      [8,110,13,23],
                      [9,100,5,1],
                      [4,5,90,5],
                      [1,3,88,10]])
    labels = ["动作片","动作片","动作片","喜剧片","喜剧片","科幻片","科幻片","科幻片","爱情片","爱情片"]
    return features, labels

if __name__ == '__main__':
    features,labels = createDataSet()
    
    input = np.array([5,100,12,6])
    k = 3
    
    label = kNN.classify(input, features, labels, k)
    print('预测结果：',label)
    


