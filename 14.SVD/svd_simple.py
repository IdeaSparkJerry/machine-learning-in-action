# -*- coding: utf-8 -*-
"""

奇异值分解(SVD)：推荐引擎 & 图像压缩
@author: Jerry
"""
import numpy as np
from numpy import linalg as la

#-------------------------------------------------加载数据集-------------------------------------------
# 行 = [Ed Peter Tracy Fan Ming Pachi Jocelyn] (品菜师)
# 列 = [鳗鱼饭 日式炸鸡排 寿司饭 烤牛肉 手撕牛肉] (菜品)
def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

# 行 = [Brett Rob Drew Scott Mary Brent Kyle Sara Shaney Brendan Leanna] (品菜师)
# 列 = [鳗鱼饭 日式炸鸡排 寿司饭 烤牛肉 三文鱼汉堡 鲁宾三明治 印度烤鸡 麻婆豆腐 宫保鸡丁 印度奶酪咖喱 俄式汉堡] (菜品) 
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

#-------------------------------------------------相似度计算------------------------------------------- 
# 相似度计算-欧氏距离    
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

# 相似度计算-皮尔逊相关系数 
def pearsSim(inA,inB):
    if len(inA) < 3 : 
        return 1.0
    return 0.5+0.5*np.corrcoef(inA, inB, rowvar = 0)[0][1]

# 相似度计算-余弦相似度  
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

#-------------------------------------------------推荐引擎------------------------------------------- 
# 计算用户对物品的估计评分值（普通方法）
def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: 
            continue
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        
        if len(overLap) == 0: 
            similarity = 0
        else: 
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    else: 
        return ratSimTotal/simTotal

# 计算用户对物品的估计评分值（基于SVD）
def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = np.mat(np.eye(4)*Sigma[:4]) 
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: 
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else: 
        return ratSimTotal/simTotal

# 推荐引擎   
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user,:].A==0)[1]#find unrated items 
    if len(unratedItems) == 0: 
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]    
    
#-------------------------------------------------图像压缩------------------------------------------- 
# 打印矩阵
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1),
            else: 
                print(0),
        print('')

# 图像压缩
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('textSet.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)
    
if __name__ == '__main__':
    dataMat = np.mat(loadExData2())
    U,Sigma,V=la.svd(dataMat)
    print('SVD计算结果')
    print('U=',U,'\nSigma=',Sigma,'\nV=',V)
    
    print('三种相似度计算结果')
    print('欧氏相似度=',ecludSim(dataMat[:,0],dataMat[:,4]))
    print('皮氏相似度=',pearsSim(dataMat[:,0],dataMat[:,4]))
    print('余弦相似度=',cosSim(dataMat[:,0],dataMat[:,4]))
    
    print(recommend(dataMat,5))
    
#    imgCompress(2)
