
"""

@author: Jerry
"""
from numpy import *
import matplotlib.pyplot as plt  
# 加载数据集
def loadDataSet(filename):
    dataMat = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))     # 将从文件中读取后默认的string类型改为float类型
        dataMat.append(fltLine)
    return dataMat

# 计算两个向量之间的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 为给定数据集构建一个包含k个随机质心的集合
def randCent(dataSet, k):
    n = shape(dataSet)[1]           # 判断数据集有几列
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])   # 找到当前列的最小值
        rangej = float(max(dataSet[:, j]) - minJ)               # 最大值减最小值
        centroids[:, j] = minJ + rangej * random.rand(k, 1)     # 计算出随机的质心
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))     # 创建簇分配结果矩阵：第一列记录簇索引值，第二列存储误差（点到质心的距离）
    centroids = createCent(dataSet, k)      # 随机分配质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):          # 遍历所有样本
            minDist = inf
            minIndex = -1           # 初始化最小值
            for j in range(k):      # 遍历所有质心，找出距离当前样本最近的那个质心
                distJI = distMeas(centroids[j,:], dataSet[i,:])     # 计算当前质心与数据点的距离
                if distJI < minDist:
                    minDist = distJI        # 找出距离当前样本最近的那个质心
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True       #
            clusterAssment[i,:] = minIndex,minDist**2   # 记录结果
        
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]   # 返回数组a中值不为k的元素的下标
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j,:]) ** 2
    while(len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsIncurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsIncurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
            
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentTosplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentTosplit
#        print('The bestCentToSplit is：', bestCentTosplit)
#        print('the len of bestClustAss is:', len(bestClustAss))
        centList[bestCentTosplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentTosplit)[0],:] = bestClustAss
    return mat(centList), clusterAssment

def dispCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:   
        return 1  

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):
        return 1 
    
    # 绘制样本点
    for i in range(numSamples):  
        markIndex = int(clusterAssment[i, 0])  #为样本指定颜色
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    
    # 绘制质心
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
    
    plt.show()
