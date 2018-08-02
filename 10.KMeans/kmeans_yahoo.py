# -*- coding: utf-8 -*-
"""

在线获取place.txt文件：由于网络原因，这部分无法运行
@author: Jerry
"""
import urllib
import json
from time import sleep

def geoGrab(stAddress, city): 
    apiStem = 'http://where.yahooapis.com/geocode?'  
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'ppp68N8t'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params) 
    yahooApi = apiStem + url_params      
    print(yahooApi)
    
    c=urllib.request.urlopen(yahooApi) 
    
    return json.loads(c.read().decode("utf-8"))

def massPlaceFind(fileName):
    fw = open('place.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print ("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: 
            print ("error fetching")
        sleep(1)
    fw.close()


if __name__ == '__main__':
    geoResults = geoGrab('1 VA Center', 'Augusta, ME')
    massPlaceFind('portlandClubs.txt')  #生成place.txt文件