# -*- coding: utf-8 -*-
"""

FPGrowth：在Twitter源中发现一些共现词 (网络问题)
@author: Jerry
"""
import twitter
import re
from time import sleep

import FPGrowth

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = 'get when you create an app'
    CONSUMER_SECRET = 'get when you create an app'
    ACCESS_TOKEN_KEY = 'get from Oauth, specific to a user'
    ACCESS_TOKEN_SECRET = 'get from Oauth, specific to a user'
    
    api = twitter.Api(consumer_key = CONSUMER_KEY,
                      consumer_secret = CONSUMER_SECRET,
                      access_token_key = ACCESS_TOKEN_KEY,
                      access_token_secret = ACCESS_TOKEN_SECRET)
    
    resultPages = []
    for i in range(1,15):
        print('feteching page %d', i)
        searchResults = api.GetSearch(searchStr, count=100, page=i)
        resultPages.append(searchResults)
        sleep(6)
    
    return resultPages

def textParse(bigString):
    urlsRemoved = re.sub('(http[s]?:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*','',bigString)
    listOfTokens = re.split(r'\W*', urlsRemoved)
    
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def mineTweets(tweetArray, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArray[i][j].text))
    
    initDict = FPGrowth.createInitSet(parsedList)
    twitterFPTree, twitterHeaderTab = FPGrowth.createTree(initDict, minSup)
    
    twitterFreqList = []
    FPGrowth.mineTree(twitterFPTree, twitterHeaderTab, set([]), twitterFreqList)
    
    return twitterFreqList
            
if __name__ == '__main__':
    lotsOfTwitters = getLotsOfTweets('RIMM')
    
    twitterFreqList = mineTweets(lotsOfTwitters, 20)