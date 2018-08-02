# -*- coding: utf-8 -*-
"""

Apriori：国会投票模式分析
@author: Jerry
"""

#from time import sleep
#from votesmart import votesmart
import Apriori

#def getActionIds():
#    actionIdList = []
#    billTitleList = []
#    fr = open('recent20bills.txt') 
#    for line in fr.readlines():
#        billNum = int(line.split('\t')[0])
#        try:
#            billDetail = votesmart.votes.getBill(billNum) #api call
#            for action in billDetail.actions:
#                if action.level == 'House' and \
#                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
#                    actionId = int(action.actionId)
#                    print ('bill: %d has actionId: %d' % (billNum, actionId))
#                    
#                    actionIdList.append(actionId)
#                    billTitleList.append(line.strip().split('\t')[1])
#        except:
#            print ("problem getting bill %d" % billNum)
#        sleep(1)                                      #delay to be polite
#    return actionIdList, billTitleList
#     
#def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
#    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
#    for billTitle in billTitleList:#fill up itemMeaning list
#        itemMeaning.append('%s -- Nay' % billTitle)
#        itemMeaning.append('%s -- Yea' % billTitle)
#    transDict = {}#list of items in each transaction (politician) 
#    voteCount = 2
#    for actionId in actionIdList:
#        sleep(3)
#        print ('getting votes for actionId: %d' % actionId)
#        try:
#            voteList = votesmart.votes.getBillActionVotes(actionId)
#            for vote in voteList:
#                if not transDict.has_key(vote.candidateName): 
#                    transDict[vote.candidateName] = []
#                    if vote.officeParties == 'Democratic':
#                        transDict[vote.candidateName].append(1)
#                    elif vote.officeParties == 'Republican':
#                        transDict[vote.candidateName].append(0)
#                if vote.action == 'Nay':
#                    transDict[vote.candidateName].append(voteCount)
#                elif vote.action == 'Yea':
#                    transDict[vote.candidateName].append(voteCount + 1)
#        except: 
#            print ("problem getting actionId: %d" % actionId)
#        voteCount += 2
#    return transDict, itemMeaning

def loadDataSet(fileName):
    fr = open(fileName)
    dataSet = [line.split() for line in fr.readlines()]
    
    return dataSet

if __name__ == "__main__":
    votingDatSet = loadDataSet('bills20DataSet.txt')
    
    L,supportData = Apriori.apriori(votingDatSet,minSupport=0.5)  
    print(L)
    
    rules=Apriori.generateRules(L,supportData,minConf=0.5)
    print(rules)
