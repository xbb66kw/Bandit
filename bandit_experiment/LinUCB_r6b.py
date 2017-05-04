
#!/usr/bin/env python

# import modules used here -- sys is a very standard one

import numpy as np
import gzip
import re
import random

from scipy import linalg

from group_cluster_bandit import DataReader
from group_cluster_bandit import ArticlesCollector
from group_cluster_bandit import Environment






class LinUCB(object):
    def __init__(self, articles_collector):
        self.articles_collector = articles_collector
        
        self.last_action = -1
        
        self.Aa = {}
        
        self.AaI = {}
        
        self.ba = {} 
        
        self.theta = {}
        
        self.k = 0
        
        self.d = 136
         
        self.alpha = 0.5
        
        self.acc_reward = 0
        

    def update(self, covariates, reward):
        
        self.Aa[self.last_action] += np.outer(covariates, covariates)
        
        self.ba[self.last_action] += reward * np.array([covariates]).T
        
        self.AaI[self.last_action] = linalg.solve(self.Aa[self.last_action],
            np.identity(self.d))
        
        self.theta[self.last_action] = np.dot(self.AaI[self.last_action],
            self.ba[self.last_action])


    def recommend(self, covariates):
        '''updating all article indexes'''
        current_articles_set = self.Aa.keys() #set of articles(string)
        extra_articles = self.articles_collector.active_articles - current_articles_set
        delete_articles = current_articles_set - self.articles_collector.active_articles
        
        for article in extra_articles:    
            self.Aa[article] = np.identity(self.d)
            self.AaI[article] = np.identity(self.d)
            self.ba[article] = np.zeros((self.d, 1))
            self.theta[article] = np.zeros((self.d, 1))
        for article in delete_articles:
            del self.Aa[article]
            del self.AaI[article]
            del self.ba[article]
            del self.theta[article]
        
        
        
        
        
        arm_keys = list(self.Aa.keys())
        x = covariates
        
        results = []
        
        for key in arm_keys:
            results.append(np.dot(self.theta[key][None,:], x)[0,0]\
                + self.alpha * np.sqrt(np.dot(x[None,:], self.AaI[key]).dot(x[:,None]))[0,0])
        
        max_ = np.argmax(reuslts)
        
        
        self.last_action = arm_keys[max_]
        
        return self.last_action, 0
    

    

 
 
 
class Agent_LinUCB(object):
    '''
    Takes Groups object and MAB object as parameters
    '''
    def __init__(self, LinUCB_object, articles_collector):
        '''
        articles_collector as the input parameter
        '''
        self.acc_reward = 0
        
        
        '''mab object is used for guests'''
        self.LinUCB_object = LinUCB_object
        
        
        self.extra_bonus = 0.0
        self.articles_collector = articles_collector
        
        
        self.last_action = '' # a string
        

    def update(self, reward, stuff):
        '''
        key is a string
        mab is a bool
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        key = stuff['key']
     
        covariates = stuff['covariates']
        
     
        self.LinUCB_object.update(covariates, reward) #self.last_action is an article string
        return 0
        
        reward = reward
        reward += self.extra_bonus
        
        
    

    def recommend(self, stuff):
        '''
        receiving a key and decide self.last_acion and self.extra_bonus
        
        key is a string
        
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        
        key = stuff['key']
        
        covariates = stuff['covariates']
        
        
        self.last_action, self.extra_bonus = self.LinUCB_object.recommend(covariates) #self.last_action is a string            
            
            
        
        return self.last_action
        
    
    def update_arms(self, stuff):
        self.articles_collector.update(stuff['extra'], stuff['delete'])
        
        


def main():
    data_reader = DataReader()
    
    
    
    AC = ArticlesCollector()
    
    
    
    LU = LinUCB(AC)
    
    E = Environment()
    A_LU = Agent_LinUCB(LU, AC)
    E.run([A_LU], data_reader)
    E.plot(len([A_LU]))
    
if __name__ == '__main__':
    main()
    
			
