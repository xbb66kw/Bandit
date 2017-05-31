#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import gzip
import re
import random

from logistic_high_di import HighDimensionalLogisticRegression



class DataReader(object):
    def __init__(self):
        
        self.articles_old = set()
        self.articles_new = set()
        self.line = None
        
        
        self.files_list = ['/Users/xbb/Desktop/bandit_experiment/r6b_18k_7day.txt']
        
        
        self. T = 0
        self.fin = open(self.files_list[self.T],'r')

    def come(self):
        '''
        extra, delete, key are string
        '''
        extra = set()
        delete = set()
        click = 0
        
        self.line = self.fin.readline()
        
        if not self.line:            
            self.T += 1
            self.fin = gzip.open(self.files_list[self.T], 'r')
            self.line = self.fin.readline()
            'If self.T >= 13, we are running out of the data.'
        
        cha = self.line
        
        
            
        matches = re.search(r"id-(\d+)\s(\d).+user\s([\s\d]+)(.+)", cha)
        
        article = int(matches.group(1))
        click = int(matches.group(2))
        covariates = np.zeros(136).astype(int)
        
        covariates[[int(elem) - 1 for elem in matches.group(3).split(' ') if elem != '']] = 1
        key = ''.join(map(str, covariates))
        
        
        
        
        ####guest
        mab = False
        if sum(covariates) == 1:
            mab = True
           
        
        
        finder = re.findall(r'\|id-(\d+)', matches.group(4))
        
        self.articles_new = set([int(result) for result in finder])
        

        
        if self.articles_new != self.articles_old:
            extra = self.articles_new - self.articles_old
            delete = self.articles_old - self.articles_new
                      
        self.articles_old = self.articles_new
        

        return {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':key}






class Environment(object):
    
    def run(self, agents, data_reader, timestamp = 70000):
        self.reward_curves = np.zeros((timestamp, len(agents)))
        self.timer = np.zeros(len(agents)).astype(int)
        self.agents = agents
        times = 0
        while np.min(self.timer) < timestamp:
            #Also in this step, arms will be refreshed
            stuff = data_reader.come()
    
            
            times += 1
            for i, agent in enumerate(agents):# agents can be [...]
                
                if int(np.sqrt(times)) == np.sqrt(times):
                    print(np.sqrt(times), times, self.timer, agent.acc_reward, '714')
                    
                if self.timer[i] < timestamp:
                    
                    agent.update_arms(stuff)
                    
                    agent.last_action = agent.recommend(stuff)
                    
                    if agent.last_action == stuff['article']:
                        reward = stuff['click']
                        agent.update(reward, stuff)
                        agent.acc_reward += reward
                        self.reward_curves[self.timer[i], i] = agent.acc_reward / (self.timer[i] + 1)
                        
                        self.timer[i] += 1
    
        print('final', times, self.timer)
        
    def plot(self, number_of_agents):
        if number_of_agents == 1:
            label_list = ['Logistic']
        elif number_of_agents == 2:
            label_list = ['Logistic', 'ucb1']
        
        collect = {}
        for j in range(len(self.reward_curves[0,:])):
            
            collect[j], = plt.plot(self.reward_curves[:,j], label=label_list[j])
            mid_ = "/Users/xbb/Desktop/bandit_experiment/model_selection_clustering/third" + str(j)
            np.save(mid_, self.reward_curves[:,j])
            
        if number_of_agents == 1:
            plt.legend(handles=[collect[0]])
        
        elif number_of_agents == 2:
            plt.legend(handles=[collect[0], collect[1]])
        else:
            plt.legend(handles=[collect[0], collect[1], collect[2]])
        
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,0.1))
        plt.show()


class ArticlesCollector(object):
    '''
    This object will be assigned to Groups and MAB object
    
    '''
    def __init__(self):
        self.__active_articles = set()
        self.__extras = set()
        self.__deletes = set()
        
    def update(self, extra, delete): 
        self.__active_articles = self.__active_articles - delete
        self.__active_articles = self.__active_articles.union(extra)
        self.__extras = extra
        self.__deletes = delete
        
    @property
    def active_articles(self):
        return self.__active_articles
    
    @property
    def extras(self):
        return self.__extras
    
    @property
    def deletes(self):
        return self.__deletes
    
    def reset(self):
        self.__deletes = set()
        self.__extras = set()
        
        


###########
####MAB####

class MAB(object):
    '''
    ArticlesCollector object will be assigned to this MAB object
    '''
    def __init__(self, articles_collector, alpha=0.2):
        self.articles_collector = articles_collector
    
        self.clicks = {}
        self.counts = {}
        
        
        self.alpha = alpha
        
        
    def recommend(self):
        '''updating all article indexes'''
        
        
        
        values = np.array([])
        articles = []
        for article in self.counts.keys():
            sum_ = sum(self.clicks.values())
            values = np.append(values, self.clicks[article] / self.counts[article] + self.alpha * np.sqrt(np.log(sum_)/(self.counts[article]+1)))
            articles.append(article)
            
        
        
        return articles[np.argmax(values)]

    def update(self, reward, article):
        '''
        article is a numbe
        '''
        
      
        self.counts[article] += 1
        self.clicks[article] += reward
        
        
    '''While the node hasnt made its own decision, it still require arms updating'''
    def articles_update(self, articles_collector):
        '''updating all article indexes'''
        
        current_articles_set = self.counts.keys()#set of articles(string)
        extra_articles = articles_collector.active_articles - current_articles_set
        delete_articles = current_articles_set - articles_collector.active_articles
        

        for article in extra_articles:    
            self.counts[article] = 1
            self.clicks[article] = 0
        for article in delete_articles:
            del self.counts[article]
            del self.clicks[article]

        


class Agent(object):
    '''
    Takes Groups object and MAB object as parameters
    '''
    def __init__(self, mab_object, articles_collector):
        
        '''
        articles_collector is the same one in the groups_object
        '''
        
        self.acc_reward = 0
       
        
        '''mab object is used for guests'''
        self.mab_object = mab_object
        
      
        self.articles_collector = articles_collector
        
        
        self.last_action = '' # a string
    
        
        
    def update(self, reward, stuff):
        
        '''
        key is a string
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        
        key = stuff['key']        
        covariates = stuff['covariates']
        
       
        
        '''MAB can share the information'''
        self.mab_object.update(reward, self.last_action) #self.last_action is an article string
        
        
    
    
    def recommend(self, stuff):
        '''
        receiving a key and decide self.last_acion and self.extra_bonus
        
        key is a string
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        
        key = stuff['key']
        covariates = stuff['covariates']
        
        self.mab_object.articles_update(self.articles_collector)
        
        self.last_action = self.mab_object.recommend()        
        
        
        return self.last_action
        
        

    def update_arms(self, stuff):
        self.articles_collector.update(stuff['extra'], stuff['delete'])
        
    







     
def main():
				A = ArticlesCollector()
				DR = DataReader()				
				E = Environment()
				
				
				
				##MAB
				M = MAB(A, 0.2)
				Ag = Agent(M, A)
				
				E.run([Ag], DR)
				E.plot(len([Ag]))
				
if __name__ == '__main__':
    main()