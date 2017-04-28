#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np

class DataReader(object):
    def __init__(self):
        
        self.articles_old = []
        self.articles_new = []
        self.line = None
        
        
        self.files_list = ['/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111003.gz', 
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111004.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111005.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111006.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111007.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111008.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111009.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111010.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111011.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111012.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111013.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111014.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111015.gz',
        '/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111016.gz']
        
        
        self. T = 5
        self.fin = gzip.open(self.files_list[self.T],'r')
        
    def come(self):
        extra = None
        delete = None
        click = 0
        
        self.line = self.fin.readline()
        
        if not self.line:
            print(self.T, '446')
            self.T += 1
            self.fin = gzip.open(self.files_list[self.T], 'r')
            self.line = self.fin.readline()
            'If self.T >= 13, we are running out of the data.'

        cha = self.line.decode('utf-8')
        

            
        matches = re.search(r"id-(\d+)\s(\d).+user\s([\s\d]+)(.+)", cha)
        
        article = int(matches.group(1))
        click = int(matches.group(2))
        covariates = np.zeros(136)
        covariates[[int(elem) - 1 for elem in matches.group(3).split(' ') if elem != '']] = 1
        
        ####guest
        mab = False
        if sum(covariates) == 1:
            mab = True
           
        
        
        finder = re.findall(r'\|id-(\d+)', matches.group(4))
    
        self.articles_new = [int(result) for result in finder]
        
        
        
        if self.articles_new != self.articles_old:
            extra = np.setdiff1d(self.articles_new, self.articles_old)
            delete = np.setdiff1d(self.articles_old, self.articles_new)
            
            
        self.articles_old = self.articles_new
        
        
        return {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab}

    def end(self):
        self.fin.close()
    
    def update_arms(self):
        pass


class ArticlesCollector(object):
    def __init__(self):
        self.active_articles = set()
    
    def update(self, extra, delete): 
        self.active_articles = self.active_articles.difference(delete)
        self.active_articles = self.active_articles.union(extra)
        
    @property
    def active_articles(self):
        return self.active_articles
        
        
class Groups(object):
    def __init__(self, articles_collector):
        self.articles_collector = articles_collector# To handle the articles appering event
        
        self.groups = {} #dictionary of list of numpy array, article:array-like keys
        #self.groups_set = {}
        
        self.all_active_nodes = {} #dictionary, key:node
        
        self.peaks = {} #article: sets of keys
        
        
    def update(self, key, node, article):
        '''
        key_node_pairs: {key: node}
        node.__group() returns sorted array-like of indexes this node belong to. -1 if there is no such index.
        
        '''
        self.all_active_nodes.update(key, node)
        
        if not self.groups[article]:
            self.groups[article] = np.array([])
        
        G = self.groups[article]
        
        
        if node.__group(article)[0] == -1:
            G = np.append(G, np.array([key]))
        elif len(node.__group(article)) == 1:
            G[node.__group(article)[0]] = np.append(G[node.__group(article)[0]], key)
        else:
            G[node.__group(article)[0]] = np.append(G[node.__group(article)[0]], key)#Add the key first.
            
            for index in node.__group(article)[1:]:
                G[node.__group(article)[0]] = np.append(G[node.__group(article)[0]], G[index])
            for index in node.__group(article)[1:]:
                del G[index]
        
        
    def update_peaks(self):
        in_use = self.articles_collector.active_articles
        
        for article in self.groups and article in in_use:
            self.peaks[article] = self.__peaks(self.groups[article])#return a list of sets
 
    
    def __peaks(self):
        pass
    
    @property
    def get_peaks(self):
        return self.peaks
    
        
    @property
    def get_groups(self):
       return self.groups

    @property
    def all_nodes(self):
        return self.all_active_nodes
        
        
class Node(object):
    def __init__(self, key, click, article):
        self.key = key
        self.counts = {}#dictionary, article: number
        
        
        self.clicks = {}#dictionary, article: number
        
        
        self.party_yet = {}
        
        
        self.d = 4 #Basic distant parameter
        
        '''self.groups will be useless after joining the party, article: set of groups'''
        self.groups = {}
        
        self.distant_sets = [] #For look-up

        self.reachable_set = set()
        
        for i in range(self.d):
            self.distant_sets.append(set())
        for i in range(self.d):
            self.distant_lists.append(np.array([]))
        
    def update(self, reward, article, groups):
        '''
        reward: binary number
        groups: Groups object 
        '''
        if not self.counts[article]:
            self.counts[article] = 1
        else:
            self.counts[article] += 1
        
        if not self.clicks[article]:
            self.clicks[article] = reward
        else:
            self.clicks[article] += reward
        
        if not self.party_yet[article]:
            self.party_yet[article] = False
            
        if not self.party_yet[article] and self.counts[article] >= 5 and self.clicks[article]:
            self.party_yet[article] = True
            self.__join_the_party(groups, article)
        
        
    def __join_the_party(self, current_groups, article):
        'Count the distants and groups, and then update curren groups object'
        groups = current_groups.get_groups[article]#list of numpy array
        nodes = current_groups.all_nodes#dictionary, key:node
        
        self.groups[article] = set()

        for group in groups:
            for key in group:
                if key in self.reachable_set:
                    self.groups[article].add(group)
                else:
                    d = NodesUtil.distant(key, self.key)
                    if d <= self.d:
                        self.__distant_update(d, key)# added in self
                        nodes[key].__distant_update(d, self.key)# adding in theirs
                       
                        
                        self.groups[article].add(group)
                    
                    
        if len(self.groups[article]) == 0:
            self.groups[article] = -1
     
        current_groups.update(self.key, self, article)
        
    
    def __group(self, article):
        return list(self.groups[article]).sort()#list-like
    
    def __distant_update(self, distant, key):
        self.distant_sets[(distant-1)].add(key)
        self.reachable_set.add(key)
        
class NodesUtil(object):
    def __init__(self, groups):
        self.groups = groups
        
    def all_nodes(self):
        return self.groups.all_nodes
        
    @staticmethod
    def distant(key1, key2):
        key1 = np.array([int(i) for i in list(key1) if i != '\n'])
        key2 = np.array([int(i) for i in list(key2) if i != '\n'])
        
        return np.sum(np.power((key1 - key2), 2))
        
def main():
    
    
        
if __name__ == '__main__':
    main()
