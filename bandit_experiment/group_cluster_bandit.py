#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import random 

import operator#for argmax dictionary

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
        self.active_articles = self.active_articles - delete
        self.active_articles = self.active_articles.union(extra)
        
    @property
    def active_articles(self):
        return self.active_articles
        
        
        
        
        
        
        
        
        
        
class Groups(object):
    def __init__(self, articles_collector):
        self.articles_collector = articles_collector# To handle the articles appering event
        
        self.groups = {} #dictionary of list of numpy array, article:array-like keys
        '''
        Used for computing the peaks
        '''
        
        '''Node reads this and scan through the updated peaks accordingly.'''
        self.update_time = 0
        
        self.all_active_nodes = {} #dictionary, key:node
        
        self.peaks = {} #article: list of keys
        '''
        Used for prediction when the upcoming node is unknown for articles
        No need for grouping the peaks
        '''
        
    def update(self, key, node, article):
        '''
        key_node_pairs: {key: node}
        node.__group() returns sorted array-like of indexes this node belong to. -1 if there is no such index.
        
        '''
        self.all_active_nodes.update(key, node)
        
        if not self.groups[article]:
            self.groups[article] = []
        G = self.groups[article]
        
        
        if node.__group(article)[0] == -1:
            G = np.append(G, np.array([key]))
        elif len(node.__group(article)) == 1:
            G[node.__group(article)[0]] = np.append(G[node.__group(article)[0]], key)
        else:
            G[node.__group(article)[0]] = np.append(G[node.__group(article)[0]], key) # Add the key first.
            
            for index in node.__group(article)[1:]:
                G[node.__group(article)[0]] = np.append(G[node.__group(article)[0]], G[index])
            for index in node.__group(article)[1:]:
                del G[index]
        
        
    def update_peaks(self):
        in_use = self.articles_collector.active_articles
        group_peaks = np.array([])
        group_values = np.array([])
        articles_index = []
        
        for article in self.groups and article in in_use:
            mid_group_peaks, mid_group_values = NodesUtil.peaks(self.groups[article], self.all_active_nodes, article) #return a list of keys
            
            for index in range(len(mid_group_peaks)):
                group_peaks = np.append(group_peaks, mid_group_peaks[index])
                group_values = np.append(group_values, mid_group_values[index])
                articles_index.append(article) #article is a string
                
                
        '''Find the k-sigma high peaks and return them as group_peaks'''
        threshold = self.k * np.std(group_values) + np.mean(group_values)
        subset = group_values >= threshold
        
        
        self.peaks = {}#refresh
        
        for index in range(len(articles_index)):
            if subset[index]:
                article = articles_index[index]
                self.peaks[article].append(group_peaks[index])
                
        self.update_time += 1
            
        #inform the active nodes the updating info.
        #self.notify_active_nodes()
    
    @property
    def get_update_time(self):
        return self.update_time
    
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
    def __init__(self, key, click, article, groups):
        '''
        groups, Groups object
        '''
        
        self.key = key
        self.groups = groups
        self.counts = {}#dictionary, article: number
        
        
        self.clicks = {}#dictionary, article: number
        
        '''for updating sacnning'''
        self.update_time = 0
        
        '''working'''
        self.recommended = set() #set of article string, recording all recommended articles
        self.recommending = set() #set of article string, recording all recommending articles
        
        
        
        
        self.party_yet = {}
        
        
        self.d = 4 #Basic distant parameter
        self.rho = 0.9 #deflator coefficient
        self.q = 0.8
        self.k = 1.5
        
        '''self.groups will be useless after joining the party, article: set of groups'''
        self.groups = {}
        
        self.distant_sets = [] #list of lists

        self.reachable_set = set()#keys are saved here as a set
        self.reachable_list = []#nodes are saved here as a list
        
        
        for i in range(self.d):
            self.distant_sets.append(set())
        '''
        for i in range(self.d):
            self.distant_lists.append(np.array([]))
        '''
        
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
        'Count the distants and groups, and then update current groups object'
        groups = current_groups.get_groups[article] #list of numpy array
        nodes = current_groups.all_nodes #dictionary, key:node
        
        self.groups[article] = set()

        for group in groups:
            for key in group:
                if key in self.reachable_set:
                    self.groups[article].add(group)
                else:
                    d = NodesUtil.distant(key, self.key)
                    if d <= self.d:
                        target_node = nodes[key]
                        self.__distant_update(d, key, target_node)# added in self
                        target_node.__distant_update(d, self.key, target_node)# adding in theirs
                       
                        
                        self.groups[article].add(group)
                    
                    
        if len(self.groups[article]) == 0:
            self.groups[article] = -1
     
        current_groups.update(self.key, self, article)
        
    
    def __group(self, article):
        return list(self.groups[article]).sort()#list-like
    
    def __distant_update(self, distant, key, node):
        self.distant_sets[(distant-1)].add(key)
        self.reachable_set.add(key)
        self.reachable_list.append(node)
    
    
    def value(self, article, current_nodes):
        '''
        current_nodes: set of keys
        '''
        values = np.array([])
        for node in self.reachable_list:
            if node.key in current_nodes: 
                values = np.append(values, append(node.average(article)))
            
        values = np.append(values, self.average(article))
        
        length = len(values)
        
        return np.sort(values)[np.floor(self.q * length)] * np.log(length+1)
        
        
    
    
    
    
    '''working'''
    def recommend(self, articles_collector):
        '''
        return the recommended article(string) and the extra_bonus(could be 0)
        '''
        if self.groups.get_update_time > self.update_time:
            self.update_time = self.groups.get_update_time
            current_recommended_articles = self.groups.get_peaks.keys()
            current_recommended_articles = current_recommended_articles - self.recommended
            current_recommended_articles = current_recommended_articles - self.recommending
            
            for article in current_recommended_articles:
                for peak in self.groups.get_peaks[article]:#peak is key, key is a string
                    if NodesUtil.distant(peak, self.key) <= self.d:
                        self.recommending.add(article)
       
       if self.recommending:
           extra_bonus = 1.5
           
           choice = random.sample(self.recommending, 1)[0]
           self.recommending.remove(choice)
           
           return choice, extra_bonus
               
       else:
           '''updating all article indexes'''
           
           current_articles_set = self.counts.keys()#set of articles(string)
           extra_articles = articles_collector.active_articles - current_articles_set
           delete_articles = current_articles_set - articles_collector.active_articles
           for article in extra_articles:    
               self.counts[article] = 0
               self.clicks[article] = 0
           for article in delete_articles:
               del self.counts[article]
               del self.clicks[article]
               
           extra_bonus = 0.0
           values = np.array([])
           articles = []
           for article in self.counts.keys():
               values = np.append(values, self.clicks[article] / self.counts[article] + self.alpha * np.sqrt(np.log(self.counts[article])/(self.counts[article]+1)))
               articles.append(article)
           
           return articles[np.argmax(values)], extra_bonus
           
           
    @property
    def neighbors(self):
        return self.reachable_set
    @property
    def key(self):
        return self.key
    
    def clicks(self, article):
        return self.clicks[article]
    def counts(self, article):
        return self.counts[article]
    def average(self, article):
        return self.clicks[article] / self.counts[article]
    
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
    
    
    
    @staticmethod
    def peaks(list_of_arrays_of_keys, dic_of_all_active_nodes, article):
        '''
        return a list of raw peaks and a list of raw peak values.
        
        '''
        
        
        active_nodes = dic_of_all_active_nodes
        
        
        group_peaks = []'''keys'''
        group_values = []
        
        
        for index in list_of_arrays_of_keys:
            list_ = list_of_arrays_of_keys[index]
            mid_ = set()
            length = len(list_)
            
            temp_path_keys = []
            temp_path_nodes = []
            total_value = 0
            
            '''Saving the subset nodes in case we use it many times'''
            subset_nodes = {}
            current_nodes = set()
            record_nodes = set()
            for key in list_:
                mid_ = active_nodes[key]
                subset_nodes.update(key, mid_)
                current_nodes.add(key)
                record_nodes.add(mid_)
                
            while current_nodes:
                
                values_dic = {}
                
                for key in current_nodes:
                    mid_ = subset_nodes[key].value(article, current_nodes)
                    values_dic.update(key, mid_)
                    average_value += mid_
                
                argmax_key = max(values_dic.iteritems(), key=operator.itemgetter(1))[0]
                
                mid_node = subset_nodes[argmax_key]#don't need to look-up twice
                temp_path_keys.append(argmax_key)
                temp_path_nodes.append(mid_node)
                
                mid_.update(mid_node.neighbors)
                mid_.add(argmax_key)
                
                current_nodes = current_nodes - mid_
            
            

            resulted_path_nodes, resulted_path_keys = NodesUtil.__path_trim(temp_path_nodes, temp_path_keys, record_nodes, article, total_value/length)
            
            
            group_values.append(NodesUtils.__path_value(resulted_path_nodes, record_nodes, article))
            group_peaks.append(resulted_path_keys)
        
        
        
        
        return group_values, group_peaks
        
        
        
 
    @staticmethod
    def __path_value(path_of_nodes, subset_nodes, article):
        '''
        subset_nodes, set of nodes
        path_of_nodes, list of nodes
        '''
        value = 0
        subset_ = subset_nodes
        length = len(path_of_nodes)
        
        for index in range(length):
            mid_ = path_of_nodes[index]
            value += mid_.value(article, subset_)
            subset_ = subset_ - mid_.neighbors  
            
        
        return value
  
  
    @staticmethod
    def __path_trim(path_of_nodes, path_of_keys, subset_nodes, article, average_value):
        '''
        article, string
        path_of_nodes, list of nodes
        path_of_keys, list of keys
        '''
        subset_ = subset_nodes
        
        full_value = NodesUtil.__path_value(path_of_nodes, subset_, article)
        
        values = []
        
        
        length = len(path_of_nodes)

        idx = np.arange(1, length)\
            - np.tri(length, length - 1, k=-1, dtype=bool)
            
        for index in range(length):
            values.append(NodesUtil.__path_value(np.array(path_of_nodes)[idx[index,:]], subset_, article))
        
        bool_vector = np.array(values) + average_value <= full_value
        trim_path_of_nodes = np.array(path_of_nodes)[bool_vector]   
        trim_path_of_keys = np.array(path_of_keys)[bool_vector]
        
        return [trim_path_of_nodes, trim_path_of_keys]#[list, list]
        
    
    
    
def main():
    
    
        
if __name__ == '__main__':
    main()
