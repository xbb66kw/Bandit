#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import random 
import gzip
import re

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
        '''
        extra, delete, key are string
        '''
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
        key = np.random.normal(0,1)
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
        

        return {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':key}

    def end(self):
        self.fin.close()
    
    def update_arms(self):
        pass

        
        
        
class Groups(object):
    def __init__(self, articles_collector):
        self.articles_collector = articles_collector# To handle the articles appering event
        
        self.groups = {} #dictionary of list of numpy array, article:array-like keys
        '''
        Used for computing the peaks
        '''
        
        '''Node reads this and scan through the updated peaks accordingly.'''
        self.update_time = 0
        
        
        '''Only when two nodes are both acive under the same article they know each other.'''
        self.all_active_nodes = {} #dictionary, key:node
        
        self.peaks = {} #article: list of keys
        '''
        Used for prediction when the upcoming node is unknown for articles
        No need for grouping the peaks
        '''
        
    def __update(self, key, node, article):
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
            
            for index in range(len(node.__group(article)[1:])):
                G[node.__group(article)[0]] = np.append(G[node.__group(article)[0]], G[index])
            for index in range(len(node.__group(article)[1:])):
                del G[index]
        
    
    def __update_peaks(self):
        in_use = self.articles_collector.active_articles
        group_peaks = np.array([])
        group_values = np.array([])
        articles_index = []
        
        
        
        for article in self.groups:  #article will be key(string) 
            if article in in_use:
                mid_group_peaks, mid_group_values = NodesUtil.__peaks(self.groups[article], self.all_active_nodes, article) #return a list of keys
            
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
        return self.peaks
    
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
    def __init__(self, key, covariates, groups):
        '''
        groups, Groups object
        '''
        self.key_integers = covariates 
        self.key = key
        self.groups_object = groups #Gropus object
        self.counts = {}  #dictionary, article: number
        
        
        self.clicks = {}  #dictionary, article: number
        
        '''for updating sacnning'''
        self.update_time = 0
        
        
        self.recommended = set() #set of article string, recording all recommended articles
        self.recommending = set() #set of article string, recording all recommending articles
        
        
        
        
        self.party_yet = {}
        
        
        self.d = 4 #Basic distant parameter
        self.rho = 0.9 #deflator coefficient
        self.q = 0.8
        self.k = 1.5
        self.__timer = 100 # for groups to update
        self.counter = 0 # for groups to update
        
        '''self.groups will be useless after joining the party, article: set of groups'''
        self.groups = {}
        
        self.distant_sets = [] #list of sets
        
        self.reachable_set = set()#keys are saved here as a set
        self.reachable_list = []#nodes are saved here as a list
        
        
        for i in range(self.d):
            self.distant_sets.append(set())
        

    def update(self, reward, article):
        '''
        reward: binary number
        
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
            self.__join_the_party(article)
        
        
    def __join_the_party(self, article):
        'Count the distants and groups, and then update current groups object'
        groups = self.groups_object.get_groups[article] #list of numpy array
        nodes = self.groups_object.all_nodes #dictionary, key:node
        
        self.groups[article] = set()
        
        for index, group in enumerate(groups):
            for key in group:
                if key in self.reachable_set:#######Reuse the information which was collected for other articles.
                    self.groups[article].add(index)######
                else:
                    target_node = nodes[key]
                    
                    d = NodesUtil.__distant(target_node.key_integers, self.key_integers)
                    if d <= self.d:    
                        self.__distant_update(d, key, target_node)# added in self
                        target_node.__distant_update(d, self.key, target_node)# adding in theirs
                        
                        
                        self.groups[article].add(index)
                    
                
        if len(self.groups[article]) == 0:
            self.groups[article] = -1
        
        self.groups_object.__update(self.key, self, article)
        self.counter += 1
        
        if self.counter >= self.__timer:
            self.counter = 0
            self.groups_object.__update_peaks()

    def __group(self, article):
        return list(self.groups[article]).sort()#list-like
    
    def __distant_update(self, distant, key, node):
        self.distant_sets[(distant-1)].add(key)
        self.reachable_set.add(key)
        self.reachable_list.append(node)
    
    
    def __value(self, article, current_nodes):
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
        
        
    
    
    
    
    def recommend(self, articles_collector):
        '''
        return the recommended article(string) and the extra_bonus(could be 0)
        '''
        if self.groups_object.get_update_time > self.update_time:
            self.update_time = self.groups_object.get_update_time
            current_recommended_articles = self.groups_object.get_peaks.keys()
            current_recommended_articles = current_recommended_articles - self.recommended
            current_recommended_articles = current_recommended_articles - self.recommending
            
            for article in current_recommended_articles:
                for peak in self.groups_object.get_peaks[article]:#peak is key, key is a string
                    if NodesUtil.__distant(self.groups_object.all_active_nodes[peak].key_integers, self.key_integers) <= self.d:
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
    def __distant(covariates1, covariates2):
        return np.sum(np.power((covariates1 - covariates2), 2))
    
    
    
    @staticmethod
    def __peaks(list_of_arrays_of_keys, dic_of_all_active_nodes, article):
        '''
        return a list of raw peaks and a list of raw peak values.
        
        Not all nodes in dic_of_all_active_nodes know each other. But it is true
        when it's with respect to the same article
        '''
        
        
        active_nodes = dic_of_all_active_nodes
        
        
        group_peaks = [] #keys
        group_values = []

        
        for index in range(len(list_of_arrays_of_keys)):
            list_ = list_of_arrays_of_keys[index]
            mid_ = set()
            length = len(list_)
            
            temp_path_keys = []
            temp_path_nodes = []
            total_value = 0
            
            '''Saving the subset nodes in case we use it many times'''
            subset_nodes = {}
            current_nodes_set = set()
            record_nodes = set()
            for key in list_:     #key is string
                mid_ = active_nodes[key]
                subset_nodes.update(key, mid_)
                current_nodes_set.add(key)
                record_nodes.add(mid_)
                
                
            while current_nodes_set:
                
                values_dic = {}
                
                for key in list_:
                    if key in current_nodes_set:
                        mid_ = subset_nodes[key].__value(article, current_nodes_set)
                        values_dic.update(key, mid_)
                        average_value += mid_
                
                argmax_key = max(values_dic.iteritems(), key=operator.itemgetter(1))[0]
                
                mid_node = subset_nodes[argmax_key]#don't need to look-up twice
                temp_path_keys.append(argmax_key)
                temp_path_nodes.append(mid_node)
                
                mid_.update(mid_node.neighbors)
                mid_.add(argmax_key)
                
                current_nodes_set = current_nodes_set - mid_
            
            
        
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
            value += mid_.__value(article, subset_)
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
        
        return [trim_path_of_nodes, trim_path_of_keys]  #[list, list]
        
        
        

class Assigner(object):
    '''
    Assigner create nodes. It takes Groups object as parameter so it can assign
    the object to the created nodes.
    '''
    def __init__(self, groups_object):
        '''saving all nodes'''
        self.all_nodes = {}
        self.all_nodes_set = set()
        self.groups_object = groups_object
        
        
    def assign(self, key, covariates):
        '''
        key is a string
        '''
     
        if key in self.all_nodes_set:
            return self.all_nodes[key]
        else:
            node = Node(key, covariates, self.groups_object)
            self.all_nodes[key] = node
            self.all_nodes_set.add(key)
            return node
            
    



class ArticlesCollector(object):
    '''
    This object will be assigned to Groups and MAB object
    
    '''
    def __init__(self):
        self.active_articles = set()
    
    def update(self, extra, delete): 
        self.active_articles = self.active_articles - delete
        self.active_articles = self.active_articles.union(extra)
        
    @property
    def active_articles(self):
        return self.active_articles
        
     
    

class MAB(object):
    '''
    ArticlesCollector object will be assigned to this MAB object
    '''
    def __init__(self, articles_collector):
        self.articles_collector = articles_collector
        
        
        
        self.clicks = {}
        self.counts = {}
        
    def recommend(self):
        '''updating all article indexes'''
        
        current_articles_set = self.counts.keys()#set of articles(string)
        extra_articles = self.articles_collector.active_articles - current_articles_set
        delete_articles = current_articles_set - self.articles_collector.active_articles
        
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
        
        
    def update(self, reward, article):
        '''
        article is a string
        '''
        self.counts[article] += 1
        self.clicks[article] += reward
        
        
        

class Agent(object):
    '''
    Takes Groups object and MAB object as parameters
    '''
    def __init__(self, groups_object, mab_object):
        '''
        articles_collector is the same one in the groups_object
        '''
        self.acc_reward = 0
        self.groups_object = groups_object
        
        '''mab object is used for guests'''
        self.mab_object = mab_object
        
        
        self.assigner = Assigner(groups_object)
        
        self.articles_collector = groups_object.articles_collector
        
        
        self.last_action = '' # a string
        self.extra_bonus = 0
        
    def update(self, reward, stuff):
        '''
        key is a string
        mab is a bool
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        key = stuff['key']
        mab = stuff['mab']
        
        if mab:
            self.mab_object.update(reward, self.last_action) #self.last_action is an article string
            return 0
        
        reward = reward
        reward += self.extra_bonus
        
        
        '''
        extra_bonus only for updating, not for the final result
        '''
        node = self.assigner(key, covariates)
        node.update(reward, self.last_action)
        
        self.extra_bonus = 0
        
    def recommend(self, stuff):
        '''
        receiving a key and decide self.last_acion and self.extra_bonus
        
        key is a string
        mab is a bool
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        key = stuff['key']
        mab = stuff['mab']
        covariates = stuff['covariates']
        
        if mab:
            self.last_action, self.extra_bonus = self.mab_object.recommand() #self.last_action is a string

            
            return self.last_action
        
        node = self.assigner(key, covariates)
        
        self.last_action, self.extra_bonus = node.recommand(self.articles_collector)
        
        return self.last_action 
        
    def update_arms(self, stuff):
        self.articles_collector.update(stuff['extra'], stuff['delete'])
        
        
        
        
    
class Environment(object):
   
    def run(self, agents, data_reader, timestamp = 2000):
        self.reward_curves = np.zeros((timestamp, len(agents)))
        self.timer = np.zeros(len(agents)).astype(int)
        self.agents = agents
        times = 0
        while np.min(self.timer) < timestamp:
            #Also in this step, arms will be refreshed
            stuff = data.come()

            
            times += 1
            for i, agent in enumerate(agents):# agents can be [...]
                
                if int(np.sqrt(times)) == np.sqrt(times):
                    print(np.sqrt(times), times, self.timer, '522')
                    
                if self.timer[i] < timestamp:
                    
                    agent.update_arms(stuff)
                    
                    agent.last_action = agent.recommand(stuff)
                    
                    if agent.last_action == stuff['article']:
                        reward = stuff['click']
                        agent.update(reward, stuff)
                        agent.acc_reward += reward
                        self.reward_curves[self.timer[i], i] = agent.acc_reward / (self.timer[i] + 1)
                        
                        self.timer[i] += 1

        print('final', times, self.timer)
        
    def plot(self, number_of_agents):
        if number_of_agents == 1:
            #label_list = ['guestucb']
            label_list = ['ogaucb']
        elif number_of_agents == 2:
            label_list = ['guestucb', 'lin_ucb']
            label_list = ['guestucb', 'ucb1']
        else:
            label_list = ['ogaucb', 'ucb', 'guestucb']

        collect = {}
        for j in range(len(self.reward_curves[0,:])):
            
            collect[j], = plt.plot(self.reward_curves[:,j], label=label_list[j])
            mid_ = "/Users/apple/Desktop/bandit_experiment/bandit_modified" + str(j)
            np.save(mid_, self.reward_curves[:,j])
            
        if number_of_agents == 1:
            plt.legend(handles=[collect[0]])
        
        elif number_of_agents == 2:
            plt.legend(handles=[collect[0], collect[1]])
        else:
            plt.legend(handles=[collect[0], collect[1], collect[2]])
        
        plt.show()


    
def main():
    data_reader = DataReader()
    print(data_reader.come())
        
if __name__ == '__main__':
    main()
