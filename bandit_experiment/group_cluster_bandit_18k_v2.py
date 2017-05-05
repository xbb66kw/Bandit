#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import random 
import gzip
import re

import operator#for argmax dictionary

class DataReader(object):
    def __init__(self):
        
        self.articles_old = set()
        self.articles_new = set()
        self.line = None
        
        
        self.files_list = ['/Users/xbb/Desktop/bandit_experiment/r6b_18k.txt']
        
        
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
        
        
        #print(key in BB, '68')
        
        ####guest
        mab = False
        if sum(covariates) == 1:
            mab = True
           
        
        
        finder = re.findall(r'\|id-(\d+)', matches.group(4))
        
        self.articles_new = set([int(result) for result in finder])
        

        
        if self.articles_new != self.articles_old:
            extra = self.articles_new - self.articles_old
            delete = self.articles_old - self.articles_new
            #print(self.articles_new, extra, '85')  
        
        #print(self.articles_new, extra, '87')                
        self.articles_old = self.articles_new
        

        return {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':key}


        
        
class Groups(object):
    def __init__(self, articles_collector):
        self.all_nodes_hitting_record = {}####TETS#######
        
        
        self.begin = False
        
        self.articles_collector = articles_collector# To handle the articles appering event
        
        self.groups = np.array([]) #list of numpy array, array-like keys
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
        
        '''For self.update_peaks'''
        self.counter = 0

        self.k = 0.5
        
        
        
    def update(self, key, node):
        '''
        key_node_pairs: {key: node}
        node.__group() returns sorted array-like of indexes this node belong to. -1 if there is no such index.
        
        '''
        self.all_active_nodes.update({key: node})
        
        
        
        
        if node.group()[0] == -1:
            self.groups.append([key])
        elif len(node.group()) == 1:
            self.groups[node.group()[0]].append(key)
        else:
            self.groups[node.group()[0]].append(key) # Add the key first.
            
            for index in range(len(node.group()[1:])):
                self.groups[node.group()[0]].extend(self.groups[index])##extend!
            for index in range(len(node.group()[1:])):
                del self.groups[index]
        
        
        
        
    def update_peaks(self):
        in_use = self.articles_collector.active_articles
        group_peaks = np.array([])
        group_values = np.array([])
        articles_index = []
        
        
        
        
            
								mid_group_values, mid_group_peaks = NodesUtil.peaks(self.groups, self.all_active_nodes) #return a list of keys
				
								for index in range(len(mid_group_peaks)):
												group_peaks = np.append(group_peaks, mid_group_peaks[index])
												group_values = np.append(group_values, mid_group_values[index])
												
                    
                    
       
        
        '''Find the k-sigma high peaks and return them as group_peaks'''
        threshold = self.k * np.std(group_values) + np.mean(group_values)
        subset = group_values >= threshold
        
        
        
        #refresh
        self.peaks = []
            
        for index in range(len(articles_index)):
            if subset[index]:
                article = articles_index[index]
                self.peaks[article].append(group_peaks[index])
                
        self.update_time += 1
        print(group_values, '193')
        print([len(self.peaks[key]) for key in self.peaks], np.sum([len(self.peaks[key]) for key in self.peaks]), '194')
        return self.peaks
        
    
    def articles_update(self, articles_collector):
    				pass
    
        '''updating the articles for self.groups and self.peaks'''    
        '''
        current_articles_set = self.groups.keys()#set of articles(string)
        extra_articles = articles_collector.active_articles - current_articles_set
        delete_articles = current_articles_set - articles_collector.active_articles
        
        for article in extra_articles:            
            self.groups[article] = []
            self.peaks[article] = []   
        for article in delete_articles:
            del self.groups[article]
            del self.peaks[article]
        '''
        

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
    def __init__(self, key, covariates, groups, mab_object):
        '''
        groups, Groups object
        '''
        self.key_integers = covariates 
        self.__key = key
        self.groups_object = groups #Gropus object
        self.mab_object = mab_object#All nodes share one single mab_object
        self.counts = {}  #dictionary, article: number
        

        self.clicks = {}  #dictionary, article: number
        
        #self.extra_bonus = {}
        
        '''for updating scanning'''
        self.update_time = 0
        
        
        self.recommended = set() #set of article string, recording all recommended articles
        self.recommending = set() #set of node, recording all recommending peaks
        
        self.__decided = False #Before the node receive any network recommendation or become a peak, it takes advice from the joint MAB 
        
        
        self.party_yet = False
        
        self.alpha = 0.5
        self.d = 3 #Basic distant parameter
        self.rho = 0.9 #deflator coefficient
        self.q = 0.8
        self.__timer = 100 # for groups to update
        
        
        
        
        '''self.groups will be useless after joining the party, set of groups'''
        self.groups = set()
        
        self.distant_sets = [] #list of sets
        
        self.reachable_set = set()#keys are saved here as a set
        self.reachable_list = []#nodes are saved here as a list
        
        
        for i in range(self.d):
            self.distant_sets.append(set())
        
        
    def update(self, reward, article, through_recommended):
        '''
        reward: binary number
        
        '''
        
        '''articles updating has already been done in the recommend step'''
        self.counts[article] += 1
        self.clicks[article] += reward
        
        '''For testing'''
        self.groups_object.counter += 1
        
        
        
        if self.groups_object.counter >= self.__timer and self.groups_object.begin:
            self.groups_object.counter = 0
            self.groups_object.update_peaks()
        
        
        
        
        ####TESRT########
        self.groups_object.all_nodes_hitting_record[self.__key] += 1
        

        if not self.party_yet and (self.clicks[article] / self.counts[article]) >= 0.06 and self.counts[article] > 4:            
            print(self.groups_object.counter, self.__key, article, 'join the party', '295')
            self.groups_object.begin = True
            self.party_yet = True
            self.__join_the_party()
    
    def __join_the_party(self):
        'Count the distants and groups, and then update current groups object'
        self.groups_object.articles_update(self.groups_object.articles_collector)
        
        groups = self.groups_object.get_groups #list of numpy array
        nodes = self.groups_object.all_nodes #dictionary, key:node
        
        #self.groups[article] = set()
        
        for index, group in enumerate(groups):
            for key in group:
                if key in self.reachable_set:#######Reuse the information which was collected for other articles.
                    self.groups.add(index)######
                else:
                    target_node = nodes[key]
                    
                    d = NodesUtil.distant(target_node.key_integers, self.key_integers)
                    if d <= self.d:    
                        self.distant_update(d, key, target_node)# added in self
                        target_node.distant_update(d, self.__key, target_node)# adding in theirs
                        
                        
                        self.groups.add(index)
                    
                
        if len(self.groups) == 0:
            self.groups = [-1]
        
        self.groups_object.update(self.__key, self)
        
    
    def group(self):
        return np.sort(list(self.groups))#list-like
    
    def distant_update(self, distant, key, node):
        self.distant_sets[(distant-1)].add(key)
        self.reachable_set.add(key)
        self.reachable_list.append(node)
    
    
    def value(self, current_nodes):
        '''
        current_nodes: set of keys
        '''
        values = np.array([])
        for node in self.reachable_list:
            if node.key in current_nodes: 
                values = np.append(values, node.overall_average)
            
        values = np.append(values, self.overall_average)
        
        length = len(values)

        return np.sort(values)[np.floor(self.q * length).astype(int)]# * np.log(length+1)
        
     
    
    
    
    
    def recommend(self, articles_collector):
        '''
        return the recommended article(string) and the extra_bonus(could be 0)
        '''
    
        if self.groups_object.get_update_time > self.update_time:
            '''refresh'''
            self.recommending = set()
            
            self.update_time = self.groups_object.get_update_time
            current_recommended_peaks = self.groups_object.get_peaks
            current_recommended_peaks = current_recommended_peaks - self.recommended
            current_recommended_peaks = current_recommended_peaks - self.recommending
            
            for node in current_recommended_peaks:
																if NodesUtil.distant(node.key_integers, self.key_integers) <= self.d:
																				self.recommending.add(node)
        
            if self.recommending:
            				'''We can check, self.recommending is supposed to be a singleton'''
                print(self.recommending, 'recommended', '406')
                
                all_neighbor_counts = {}
                all_neighbor_clicks = {}
                for article in self.counts.keys():
                				all_neighbor_counts[article] = 0
                				all_neighbor_clicks[article] = 0
                for node in self.recommending:
                				for key in node.neighbors:
                								counts_mid = self.groups_object.all_active_nodes[key]
                								clicks_mid = self.groups_object.all_active_nodes[key]
                								for article in counts_mid.keys():
																												all_neighbor_counts[article] += counts_mid.counts_dictionary[article]
																												all_neighbor_clicks[article] += clicks_mid.clicks_dictionary[article]
                								

                
                values = np.array([])
                articles = []
                for article in self.counts.keys():
                    values = np.append(values, all_neighbor_clicks[article] / all_neighbor_counts[article] + self.alpha * np.sqrt(1/(all_neighbor_counts[article]+1)))
                    articles.append(article)
                
                
                
                
                return articles[np.argmax(values)],  True
        
        return self.mab_object.recommend()
    
    '''While the node hasnt made its own decision, it still require arms updating'''
    def articles_update(self, articles_collector):
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
            
    @property
    def decided(self):
        return self.__decided
        
    @property
    def neighbors(self):
        return self.reachable_set
        
    @property
    def key(self):
        return self.__key
    
    def clicks(self, article):
        return self.clicks[article]
    def counts(self, article):
        return self.counts[article]
        
    @property
    def clicks_dictionary(self):    				
    				return self.clicks
    @property
    def counts_dictionary(self):    				
    				return self.counts
    @property    
    def overall_average(self):
    				average = sum(self.clicks.values()) / sum(self.counts.values())
    				return average
    				
    def average(self, article):
        return self.clicks[article] / self.counts[article]       
    
    

class NodesUtil(object):
    def __init__(self, groups):
        self.groups = groups
        
    def all_nodes(self):
        return self.groups.all_nodes
        
    @staticmethod
    def distant(covariates1, covariates2):
        return np.sum(np.power((covariates1 - covariates2), 2))
    
    
    
    @staticmethod
    def peaks(list_of_arrays_of_keys, dic_of_all_active_nodes):
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
            
            length = len(list_)
            
            temp_path_keys = []
            temp_path_nodes = []
            
            '''compute the averate_value'''
            total_value = 0
            round_ = 0
            
            '''Saving the subset nodes in case we use it many times'''
            subset_nodes = {}
            current_nodes_set = set()
            record_nodes = set()
            for index, key in enumerate(list_):     #key is string
                mid_ = active_nodes[key]
                subset_nodes.update({key:mid_})
                current_nodes_set.add(key)
                record_nodes.add(key)
                
                mid_3 = set()
                
            while current_nodes_set:
                
                values_dic = {}
                
                
                for key in list_:
                    if key in current_nodes_set:
                        mid_2 = subset_nodes[key].value(current_nodes_set)
                        values_dic.update({key: mid_2})
                        if round_ <= 0:
                            total_value += mid_2
                
                argmax_key = max(values_dic, key=values_dic.get)
                
                mid_node = subset_nodes[argmax_key]#don't need to look-up twice
                temp_path_keys.append(argmax_key)
                temp_path_nodes.append(mid_node)

                mid_3.update(mid_node.neighbors)
                mid_3.add(argmax_key)
                
                current_nodes_set = current_nodes_set - mid_3
                
                round_ += 1
            
        
            resulted_path_nodes, resulted_path_keys = NodesUtil.__path_trim(temp_path_nodes, temp_path_keys, record_nodes, total_value/length)
            
            
            group_values.append(NodesUtil.__path_value(resulted_path_nodes, record_nodes)/len(resulted_path_keys))
            group_peaks.append(resulted_path_keys)
        
        
        
        
        return group_values, group_peaks
        
        
        

    @staticmethod
    def __path_value(path_of_nodes, subset_nodes):
        '''
        subset_nodes, set of nodes
        path_of_nodes, list of nodes
        '''
        value = 0
        subset_ = subset_nodes
        length = len(path_of_nodes)
        
        for index in range(length):
            mid_ = path_of_nodes[index]
            value += mid_.value(subset_)
            subset_ = subset_ - mid_.neighbors.union([mid_])
     
        return value
  
  
    @staticmethod
    def __path_trim(path_of_nodes, path_of_keys, subset_nodes, average_value):
        '''
        article, string
        path_of_nodes, list of nodes
        path_of_keys, list of keys
        '''
        subset_ = subset_nodes
        
        full_value = NodesUtil.__path_value(path_of_nodes, subset_)
        
        values = []
        
        
        length = len(path_of_nodes)
        
        idx = np.arange(1, length)\
            - np.tri(length, length - 1, k=-1, dtype=bool)
            
        for index in range(length):
            values.append(NodesUtil.__path_value(np.array(path_of_nodes)[idx[index,:]], subset_))
        
        bool_vector = np.array(values) + average_value <= full_value
        
        ####print(article, len(path_of_nodes), values, average_value, average_value, bool_vector, '591')
        
        '''Keep all elements. Sometimes bool_vector will be all False, this is 
        probability because average_value consider all neighbors of a given 
        nodes, path_value considers only diminishing nodes values '''
        if not any(bool_vector):
            bool_vector = np.array([not elem for elem in bool_vector])
        
        trim_path_of_nodes = np.array(path_of_nodes)[bool_vector]   
        trim_path_of_keys = np.array(path_of_keys)[bool_vector]
        
        return [trim_path_of_nodes, trim_path_of_keys]  #[list, list]
        
        
        

class Assigner(object):
    '''
    Assigner create nodes. It takes Groups object as parameter so it can assign
    the object to the created nodes.
    '''
    def __init__(self, groups_object, mab_object):
        '''saving all nodes'''
        self.all_nodes = {}
        self.all_nodes_set = set()
        self.mab_object = mab_object
        

        self.groups_object = groups_object
        
        
    def assign(self, key, covariates):
        '''
        key is a string
        '''
        
        if key in self.all_nodes_set:
            return self.all_nodes[key], True
            
        else:
            node = Node(key, covariates, self.groups_object, self.mab_object)
            self.all_nodes[key] = node
            
            self.groups_object.all_nodes_hitting_record[key] = 0 ####TETS
            
            self.all_nodes_set.add(key)
            return node, False
            
    



class ArticlesCollector(object):
    '''
    This object will be assigned to Groups and MAB object
    
    '''
    def __init__(self):
        self.__active_articles = set()
    
    def update(self, extra, delete): 
        self.__active_articles = self.__active_articles - delete
        self.__active_articles = self.__active_articles.union(extra)

    @property
    def active_articles(self):
        return self.__active_articles
        
     
    

class MAB(object):
    '''
    ArticlesCollector object will be assigned to this MAB object
    '''
    def __init__(self, articles_collector):
        self.articles_collector = articles_collector
    
        self.clicks = {}
        self.counts = {}
        
        
        self.alpha = 0.5
        
        
    def recommend(self):
        '''updating all article indexes'''
      
      
  
        values = np.array([])
        articles = []
        for article in self.counts.keys():
            values = np.append(values, self.clicks[article] / self.counts[article] + self.alpha * np.sqrt(1/(self.counts[article]+1)))
            articles.append(article)
            
            
        #print(articles[np.argmax(values)], '726')
        
        
        return articles[np.argmax(values)], False #through_recommended
        
        
    def update(self, reward, article):
        '''
        article is a string
        '''
        
        
        
        #print(self.counts, '714')
        self.counts[article] += 1
        self.clicks[article] += reward
        #print(self.counts[article], self.clicks[article], article, '703')
        
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
    def __init__(self, groups_object, mab_object):
        
        '''
        articles_collector is the same one in the groups_object
        '''
        
        self.acc_reward = 0
        self.groups_object = groups_object
        
        '''mab object is used for guests'''
        self.mab_object = mab_object
        
        
        self.assigner = Assigner(groups_object, mab_object)
        
        self.articles_collector = groups_object.articles_collector
        
        
        self.last_action = '' # a string
        
        
        
        self.through_recommended = False
        self.recommend_counter = {'reco':0, 'mab':0}
        
    def update(self, reward, stuff):
        
        '''
        key is a string
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        
        key = stuff['key']        
        covariates = stuff['covariates']
        

        '''
        extra_bonus only for updating, not for the final result
        '''
        
        node, used_node = self.assigner.assign(key, covariates)
        
        
        '''MAB can share the information'''
        self.mab_object.update(reward, self.last_action) #self.last_action is an article string
        
        node.update(reward, self.last_action, self.through_recommended)

        
        '''Recommending counter'''
        if self.through_recommended:
            self.recommend_counter['reco'] += 1
        else:
            self.recommend_counter['mab'] += 1
        #print(self.recommend_counter, '825')#OK
        
 
    def recommend(self, stuff):
        '''
        receiving a key and decide self.last_acion and self.extra_bonus
        
        key is a string
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        
        key = stuff['key']
        covariates = stuff['covariates']
        
        
        node, used_node = self.assigner.assign(key, covariates)
        node.articles_update(self.articles_collector)
        self.mab_object.articles_update(self.articles_collector)
        
        self.last_action, self.through_recommended = node.recommend(self.articles_collector)        
        
        
        return self.last_action
        
        

    def update_arms(self, stuff):
        self.articles_collector.update(stuff['extra'], stuff['delete'])
        
        



    
class Environment(object):
   
    def run(self, agents, data_reader, timestamp = 70000):
        self.reward_curves = np.zeros((timestamp, len(agents)))
        self.timer = np.zeros(len(agents)).astype(int)
        self.agents = agents
        times = 0
        while np.min(self.timer) < timestamp:
            #Also in this step, arms will be refreshed
            stuff = data_reader.come()

            if not stuff['mab']:
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
            label_list = ['Groups']
        elif number_of_agents == 2:
            label_list = ['guestucb', 'lin_ucb']
            label_list = ['guestucb', 'ucb1']
        else:
            label_list = ['ogaucb', 'ucb', 'guestucb']

        collect = {}
        for j in range(len(self.reward_curves[0,:])):
            
            collect[j], = plt.plot(self.reward_curves[:,j], label=label_list[j])
            mid_ = "/Users/apple/Desktop/bandit_experiment/bandit_modified" + str(j)
            #np.save(mid_, self.reward_curves[:,j])
            
        if number_of_agents == 1:
            plt.legend(handles=[collect[0]])
        
        elif number_of_agents == 2:
            plt.legend(handles=[collect[0], collect[1]])
        else:
            plt.legend(handles=[collect[0], collect[1], collect[2]])
        
        plt.show()


    
def main():
    data_reader = DataReader()
    
    
    
    AC = ArticlesCollector()
    G = Groups(AC)
    
    M = MAB(AC)
    
    AG = Agent(G, M)
    
    E = Environment()
    E.run([AG], data_reader)
    E.plot(len([AG]))
if __name__ == '__main__':
    main()