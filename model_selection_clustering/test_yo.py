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
        
        
        self.files_list = ['/Users/apple/Desktop/bandit_experiment/model_selection_clustering/r6b_18k.txt']
        
        
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






class Util(object):
    
    @staticmethod
    def zeros_and_ones(relevents, coefficients):
        '''relevents and coefficients are ndarray'''
        '''return ones and zeros' indexes'''
        coef_ = coefficients[0,relevents]
        ones = coef_ >= 0
        zeros = coef_ < 0
        return list(relevents[ones]), list(relevents[zeros])
        
        
        
class Graph(object):    
    def __init__(self, articles_set, list_keys_set, collector):
        '''
        articles_set: a set of articles(numbers)
        list_keys_set: exhaustive sets
        articles_set: a set of numbers
        '''
        self.__articles = articles_set
        self.__article_numbers = [{}] * len(list_keys_set) #list of dictionaries. {article:numbers, } 
        self.__article_clicks = [{}] * len(list_keys_set)
        
        
        self.__collector = collector
        
        self.__all_keys = set()
        self.__list_keys_set = []
        self.__numbers_keys_set = [0] * len(list_keys_set)
        
        self.__clicks_keys_set = [0] *  len(list_keys_set)
        
        

        for index, keyset in enumerate(list_keys_set):
            self.__list_keys_set.append(keyset)
            print(index, len(articles_set), 'length of article sets', len(keyset), '122')
            self.__all_keys.update(keyset)
            
            X, y = self.__collector.get_article_covariates(articles_set, keyset)
            self.__numbers_keys_set[index] += len(y)
            self.__clicks_keys_set[index] += sum(y)
        
    
        '''initializeing article-specific clicks'''
        for article in self.__articles:
            for index, keyset in enumerate(list_keys_set):
                X, y = self.__collector.get_article_covariates(article, keyset)
                self.__article_numbers[index].update({article:len(y)})
                self.__article_clicks[index].update({article:sum(y)})
            
        
    def update(self, article, key, click):
        '''new key will join the party in the next round'''
        if article in self.__articles:
            for index, keyset in enumerate(self.__list_keys_set):
                if key in keyset:
                    self.__numbers_keys_set[index] += 1
                    self.__clicks_keys_set[index] += click
                    if article in self.__article_numbers[index].keys(): 
                        self.__article_numbers[index][article] += 1
                        self.__article_clicks[index][article] += click
                    else:
                        self.__article_numbers[index][article] = 1
                        self.__article_clicks[index][article] = click
                    return None
                    
    def get_data(self, key):
        '''return playing times and clicks for this articles set'''
        '''
        key: string
        '''
        for index, keyset in enumerate(self.__list_keys_set):
            if key in keyset:
                #print(self.__numbers_keys_set[index], self.__clicks_keys_set[index])
                return self.__numbers_keys_set[index], self.__clicks_keys_set[index]
                

        '''if there's no such key, return the first'''        
        return self.__numbers_keys_set[0], self.__clicks_keys_set[0]
        
        

    @property
    def all_keys(self):
        return self.__all_keys
    @property
    def get_articles(self):
        return self.__articles
    
    def get_detailed_data(self, key):
        '''return detailed numbers and clicks data''' 
        for index, keyset in enumerate(self.__list_keys_set):
            if key in keyset:               
                return self.__article_numbers[index], self.__article_clicks[index]
    
        
        '''if there's no such key'''
        return self.__article_numbers[0], self.__article_clicks[0]
        
        
    
    
class model_UCB1(object):
    def __init__(self, collector, articles_collector, list_graphs):
        '''list_graphs: list of graph objects'''
        
        self.__articles_collector = articles_collector
        self.__collector = collector
        self.__list_graphs = list_graphs
        
        
        self.last_action = -1
        
    
        self.alpha = 0.5
        
        self.first_round = -1
        
        
        self.counter = 0
    def update_graphs(self, list_graphs):
        self.__list_graphs = list_graphs
        
    def update(self, key, article, click):
        self.counter += 1
        
        '''Newly coming articles will not be counted in action'''
        for graph in self.__list_graphs:
            if article in graph.get_articles:
                graph.update(article, key, click)
                return None
    
    
    def recommend(self, key):
        
        if self.__collector.update_article and self.counter > 100:
            self.__collector.update_article = False
            list_graphs = self.__collector.model_selection_graphs()
            print(len(list_graphs), self.first_round, '219')
            self.update_graphs(list_graphs)
            
            
            ##TESTING
            numbers = [0] * len(self.__list_graphs)
            clicks = [0] * len(self.__list_graphs)
            for index, graph in enumerate(self.__list_graphs):
                numbers[index], clicks[index] = graph.get_data(key)
            print(numbers, clicks, '238')
        
        #print(len(self.__collector.all_keys), len(self.__list_graphs[0].all_keys), len(self.__list_graphs))
        if not key in self.__list_graphs[0].all_keys:
            numbers_object = self.__collector.article_numbers
            clicks_object = self.__collector.article_clicks        
            
            articles = numbers_object.keys()
            articles = articles & self.__articles_collector.active_articles #intersection! 
            extras = self.__articles_collector.active_articles - articles
            articles.update(extras)
            
            numbers = [0] * len(articles)
            clicks = [0] * len(articles)
            articles_ = []
            for index, article in enumerate(articles):
                if article in extras:
                    numbers[index] = 0
                    clicks[index] = 0
                else:     
                    numbers[index] = numbers_object[article]
                    clicks[index] = clicks_object[article]
                articles_.append(article)
            
            max_index = model_UCB1.max_finder(numbers, clicks)
            self.last_action = articles_[max_index]
            
            return self.last_action
            
        #first round
        numbers = [0] * len(self.__list_graphs)
        clicks = [0] * len(self.__list_graphs)
        for index, graph in enumerate(self.__list_graphs):
            numbers[index], clicks[index] = graph.get_data(key)
        
        
        max_index = model_UCB1.max_finder(numbers, clicks)
        self.first_round = max_index #TESTING
        #second round
        numbers_object, clicks_object = self.__list_graphs[max_index].get_detailed_data(key)
        articles = numbers_object.keys()
        
        articles = articles & self.__articles_collector.active_articles #intersection! 
        #extras = self.__articles_collector.active_articles - articles
        #articles.update(extras)
        
        numbers = [0] * len(articles)
        clicks = [0] * len(articles)
        articles_ = []
        for index, article in enumerate(articles):
            numbers[index] = numbers_object[article]
            clicks[index] = clicks_object[article]
            articles_.append(article)
        
        
        
        #print(numbers, clicks, '270')
        
        
        
        max_index = model_UCB1.max_finder(numbers, clicks)
        self.last_action = articles_[max_index]
        
        #print(self.last_action, '241')
        
        return self.last_action
    

    @staticmethod
    def max_finder(numbers, clicks):
        '''
        numbers: list
        clicks: list
        '''
        numbers = np.array(numbers)
        sum_ = np.sum(numbers)
        clicks = np.array(clicks)
        ###ALPHA!!
        return np.argmax(clicks / (numbers + 1) + 0.5 * np.sqrt(1/(numbers+1)))
        
    
    
    
      





        
class Collector(object):
    
    def __init__(self):
        self.__overall_key_covariates = {}##need to be updated, but I go hueristically. {key:covariates, }
        self.__overall_key_numbers = {}##need to be updated
        self.__overall_key_clicks = {}##need to be updated
        self.__article_key_covariates = {}##need to be updated
        self.__article_key_numbers = {}##need to be updated. {article:{key:numbers, }, }
        self.__article_key_clicks = {}##need to be updated
        
        self.__article_numbers = {}#{article: numbers, }
        self.__article_clicks = {}#{article: clicks, }
        
        self.__all_active_keys = set()##need to be updated, but I go hueristically
        self.__all_active_articles = set()##need to be updated
        
        self.__nonzero_sets = {}##need to be updated, but I go hueristically
        for index in range(136):
            self.__nonzero_sets[index] = set()
        
        
        
        
        self.update_article = False
                
                
                
    def __update_nonzero_sets(self, covariates, key):
        for index, elem in enumerate(covariates):
            if elem != 0: 
                self.__nonzero_sets[index].add(key)
    
    '''get any nonzero sets'''
    def get_any_nonzero_sets(self, index):
        '''index is a list'''
        result = set()
        
        for i in index:
            if i != 0:
                result = result | self.__nonzero_sets[i]
                
        return result
    
    def get_any_zero_sets(self, index):
        '''index is a list'''
        result = self.__all_active_keys
        
        for i in index:
            if i != 0:
                result = result & self.__nonzero_sets[i]
       
        result = self.__all_active_keys - result
        
        return result
    
    
    
    def get_nonzero_sets(self, index):
        '''index is a list'''
        result = self.__all_active_keys
        
        for i in index:
            if i != 0:
                result = result & self.__nonzero_sets[i]
            #print(i, self.__nonzero_sets[i], '46')
        return result
    
    def get_zero_sets(self, index):
        '''index is a list'''
        result = set()
        
        
        
        for i in index:
            if i != 0:
                result.update(self.__nonzero_sets[i])
        
        result = self.__all_active_keys - result
        
        return result
    
    
    def update_key(self, key, covariates, article, click):
        
        self.__article_numbers[article] += 1
        self.__article_clicks[article] += click
       
        
        self.__all_active_keys.add(key)
        self.__update_nonzero_sets(covariates, key)
        self.__all_active_articles.add(article)
        
        
        self.__add_overall_key_covariates(key, covariates)
        self.__add_overall_key_numbers(key)
        self.__add_overall_key_clicks(key, click)
        self.__add_article_key_covariates(article, key, covariates)
        self.__add_article_key_numbers(article, key)
        self.__add_article_key_clicks(article, key, click)
        
        
    
    
    def update_articles(self, articles_collector):
        '''Some minor changes have been ignored'''

        deletes = self.__all_active_articles - articles_collector.active_articles
        for article in deletes:
            
            del self.__article_key_covariates[article]
            numbers = self.__article_key_numbers[article]
            del self.__article_key_numbers[article]
            clicks = self.__article_key_clicks[article]
            del self.__article_key_clicks[article]
            
            
            del self.__article_numbers[article]
            del self.__article_clicks[article]
            
            
            self.__all_active_articles.remove(article)
            
            self.update_article = True
            
            
            for key in numbers.keys():
                self.__overall_key_numbers[key] -= numbers[key]
                self.__overall_key_clicks[key] -= clicks[key]
               
        
        extras = articles_collector.active_articles - self.__all_active_articles
        for article in extras:
            
            
            
            self.update_article = True
            
            
            self.__article_key_covariates[article] = {}
            self.__article_key_numbers[article] = {}
            self.__article_key_clicks[article] = {}
            
            self.__article_numbers[article] = 0
            self.__article_clicks[article] = 0
            
            self.__all_active_articles.add(article)
            
 
 
    def __add_overall_key_covariates(self, key, covariates):
        'covariates is a list of numbers'
        self.__overall_key_covariates.update({key:covariates})
        
        
    def __add_overall_key_numbers(self, key):
        if not key in self.__overall_key_numbers.keys(): 
            self.__overall_key_numbers.update({key:1})
        else:
            self.__overall_key_numbers[key] += 1
    
    def __add_overall_key_clicks(self, key, click):
        if not key in self.__overall_key_clicks.keys(): 
            self.__overall_key_clicks.update({key:click})
        else:
            self.__overall_key_clicks[key] += click
    
    
    
    def __add_article_key_covariates(self, article, key, covariates):
        'covariates is a list of numbers'
        if not article in self.__article_key_covariates.keys():            
            self.__article_key_covariates[article] = {}
            self.__article_key_covariates[article].update({key:covariates})
        else:
            self.__article_key_covariates[article].update({key:covariates})
        
    def __add_article_key_numbers(self, article, key):
        if not article in self.__article_key_numbers.keys():
            self.__article_key_numbers[article] = {}
            self.__article_key_numbers[article].update({key:1})
        else:
            if not key in self.__article_key_numbers[article].keys():
                self.__article_key_numbers[article].update({key:1})
            else:
                self.__article_key_numbers[article][key] += 1
    
    
    
    def __add_article_key_clicks(self, article, key, click):
        if not article in self.__article_key_clicks.keys():
            self.__article_key_clicks[article] = {}
            self.__article_key_clicks[article].update({key:click})
        else:
            if not key in self.__article_key_clicks[article].keys():
                self.__article_key_clicks[article].update({key:click})
            else:
                self.__article_key_clicks[article][key] += click
    
    def get_article_covariates(self, articles, key_subsets):
        design_ = []
        y_ = []

        if not isinstance(articles, int):
            for article in articles:
                
                key_subset = self.__article_key_numbers[article].keys() & key_subsets
                
                len_ = len(key_subset)
                key_subset = iter(key_subset)
                
                for i in range(len_):
                    key = next(key_subset)
                    
                    row_ = self.__article_key_numbers[article][key]
                    click_ = self.__article_key_clicks[article][key]
                   
                    design_.extend([self.__article_key_covariates[article][key],]*row_)
                    y_.extend([1]*click_)
                    y_.extend([0]*(row_ - click_))
        else:
            article = articles
            key_subsets = self.__article_key_numbers[article].keys() & key_subsets
            
            len_ = len(key_subsets)
            key_subsets = iter(key_subsets)
            
            for i in range(len_):
                key = next(key_subsets)
                
                row_ = self.__article_key_numbers[article][key]
                click_ = self.__article_key_clicks[article][key]    
                    
                design_.extend([self.__article_key_covariates[article][key],]*row_)
                y_.extend([1]*click_)
                y_.extend([0]*(row_ - click_))
                
        return np.array(design_), np.array(y_)
        
        

    def get_covariates(self, key_subsets):
        design_ = []
        y_ = []
        
        len_ = len(key_subsets)
        key_subsets = iter(key_subsets)
        
        for i in range(len_):
            key = next(key_subsets)
            
            row_ = self.__overall_key_numbers[key]
            click_ = self.__overall_key_clicks[key]    
                
            design_.extend([self.__overall_key_covariates[key],]*row_)
            y_.extend([1]*click_)
            y_.extend([0]*(row_ - click_))
            
            
        return np.array(design_), np.array(y_)
        
    
    def model_selection_graphs(self):
        '''return list of graph(s)'''
        
        keys_set = self.__all_active_keys
    
    
        Logistic = HighDimensionalLogisticRegression(fit_intercept=False)
 
        
        articles = set()
        for article in self.all_articles:   
            X, y = self.get_article_covariates(article, keys_set)
            
            if len(y) > 5:
                Logistic.fit(X, y)
                
                if len(Logistic.model_) > 1:
                    articles.add(article)
           
           
           
        
        
        if len(articles) > 0:
        
            X, y = self.get_article_covariates(articles, keys_set)
            
            Logistic.fit(X, y)
            
            '''Combo!''' 
            ones, zeros = Util.zeros_and_ones(Logistic.model_, Logistic.coef_)
            second_graph_second_key_set = self.get_nonzero_sets(ones) & self.get_zero_sets(zeros) #keys set
      
            if len(second_graph_second_key_set) == len(keys_set) or len(second_graph_second_key_set) == 0:                
                return [Graph(self.all_articles - articles, [keys_set], self), Graph(articles, [keys_set], self)]
            
            '''Make sure there's no need for classification anymore'''
            X, y = self.get_article_covariates(articles, keys_set - second_graph_second_key_set)
            
            Logistic.fit(X, y)
            
            '''Combo!''' 
            ones, zeros = Util.zeros_and_ones(Logistic.model_, Logistic.coef_)
            second_graph_extra_key_set = self.get_nonzero_sets(ones) & self.get_zero_sets(zeros) #keys set
            second_graph_first_key_set = keys_set - second_graph_second_key_set - second_graph_extra_key_set
            
            if len(second_graph_first_key_set) == 0:
                print(len(second_graph_second_key_set), len(keys_set), '598')
                graph1 = Graph(self.all_articles - articles, [keys_set], self)#we dont care about non-distinguished subsets
                
                list_keys_set2 = [keys_set - second_graph_second_key_set, second_graph_second_key_set]
                graph2 = Graph(articles, list_keys_set2, self)#collector

            
            
                return [graph1, graph2]
            
            

            graph1 = Graph(self.all_articles - articles, [keys_set], self)#we dont care about non-distinguished subsets
            
            list_keys_set2 = [second_graph_first_key_set, second_graph_second_key_set | second_graph_extra_key_set]
            graph2 = Graph(articles, list_keys_set2, self)#collector

            print(len(second_graph_second_key_set), len(second_graph_extra_key_set), len(keys_set), '615')
            
            return [graph1, graph2]
            
        else:
            
            X, y = self.get_article_covariates(self.all_articles, keys_set)
            
            Logistic.fit(X, y)
            '''Combo!'''         
            ones, zeros = Util.zeros_and_ones(Logistic.model_, Logistic.coef_)        
            first_key_set = self.get_nonzero_sets(ones) & self.get_zero_sets(zeros) #keys set
           
            if len(first_key_set) == len(keys_set):
                return [Graph(self.all_articles, [keys_set], self)]
    
            
            X, y = self.get_article_covariates(self.all_articles, keys_set - first_key_set)

            Logistic.fit(X, y)
            '''Combo!'''
            if len(Logistic.model_) == 1:      
                list_keys_set = [keys_set - first_key_set, first_key_set]
                graph = Graph(self.all_articles, list_keys_set, self)#collector
                
                return [graph]
            else:
                ones, zeros = Util.zeros_and_ones(Logistic.model_, Logistic.coef_)
                second_key_set = self.get_nonzero_sets(ones) & self.get_zero_sets(zeros) #keys set
               
                
                list_keys_set = [keys_set - first_key_set - second_key_set,\
                    second_key_set | first_key_set]#sum of the two subsets!
                graph = Graph(self.all_articles, list_keys_set, self)#collector
                
                return [graph]
            
            
            
            
            
            
    
    
    
    @property
    def overall_key_covariates(self):
        return self.__overall_key_covariates
    @property        
    def overall_key_numbers(self):
        return self.__overall_key_numbers
    @property
    def overall_key_clicks(self):
        return self.__overall_key_clicks
    @property
    def article_key_covariates(self):
        return self.__article_key_covariates
    @property
    def article_key_numbers(self):
        return self.__article_key_numbers
    @property
    def article_key_clicks(self):
        return self.__article_key_clicks
    @property
    def all_articles(self):
        return self.__all_active_articles
    @property
    def all_keys(self):
        return self.__all_active_keys
    @property
    def article_numbers(self):
        return self.__article_numbers
    @property
    def article_clicks(self):
        return self.__article_clicks






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
 
 





class Agent_model_UCB1(object):
    '''
    Takes Groups object and MAB object as parameters
    '''
    def __init__(self, articles_collector, collector):
        '''
        articles_collector as the input parameter
        '''
        self.acc_reward = 0
        
        
        self.collector = collector
        '''mab object is used for guests'''#list_keys_set
        self.model_UCB1_object = model_UCB1(collector, articles_collector, [Graph(collector.all_articles, [collector.all_keys], collector)])
        
        
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
        
        
        article = self.last_action
        
        
        self.model_UCB1_object.update(key, article, reward) #self.last_action is an article number
        self.collector.update_key(key, covariates, article, reward)
        
        
        
    def recommend(self, stuff):
        '''
        receiving a key and decide self.last_acion and self.extra_bonus
        
        key is a string
        
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        
        key = stuff['key']
        
        covariates = stuff['covariates']
        
        #updating the articles. Only collector need to do this
        self.collector.update_articles(self.articles_collector)

        
        self.last_action = self.model_UCB1_object.recommend(key) #self.last_action is a number         
            
            
        
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
    DR = DataReader()
    C = Collector()
    A = ArticlesCollector()
    E = Environment()
    AG = Agent_model_UCB1(A, C)
    
    E.run([AG], DR)
    E.plot(len([AG]))
if __name__ == '__main__':
    main()
