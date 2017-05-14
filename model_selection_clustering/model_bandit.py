
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
        self.__articles = set()
        self.__articles.update(articles_set)#Needed to be updated
       
        self.__article_numbers = [{} for i in range(len(list_keys_set))]
        self.__article_clicks = [{} for i in range(len(list_keys_set))]#Needed to be updated
        
        
        self.__collector = collector
        
        self.__all_keys = set() #Needed to be updated
        self.__list_keys_set = [] #list of sets
        
        self.__numbers_keys_set = [0 for i in range(len(list_keys_set))]
        
        self.__clicks_keys_set = [0 for i in range(len(list_keys_set))]
        
        
        
        for index, keyset in enumerate(list_keys_set):
            self.__list_keys_set.append(keyset)
            print(index, len(articles_set), 'length of article sets', len(keyset), '126')
            self.__all_keys.update(keyset)
            
            X, y = self.__collector.get_article_covariates(articles_set, keyset)
            self.__numbers_keys_set[index] += len(y)
            self.__clicks_keys_set[index] += sum(y)
            
        
       
        '''initializeing article-specific clicks'''
        for article in self.__articles:
            #print('graph init', '137')
            for index, keyset in enumerate(list_keys_set):
                X, y = self.__collector.get_article_covariates(article, keyset)
                self.__article_numbers[index].update({article:len(y)})
                self.__article_clicks[index].update({article:sum(y)})

        print(self.__article_numbers, '145')
        print(self.__article_clicks, '146')
        
    '''Updating for newly coming key'''
    def update(self, article, key, click):
        
        if article in self.__articles:
            if key in self.__all_keys:
                for index, keyset in enumerate(self.__list_keys_set):
                    if key in keyset:
                        self.__numbers_keys_set[index] += 1
                        self.__clicks_keys_set[index] += click
                        
                        self.__article_numbers[index][article] += 1
                        self.__article_clicks[index][article] += click
                    
            else:#Adding key to the first set
                index = 0
                self.__all_keys.add(key)
                self.__list_keys_set[index].add(key)
                
                self.__numbers_keys_set[index] += 1
                self.__clicks_keys_set[index] += click
    
                self.__article_numbers[index][article] += 1
                self.__article_clicks[index][article] += click
                
                    
    def get_data(self, key):
        '''return playing times and clicks for this articles set'''
        '''
        key: string
        '''
        for index, keyset in enumerate(self.__list_keys_set):
            if key in keyset:
                return self.__numbers_keys_set[index], self.__clicks_keys_set[index]
                
       
    
    @property
    def all_keys(self):
        return self.__all_keys
    @property
    def get_articles(self):
        return self.__articles
    @property
    def get_list_keys_sets(self):
        return self.__list_keys_set
    
    def get_detailed_data(self, key):
        '''return detailed numbers and clicks data''' 
        for index, keyset in enumerate(self.__list_keys_set):
            if key in keyset:               
                return self.__article_numbers[index], self.__article_clicks[index]
    
        
        
    
    def update_articles(self, articles_collector, first_graph, delete_articles=None, extra_articles=None):        
      
        if first_graph:
            
            self.__articles.update(extra_articles)
            
            for article in extra_articles:
                len_ = len(self.__article_numbers)
                
                for index in range(len_):
                    #set_ is a dictionary of articles and numbers.
                    self.__article_numbers[index][article] = 0
                    self.__article_clicks[index][article] = 0
        
        for article in delete_articles:
            if article in self.__articles:
                len_ = len(self.__article_numbers)
                
   
                for index in range(len_):
                    #print(index, '218')
                    #if article in self.__article_numbers[index].keys():
                    number = self.__article_numbers[index][article]
    
                    del self.__article_numbers[index][article]
                    click = self.__article_clicks[index][article]
                    del self.__article_clicks[index][article]
                    
    
                    
                    self.__numbers_keys_set[index] -= number    
                    self.__clicks_keys_set[index] -= click
                    
                
                self.__articles.remove(article)
            
            #destroy this graph
            if not self.__articles:
                print('destroyed graph 223!!!!')
                return True
        
        
        
        return False
        
        
    @property
    def test(self):
        return self.__article_numbers
        

        
    
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
        
        '''
        Newly coming article comes into the first graph.
        Newly coming key comes into the first key set of the corresponding article graph.
        '''
        for graph in self.__list_graphs:
            if article in graph.get_articles:
                graph.update(article, key, click)
                
    
    
    def recommend(self, key):
        #Because not all graphs contain each key.
        #IF there's any graph being without such key, use overall MAB.
        whole_key = self.__collector.all_keys
        
        for graph in self.__list_graphs:
            whole_key = whole_key & graph.all_keys
        
        #If the it's a new key
        if not key in whole_key:
            numbers_object = self.__collector.article_numbers
            clicks_object = self.__collector.article_clicks        
            
            articles = numbers_object.keys()
            articles = articles & self.__articles_collector.active_articles #intersection! 
            extras = self.__articles_collector.active_articles - articles
            articles.update(extras)
            
            numbers = [0 for i in range(len(articles))]
            clicks = [0 for i in range(len(articles))]
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
        numbers = [0 for i in range(len(self.__list_graphs))]
        clicks = [0 for i in range(len(self.__list_graphs))]
        for index, graph in enumerate(self.__list_graphs):
            numbers[index], clicks[index] = graph.get_data(key)
        
        
        max_index = model_UCB1.max_finder_cluster(numbers, clicks)
        self.first_round = max_index #TESTING
        #second round
        numbers_object, clicks_object = self.__list_graphs[max_index].get_detailed_data(key)
        articles = numbers_object.keys()
        
        articles = articles & self.__articles_collector.active_articles #intersection! 
        
        
        numbers = [0 for i in range(len(articles))]
        clicks = [0 for i in range(len(articles))]
        articles_ = []
        for index, article in enumerate(articles):
            numbers[index] = numbers_object[article]
            clicks[index] = clicks_object[article]
            articles_.append(article)
        
        
        
        #print(numbers, clicks, '270')
        
        
        
        max_index = model_UCB1.max_finder_within(numbers, clicks)
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
    
    
    @staticmethod
    def max_finder_cluster(numbers, clicks):
        '''
        numbers: list
        clicks: list
        '''
        numbers = np.array(numbers)
        sum_ = np.sum(numbers)
        clicks = np.array(clicks)
        ###ALPHA!!
        return np.argmax(clicks / (numbers + 1) + 0.5 * np.sqrt(1/(numbers+1)))
        
    
    
    @staticmethod
    def max_finder_within(numbers, clicks):
        '''
        numbers: list
        clicks: list
        '''
        numbers = np.array(numbers)
        sum_ = np.sum(numbers)
        clicks = np.array(clicks)
        ###ALPHA!!
        return np.argmax(clicks / (numbers + 1) + 0.5 * np.sqrt(1/(numbers+1)))
    
    
    #Updating the article, not the keys
    def update_articles(self, articles_collector):
        destroying = [False]
        
        extra_articles = articles_collector.active_articles
        for graph in self.__list_graphs:
            extra_articles = extra_articles - graph.get_articles
         
        delete_articles = set()
        mid_ = set()
        for graph in self.__list_graphs:
            mid_ = mid_ | graph.get_articles
        
        delete_articles = mid_ - articles_collector.active_articles
        
        
        
        
        
        for index in range(len(self.__list_graphs)): 
            if index == 0:
                
               
                self.__list_graphs[index].update_articles(articles_collector, True, delete_articles, extra_articles)
            else:
                destroying.append(self.__list_graphs[index].update_articles(articles_collector, False, delete_articles))
                
        list_mid_graphs = []
        for index, bool_ in enumerate(destroying):
            if not bool_:
                list_mid_graphs.append(self.__list_graphs[index])
        
        #Remove those article empty graph 
        self.__list_graphs = list_mid_graphs
        #Reset the article collector
        #articles_collector.reset()
        
        
    @property
    def get_graphs(self):
        return self.__list_graphs
    
    
    
    
    
    
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
        
        self.__zero_sets = {}##need to be updated, but I go hueristically
        for index in range(136):
            self.__zero_sets[index] = set()
        
        
       
                
                
    def __update_nonzero_sets(self, covariates, key):
        for index, elem in enumerate(covariates):
            if elem != 0: 
                self.__nonzero_sets[index].add(key)
    
    
    def __update_zero_sets(self, covariates, key):
        for index, elem in enumerate(covariates):
            if elem != 1: 
                self.__zero_sets[index].add(key)
    
    
    
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
    
    
    
    def get_sets(self, ones, zeros):
        
        if len(ones) + len(zeros) == 1:
            return set()
        
        return self.get_nonzero_sets(ones) & self.get_zero_sets(zeros)
    
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
        result = self.__all_active_keys
        
        for i in index:
            if i != 0:
                result = result & self.__zero_sets[i]
            #print(i, self.__nonzero_sets[i], '46')
        return result
    
    
    def update_key(self, key, covariates, article, click):
        
        self.__article_numbers[article] += 1
        self.__article_clicks[article] += click
       
        
        self.__all_active_keys.add(key)
        self.__update_nonzero_sets(covariates, key)
        self.__update_zero_sets(covariates, key)
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
            
        
            '''
            for key in numbers.keys():
                self.__overall_key_numbers[key] -= numbers[key]
                self.__overall_key_clicks[key] -= clicks[key]
            ''' 
        
        extras = articles_collector.active_articles - self.__all_active_articles
        for article in extras:
            
            
       
            
            
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
            article = articles#int
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
        
    
    
    
    
    '''Refresh the newly coming keys (in all graphs)'''
    def model_selection_graphs(self, list_graphs):
        len_ = len(list_graphs)
        final_graphs_list = []
        
        Logistic = HighDimensionalLogisticRegression(fit_intercept=False)
        
        #Pick the first graph for the first round process
        graph = list_graphs[0]
        keys_set = graph.all_keys
        
        
        
        list_of_keys_sets = graph.get_list_keys_sets
        
        
        articles = set()
        
        for key_set_in_the_list in list_of_keys_sets:    
            for article in graph.get_articles:
                
                X, y = self.get_article_covariates(article, key_set_in_the_list)
            
                if len(y) > 5:
                    Logistic.fit(X, y)
                    
                    if len(Logistic.model_) > 1:
                        articles.add(article)
        graphs = []
        if len(articles) > 0:            
            for article in articles:
                article_ = set([article])
                X, y = self.get_article_covariates(article, keys_set)
                
                Logistic.fit(X, y)
                
                
                
                '''Combo!'''
                ones, zeros = Util.zeros_and_ones(Logistic.model_, Logistic.coef_)
                second_key_set = self.get_sets(ones, zeros) #keys set
                
                '''Make sure there's no need for classification anymore'''
                X, y = self.get_article_covariates(article, keys_set - second_key_set)
                
                Logistic.fit(X, y)
                
                '''Combo!'''
                ones, zeros = Util.zeros_and_ones(Logistic.model_, Logistic.coef_)
                third_key_set = self.get_sets(ones, zeros) #keys set
                
                
                if len(third_key_set) != len(second_key_set) and len(third_key_set) != 0:
                    list_key_sets = [keys_set - third_key_set - second_key_set, second_key_set - third_key_set, second_key_set]
                else: 
                    list_key_sets = [keys_set - second_key_set, second_key_set]
                
                
                list_key_sets_2 = []
                
                for elem in list_key_sets:
                    if len(elem) > 0:
                        list_key_sets_2.append(elem)
                
                graph2 = Graph(article_, list_key_sets_2, self)#collector
                graphs.append(graph2)
        
        
        list_ = []
        for key_set_in_the_list in list_of_keys_sets:
            ####For the first graph
            X, y = self.get_article_covariates(graph.get_articles - articles, key_set_in_the_list)
            
            Logistic.fit(X, y)
            
            '''Combo!''' 
            ones, zeros = Util.zeros_and_ones(Logistic.model_, Logistic.coef_)
            second_key_set = self.get_sets(ones, zeros) #keys set
            
            
            '''Make sure there's no need for classification anymore'''
            X, y = self.get_article_covariates(graph.get_articles - articles, key_set_in_the_list - second_key_set)
            
            '''Combo!''' 
            ones, zeros = Util.zeros_and_ones(Logistic.model_, Logistic.coef_)
            third_key_set = self.get_sets(ones, zeros) #keys set
            
            
            
            if len(third_key_set) != len(second_key_set) and len(third_key_set) != 0:
                list_key_sets = [keys_set - third_key_set - second_key_set, second_key_set - third_key_set, second_key_set]
            else: 
                list_key_sets = [keys_set - second_key_set, second_key_set]
        
            
            
            for elem in list_key_sets:
                if len(elem) > 0:
                    list_.append(elem)
                    
        graph1 = Graph(graph.get_articles - articles, list_, self)#collector
        
        
        final_graphs_list.append(graph1)
        final_graphs_list.extend(graphs)
        
            
    
    ####
        for graph in list_graphs[1:]:
            list_of_keys_sets = graph.get_list_keys_sets
            list_ = []
            for key_set_in_the_list in list_of_keys_sets:
                article = graph.get_articles
                keys_set = graph.all_keys
                
                
                X, y = self.get_article_covariates(article, key_set_in_the_list)
                    
                Logistic.fit(X, y)
                
                
                
                '''Combo!'''
                ones, zeros = Util.zeros_and_ones(Logistic.model_, Logistic.coef_)
                second_key_set = self.get_sets(ones, zeros) #keys set
                
                '''Make sure there's no need for classification anymore'''
                X, y = self.get_article_covariates(article, key_set_in_the_list - second_key_set)
                
                Logistic.fit(X, y)
                
                '''Combo!'''
                ones, zeros = Util.zeros_and_ones(Logistic.model_, Logistic.coef_)
                third_key_set = self.get_sets(ones, zeros) #keys set
                
                
                if len(third_key_set) != len(second_key_set) and len(third_key_set) != 0:
                    list_key_sets = [keys_set - third_key_set - second_key_set, second_key_set - third_key_set, second_key_set]
                else: 
                    list_key_sets = [keys_set - second_key_set, second_key_set]
                
                
                
                
                
                for elem in list_key_sets:
                    if len(elem) > 0:
                        list_.append(elem)
                        
                #position if list of graphs matters!
               
            graph2 = Graph(article, list_, self)#collector
            final_graphs_list.append(graph2)

        return final_graphs_list
            
    
    
    
    
    #####
    def graph_refresher(self):
        '''return list of one graph'''
        
        
        graph = Graph(self.all_articles, [self.__all_active_keys], self)#collector
        
    
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
        
        self.first_time = True
        self.counter = 0
        self.counter_usual = 0
        
    def update(self, reward, stuff):
        '''
        key is a string
        mab is a bool
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        key = stuff['key']
        
        covariates = stuff['covariates']

        
        
        article = self.last_action
        
        if not self.first_time:
            self.model_UCB1_object.update(key, article, reward) #self.last_action is an article number
        self.collector.update_key(key, covariates, article, reward)
        
        #Updating model
        self.counter += 1
        self.counter_usual += 1
        if self.counter >= 500:
            if self.first_time:
                list_graphs = self.collector.graph_refresher()
                self.model_UCB1_object.update_graphs(list_graphs)
                self.first_time = False
                
        if self.counter_usual >= 501:

            list_graphs = self.collector.model_selection_graphs(self.model_UCB1_object.get_graphs)
            self.model_UCB1_object.update_graphs(list_graphs)
            self.counter_usual = 0

    def recommend(self, stuff):
        '''
        receiving a key and decide self.last_acion and self.extra_bonus
        
        key is a string
        
        stuff, {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete, 'mab':mab, 'key':None}
        '''
        
        key = stuff['key']
        
        covariates = stuff['covariates']
        
        
        #updating the articles. Only collector has to do this
   
        self.collector.update_articles(self.articles_collector)
        
        #for elem in self.__article_numbers:            
            #print(562265 in elem.keys(), '217')
                
        #print(562265 in elem.keys(), '1042')
        
        
        if self.counter >= 500:
           
            self.model_UCB1_object.update_articles(self.articles_collector)#add the newly coming article to the first graph
        
        
       
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
            label_list = ['Logistic']
        elif number_of_agents == 2:
            label_list = ['Logistic', 'ucb1']
        
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




###########
####MAB####

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
    DR = DataReader()
    C = Collector()
    A = ArticlesCollector()
    E = Environment()
    AG = Agent_model_UCB1(A, C)
    
    
    
    ##MAB
    M = MAB(A)
    Ag = Agent(M, A)
    
    
    E.run([AG, Ag], DR)
    E.plot(len([AG, Ag]))
if __name__ == '__main__':
    main()
