#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numpy.linalg import inv
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
    def __init__(self, articles_set, list_keys_set, collector, model=None):
								'''
								articles_set: a set of articles(numbers)
								list_keys_set: exhaustive sets
								articles_set: a set of numbers
								'''
								self.__articles = set()
								self.__articles.update(articles_set)
								
								self.__article_numbers = [{} for i in range(len(list_keys_set))]
								self.__article_clicks = [{} for i in range(len(list_keys_set))]
								
								
								self.__collector = collector
								
								self.__all_keys = set()
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
												for index, keyset in enumerate(list_keys_set):
																X, y = self.__collector.get_article_covariates(article, keyset)
																self.__article_numbers[index].update({article:len(y)})
																self.__article_clicks[index].update({article:sum(y)})
								
								print(self.__article_numbers, '145')
								print(self.__article_clicks, '146')
								
								
								'''Create inverse matrix with model_ information'''
								global MODEL
								self.__model = np.array(MODEL)
								model = np.array(MODEL)###
								if len(model) > 0:												
												self.__article_inverses = {}
												mid_overall = np.eye(model.shape[0])
												for article in self.__articles:        				
																keys_list = [key for sub_list in list_keys_set for key in sub_list]
																design, y = self.__collector.get_article_covariates(article, keys_list)
																
																if design.shape[0]:
																				sub_d = design[:,model]
																				
																				matrix_ = np.dot(sub_d.T, sub_d)        				
																				mid_ = inv(matrix_ + np.eye(model.shape[0]))
																				
																				self.__article_inverses[article] = mid_
																				mid_overall += matrix_
																else:
																				self.__article_inverses[article] = np.eye(model.shape[0])
																				
																								
												self.__overall_inverse = inv(mid_overall)
								

    '''Updating for newly coming key'''
    def update(self, article, key, click, cov):
								
								if article in self.__articles:												
												if len(self.__model) > 0:
																cov = cov[self.__model]
																self.__article_inverses[article] = inv(inv(self.__article_inverses[article]) + np.dot(cov[:,None], cov[None,:]))
																self.__overall_inverse = inv(inv(self.__overall_inverse) + np.dot(cov[:,None], cov[None,:]))
																
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
                index_ = index    
        return self.__numbers_keys_set[index_], self.__clicks_keys_set[index_]
                
    
    @property
    def model(self):
    				return self.__model
    @property
    def get_overall_inverse(self):
    				return self.__overall_inverse
    @property
    def get_article_inverses(self):
    				return self.__article_inverses
    
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
    
    def get_all_data(self):#now we don't use differentiated consumers
    				return self.__article_numbers[0], self.__article_clicks[0]
    def get_special_data(self):#now we don't use differentiated consumers
    				return self.__article_numbers[1], self.__article_clicks[1]
    
    def update_articles(self, articles_collector, first_graph, delete_articles=None, extra_articles=None):        
								
								if first_graph:
												
												self.__articles.update(extra_articles)
												
												for article in extra_articles:
																len_ = len(self.__article_numbers)
																
																for index in range(len_):
																				#set_ is a dictionary of articles and numbers.
																				self.__article_numbers[index][article] = 0
																				self.__article_clicks[index][article] = 0
																if len(self.__model) > 0:                        
																				self.__article_inverses[article] = np.eye(self.__model.shape[0])
																
								for article in delete_articles:																								
												if article in self.__articles:
																if len(self.__model) > 0:                        
																				del self.__article_inverses[article]
																
																len_ = len(self.__article_numbers)
																
								
																for index in range(len_):                    
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
				def __init__(self, collector, articles_collector, list_graphs, alpha = None):
								'''list_graphs: list of graph objects'''
								
								self.__articles_collector = articles_collector
								self.__collector = collector
								self.__list_graphs = list_graphs
								
								
								self.last_action = -1
								self.special_recommend = False
								self.articles_counter = {}
					
								self.alpha = alpha
								
								self.first_round = -1
								
								
								self.counter = 0
								self.counter_special = 0
				def update_graphs(self, list_graphs):
								self.__list_graphs = list_graphs
								
				def update(self, key, article, click, cov):
								self.counter += 1
								
								
								
								if self.special_recommend:
												#print('test', '333')
												self.articles_counter[article] += 1
												self.counter_special += 1
								'''
								Newly coming article comes into the first graph.
								Newly coming key comes into the first key set of the corresponding article graph.
								'''
								for graph in self.__list_graphs:
												if article in graph.get_articles:
																graph.update(article, key, click, cov)
																
				
				
				def recommend(self, key, cov):
								#For starting
								if self.counter <= 100:
												numbers_object = self.__collector.article_numbers
												articles = []												
												articles.extend(numbers_object.keys())
												return articles[2], False
								#For new articles
								global FEW
								
								global MODEL
								
								
								
								
								if len(self.__list_graphs[0].model) > 0:
												model_ = MODEL#self.__list_graphs[0].model											
								else:
												model_ = []
								
								
								#special event
								if sum(cov[np.array(model_).astype(int)]) > 2/3 * len(np.array(model_)):
												
												#if int(self.counter / 1000) == self.counter / 1000:
																#print('number of special assignment:', self.counter_special, self.articles_counter, '367')
												
												for article in self.__collector.all_articles:
																
																if not article in self.articles_counter.keys():
																				self.articles_counter.update({article:0})
																
																
																if self.articles_counter[article] <= GREEDY:																																																																																										
																				self.last_action = article
																				
																				
																				return self.last_action, True
																				
								if len(self.__list_graphs) > 1:
												
												group1_numbers, group1_clicks = self.__list_graphs[0].get_all_data()
												group2_numbers, group2_clicks = self.__list_graphs[1].get_all_data()
												numbers = [0, 0]
												clicks = [0, 0]
												
												for article in group1_numbers.keys():
																numbers[0] += group1_numbers[article]
																clicks[0] += group1_clicks[article]
												for article in group2_numbers.keys():
																numbers[1] += group2_numbers[article]
																clicks[1] += group2_clicks[article]
												
												max_index = model_UCB1.max_finder_cluster(cov, self.__list_graphs, numbers, clicks, total = self.counter, alpha = None, model=model_)
												
												graph = self.__list_graphs[max_index]
												#round 2
												
												group_numbers, group_clicks = graph.get_all_data()
												
												numbers = [0 for i in range(len(group_numbers.keys()))]
												clicks = [0 for i in range(len(group_numbers.keys()))]
												articles_ = []
												
												for index, article in enumerate(group_numbers.keys()):
																numbers[index] += group_numbers[article]
																clicks[index] += group_clicks[article]
																articles_.append(article)
																
												max_index = model_UCB1.max_finder_expedience(cov, graph, numbers, clicks, self.counter, self.alpha, model_)
												self.last_action = articles_[max_index]
												
												return self.last_action, False
								else:
												graph = self.__list_graphs[0]
												
												group_numbers, group_clicks = graph.get_all_data()
												
												numbers = [0 for i in range(len(group_numbers.keys()))]
												clicks = [0 for i in range(len(group_numbers.keys()))]
												articles_ = []
												
												for index, article in enumerate(group_numbers.keys()):
																numbers[index] += group_numbers[article]
																clicks[index] += group_clicks[article]
																articles_.append(article)
																
												max_index = model_UCB1.max_finder_expedience(cov, graph, numbers, clicks, self.counter, self.alpha, model_)
												self.last_action = articles_[max_index]
												
												return self.last_action, False
												
				
				
				@staticmethod
				def max_finder_cluster(cov, graphs, numbers, clicks, total = None, alpha = None, model=None):
								'''
								numbers: list
								clicks: list
								'''
								
								numbers = np.array(numbers)
								sum_ = np.sum(numbers)
								clicks = np.array(clicks)
								
								#alpha = 0.05
								array_ = (clicks)/ (numbers + 1) + ALPHA_BETWEEN * np.sqrt(np.log(sum_+1)/(numbers+1))
								
								return np.random.choice(np.where(array_ == array_.max())[0])
								
								
				@staticmethod
				def max_finder_expedience(cov, graph, numbers, clicks, total = None, alpha = None, model=None):
								'''
								numbers: list
								clicks: list
								'''
								numbers = np.array(numbers)
								sum_ = np.sum(numbers)
								clicks = np.array(clicks)
								
								
								#deter = np.maximum(numbers - 20, np.zeros(len(numbers)))
								#if any(deter == 0):
								#				return np.random.choice(np.where(deter == 0.0)[0])
												
								array_ = (clicks)/ (numbers + 1) + ALPHA_WITHIN * np.sqrt(np.log(sum_+1)/(numbers+1))# + directions

								return np.random.choice(np.where(array_ == array_.max())[0])				
				
								
								
								
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
								
								
								
				@property
				def get_graphs(self):
								return self.__list_graphs
				
				
				
				
				

class Collector(object):
    
    def __init__(self):
        self.__overall_key_covariates = {}
        self.__overall_key_numbers = {}
        self.__overall_key_clicks = {}
        self.__article_key_covariates = {}
        self.__article_key_numbers = {}#{article:{key:numbers, }, }
        self.__article_key_clicks = {}
        
        self.__article_numbers = {}#{article: numbers, }
        self.__article_clicks = {}#{article: clicks, }
        
        self.__all_active_keys = set()
        self.__all_active_articles = set()
        
        self.__nonzero_sets = {}
        for index in range(136):
            self.__nonzero_sets[index] = set()
        
        self.__zero_sets = {}
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
    
    
    
    def get_sets(self, ones, zeros, keyset):
        
        if len(ones) + len(zeros) == 1:
            return set()
        
        return self.get_nonzero_sets(ones) & self.get_zero_sets(zeros) & keyset
    
    def get_nonzero_sets(self, index):
        '''index is a list'''
        result = self.__all_active_keys
        
        for i in index:
            if i != 0:
                result = result & self.__nonzero_sets[i]            
        return result
    
    def get_zero_sets(self, index):
        '''index is a list'''
        result = self.__all_active_keys
        
        for i in index:
            if i != 0:
                result = result & self.__zero_sets[i]            
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
            clicks = self.__article_key_clicks[article]
            
            for key in numbers.keys():
                self.__overall_key_numbers[key] -= numbers[key]
                self.__overall_key_clicks[key] -= clicks[key]
                
            del self.__article_key_numbers[article]
            del self.__article_key_clicks[article]
            
            
            del self.__article_numbers[article]
            del self.__article_clicks[article]
                        
            self.__all_active_articles.remove(article)
                                
            
                                
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
								
								final_graphs_list = []
								
								Logistic = HighDimensionalLogisticRegression(fit_intercept=False)
																
								keys_set = self.all_keys
								
								articles = set()
								articles_emergency = set()
								article_indexes = []
								article_compe = []
								
								article_not_all_zeros = set()
								
								global MODEL
								global FEW
								#MODEL = []
								model_bool = np.array([]).astype(int)
								FEW = []
								#FEW2 = []
								for article in self.all_articles:
												X, y = self.get_article_covariates(article, keys_set)
												
												#if len(y) > 5:
												article_indexes.append(article)
												
												if len(y) == 0:
																article_compe.append(0)
												else:
																article_compe.append(np.mean(y))
																
												if len(y) <= 15:																																																																										
																FEW.append(article)
																
								n_ = len(article_compe)
								
								std = np.std(article_compe)
								ave = np.mean(article_compe)
								skewness = stats.skew(article_compe)
								
								global para1
								global para2
								global para3
								
								ordered_compe = np.sort(article_compe)[::-1]
								
								l = 0
								if len(article_compe) > 0 and sum(article_compe) != 0.0:
												while len(ordered_compe) > l and ordered_compe[l] >= (ave + para1 * std + para2 * l * std + para3 * (l + 1) * abs(skewness)):
																l = l + 1						
								else:
												return [Graph(self.all_articles, [keys_set], self, model_bool)]
								
								print(articles, skewness, std, ave, l, len(article_compe), '902')
								
								
																																				
								places = np.sort(article_compe)[(n_-l):n_]
								article_indexes = np.array(article_indexes)
								
								
								result = set([int(article) for article in article_indexes[np.in1d(article_compe, places)]])
								#result.update(articles)
								
								X, y = self.get_article_covariates(self.all_articles, keys_set)
								
								Logistic.fit(X, y)
								
								if len(Logistic.model_) > 1:																
												model_bool = np.append(model_bool, Logistic.model_[Logistic.coef_[Logistic.coef_ != 0] > 0])																
												model_bool = np.setdiff1d(model_bool, 0)
																
								if model_bool.shape[0] > 0:												
												model_bool.flatten()		
												model_bool = np.unique(model_bool)													
								else:
												model_bool = None
								
								
								
								if not model_bool is None:												
												MODEL.extend(model_bool)
												MODEL = list(np.unique(MODEL))												
												
								print(model_bool, MODEL, '893')
												
								if l > 0 and l < len(self.all_articles) and len(MODEL) > 0:
												graph2 = Graph(result, [keys_set], self, model_bool)#collector
												graph3 = Graph(self.all_articles - result, [keys_set], self, model_bool)#collector
												
												final_graphs_list.append(graph3)
												final_graphs_list.append(graph2)
												
												return final_graphs_list
												
								else:
												return [Graph(self.all_articles, [keys_set], self, model_bool)]
								
								
								
								

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
    def __init__(self, articles_collector, collector, alpha = None):
        '''
        articles_collector as the input parameter
        '''
        self.acc_reward = 0
        
        
        self.collector = collector
        '''mab object is used for guests'''#list_keys_set
        self.model_UCB1_object = model_UCB1(collector, articles_collector, [Graph(collector.all_articles, [collector.all_keys], collector)], alpha)
        
        
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
            self.model_UCB1_object.update(key, article, reward, covariates) #self.last_action is an article number
        self.collector.update_key(key, covariates, article, reward)
        
        #Updating model
        self.counter += 1
        self.counter_usual += 1
        if self.counter >= 29:
            if self.first_time:
                list_graphs = self.collector.graph_refresher()
                self.model_UCB1_object.update_graphs(list_graphs)
                self.first_time = False
                
        if self.counter_usual >= 30:

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
        
                   
        self.collector.update_articles(self.articles_collector)
        
        #Block the first 100 round for program satability
        if self.counter >= 99:           
            self.model_UCB1_object.update_articles(self.articles_collector)#add the newly coming article to the first graph
        
        
        
        self.last_action, special = self.model_UCB1_object.recommend(key, covariates) #self.last_action is a number         
        #print(self.last_action, special, '1118')
        self.model_UCB1_object.special_recommend = special
        
        
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
            mid_ = "/Users/xbb/Desktop/bandit_experiment/model_selection_clustering/logistic_1_" + str(j)
            #np.save(mid_, self.reward_curves[:,j])
            
        if number_of_agents == 1:
            plt.legend(handles=[collect[0]])
        
        elif number_of_agents == 2:
            plt.legend(handles=[collect[0], collect[1]])
        else:
            plt.legend(handles=[collect[0], collect[1], collect[2]])
        
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,0.085))
        plt.show()



para1 = 0.8
para2 = 0.1
para3 = 0.01
MODEL = []
FEW1 = []
FEW2 = []
ALPHA_WITHIN = 0.05
ALPHA_BETWEEN = 0.1
GREEDY = 13

def main():
				
				DR = DataReader()
				C = Collector()
				A = ArticlesCollector()
				E = Environment()
				alpha = 0.06
				AG = Agent_model_UCB1(A, C, alpha)
				
				
				E.run([AG], DR)
				E.plot(len([AG]))
				
				
if __name__ == '__main__':
    main()