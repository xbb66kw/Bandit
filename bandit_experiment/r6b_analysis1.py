#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import gzip
import re


def main():
				
				
				files_list = ['/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111003.gz', 
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111004.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111005.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111006.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111007.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111008.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111009.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111010.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111011.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111012.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111013.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111014.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111015.gz',
								'/Users/xbb/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111016.gz']
								
				T = 1
				fin = gzip.open(files_list[T], 'r')
				
				
				click_names = {}
				non_click_names = {}
				article_lists = {}
				total_click_through = 0
				every_body = {}
				S = 0
				while T < 2:# and S < 20000:#len(files_list):
								S += 1
								x = fin.readline()
								cha = x.decode('utf-8')
								if not x:
												T += 1
												fin = gzip.open(files_list[T], 'r')
												x = fin.readline()
												cha = x.decode('utf-8')
											
								matches = re.search(r"id-(\d+)\s(\d).+user\s([\s\d]+)(.+)", cha)
								
								article = int(matches.group(1))
								
								if article not in article_lists:
												article_lists[article] = 1
												every_body[article] = {}
												
								click = int(matches.group(2))
								covariates = np.zeros(136)
								covariates[[int(elem) - 1 for elem in matches.group(3).split(' ') if elem != '']] = 1
								
							
								sentence = [str(int(cha)) for cha in covariates]
								every_body[article][''.join(sentence)] = 1
								
								
								if click and not ''.join(sentence) in click_names:
												click_names[''.join(sentence)] = {}
												click_names[''.join(sentence)][article] = 1
												total_click_through += 1
								elif click and not article in click_names[''.join(sentence)]:				
												click_names[''.join(sentence)][article] = 1
								elif click:
												click_names[''.join(sentence)][article] += 1
								
								if not click and not ''.join(sentence) in non_click_names:
												non_click_names[''.join(sentence)] = {}
												non_click_names[''.join(sentence)][article] = 1
								elif not click and not article in non_click_names[''.join(sentence)]:				
												non_click_names[''.join(sentence)][article] = 1
								elif not click:
												non_click_names[''.join(sentence)][article] += 1
								
				'''
				s = 0
				total_user = 0
				for i, j in enumerate(click_names):
								if j in non_click_names:
												print(j, click_names[j], non_click_names[j])
								else:
												print(j, click_names[j], 'no key')
				'''
				
				list_ = article_lists.keys()
				for article in list_:
								print(article, '------------------------------------------')
								for i, j in enumerate(click_names):
												if j in non_click_names and article in click_names[j]\
																and click_names[j][article] * sum(non_click_names[j].values()) / 22 >= 0.06:
																#and sum(non_click_names[j].values()) <= 9000:
																
																print(j, click_names[j], sum(non_click_names[j].values()))
																
				print("------------------------------", '\n\n\n\n\n\n\n\n\n\n\n')												
				for article in list_:
								print(article, '------------------------------------------')
								for i, j in enumerate(click_names):
												if j in non_click_names and article in click_names[j]\
																and click_names[j][article] * sum(non_click_names[j].values()) / 22 >= 0.06:
																#and sum(non_click_names[j].values()) <= 9000:
																
																print(j, sum(non_click_names[j].values()))												
																
				print("Third------------------------------", '\n\n\n\n\n\n\n\n\n\n\n')												
				for article in list_:
								print(article, '------------------------------------------')
								for i, j in enumerate(click_names):
												if j in non_click_names and article in click_names[j]\
																and click_names[j][article] * sum(non_click_names[j].values()) / 22 >= 0.06:
																#and sum(non_click_names[j].values()) <= 9000:
																
																print(j)
				
				
				print("Foruth------------------------------", '\n\n\n\n\n\n\n\n\n\n\n')												
				for article in list_:
								print(article, '------------------------------------------')
								for i, j in enumerate(click_names):
												if j in non_click_names and article in click_names[j]\
																and click_names[j][article] * sum(non_click_names[j].values()) / 22 < 0.06:
																
																print(j)	
				'''
				for i, j in enumerate(non_click_names):
								if j not in click_names:
												print(print(j, sum(non_click_names[j].values())), 'never-click')
				'''
				'''
				print("Fifth------------------------------", '\n\n\n\n\n\n\n\n\n\n\n')												
				for article in list_:
								print(article, '------------------------------------------')
								for i, j in enumerate(every_body[article]):																
												print(j)
				'''
				print("Sixth------------------------------", '\n\n\n\n\n\n\n\n\n\n\n')												
				for article in list_:
								print(article, '------------------------------------------')
								for i, j in enumerate(click_names):
												if j in non_click_names and article in click_names[j]\
																and click_names[j][article] * sum(non_click_names[j].values()) / 22 >= 0.06\
																and sum(non_click_names[j].values()) > 300:
																
																print(j, sum(non_click_names[j].values()))
				
				print('total users', S)
				print('total click-through member', total_click_through)
				
if __name__ == '__main__':
    main()
    
