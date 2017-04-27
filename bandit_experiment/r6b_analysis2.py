#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import matplotlib.pyplot as plt
import numpy as np
import gzip
import re

class Test(object):
				def __init__(self):
								print('test1')


def main():
				k = 50
				K = 5
				f = open("/Users/xbb/Desktop/bandit_experiment/extracted_data_568610.txt")
				'/Users/xbb/Desktop/bandit_experiment/extracted_data_568610.txt'
				
				cha = f.readline()
				nodes = {}
				i = 0
				line_number = 0
				while cha:
				
								
								result = np.array([int(i) for i in list(cha) if i != '\n'])
								nodes[i] = result
								i += 1
								cha = f.readline()
								line_number += 1
				
				
				select = np.random.choice(line_number, k, replace = False)
				total = 0
				for index, key in enumerate(nodes):
								if any_connection_in_k(select, nodes[key], nodes, K):
												total += 1
				
				print(total)
				
def distant(A, B):
				return np.sum(np.power((A - B), 2))


def any_connection_in_k(select, node, nodes, K):
				'node: array-like'
				for key in select:
								if distant(node, nodes[key]) <= K:
												return True
												
				return False
if __name__ == '__main__':
    main()
    
