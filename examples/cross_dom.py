#!/usr/bin/env python
#Bandit setting with K arms. With context given in each stage.
import sys
import os.path
import re
'''
Not sure it's a proper way to do it.
'''
path = os.path.dirname(__file__)
sys.path.append(re.sub(r'examples', '', path))


import numpy as np
from bandit.OGA_path import OGA
from bandit.Bandit_oracle import bandit_oracle
from bandit.bandit_low import bandit_low, bandit_low_cross
from bandit.bandit_high import bandit_high
from bandit.bandit_bylearning import bandit_bylearning



def risk_calculator(x, e, coef_, selection):
				len_ = len(x[:,0])
				sum_ = 0
				band = np.arange(0, len_+1, 200)
				results = np.zeros(int((len_+1) / 200))
				j = 0
				
				for i in range(len_):
								
								sum_ = sum_ + x[i,:].dot(coef_[selection[i]]) + e[i,selection[i]]
								if i in band:
												results[j] = sum_
												j = j + 1
				
				return results

#Generate the cross terms in context
def cross_gene_(context):
				n = len(context[:,0])
				context = np.array(np.hstack([np.matrix(np.ones(n)).T, context]))
				
				len_coef = len(context[0,:])
				estimates = np.zeros(len_coef)
				
				name_index = range(len_coef)
				total_ = len_coef * len_coef / 2
				extended_context = np.matrix(np.empty((0,n), int)).T
				
				if n > np.sqrt(total_):
								for i in range(len_coef):
												for j in range(len_coef)[i:]:
																extended_context = np.hstack([extended_context, np.matrix(context[:,i] * context[:,j]).T ])
				else: 
								extended_context = context
				
				return np.array(extended_context)[:,1:]
				
				
def main():
				
				np.random.seed(47)	
				n = 10000
				p = 10
				n_arm = 2
				divide = 1
				coef_ = np.array([np.zeros(p + int((p + 1) * p / 2)) for _ in range(int(n_arm))])
				coef_[0][11] = -6.0
				coef_[0][1] = 4.0
				
				coef_[1][1] = 4.0
				coef_[1][0] = -6.0
				#coef_[1][11] = 4
				'''
				data
				'''
				R = 50
				size = int((n+1) / 200)
				X_oracle = np.zeros(R * size).reshape(size, R)
				X_low = np.zeros(R * size).reshape(size, R)
				X_cross = np.zeros(R * size).reshape(size, R)
				X_random = np.zeros(R * size).reshape(size, R)
				
				for r_ in range(R):
								
				
				
								x = np.random.uniform(-1,1,p * n).reshape(n, p)
								for j in range(n):
												x[j,:] = x[j,:] / np.sqrt(np.dot(x[j,:], x[j,:])) * 2
							
								e = np.random.normal(0,0.05,n_arm * n).reshape(n,n_arm)
								
								
								
								
								for i in range(n_arm):
												coef_[i] = coef_[i] / np.sqrt(np.dot(coef_[i], coef_[i])) * 1
								#print(coef_)
								X = cross_gene_(x)
								#print(X[0:6, 0:20])
								
								select_cross_low = bandit_low_cross(x, e, coef_)
								select_cross = bandit_high(X, e, coef_) #Consider the cross-terms
								select_oracle = bandit_oracle(X, coef_)
								select_random = np.random.randint(0, coef_.shape[0], n)
								
								'''
								print(select_cross[0][500:600])
								
								print(risk_calculator(X, e, coef_, select_oracle)[10], 'oracle')
								print(risk_calculator(X, e, coef_, select_cross[0])[10], 'cross')
								print(risk_calculator(X, e, coef_, select_cross_low[0])[10], 'cross_low')
								print(risk_calculator(X, e, coef_, select_random), 'random')
								'''
								
								X_oracle[:, r_] = risk_calculator(X, e, coef_, select_oracle)
								X_low[:, r_] = risk_calculator(X, e, coef_, select_cross_low[0])
								X_cross[:, r_] = risk_calculator(X, e, coef_, select_cross[0])
								X_random[:, r_] = risk_calculator(X, e, coef_, select_random)
				'''
				Save the simulating data
				'''
								
				'
				np.save("/Users/xbb/Desktop/OGA-test/data/cross_oracle.npy", X_oracle)
				np.save("/Users/xbb/Desktop/OGA-test/data/cross_low.npy", X_low)
				np.save("/Users/xbb/Desktop/OGA-test/data/cross_cross.npy", X_cross)
				np.save("/Users/xbb/Desktop/OGA-test/data/cross_random.npy", X_random)
				'
				

if __name__ == '__main__':
    main()
    
