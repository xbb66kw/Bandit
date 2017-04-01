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
				p = 100
				n_arm = 6
				divide = 1
				A = [np.empty((0, p), int) for _ in range(int(n_arm / divide))]
			
				for j in range(int(n_arm / divide)):
								A[j] = np.hstack([np.random.uniform(-5,10, p)])		
								A[j] = A[j] / np.sqrt(np.dot(A[j],A[j]))
				coef_ = np.array(A)
				
				
				
				#non-sparse example. High-dimension tends to perform badly
				A = [np.empty((0, p), int) for _ in range(int(n_arm / divide))]
				b_ = 0
				for j in range(int(n_arm / divide)):
								A[j] = np.zeros(p)
								if j < int(n_arm / divide) / 3:
												A[j][:int(p / 3)] = np.hstack([np.random.uniform(b_,10, int(p / 3))])
								elif j > int(n_arm / divide) / 3 and  j <= int(n_arm / divide) / 3 * 2:
												A[j][int(p / 3) + 1:int(2 * p / 3)] = np.hstack([np.random.uniform(b_,10, int(2 * p / 3) - int(p / 3) - 1)])
								else: 
												A[j][int(2 * p / 3) + 1:] = np.hstack([np.random.uniform(b_,10, p - int(2 * p / 3) - 1)])
								A[j] = A[j] / np.sqrt(np.dot(A[j],A[j]))
				coef_ = np.array(A)
				
				
				'''
				data
				'''
				R = 50
				size = int((n+1) / 200)
				X_oracle = np.zeros(R * size).reshape(size, R)
				X_low = np.zeros(R * size).reshape(size, R)
				X_high = np.zeros(R * size).reshape(size, R)
				X_bylearning = np.zeros(R * size).reshape(size, R)
				X_random = np.zeros(R * size).reshape(size, R)
				
				for r_ in range(R):
								
								
								
								
								x = np.random.uniform(-1,1,p * n).reshape(n, p)
								for j in range(n):
												x[j,:] = x[j,:] / np.sqrt(np.dot(x[j,:], x[j,:])) * 2
								
								e = np.random.normal(0,0.05,n_arm * n).reshape(n,n_arm)
								
								
								
								
								
								
								select_1 = bandit_oracle(x, coef_)
								select_2 = bandit_low(x, e, coef_)
								select_3 = bandit_high(x, e, coef_)
								select_4 = bandit_bylearning(x, e, coef_)
								select_random = np.random.randint(0, coef_.shape[0], n)
								'''
								print(risk_calculator(x, e, coef_, select_1).round(4))
								print(risk_calculator(x, e, coef_, select_2[0]).round(4), 'low')
								print(risk_calculator(x, e, coef_, select_3[0]).round(4), 'high')
								print(risk_calculator(x, e, coef_, select_4[0]).round(4), 'bylearning')
								print(risk_calculator(x, e, coef_, select_random), 'random')
								'''
								X_oracle[:, r_] = risk_calculator(x, e, coef_, select_1)
								X_low[:, r_] = risk_calculator(x, e, coef_, select_2[0])
								X_high[:, r_] = risk_calculator(x, e, coef_, select_3[0])
								X_bylearning[:, r_] = risk_calculator(x, e, coef_, select_4[0])
								X_random[:, r_] = risk_calculator(x, e, coef_, select_random)
				
				
				'''
				Save the simulating data
				'''
				'''
				np.save("/Users/xbb/Desktop/OGA-test/data/low_oracle.npy", X_oracle)
				np.save("/Users/xbb/Desktop/OGA-test/data/low_low.npy", X_low)
				np.save("/Users/xbb/Desktop/OGA-test/data/low_high.npy", X_high)
				np.save("/Users/xbb/Desktop/OGA-test/data/low_bylearning.npy", X_bylearning)
				np.save("/Users/xbb/Desktop/OGA-test/data/low_random.npy", X_random)
				'''
if __name__ == '__main__':
    main()
    
