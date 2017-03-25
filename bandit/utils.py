import numpy as np
from bandit.OGA_path import OGA
def high_update(context, results):

				len_coef = len(context[0,:])
				estimates = np.zeros(len_coef)
				#print(context[0,:])				
				if len(context[:,0]) > np.sqrt(len_coef):
												#High-dimension regression technique is used here.
												
												path_ = OGA(np.array(context),results.ravel())
												
												temp_context_ = context[:,path_]
												
												
												estimates[path_] = np.linalg.inv(
																				temp_context_.transpose().dot(temp_context_)
																).dot(temp_context_.transpose()
																				).dot(results.ravel())
				else:
								#Ridge regression technique is used here.
								estimates = np.linalg.inv(
																context.transpose().dot(context) + np.eye(len_coef)
												).dot(context.transpose()
																).dot(results.ravel())
																
				#print(estimates)
				return estimates
				
def low_update(context, results):
				
				len_coef = len(context[0,:])
				estimates = np.zeros(len_coef)
				
				#Ridge regression technique is used here.
				estimates = np.linalg.inv(
												context.transpose().dot(context) + np.eye(len_coef) * 0.1
								).dot(context.transpose()
												).dot(results.ravel())
																				
				return estimates
				
def cross_update(context, result):
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
																#print(np.matrix(context[:,i]* context[:,j]).T)
																extended_context = np.hstack([extended_context, np.matrix(context[:,i] * context[:,j]).T ])
								#print(np.array(extended_context))
								estimates = high_update(np.array(extended_context)[:,1:], result)								
								
				else:
								#Ridge regression technique is used here.
								estimates = np.linalg.inv(
																context.transpose().dot(context) + np.eye(len_coef)
												).dot(context.transpose()
																).dot(results.ravel())
				#path_ = OGA(np.array(extended_context)[:,1:],result.ravel())
				#print(path_)
				#array-like, shape = (len_coef, 1), list.
				return estimates[np.nonzero(estimates)], [i for i, e in enumerate(estimates) if e != 0]


def choice_calculator(estimates, data_, beta, K, p):
				
				mid = np.zeros(K)
				
				for j in range(K):
								mid[j] = (np.dot(estimates[j], data_)) 
				
				choice = np.ones(K) * p
				choice[mid.argmax()] = (1 - (K - 1) * p)
				return choice
