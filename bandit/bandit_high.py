import numpy as np
from bandit.OGA_path import OGA
def bandit_high(data, residuals, coef):
				"""Run the high-dimensional algorithm on the context.
				
				Parameters
				----------
				data : {array-like}, shape = (n_sample, n_context)
								Context.
								
				residuals : {array-like}, shape = (n_sample, n_context * K)
								Random terms for each arm at every stage. 'K arms' is necessary 
								if we want to compare to the oracle.
								
				coef : {array-like}, shape = (n_context * K)
								Coefficients of each arms.
				
				Returns
				-------
				X : {array-like}, shape = (n_sample)
								The selected arms' indexes at each stage.
				
				y : float
								Sample risk.
				"""
				len_ = len(data[:,1])
				coef_ = coef
				len_coef = len(data[1,:])
				data_ = data
				residual_ = residuals
				beta = 2.0
				
				K = coef_.shape[0]
				p = 0.025 / K
				estimates = np.zeros(len_coef * K).reshape(K, len_coef)
				mid = np.zeros(K)
				selection = np.zeros(len_).astype(int)
				weights = np.zeros(K)
				sum_ = np.zeros(1)
				
				
				#Context containers
				arms_context = [np.empty((0, len_coef), int) for _ in range(K)]
				results = [np.empty((0, 1), int) for _ in range(K)]
				count = 0
				
				for k in range(len_):
								test_ = np.zeros(K)
								for j in range(K):
												if k > K and np.sqrt(k-j).is_integer():
																test_[j] = 1
								
								if int(np.sqrt(k)) != np.sqrt(k) and k >  K * np.sqrt(len_coef) and sum(test_) == 0:
												weights = choice_calculator(estimates, data_[k,:], beta, K, p / K)
												
												
												selection[k] = np.random.choice(range(K), 1, p=weights)
												
								else:
												#print(k)
												#loop_ = 0
												#while loop_ < K:
																#loop_ = loop_ + 1
												selection[k] = int(count % K)
												count = count + 1
												for elem in arms_context:
																pass#print(elem.shape)
												
												arms_context[selection[k]] = np.vstack([
																				arms_context[selection[k]],
																				data_[k,]
																])
												
												#print(coef_)
												results[selection[k]] = np.vstack([
																				results[selection[k]],
																				np.dot(coef_[selection[k]], data_[k]) + 
																								residual_[k, selection[k]]
																])
												
												
												estimates[selection[k]] = np.zeros(len_coef)
												estimates[selection[k]] = high_update(arms_context[selection[k]], 
																results[selection[k]])
								
				return selection, estimates

def choice_calculator(estimates, data_, beta, K, p):
				
				mid = np.zeros(K)
				
				for j in range(K):
								mid[j] = (np.dot(estimates[j], data_)) 
				
				choice = np.ones(K) * p
				choice[mid.argmax()] = (1 - (K - 1) * p)
				return choice


def high_update(context, results):
				#print(context)
				len_coef = len(context[0,:])
				estimates = np.zeros(len_coef)
				
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
				