import numpy as np
from bandit.OGA_path import OGA
from bandit.utils import high_update, low_update, choice_calculator

def bandit_bylearning(data, residuals, coef, n_algo=2):
				"""Run the bylearning algorithm on the context.
				
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
				p = 0.05 / K
				estimates = [np.zeros(len_coef * K).reshape(K, len_coef)
								for _ in range(n_algo)]
		
				choice = [np.zeros(K) for _ in range(n_algo)]
				mid = np.zeros(K)
				selection = np.zeros(len_).astype(int)
				belief = np.ones(n_algo) / n_algo
				sum_ = np.zeros(1)
				#belief = np.array([0,1])
				final_distribution = np.zeros(K)
				
				#Context containers
				arms_context = [np.empty((0, len_coef), int) for _ in range(K)]
				results = [np.empty((0, 1), int) for _ in range(K)]
				count = 0
				loop_ = 0
				for k in range(len_):
								test_ = np.zeros(K)
								for j in range(K):
												if k > K and np.sqrt(k-j).is_integer():
																test_[j] = 1
								
								if k != 0 and not np.sqrt(k).is_integer() and k >  K * np.sqrt(len_coef) and sum(test_) == 0:
												for j in range(n_algo):
																choice[j] = choice_calculator(estimates[j], 
																				data_[k,:], beta, K, p / K)
												
												sum_ = np.zeros(K)
												for j in range(n_algo):
																sum_ = sum_ +  belief[j] * choice[j] / sum(belief)
																
												final_distribution = sum_ #* (1 - K * p) + np.ones(K) * p
												
												selection[k] = np.random.choice(range(K), 1, p=final_distribution)
												
												
												outcome = np.dot(coef_[selection[k]], data_[k]) \
																+ residual_[k, selection[k]]
											
											
												#Calculate the outcome and update the belief:
												mid_ = np.zeros(n_algo)
												for j in range(n_algo):
																mid_[j] = outcome / final_distribution[j] * choice[j][selection[k]]
												for j in range(n_algo):
																belief[j] = belief[j] * np.exp(mid_[j] * p / K)
												
												if max(belief) >= 1e10:
																#belief =  belief / 1e10
																belief =  np.ones(n_algo) / n_algo
																#print('hi')
												#belief = np.array([0,1])
								else:
												
												#print(k)
												#if (k/2).is_integer():
												
												if (k / 2).is_integer() and np.sqrt(k).is_integer() and\
																np.sqrt(loop_).is_integer() :
																#print(k)
																#print(loop_)
																belief =  np.ones(n_algo) / n_algo
												loop_ = loop_ + 1
												#print(k)
												#loop_ = 0
												#while loop_ < K:
												#loop_ = loop_ + 1
												selection[k] = int(count % K)
												count = count + 1
												#np.random.choice(range(K), 1)
												
												
												arms_context[selection[k]] = np.vstack([
																				arms_context[selection[k]],
																				data_[k,]
																])
												
												outcome = np.dot(coef_[selection[k]], data_[k]) \
																+ residual_[k, selection[k]]
												results[selection[k]] = np.vstack([
																				results[selection[k]], outcome])
												
												
												'''
												
												
												'''
												
												estimates[0][selection[k]] = high_update(arms_context[selection[k]], 
																results[selection[k]])
												estimates[1][selection[k]] = low_update(arms_context[selection[k]], 
																results[selection[k]])
								
												'''
												comparing = np.zeros(n_algo)
												for j in range(n_algo):
																
																comparing[j] = accumulative_risk(arms_context, results, 
																				estimates[j])
												j_win = comparing.argmin()
												belief = np.zeros(n_algo)
												belief[j_win] = 1
												'''
												
				print(belief)#, comparing)
				return selection, estimates