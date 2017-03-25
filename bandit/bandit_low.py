import numpy as np
from bandit.utils import low_update, choice_calculator

def bandit_low(data, residuals, coef):
				"""Run the low-dimensional algorithm on the context.
				
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
								
								
								if int(np.sqrt(k)) != np.sqrt(k) and k >  K * np.sqrt(len_coef) and sum(test_) == 0:#int(k / 10) != k / 10: #int(np.sqrt(k)) != np.sqrt(k):
												weights = choice_calculator(estimates, data_[k,:], beta, K, p / K)
												#weights * (1 - K * p) + np.ones(K) * p
												
												
												selection[k] = np.random.choice(range(K), 1, p=weights)
												
								else:
												#print(k)
												#loop_ = 0
												#while loop_ < K:
												#loop_ = loop_ + 1
												selection[k] = int(count % K)
												count = count + 1
								
												arms_context[selection[k]] = np.vstack([
																				arms_context[selection[k]],
																				data_[k,]
																])
												
												results[selection[k]] = np.vstack([
																				results[selection[k]],
																				np.dot(coef_[selection[k]], data_[k]) + 
																								residual_[k, selection[k]]
																])
												
												#Ridge regression technique is used here. lstsq(...) is not ok.
												estimates[selection[k]] = low_update(arms_context[selection[k]], 
																results[selection[k]])
												
				return selection, estimates


def bandit_low_cross(data, residuals, coef):
				"""Run the low-dimensional algorithm on the context.
				
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
				coef_ = coef #including the cross-terms
				len_coef = len(data[1,:])
				data_ = data
				
				cross_data_ = cross_gene_(data_)
				
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
								
								
								if int(np.sqrt(k)) != np.sqrt(k) and k >  K * np.sqrt(len_coef) and sum(test_) == 0:#int(k / 10) != k / 10: #int(np.sqrt(k)) != np.sqrt(k):
												weights = choice_calculator(estimates, data_[k,:], beta, K, p / K)
												#weights * (1 - K * p) + np.ones(K) * p
												
												
												selection[k] = np.random.choice(range(K), 1, p=weights)
												
								else:
												#print(k)
												#loop_ = 0
												#while loop_ < K:
												#loop_ = loop_ + 1
												selection[k] = int(count % K)
												count = count + 1
								
												arms_context[selection[k]] = np.vstack([
																				arms_context[selection[k]],
																				data_[k,]
																])
												
												results[selection[k]] = np.vstack([
																				results[selection[k]],
																				np.dot(coef_[selection[k]], cross_data_[k]) + 
																								residual_[k, selection[k]]
																])
												
												#Ridge regression technique is used here. lstsq(...) is not ok.
												estimates[selection[k]] = low_update(arms_context[selection[k]], 
																results[selection[k]])
												
				return selection, estimates


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
																
				return np.array(extended_context)[:,1:]
				
				