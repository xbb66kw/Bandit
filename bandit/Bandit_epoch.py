#Bandit setting with K arms. With context given in each stage.


import numpy as np
from bandit.OGA_path import OGA
from bandit.Bandit_oracle import bandit_oracle
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
												'''estimates[selection[k]] = np.linalg.inv(
																				arms_context[selection[k]].transpose().dot(
																								arms_context[selection[k]]) + np.eye(len_coef)
																).dot(arms_context[selection[k]].transpose()).dot(
																								results[selection[k]].ravel())'''
				return selection, estimates

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
												
												
												estimates[selection[k]] = np.zeros(len_coef)
												estimates[selection[k]] = high_update(arms_context[selection[k]], 
																results[selection[k]])
								
				return selection, estimates


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
				p = 0.1 / K
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
																belief =  belief / 1e10
												
																#print('hi')
												#belief = np.array([0,1])
								else:
												#print(k)
												#if (k/2).is_integer():
												belief =  np.ones(n_algo) / n_algo
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


def accumulative_risk(x, results, estimates):
				
				result = 0
				len_ = len(estimates)
				for j in range(len_):
								result = result + np.sum(np.power(np.dot(x[j], estimates[j]) - results[j], 2)) / len(x[j][:,1])
				
				return result

def risk_calculator(x, e, coef_, selection):
				len_ = len(x[:,0])
				sum_ = 0
				for i in range(len_):
								sum_ = sum_ + x[i,:].dot(coef_[selection[i]]) + e[i,selection[i]]
				
				return sum_


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
				
def low_update(context, results):
				
				len_coef = len(context[0,:])
				estimates = np.zeros(len_coef)
				
				#Ridge regression technique is used here.
				estimates = np.linalg.inv(
												context.transpose().dot(context) + np.eye(len_coef) * 0.1
								).dot(context.transpose()
												).dot(results.ravel())
																				
				return estimates
				
def main():
				''' 
				high-dimension winning case:
				n = 1000
				p = 100
				n_arm = 3
				divide = 1
				x = np.random.uniform(0,1,p * n).reshape(n, p)#np.abs(np.random.normal(0,0.2,p * n).reshape(n,p))
				
				for j in range(n):
								x[j,:] = x[j,:] / np.sqrt(np.dot(x[j,:], x[j,:])) 
			
				e = np.random.normal(0,0.01,n_arm * n).reshape(n,n_arm)
				
				coef_high = np.array([np.zeros(p) for _ in range(int(n_arm / divide))])
				for j in range(n_arm):
								#print(coef_high[j])
								#print(coef_high[j][np.random.choice(range(p), 3, replace=False).astype(int)])
								coef_high[j][np.random.choice(range(p), 3, replace=False).astype(int)] \
												= np.random.uniform(0,10, 3)
								coef_high[j] = coef_high[j] / np.sqrt(np.dot(coef_high[j], coef_high[j]))

				coef_ = coef_high
				'''
				n = 2000
				p = 100
				n_arm = 8
				divide = 1
				#x = np.random.normal(0,1,p * n).reshape(n,p)
				x = np.random.uniform(0,1,p * n).reshape(n, p)
				for j in range(n):
								x[j,:] = x[j,:] / np.sqrt(np.dot(x[j,:], x[j,:])) 
			
				e = np.random.normal(0,0.01,n_arm * n).reshape(n,n_arm)
				'''
				low, not so good an example
				n = 2000
				p = 100
				n_arm = 8
				divide = 1
				x = np.random.normal(0,1,p * n).reshape(n,p)
				#x = np.random.uniform(0,1,p * n).reshape(n, p)
				for j in range(n):
								x[j,:] = x[j,:] / np.sqrt(np.dot(x[j,:], x[j,:])) 
			
				e = np.random.normal(0,0.01,n_arm * n).reshape(n,n_arm)
				
				A = [np.empty((0, p), int) for _ in range(int(n_arm / divide))]
			
				for j in range(int(n_arm / divide)):
								A[j] = np.hstack([np.random.uniform(-5,10, p)])		
								A[j] = A[j] / np.sqrt(np.dot(A[j],A[j]))
				coef_ = np.array(A)
				'''
				
				
				#non-sparse example. High-dimension tends to perform badly
				A = [np.empty((0, p), int) for _ in range(int(n_arm / divide))]
				b_ = 0
				for j in range(int(n_arm / divide)):
								#A[j] = np.hstack([np.random.uniform(0,10, p)])
								A[j] = np.zeros(p)
								if j < int(n_arm / divide) / 3:
												#print(A[j][:int(p / 2)])
												A[j][:int(p / 3)] = np.hstack([np.random.uniform(b_,10, int(p / 3))])
								elif j > int(n_arm / divide) / 3 and  j <= int(n_arm / divide) / 3 * 2:
												A[j][int(p / 3) + 1:int(2 * p / 3)] = np.hstack([np.random.uniform(b_,10, int(2 * p / 3) - int(p / 3) - 1)])
								else: 
												A[j][int(2 * p / 3) + 1:] = np.hstack([np.random.uniform(b_,10, p - int(2 * p / 3) - 1)])
								A[j] = A[j] / np.sqrt(np.dot(A[j],A[j]))
				coef_ = np.array(A)
				#coef_ = np.array(np.vstack([A, [_ / 3 for _ in A]])) * 1 #Cool!
				
				
				coef_high = np.array([np.zeros(p) for _ in range(int(n_arm))])
				for j in range(n_arm):
								#print(coef_high[j])
								#print(coef_high[j][np.random.choice(range(p), 3, replace=False).astype(int)])
								coef_high[j][np.random.choice(range(p), 3, replace=False).astype(int)] \
												= np.random.uniform(0,10, 3)
								coef_high[j] = coef_high[j] / np.sqrt(np.dot(coef_high[j], coef_high[j]))

				#coef_ = coef_high
				for elem in coef_:
								print(elem.round(4))
				
				
				select_1 = bandit_oracle(x, coef_)
				select_2 = bandit_low(x, e, coef_)
				select_3 = bandit_high(x, e, coef_)
				select_4 = bandit_bylearning(x, e, coef_)
				#select_5 = bandit_low(x, e, coef_) These algorithm possess some randomness
				#random
				
				print(risk_calculator(x, e, coef_, select_1))
				print(risk_calculator(x, e, coef_, select_2[0]), 'low')
				print(risk_calculator(x, e, coef_, select_3[0]), 'high')
				print(risk_calculator(x, e, coef_, select_4[0]), 'bylearning')
				select_random = np.random.randint(0, coef_.shape[0], n)
				print(risk_calculator(x, e, coef_, select_random), 'random')
				select_random = np.random.randint(0, coef_.shape[0], n)
				print(risk_calculator(x, e, coef_, select_random), 'random')
				#print(risk_calculator(x, e, coef_, select_5[0]))
				#print(select_2[1].round(4))
				#print(select_4[0])				
				#print(select_2[0], 'low')
				#print(select_3[0], 'high')
				for elem in select_3[1]:
								pass#print(elem.round(4))
				
				for elem in select_4[1]:
								for _ in elem:
												pass#print(_.round(4))
				#for elem in select_4[1][1]:
				#				print(elem.round(4))
				
if __name__ == '__main__':
    main()
    
