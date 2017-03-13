#Bandit setting with K arms. With context given in each stage.


import numpy as np
from OGA_path import OGA

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
				beta = 1.0
				K = coef_.shape[0]
				
				estimates = np.zeros(len_coef * K).reshape(K, len_coef)
				mid = np.zeros(K)
				selection = np.zeros(len_).astype(int)
				weights = np.zeros(K)
				sum_ = np.zeros(1)
						
				#Context containers
				arms_context = [np.empty((0, len_coef), int)] * K
				results = [np.empty((0, 1), int)] * K
	
			
				for k in range(len_):
								weights = weights_calculator(estimates, data_[k,:], beta, K, len_)
					
								selection[k] = np.random.choice(range(K), 1, p=weights)
								
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
								estimates[selection[k]] = np.linalg.inv(
																arms_context[selection[k]].transpose().dot(
																				arms_context[selection[k]]) + np.eye(len_coef)
												).dot(arms_context[selection[k]].transpose()).dot(
																				results[selection[k]].ravel())
								
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
				beta = 1.0
				K = coef_.shape[0]
				
				estimates = np.zeros(len_coef * K).reshape(K, len_coef)
				mid = np.zeros(K)
				selection = np.zeros(len_).astype(int)
				weights = np.zeros(K)
				sum_ = np.zeros(1)
				
				
				#Context containers
				arms_context = [np.empty((0, len_coef), int)] * K
				results = [np.empty((0, 1), int)] * K
		
				
				for k in range(len_):
								weights = weights_calculator(estimates, data_[k,:], beta, K, len_)
								
								#print(mid)
								selection[k] = np.random.choice(range(K), 1, p=weights)
								
								arms_context[selection[k]] = np.vstack([
																arms_context[selection[k]],
																data_[k,]
												])
								
								results[selection[k]] = np.vstack([
																results[selection[k]],
																np.dot(coef_[selection[k]], data_[k]) + 
																				residual_[k, selection[k]]
												])
								
								
								if len(arms_context[selection[k]][:,1]) > np.sqrt(len_coef):
												#High-dimension regression technique is used here.
												path_ = OGA(np.array(arms_context[selection[k]]),
																results[selection[k]].ravel())
												
												temp_context_ = arms_context[selection[k]][:,path_]
												
												estimates[selection[k]] = np.zeros(len_coef)
												estimates[selection[k]][path_] = np.linalg.inv(
																				temp_context_.transpose().dot(
																								temp_context_)
																).dot(temp_context_.transpose()
																				).dot(results[selection[k]].ravel())
								else:
												#Ridge regression technique is used here.
												estimates[selection[k]] = np.linalg.inv(
																				arms_context[selection[k]].transpose().dot(
																								arms_context[selection[k]]) + np.eye(len_coef)
																).dot(arms_context[selection[k]].transpose()
																				).dot(results[selection[k]].ravel())
																	
				
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
				beta = 1.0
				K = coef_.shape[0]
				
				estimates = [np.zeros(len_coef * K).reshape(K, len_coef)] * n_algo
				weights = [np.zeros(K)] * n_algo
				mid = np.zeros(K)
				selection = np.zeros(len_).astype(int)
				belief = np.ones(n_algo) / n_algo
				sum_ = np.zeros(1)
				
				final_distribution = np.zeros(K)
				
				#Context containers
				arms_context = [np.empty((0, len_coef), int)] * K
				results = [np.empty((0, 1), int)] * K
		
				
				for k in range(len_):
								for j in range(n_algo):
												weights[j] = weights_calculator(estimates[j], 
																data_[k,:], beta, K, len_)
								
								sum_ = 0
								for j in range(n_algo):
												sum_ = sum_ + belief[j] * weights[j]
												
								final_distribution = sum_ 
								selection[k] = np.random.choice(range(K), 1, p=final_distribution)
								
								arms_context[selection[k]] = np.vstack([
																arms_context[selection[k]],
																data_[k,]
												])
								
								#Calculate the outcome and update the belief:
								outcome = np.dot(coef_[selection[k]], data_[k]) \
												+ residual_[k, selection[k]]
								results[selection[k]] = np.vstack([
																results[selection[k]], outcome])
								
								mid_ = np.zeros(n_algo)
								for j in range(n_algo):
												mid_[j] = outcome / final_distribution[j] * weight[j][selection[k]]
								
								mid_
								
								#Update estimates:
								
								if len(arms_context[selection[k]][:,1]) > np.sqrt(len_coef):
												#High-dimension regression technique is used here.
												path_ = OGA(np.array(arms_context[selection[k]]),
																results[selection[k]].ravel())
												
												temp_context_ = arms_context[selection[k]][:,path_]
												
												estimates[selection[k]] = np.zeros(len_coef)
												estimates[selection[k]][path_] = np.linalg.inv(
																				temp_context_.transpose().dot(
																								temp_context_)
																).dot(temp_context_.transpose()
																				).dot(results[selection[k]].ravel())
								else:
												#Ridge regression technique is used here.
												estimates[selection[k]] = np.linalg.inv(
																				arms_context[selection[k]].transpose().dot(
																								arms_context[selection[k]]) + np.eye(len_coef)
																).dot(arms_context[selection[k]].transpose()
																				).dot(results[selection[k]].ravel())
																	
				
				return selection, estimates

def bandit_oracle(data, residuals, coef):
				"""Run the oracle algorithm on the context.
				
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
				K = coef_.shape[0]
				
				mid = np.zeros(K)
				selection = np.zeros(len_).astype(int)

				
				
				#Context containers
				arms_context = [np.empty((0, len_coef), int)] * K
				results = [np.empty((0, 1), int)] * K
		
				
				for k in range(len_):
								for j in range(K):	
												mid[j] = np.abs(np.dot(coef_[j], data_[k,:])) 
								selection[k] = mid.argmax()
				
				return selection


def risk_calculator(x, e, coef_, selection):
				len_ = len(x[:,1])
				sum_ = 0
				for i in range(len_):
								sum_ = sum_ + x[i,:].dot(coef_[selection[i]]) + e[i,selection[i]]
				
				return sum_


def weights_calculator(estimates, data_, beta, K, len_):
				
				mid = np.zeros(K)
				
				for j in range(K):
								mid[j] = (beta * np.abs(np.dot(estimates[j], data_)) 
												+ 1 / (len_ / 10 + 1))
				sum_ = np.exp(mid).sum()
				weights = np.exp(mid) / sum_
				
				return weights
				
def main():
				n = 1000
				p = 150
				n_arm = 3
				x = np.abs(np.random.normal(0,0.2,p * n).reshape(n,p))
				e = np.random.normal(0,1,n_arm * n).reshape(n,n_arm)
				
				#non-sparse example. High-dimension tends to perform badly
				coef_ = np.array([np.arange(p),
								np.arange(p)[::-1]*2,
								np.arange(p)*2]) / 300
				
				coef_high = np.array([np.hstack([[2,6,4], np.zeros(p-3)]),
								np.hstack([[2,4,6], np.zeros(p-3)])[::-1],
				np.hstack([[6,2,4], np.zeros(p-3)])])
				coef_ = coef_high
				print(coef_)
				#print(sum(A[0] == 0))
				#print(sum(A[0] == 1))
				#print(sum(A[0] == 2))
				#print(A[1])
				#print(bandit_oracle(x, e, coef_, 3))
				
				select_1 = bandit_oracle(x, e, coef_)
				select_2 = bandit_low(x, e, coef_)
				select_3 = bandit_high(x, e, coef_)
				print(risk_calculator(x, e, coef_, select_1))
				print(risk_calculator(x, e, coef_, select_2[0]))
				print(risk_calculator(x, e, coef_, select_3[0]))
				#print(select_2[1].round(4))
				print(select_3[0].round(4))				
				print(select_3[1].round(4))
				
				
if __name__ == '__main__':
    main()
    
