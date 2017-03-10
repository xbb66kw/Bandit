#Bandit setting with K arms. With context given in each stage.


import numpy as np
from OGA_path import OGA

def bandit_low(data, residuals, coef, K):
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
				beta = 2
				
				estimates = np.zeros(len_coef * K).reshape(K, len_coef)
				mid = np.zeros(K)
				selection = np.zeros(len_).astype(int)
				weights = np.zeros(K)
				sum_ = np.zeros(1)
						
				#Context containers
				arms_context = [np.empty((0, len_coef), int)] * K
				results = [np.empty((0, 1), int)] * K
	
			
				for k in range(len_):
								for j in range(K):
												mid[j] = (beta * np.abs(np.dot(estimates[j], data_[k,:])) 
																+ 1 / (k + 1))
								sum_ = np.exp(mid).sum()
								weights = np.exp(mid) / sum_
								print(weights.round(4))
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
								
								#Ridge regression technique is used here.
								estimates[selection[k]] = np.linalg.inv(
																arms_context[selection[k]].transpose().dot(
																				arms_context[selection[k]]) + np.eye(len_coef)
												).dot(arms_context[selection[k]].transpose()).dot(
																				results[selection[k]].ravel())
		
				return selection, estimates

def bandit_high(data, residuals, coef, K):
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
				beta = 2
				
				estimates = np.zeros(len_coef * K).reshape(K, len_coef)
				mid = np.zeros(K)
				selection = np.zeros(len_).astype(int)
				weights = np.zeros(K)
				sum_ = np.zeros(1)
				
				
				#Context containers
				arms_context = [np.empty((0, len_coef), int)] * K
				results = [np.empty((0, 1), int)] * K
		
				
				for k in range(len_):
								for j in range(K):
												mid[j] = (beta * np.abs(np.dot(estimates[j], data_[k,:])) 
																+ 1 / (k + 1))
								sum_ = np.exp(mid).sum()
								weights = np.exp(mid) / sum_
								#print(weights.round(4))
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
												#print(arms_context[selection[k]], results[selection[k]])
												path_ = OGA(np.array(arms_context[selection[k]]),
																results[selection[k]].ravel())
												
												temp_context_ = arms_context[selection[k]][:,path_]
												
												#print(temp_context_.transpose(), results[selection[k]].ravel(), estimates[selection[k]])
												
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


def main():
				x = np.abs(np.random.normal(0,1,5000).reshape(1000,5))
				e = np.random.normal(0,1,3000).reshape(1000,3)
				coef_ = np.array([[1,2,3,4,5],
								[10,8,6,4,2],
								[2,4,6,8,10]]) / 10
				A = bandit_high(x, e, coef_, 3)
				print(sum(A[0] == 0))
				print(sum(A[0] == 1))
				print(sum(A[0] == 2))
				print(A[1])

if __name__ == '__main__':
    main()
    
