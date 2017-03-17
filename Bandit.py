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
	

				for k in range(len_):
								weights = weights_calculator(estimates, data_[k,:], beta, K, p / K)
								weights * (1 - K * p) + np.ones(K) * p
								'''
								for elem in range(K):
												print(weights[elem].round(3))
								print(estimates)
								print('-----')
								'''
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
		
				
				for k in range(len_):
								weights = weights_calculator(estimates, data_[k,:], beta, K, p / K)
								
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
								
								#estimates[selection[k]] = high_update(arms_context[selection[k]], 
									#			results[selection[k]])
								
								if len(arms_context[selection[k]][:,1]) > np.sqrt(len_coef):
												#High-dimension regression technique is used here.
												path_ = OGA(np.array(arms_context[selection[k]]),
																results[selection[k]].ravel())
												#print(k)
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
				beta = 2.0
				K = coef_.shape[0]
				p = 0.025 / K
				estimates = [np.zeros(len_coef * K).reshape(K, len_coef)
								for _ in range(n_algo)]
		
				weights = [np.zeros(K) for _ in range(n_algo)]
				mid = np.zeros(K)
				selection = np.zeros(len_).astype(int)
				belief = np.ones(n_algo) / n_algo
				sum_ = np.zeros(1)
				
				final_distribution = np.zeros(K)
				
				#Context containers
				arms_context = [np.empty((0, len_coef), int) for _ in range(K)]
				results = [np.empty((0, 1), int) for _ in range(K)]
				
				for k in range(len_):
								for j in range(n_algo):
												weights[j] = weights_calculator(estimates[j], 
																data_[k,:], beta, K, p / K)
								
								sum_ = np.zeros(K)
								for j in range(n_algo):
												sum_ = sum_ +  belief[j] * weights[j] / sum(belief)
												
								final_distribution = sum_ * (1 - K * p) + np.ones(K) * p
								#print(final_distribution,'final')
								#print(weights)
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
												mid_[j] = outcome / final_distribution[j] * weights[j][selection[k]]
												#print(1/final_distribution[j] * weights[j][selection[k]])
								#print(belief)
								#print(outcome, 'outcome')
								#A base p
								#print(mid_)
								#for j in range(n_algo):
												#mid_[j] = min(mid_[j], 20)
			
								for j in range(n_algo):
												belief[j] = belief[j] * np.exp(mid_[j] * 0.001)
								#belief = np.array([1,0])
								#print(belief)
								#print(mid_)
								#belief = np.array([0,1])
								#Update estimates:
								estimates[0][selection[k]] = high_update(arms_context[selection[k]], 
												results[selection[k]])
								#print(estimates[0])
								estimates[1][selection[k]] = low_update(arms_context[selection[k]], 
												results[selection[k]])
								
								
								
																	
				print(belief, weights)
				return selection, estimates

def bandit_oracle(data, coef):
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
				len_ = len(data[:,0])
				coef_ = coef
				len_coef = len(data[0,:])
				data_ = data
				K = coef_.shape[0]
				
				mid = np.zeros(K)
				selection = np.zeros(len_).astype(int)

				
				for k in range(len_):
								for j in range(K):	
												mid[j] = np.dot(coef_[j], data_[k,:])
								selection[k] = mid.argmax()
				
				return selection


def risk_calculator(x, e, coef_, selection):
				len_ = len(x[:,0])
				sum_ = 0
				for i in range(len_):
								sum_ = sum_ + x[i,:].dot(coef_[selection[i]]) + e[i,selection[i]]
				
				return sum_


def weights_calculator(estimates, data_, beta, K, p):
				
				mid = np.zeros(K)
				beta = 7.0
				for j in range(K):
								#print(np.dot(estimates[j], data_) )
								mid[j] = (np.dot(estimates[j], data_) *  beta) 
				sum_ = sum(np.exp(mid))
				weights = (np.exp(mid) / sum_)# * (1 - K * p) + p
				
				return weights


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
												context.transpose().dot(context) + np.eye(len_coef)
								).dot(context.transpose()
												).dot(results.ravel())
																				
				return estimates
				
def main():
				n = 1000
				p = 30
				q = 15
				n_arm = 10
				divide = 1
				x = np.random.uniform(0,1,p * n).reshape(n, p)#np.abs(np.random.normal(0,0.2,p * n).reshape(n,p))
				#x = np.random.normal(0,1,p * n)
				#x[abs(x)>1.5] = 1.5
				#x = x.reshape(n,p)
				for j in range(n):
								x[j,:] = x[j,:] / np.sqrt(np.dot(x[j,:], x[j,:])) 
				#print(x)
				e = np.random.normal(0,0.00,n_arm * n).reshape(n,n_arm)
				
				#non-sparse example. High-dimension tends to perform badly
				A = [np.empty((0, p), int) for _ in range(int(n_arm / divide))]
			
				for j in range(int(n_arm / divide)):
								A[j] = np.hstack([np.random.uniform(0,10, p)])
								A[j] = A[j] / np.sqrt(np.dot(A[j],A[j]))
				coef_ = np.array(A)
				#coef_ = np.array(np.vstack([A, [_ / 2 for _ in A]])) * 1 #Cool!
				
				
				coef_high = np.array([np.hstack([[2,6,4], np.zeros(p-3)]),
								np.hstack([[2,4,6], np.zeros(p-3)])[::-1],
				np.hstack([[6,2,4], np.zeros(p-3)])])
				coef_ = coef_high
				for elem in coef_:
								print(elem.round(4))
				#print(sum(A[0] == 0))
				#print(sum(A[0] == 1))
				#print(sum(A[0] == 2))
				#print(A[1])
				#print(bandit_oracle(x, e, coef_, 3))
				
				select_1 = bandit_oracle(x, coef_)
				select_2 = bandit_low(x, e, coef_)
				select_3 = bandit_high(x, e, coef_)
				select_4 = bandit_bylearning(x, e, coef_)
				#select_5 = bandit_low(x, e, coef_) These algorithm possess some randomness
				print(risk_calculator(x, e, coef_, select_1))
				print(risk_calculator(x, e, coef_, select_2[0]))
				print(risk_calculator(x, e, coef_, select_3[0]))
				print(risk_calculator(x, e, coef_, select_4[0]))
				#print(risk_calculator(x, e, coef_, select_5[0]))
				#print(select_2[1].round(4))
				#print(select_4[0])				
				#print(select_2[0])
				'''
				for elem in select_2[1]:
								print(elem.round(4))
				
				for elem in select_4[1]:
								for _ in elem:
												pass#print(_.round(4))
				for elem in select_4[1][1]:
								print(elem.round(4))
				'''
if __name__ == '__main__':
    main()
    
