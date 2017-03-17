import numpy as np

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
