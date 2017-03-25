#Bandit setting with K arms. With context given in each stage.


import numpy as np
from bandit.OGA_path import OGA
from bandit.Bandit_oracle import bandit_oracle
from bandit.bandit_low import bandit_low, bandit_low_cross
from bandit.bandit_high import bandit_high
from bandit.bandit_bylearning import bandit_bylearning



def risk_calculator(x, e, coef_, selection):
				len_ = len(x[:,0])
				sum_ = 0
				for i in range(len_):
								sum_ = sum_ + x[i,:].dot(coef_[selection[i]]) + e[i,selection[i]]
				
				return sum_

def risk_calculator_cross(x, e, coef_, selection):
				data_ = cross_gene_(x)
				len_ = len(x[:,0])
				sum_ = 0
				for i in range(len_):
								sum_ = sum_ + data_[i,:].dot(coef_[selection[i]]) + e[i,selection[i]]
				
				return sum_


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
				
				
def main():
				'''
				high-dimension winning case:
				n = 2000
				p = 200
				n_arm = 6
				divide = 1
				x = np.random.uniform(0,1,p * n).reshape(n, p)#np.abs(np.random.normal(0,0.2,p * n).reshape(n,p))
				
				for j in range(n):
								x[j,:] = x[j,:] / np.sqrt(np.dot(x[j,:], x[j,:])) * 2
			
				e = np.random.normal(0,0.05,n_arm * n).reshape(n,n_arm)
				
				coef_high = np.array([np.zeros(p) for _ in range(int(n_arm / divide))])
				for j in range(n_arm):
								#print(coef_high[j])
								#print(coef_high[j][np.random.choice(range(p), 3, replace=False).astype(int)])
								coef_high[j][np.random.choice(range(p), 3, replace=False).astype(int)] \
												= np.random.uniform(0,10, 3)
								coef_high[j] = coef_high[j] / np.sqrt(np.dot(coef_high[j], coef_high[j]))

				coef_ = coef_high
				'''
				n = 1000
				p = 100
				n_arm = 6
				divide = 1
				x = np.random.uniform(0,1,p * n).reshape(n, p)
				for j in range(n):
								x[j,:] = x[j,:] / np.sqrt(np.dot(x[j,:], x[j,:])) * 2
			
				e = np.random.normal(0,0.05,n_arm * n).reshape(n,n_arm)
				'''
				low, 
				n = 2000
				p = 100
				n_arm = 6
				divide = 1
				x = np.random.uniform(0,1,p * n).reshape(n, p)
				for j in range(n):
								x[j,:] = x[j,:] / np.sqrt(np.dot(x[j,:], x[j,:])) * 2
			
				e = np.random.normal(0,0.05,n_arm * n).reshape(n,n_arm)
				
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
								A[j] = np.zeros(p)
								if j < int(n_arm / divide) / 3:
												A[j][:int(p / 3)] = np.hstack([np.random.uniform(b_,10, int(p / 3))])
								elif j > int(n_arm / divide) / 3 and  j <= int(n_arm / divide) / 3 * 2:
												A[j][int(p / 3) + 1:int(2 * p / 3)] = np.hstack([np.random.uniform(b_,10, int(2 * p / 3) - int(p / 3) - 1)])
								else: 
												A[j][int(2 * p / 3) + 1:] = np.hstack([np.random.uniform(b_,10, p - int(2 * p / 3) - 1)])
								A[j] = A[j] / np.sqrt(np.dot(A[j],A[j]))
				coef_ = np.array(A)
				#coef_ = np.array(np.vstack([A, [_ / 3 for _ in A]])) * 1 #Cool!
				
				#Sparse example:
				coef_high = np.array([np.zeros(p) for _ in range(int(n_arm))])
				for j in range(n_arm):
								#print(coef_high[j])
								#print(coef_high[j][np.random.choice(range(p), 3, replace=False).astype(int)])
								coef_high[j][np.random.choice(range(p), 3, replace=False).astype(int)] \
												= np.random.uniform(0,10, 3)
								coef_high[j] = coef_high[j] / np.sqrt(np.dot(coef_high[j], coef_high[j]))

				#coef_ = coef_high

				
				
				select_1 = bandit_oracle(x, coef_)
				select_2 = bandit_low(x, e, coef_)
				select_3 = bandit_high(x, e, coef_)
				select_4 = bandit_bylearning(x, e, coef_)
				
				
				print(risk_calculator(x, e, coef_, select_1))
				print(risk_calculator(x, e, coef_, select_2[0]), 'low')
				print(risk_calculator(x, e, coef_, select_3[0]), 'high')
				print(risk_calculator(x, e, coef_, select_4[0]), 'bylearning')
				select_random = np.random.randint(0, coef_.shape[0], n)
				print(risk_calculator(x, e, coef_, select_random), 'random')
				select_random = np.random.randint(0, coef_.shape[0], n)
				print(risk_calculator(x, e, coef_, select_random), 'random')
				
				
				
				
				
				
				
				'''
				'''
				
				
				
				#Cross-terms;

				
				n = 1000
				p = 10
				n_arm = 2
				divide = 1
				x = np.random.uniform(0,1,p * n).reshape(n, p)
				for j in range(n):
								x[j,:] = x[j,:] / np.sqrt(np.dot(x[j,:], x[j,:])) * 2
			
				e = np.random.normal(0,0.05,n_arm * n).reshape(n,n_arm)
				
				
				coef_ = np.array([np.zeros(p + int((p + 1) * p / 2)) for _ in range(int(n_arm))])
				coef_[0][10] = 4.2
				coef_[0][0] = -2.0
				#coef_[1][1] = -2.0
				coef_[1][11] = 4.0
				
				for i in range(n_arm):
								coef_[i] = coef_[i] / np.sqrt(np.dot(coef_[i], coef_[i]))
				print(coef_)
				X = cross_gene_(x)
				
				select_cross_low = bandit_low_cross(x, e, coef_)
				select_cross = bandit_high(X, e, coef_)
				select_1 = bandit_oracle(X, coef_)
				select_random = np.random.randint(0, coef_.shape[0], n)
				#print(select_1)
				print(risk_calculator_cross(x, e, coef_, select_1), 'oracle')
				print(risk_calculator_cross(x, e, coef_, select_cross[0]), 'cross')
				print(risk_calculator_cross(x, e, coef_, select_cross_low[0]), 'cross_low')
				print(risk_calculator_cross(x, e, coef_, select_random), 'random')
				

if __name__ == '__main__':
    main()
    
