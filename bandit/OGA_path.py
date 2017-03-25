#OGA path implemented in python


import numpy as np


def OGA(X, y, Kn=0):
				"""
								Compute the OGA path with length Kn.
								----------
								X : {array-like}, shape = (n, p)
												Covariates.
								y : {array-like}, shape = (n)
												Dependent data.
								Kn : int
												Length of the path.
				"""
				
				#Taking care of parameters. se_catcher is not necessarily needed here.
				len_ = len(X[:,1])
				p_ = len(X[1,:])
				Kn = np.sqrt(len(X[1,:])).astype(np.int64)
				jhat = np.zeros(Kn).astype(np.int64) - 1
				se_ = np.zeros(p_)
				x_hat = np.zeros(len_ * Kn).reshape(len_, Kn)
				se_catcher = np.zeros(Kn)
				
				
				#Nomalizing
				y = y - np.mean(y)
				X = X - np.mean(X, axis = 0)
				normalizer = np.dot(X.transpose(), X)[range(p_), range(p_)]
				X = X / np.sqrt(normalizer)
				#print(X, normalizer)
				
				#Run first step separately 
				for j in range(p_):
								se_[j] = np.abs(np.dot(y, X[:,j]))
				
				jhat[0] = np.argmax(se_)
				x_hat[:,0] = X[:,jhat[0]]
				u = y - x_hat[:,0] * np.dot(x_hat[:,0], y)
				se_catcher[0] = np.dot(y-u, y-u)
				
				
				#The other (Kn - 1) steps
				for k in np.arange(Kn)[np.arange(Kn) != 0]:
								for j in range(p_):
												se_[j] = np.abs(np.dot(u, X[:,j]))
								se_[jhat[jhat != -1]] = 0 #Just to make sure
								jhat[k] = np.argmax(se_)
								#Orthogonalizing
								x_hat[:,k] = X[:,jhat[k]] - np.dot(x_hat[:,range(k)], np.dot(x_hat[:,range(k)].transpose(), X[:,jhat[k]]))
								x_hat[:,k] = x_hat[:,k] / np.sqrt(np.dot(x_hat[:,k], x_hat[:,k])) 
								u = u - x_hat[:,k] * np.dot(x_hat[:,k], u)
				
								se_catcher[k] = np.dot(u, u)
				
				HDICs = np.array([len_ * np.log(se_catcher / len_) + (np.arange(Kn) + 1) * np.log(len_) * np.log(p_)])[0]
				
				HDIC_min = HDICs.argmin()
				HDIC_path = jhat[0:(HDIC_min+1):1]
				HDIC_check = np.zeros(HDIC_min + 1)
				
				if HDIC_min != 0:
								for j in range(HDIC_min + 1):
												X_resi = X[:,HDIC_path[np.arange(len(HDIC_path))!=j]]	
												HDIC_check[j] = len_ * np.log(np.linalg.lstsq(X_resi, y)[1] / len_) + (HDIC_min + 1 - 1) * np.log(len_) * np.log(p_)
								#print( (HDIC_check > HDICs[HDIC_min]) == 0)
								three_stage_keep = HDIC_path[(HDIC_check > HDICs[HDIC_min]) == 1]
								#print(three_stage_keep)
				else:
								three_stage_keep = np.array([0])
				#print(three_stage_keep)
				return three_stage_keep
				'''if len(jhat) > 4:
								return jhat[np.arange(3)]#three_stage_keep
				else:
								return jhat'''

def main():
    #Testing example, y = (b1 * x1) + ... + (b5 * x5) + e 
    n = 100
    p = 50
    q = 50
    coef_ = np.array([4.2, -3.7, 5.1, -5.7, 3.5])
    #coef_ = np.hstack([np.random.uniform(3,5, int(q/2)), np.random.uniform(-3,-5, int(q/2))])
    X = np.random.normal(0,1.5,n * p).reshape(n,p)
    y = np.random.normal(0,1,n) + np.dot(X[:,0:q], coef_.transpose())
    #print(coef_)
    
    print(OGA(X, y))
    
    #print(np.linalg.inv(np.dot(X.transpose(), X)).dot(X.transpose()).dot(y)) The same
    print(np.linalg.lstsq(X, y)[0])

if __name__ == '__main__':
    main()
    
