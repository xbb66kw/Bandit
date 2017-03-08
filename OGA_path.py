#OGA path implemented in python


import numpy as np


def OHT(X, y, Kn=0):
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
								print(se_[range(10)])
								#se_[jhat[jhat != -1]] = 0 #Just to make sure
								jhat[k] = np.argmax(se_)
								#Orthogonalizing
								x_hat[:,k] = X[:,jhat[k]] - np.dot(x_hat[:,range(k)], np.dot(x_hat[:,range(k)].transpose(), X[:,jhat[k]]))
								x_hat[:,k] = x_hat[:,k] / np.sqrt(np.dot(x_hat[:,k], x_hat[:,k])) 
								u = u - x_hat[:,k] * np.dot(x_hat[:,k], u)
								
								se_catcher[k] = np.dot(y-u, y-u)
				
				#return the OGA Kn path
				return jhat

def main():
				#Testing example, y = (b1 * x1) + ... + (b5 * x5) + e 
				n = 200
				p = 2000
				q = 5
				coef_ = np.array([4.2, -3.7, 5.1, -5.7, 3.5])
				X = np.random.normal(0,1.5,n * p).reshape(n,p)
				y = np.random.normal(0,1,n) + np.dot(X[:,0:q], coef_.transpose())
				print(OHT(X, y))

if __name__ == '__main__':
    main()
    
