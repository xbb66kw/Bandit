

class LinUCB(object):
				def __init__(self):
								self.last_action = -1
								
								self.Aa = {}
								
								self.AaI = {}
								
								self.ba = {} 
								
								self.theta = {}
								
								self.k = 0
								
								self.d = 136
									
								self.alpha = 0.2
								
								self.acc_reward = 0
								
								self.active_arms = []									
				def update(self, covariates, reward, mab=False):
								
								self.Aa[self.last_action] += np.outer(covariates, covariates)
								
								self.ba[self.last_action] += reward * np.array([covariates]).T
								
								self.AaI[self.last_action] = linalg.solve(self.Aa[self.last_action],
												np.identity(self.d))
								#print(self.AaI)
								#print(self.Aa)
								self.theta[self.last_action] = np.dot(self.AaI[self.last_action],
												self.ba[self.last_action])
								

				def update_arms(self, stuff):							
								if stuff['extra'] is not None:
												
												self.active_arms = np.append(self.active_arms, stuff['extra']).astype(int)
												
												for key in stuff['extra']:
																
																self.Aa[key] = np.identity(self.d)
																self.AaI[key] = np.identity(self.d)
																self.ba[key] = np.zeros((self.d, 1))
																self.theta[key] = np.zeros((self.d, 1))
								if stuff['delete'] is not None: 
												self.active_arms = np.setdiff1d(self.active_arms, stuff['delete']).astype(int)
								
								
								
								
								
								
								self.k = len(self.active_arms)

				def recommand(self, covariates, mab=False):
								arms = self.active_arms
								x = np.array([covariates]).T
								xT = x.T
								
								tmp_AaI = [self.AaI[arm] for arm in arms]
								
								tmp_theta = [self.theta[arm] for arm in arms]
								
							
								
								max_ = np.argmax(np.dot(xT, tmp_theta)[0,:,:]
												+ self.alpha * np.sqrt(np.dot(np.dot(xT, tmp_AaI)[0,:,:], x)))
								
								self.last_action = arms[max_]
								
								return arms[max_]
								