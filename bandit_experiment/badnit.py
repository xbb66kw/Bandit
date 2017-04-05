#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


class LinUCB(object):
				def __init__(self):
								self.last_action = 0
								
								self.Aa = {}
								
								self.AaI = {}
								
								self.ba = {} 
								
								self.theta = {}
								
								self.k = 0
								
								self.d = 0
								
								self.alpha = 0.5
								
								self.acc_reward = 0
				
				def set_bandits(self, arms = {
																								0:[1,0,0,0],
																								1:[0,1,0,0],
																								2:[0,0,1,0],
																								3:[0,0,0,1],
																				}):
								self.k = len(arms)
								self.d = len(arms[0])
								
								for key in arms:
												self.Aa[key] = np.identity(self.d)
												self.AaI[key] = np.identity(self.d)
												self.ba[key] = np.zeros((self.d, 1))
												self.theta[key] = np.zeros((self.d, 1))
												
				def update(self, covariates, reward):
								
								self.Aa[self.last_action] += np.outer(covariates, covariates)
								
								self.ba[self.last_action] += reward * np.array([covariates]).T
								
								self.AaI[self.last_action] = linalg.solve(self.Aa[self.last_action],
												np.identity(self.d))
								#print(self.AaI)
								#print(self.Aa)
								self.theta[self.last_action] = np.dot(self.AaI[self.last_action],
												self.ba[self.last_action])
				

				def recommand(self, covariates, arms = {
																								0:[1,0,0,0],
																								1:[0,1,0,0],
																								2:[0,0,1,0],
																								3:[0,0,0,1],
																				}):
								x = np.array([covariates]).T
								xT = x.T
								
								tmp_AaI = np.array([self.AaI[arm] for arm in arms])
								
								tmp_theta = np.array([self.theta[arm] for arm in arms])
								
								#shape = (1, k, 1)
								
								#print(tmp_theta, tmp_AaI, xT)
								#print(np.dot(xT, tmp_theta))
								#print(np.dot(np.dot(xT, tmp_AaI)[0,:,:], x))
								#print('-----')
								#self.last_action = arms[np.argmax(np.dot(xT, tmp_theta)[0,:,:].T + 
								#				self.alpha * np.dot(np.dot(xT, tmp_AaI)[0,:,:], x))]
								self.last_action = np.argmax(np.dot(xT, tmp_theta)[0,:,:]
												+ self.alpha * np.sqrt(np.dot(np.dot(xT, tmp_AaI)[0,:,:], x)))
								#print(np.dot(xT, tmp_theta)[0,:,:].T + 
								#				self.alpha * np.dot(np.dot(xT, tmp_AaI)[0,:,:], x))
								return self.last_action

class Environment(object):
								
				
				def run_synthetic(self, agents, bandits, user, timestamp = 2000):
								self.reward_curves = np.zeros((timestamp, len(agents)))
								
								for t in range(timestamp):
												covariates = user.come()
												
												for i, agent in enumerate(agents):# agents can be [...]
																#print(agents, agent, i)
																agent.last_action = agent.recommand(covariates, bandits.arms)
																
																reward = bandits.pull(covariates, agent.last_action)
																
																agent.update(covariates, reward)
																
																agent.acc_reward += reward
																
																#print(agent.theta)
																self.reward_curves[t, i] = agent.acc_reward / (t + 1)
																
				def plot(self):

								for j in range(len(self.reward_curves[0,:])):
												plt.plot(self.reward_curves[:,j])
								plt.show()
								
class Bandit(object):
				
				def __init__(self, arms = {
																								0:[1,0,0,0],
																								1:[0,1,0,0],
																								2:[0,0,1,0],
																								3:[0,0,0,1],
																				}):
				
								self.arms = arms
								
				
				def pull(self, covariates, arm_index):
								#print(covariates, arm_index)
								#print(np.dot(self.arms[arm_index], covariates))
								return np.random.binomial(1,np.dot(self.arms[arm_index], covariates),1)
				
				@property
				def get_arms(self):
								return self.arms

class User(object):
				def __init__(self, d=4):
								self.d = d
								
				def come(self):
								return np.random.uniform(0, 1, self.d)
								
def main():
				n = 10# There will be some never-played arms. (Depend on alpha)
				arms = {}
				for key in range(n):
								arms[key] = np.identity(n)[key,:]
				
				
				A = LinUCB()
				
				bandits = Bandit(arms)
				A.set_bandits(bandits.get_arms)
				
				user = User(n)
				E = Environment()
				E.run_synthetic([A], bandits, user)
				
				E.plot()
				#print(A.recommand([4,5,6,7], [1,2,3]))
				pass



if __name__ == '__main__':
    main()