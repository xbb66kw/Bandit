#!/usr/bin/env python

import re
import gzip
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg




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
    jhat = np.array([]).astype(int)
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
    
    jhat = np.append(jhat, np.argmax(se_))
    x_hat[:,0] = X[:,jhat[0]]
    u = y - x_hat[:,0] * np.dot(x_hat[:,0], y)
    
    se_catcher[0] = np.dot(y-u, y-u)
    
    
    #The other (Kn - 1) steps
    for k in np.arange(Kn)[np.arange(Kn) != 0]:
        for j in range(p_):
            se_[j] = np.abs(np.dot(u, X[:,j]))
        if sum(se_) <= 1e-10:
            if jhat.size:
                return jhat[0:k]
            else:
                return [0]
        se_[jhat] = 0 #Just to make sure
        
        jhat = np.append(jhat, np.argmax(se_))
        #Orthogonalizing
        x_hat[:,k] = X[:,jhat[k]] - np.dot(x_hat[:,range(k)], np.dot(x_hat[:,range(k)].transpose(), X[:,jhat[k]]))
        'Issue!!'
        
        x_hat[:,k] = x_hat[:,k] / np.sqrt(np.dot(x_hat[:,k], x_hat[:,k])) 
        u = u - x_hat[:,k] * np.dot(x_hat[:,k], u)
    
        se_catcher[k] = np.dot(u, u)
    
    HDICs = np.array([len_ * np.log(se_catcher / len_) + (np.arange(Kn) + 1) * np.log(len_) * np.log(p_)])[0]
    
    HDIC_min = HDICs.argmin()
    HDIC_path = jhat[0:(HDIC_min+1):1]
    HDIC_check = np.zeros(HDIC_min + 1)
    '''
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
    '''
    return jhat    
    return three_stage_keep
    '''if len(jhat) > 4:
        return jhat[np.arange(3)]#three_stage_keep
    else:
        return jhat'''








class LinUCB(object):
    def __init__(self):
        self.last_action = -1
        
        self.Aa = {}
        
        self.AaI = {}
        
        self.ba = {} 
        
        self.theta = {}
        
        self.k = 0
        
        self.d = 136
         
        self.alpha = 0.3
        
        self.acc_reward = 0
        
        self.active_arms = []         
    def update(self, covariates, reward):
        
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

    def recommand(self, covariates):
        arms = self.active_arms
        x = np.array([covariates]).T
        xT = x.T
        
        tmp_AaI = [self.AaI[arm] for arm in arms]
        
        tmp_theta = [self.theta[arm] for arm in arms]
        
       
        
        max_ = np.argmax(np.dot(xT, tmp_theta)[0,:,:]
            + self.alpha * np.sqrt(np.dot(np.dot(xT, tmp_AaI)[0,:,:], x)))
        
        self.last_action = arms[max_]
        
        return arms[max_]
        
        
        
        
        
        
        

class modified_LinUCB(object):
    def __init__(self):
        self.last_action = -1 
        
        self.Aa = {}
        
        self.AaI = {}
        
        self.X = {}
        self.y = {}
        
        self.ba = {} 
        
        self.theta = {}
        
        self.k = 0
        
        self.d = 136
         
        self.alpha = 0.3
        print(self.alpha, 'alpha', '206')
        self.acc_reward = 0
        'keys of arms'
        self.active_arms = []
        
        self.warm_up = {}
    
    def update(self, covariates, reward):
        self.Aa[self.last_action] += np.outer(covariates, covariates)
        
        self.ba[self.last_action] += reward * np.array([covariates]).T
        
        self.AaI[self.last_action] = linalg.solve(self.Aa[self.last_action],
            np.identity(self.d))        
        
        if not self.X[self.last_action].size:
            self.X[self.last_action] = np.array(covariates)[None,:]
            self.y[self.last_action] = np.array(reward)
            self.theta[self.last_action] = np.dot(self.AaI[self.last_action],
                self.ba[self.last_action])
            
        else:    
            
            if (self.warm_up[self.last_action] <= 3 * self.d):
                
                self.X[self.last_action] = np.append(self.X[self.last_action], covariates[None,:], axis=0)
                self.y[self.last_action] = np.append(self.y[self.last_action], reward)
                
                
                if 3 < self.warm_up[self.last_action] and not all(self.y[self.last_action] == self.y[self.last_action][0]):
                    
                    X = self.X[self.last_action]
                    length_ = len(self.X[self.last_action][:,0])
                    indexes_ = [index for index, value in enumerate(np.sum(self.X[self.last_action], axis=0)) if value == length_]
                    indexes_ = np.append(indexes_, [index for index, value in enumerate(np.sum(self.X[self.last_action], axis=0)) if value == 0])

                    indexes_ = np.sort(np.unique(indexes_)).astype(int)
                    
                    rest_indexes_ = np.setdiff1d(range(136), indexes_)
                    trivial_indexes = []
                    for i in rest_indexes_:
                        for j in range(i+1, 136):
                            
                            if all(X[:,i] == X[:,j]):
                                trivial_indexes = np.append(trivial_indexes, j)
                            
                    
                    trivial_indexes = np.unique(trivial_indexes).astype(int)
                   
                    nontrivial_indexes = np.setdiff1d(range(136), indexes_)
                    nontrivial_indexes = np.setdiff1d(nontrivial_indexes, trivial_indexes)
                    
                    
                    mid_indexes_X = OGA(X[:,nontrivial_indexes], self.y[self.last_action], Kn=np.floor(np.sqrt(len(self.y[self.last_action]))))
                    
                    mid_X = X[:,nontrivial_indexes][:,mid_indexes_X]
                    'Add the indexes back!'
                   
                    theta = np.dot(linalg.solve(np.dot(mid_X.T, mid_X), np.identity(len(mid_indexes_X))), np.dot(mid_X.T, self.y[self.last_action]))[:,None]
                   
                    self.theta[self.last_action] = np.zeros(self.d)[:,None]
        
                    self.theta[self.last_action][mid_indexes_X,:] = theta
                else:
                    
                    self.theta[self.last_action] = np.dot(self.AaI[self.last_action],
                        self.ba[self.last_action])
                   
        
        
        
            
            else:
                self.theta[self.last_action] = np.dot(self.AaI[self.last_action],
                    self.ba[self.last_action])
        
        self.warm_up[self.last_action] += 1
        
    def update_arms(self, stuff):       
        if stuff['extra'] is not None:
            
            self.active_arms = np.append(self.active_arms, stuff['extra']).astype(int)
            
            for key in stuff['extra']:
                self.X[key] = np.array([])
                self.y[key] = np.array([])
                self.warm_up[key] = 0
            
                self.Aa[key] = np.identity(self.d)
                self.AaI[key] = np.identity(self.d)
                self.ba[key] = np.zeros((self.d, 1))
                self.theta[key] = np.zeros((self.d, 1))
        if stuff['delete'] is not None: 
            self.active_arms = np.setdiff1d(self.active_arms, stuff['delete']).astype(int)
        
        
        
        
        
        
        self.k = len(self.active_arms)
    
    def recommand(self, covariates):
        arms = self.active_arms
        x = np.array([covariates]).T
        xT = x.T
        
        tmp_AaI = [self.AaI[arm] for arm in arms]
        
        tmp_theta = [self.theta[arm] for arm in arms]
        
        
        
        max_ = np.argmax(np.dot(xT, tmp_theta)[0,:,:]
            + self.alpha * np.sqrt(np.dot(np.dot(xT, tmp_AaI)[0,:,:], x)))
        
        self.last_action = arms[max_]
        
        return arms[max_]
    
class Environment(object):
    
    
    def run_synthetic(self, agents, bandits, user, timestamp = 100):
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
     
    def run(self, agents, data, timestamp = 2000):
        self.reward_curves = np.zeros((timestamp, len(agents)))
        self.timer = np.zeros(len(agents)).astype(int)
        
        times = 0
        while np.min(self.timer) < timestamp:
            #Also in this step, arms will be refreshed
            stuff = data.come()
            covariates = stuff['covariates']
            
  
            for i, agent in enumerate(agents):# agents can be [...]
                if self.timer[i] < timestamp:
                    times += 1
                    agent.update_arms(stuff)
                    
                    agent.last_action = agent.recommand(covariates)
                    
                    if agent.last_action == stuff['article']:
                        reward = stuff['click']
                        agent.update(covariates, reward)
                        agent.acc_reward += reward
                        self.reward_curves[self.timer[i], i] = agent.acc_reward / (self.timer[i] + 1)
                        
                        self.timer[i] += 1

        print('final', times)
    def plot(self):
        label_list = ['ogaucb', 'ucb']
        collect = {}
        for j in range(len(self.reward_curves[0,:])):
            
            collect[j], = plt.plot(self.reward_curves[:,j], label=label_list[j])
            
        plt.legend(handles=[collect[0], collect[1]])

        plt.show()
        

class DataReader(object):
    def __init__(self):
        self.fin = gzip.open('/Users/apple/Desktop/bandit_experiment/Webscope_R6B/ydata-fp-td-clicks-v2_0.20111002.gz','r')
        self.articles_old = []
        self.articles_new = []
        self.line = None
        
        
    def come(self):
        extra = None
        delete = None
        click = 0
        
        self.line = self.fin.readline()
        
        

        cha = self.line.decode('utf-8')
        matches = re.search(r"id-(\d+)\s(\d).+user\s([\s\d]+)(.+)", cha)
        
        article = int(matches.group(1))
        click = int(matches.group(2))
        covariates = np.zeros(136)
        covariates[[int(elem) - 1 for elem in matches.group(3).split(' ') if elem != '']] = 1 
        
        
        finder = re.findall(r'\|id-(\d+)', matches.group(4))
    
        self.articles_new = [int(result) for result in finder]
        
        
        
        if self.articles_new != self.articles_old:
            extra = np.setdiff1d(self.articles_new, self.articles_old)
            delete = np.setdiff1d(self.articles_old, self.articles_new)
            
            
        self.articles_old = self.articles_new
        
        
        return {'covariates':covariates, 'article':article, 'click':click, 'extra':extra, 'delete':delete}

    def end(self):
        self.fin.close()
    
    def update_arms(self):
        pass
    
def main():
    '''
    n = 5# There will be some never-played arms. (Depend on alpha)
    arms = {}
    for key in range(n):
        arms[key] = np.identity(n)[key,:]
    
    
    A = modified_LinUCB()
    
    bandits = Bandit(arms)
    A.set_bandits(bandits.get_arms)
    
    user = User(n)
    E = Environment()
    E.run_synthetic([A], bandits, user)
    E.plot()
    '''
    A = modified_LinUCB()
    B = LinUCB()
    E = Environment()

    data_reader = DataReader()
    E.run([A, B], data_reader)
    
    E.plot()



if __name__ == '__main__':
    main()
