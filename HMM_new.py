import numpy as np
import matplotlib.pyplot as plt
import mfcc
import os
import scipy.stats as st
import numpy as np
from enum import Enum
sample_rate = 16000

class DetectedWord(Enum):
    ODESSA = 0
    TURN_ON_THE_LIGHTS = 1 
    TURN_OFF_THE_LIGHTS = 2 
    WHAT_TIME_IS_IT = 3 
    PLAY_MUSIC = 4
    STOP_MUSIC = 5 

class gmmhmm:
    def __init__(self, n_states):
        self.n_states = n_states
        self.random_state = np.random.RandomState(0)
        
        #Normalize random initial state
        self.prior = self._normalize(self.random_state.rand(self.n_states, 1))  # Initialize prior probability "Pi"
        self.A = self._stochasticize(self.random_state.rand(self.n_states, self.n_states)) # Initialize "A" probability
        
# Initializing mu, covariance and dimension matrix to calcuate "B" matrix       
        self.mu = None
        self.covs = None
        self.n_dims = None
           
    def _forward(self, B):# B is basically bj(o(t))
        log_likelihood = 0.
        T = B.shape[1]  # B.shape = (n_state x Total_time)
        alpha = np.zeros(B.shape) # n_states x Total_Time
        for t in range(T):
            if t == 0:
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])
         
            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, alpha
    
    def _backward(self, B):
        T = B.shape[1]
        beta = np.zeros(B.shape)
           
        beta[:, -1] = np.ones(B.shape[0])
            
        for t in range(T - 1)[::-1]:
            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
            beta[:, t] /= np.sum(beta[:, t])
        return beta
# The product alpha_t(i)*beta_t(i) gives the probability of whole observation with the condition that at time t it was
# in ith state. Alone alpha upto time T can't give this probability.
       
    
    def _state_likelihood(self, obs):
        obs = np.atleast_2d(obs)
        B = np.zeros((self.n_states, obs.shape[1]))
        for s in range(self.n_states): # Probability of getting the observation (o1,o2,...oT) when it is in state "s"
            #Needs scipy 0.14
            np.random.seed(self.random_state.randint(1))
            B[s, :] = st.multivariate_normal.pdf(
                obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T)
        return B
    
    def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)
    
    def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)
    
    def _em_init(self, obs):
        #Using this _em_init function allows for less required constructor args
        if self.n_dims is None:
            self.n_dims = obs.shape[0]
        if self.mu is None:
            subset = self.random_state.choice(np.arange(self.n_dims), size=self.n_states, replace=False)
            self.mu = obs[:, subset]
        if self.covs is None:
            self.covs = np.zeros((self.n_states, self.n_dims, self.n_dims))
            # Calculate the diagonal covariance matrix
            diag_cov = np.diag(np.diag(np.cov(obs)))
            expanded_diag_cov = np.expand_dims(diag_cov, axis=0)
            broadcastable_diag_cov = np.repeat(expanded_diag_cov, self.n_states, axis=0)
            self.covs += np.diag(np.diag(np.cov(obs)))[:, :, None]#broadcastable_diag_cov#
        return self
    
    def _em_step(self, obs): 
        obs = np.atleast_2d(obs)
        B = self._state_likelihood(obs)
        T = obs.shape[1]
        
        log_likelihood, alpha = self._forward(B)
        beta = self._backward(B)
        
        xi_sum = np.zeros((self.n_states, self.n_states))
        gamma = np.zeros((self.n_states, T)) # gamma is the probability that it is in ith state at time t, given the
        # observations and the model. gamma.shape = (n_state, T) 
        
        for t in range(T - 1):
            partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)
            xi_sum += self._normalize(partial_sum)
            partial_g = alpha[:, t] * beta[:, t]
            gamma[:, t] = self._normalize(partial_g)
              
        partial_g = alpha[:, -1] * beta[:, -1]
        gamma[:, -1] = self._normalize(partial_g)
        
        expected_prior = gamma[:, 0]
        expected_A = self._stochasticize(xi_sum)
        
        expected_mu = np.zeros((self.n_dims, self.n_states))
        expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
        
        gamma_state_sum = np.sum(gamma, axis=1)
        #Set zeros to 1 before dividing
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
        
        for s in range(self.n_states):
            gamma_obs = obs * gamma[s, :]
            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s], expected_mu[:, s].T)
            #Symmetrize
            partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)
        
        #Ensure positive semidefinite by adding diagonal loading
        expected_covs += .01 * np.eye(self.n_dims)[:, :, None]
        
        self.prior = expected_prior
        self.mu = expected_mu
        self.covs = expected_covs
        self.A = expected_A
        return log_likelihood
    
    def train(self, obs, n_iter=15):
        if len(obs.shape) == 2:
            for i in range(n_iter):
                self._em_init(obs)
                log_likelihood = self._em_step(obs)
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            for n in range(count):
                for i in range(n_iter):
                    self._em_init(obs[n, :, :])
                    log_likelihood = self._em_step(obs[n, :, :])
        return self
    
    def test(self, obs):
        if len(obs.shape) == 2:
            B = self._state_likelihood(obs)
            log_likelihood, _ = self._forward(B)
            return log_likelihood
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            out = np.zeros((count,))
            for n in range(count):
                B = self._state_likelihood(obs[n, :, :])
                log_likelihood, _ = self._forward(B)
                out[n] = log_likelihood
            return out

directory = "OdessaRecordings" 
train_file_paths = ['train 1.txt']#['train 1.txt','train 2.txt','train 3.txt','train 4.txt','train 5.txt']
val_file_paths = ['validation 1.txt']#['validation 1.txt','validation 2.txt','validation 3.txt','validation 4.txt','validation 5.txt']

def get_enum_val(ip_str):
    print(ip_str)

X_train_fold = []
y_train_fold = []
for train_file in train_file_paths:
    with open(f"Folds/{train_file}",'r') as file:
        train_set = file.readlines()
    X_train = []
    y_train = []#, y_test = all_labels[train_index], all_labels[test_index]

    for sample in train_set:
        sample = sample.replace('.wav\n','')
        features = mfcc.calculations_for_onerecording(False, directory=directory,filename=f"{sample}",sample_rate=sample_rate)
        X_train.append(features)
        label = sample.split('_')[0]
        y_train.append(str(label))
    X_train_fold.append(X_train)
    y_train_fold.append(y_train)

X_test_fold = []
y_test_fold = []
for test_file in val_file_paths:
    with open(f"Folds/{test_file}",'r') as file:
        test_set = file.readlines()
    X_test = []
    y_test = []
    for sample in test_set:
        sample = sample.replace('.wav\n','')
        features = mfcc.calculations_for_onerecording(False, directory=directory,filename=f"{sample}",sample_rate=sample_rate)
        X_test.append(features)
        label = sample.split('_')[0]
        y_test.append(str(label))
    y_test_fold.append(y_test)
    X_test_fold.append(X_test)

ys = set(y_test)
ms = [gmmhmm(7) for y in ys]

for i in range(5):
    print(i)
    X_train = np.array(X_train_fold[i])
    y_train = np.array(y_train_fold[i])
    X_test = np.array(X_test_fold[i])
    y_test = np.array(y_test_fold[i])

    _ = [model.train(X_train[y_train == y, :, :]) for model, y in zip(ms, ys)]
    ps1 = [model.test(X_test) for model in ms]
    res1 = np.vstack(ps1)
    predicted_label1 = np.argmax(res1, axis=0)
    dictionary = ['odessa', 'turn_on_the_lights', 'turn_off_the_lights', 'what_time_is_it', 'play_music', 'stop_music']
    spoken_word = []
    for i in predicted_label1:
        spoken_word.append(dictionary[i])
    print(spoken_word)
    missed = (predicted_label1 != y_test)
    print('Test accuracy: %.2f percent' % (100 * (1 - np.mean(missed))))

