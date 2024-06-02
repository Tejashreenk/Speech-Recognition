import numpy as np
import pathlib

import librosa
import os
import warnings
import copy
# import tqdm
# from mfcc import _WindowingMFCC
# from delta import compute_delta
from scipy.stats import multivariate_normal
import pdb
from sklearn.preprocessing import StandardScaler


def multivariate_normal_logpdf(x, mean, covar):
    """
    This function calculates the log of the probability density function of a multivariate normal distribution.
    """
    x_diff = x - mean
    # breakpoint()
    inv_covar = np.linalg.inv(covar + 1e-10)
    log_det_covar = np.linalg.slogdet(covar)[1]
    try:
        quadratic_term = -0.5 * np.dot(x_diff.T, np.dot(inv_covar, x_diff))
    except ValueError :
        breakpoint()
    normalization_term = -0.5 * (len(x) * np.log(2 * np.pi) + log_det_covar)
    # breakpoint()
    return quadratic_term + normalization_term







class HMMGaus():
    def __init__(self, n_states, A, mu, sigma, pi):
        self.n_states = n_states
        self.A = A
        self.mu = mu
        self.sigma = sigma
        self.pi = pi
    
    def Forward(self, observation):
        '''
        alphas should be in log domain to avoid underflow and overflow
        assume that every other thing is in log domain
        '''
        #initialize empty or zero alphas 
        #observations are of shape (T, 26)
        alpha = np.zeros((self.n_states, observation.shape[0]))

        #calculating the emission_probabilities
        b = np.zeros((self.n_states,observation.shape[0] ))

        for states in range(self.n_states):
            for times in range(observation.shape[0]):
                #print("shape of mu passed is ",self.mu[states].shape)
                #print("shape of sigmas passed is ", self.sigma[states].shape)
                b[states, times] = multivariate_normal_logpdf(x = observation[times, :], mean= self.mu[states], covar= self.sigma[states])
        self.b = np.asarray(b)
    
        #calculating the alpha_t=1 (i)
        alpha[:, 0] = self.pi + self.b[:, 0]  #pi_{state = i} * b_{state = i} (O_{1})

        #calculating the alpha_t=2:T(i)
        self.norm_sum = np.zeros((1, alpha.shape[1])) # (1, T)
        for times in range(1, observation.shape[0]):
            alpha_state_sum = 0
            for state in range(self.n_states): #i--->j
                multerm = alpha[:, times-1] * self.A[:, state]
                sum_term = 0
                for iters in range(alpha[ :, times-1].shape[0]):
                    sum_term = np.logaddexp(sum_term, multerm[iters])
                alpha[state, times] = sum_term + self.b[states, times]
                alpha_state_sum = np.logaddexp(alpha_state_sum, alpha[state, times])
            self.norm_sum[0, times] = np.logaddexp(self.norm_sum[0, times], alpha_state_sum)
        self.alpha = alpha - self.norm_sum
        # return self.alpha
    
    def Backward(self, observation):
        '''
        betas should be in log domain to avoid underflow and overflow
        assume that every other thing is in log domain
        '''

        #initializing empty beta array
        #observations are of shape (T, 26)
        beta = np.zeros((self.n_states, observation.shape[0])) 

        #calculating the initial condition of beta
        beta[:, -1] = np.zeros(beta[:, -1].shape[0]) #beta_{t=T}(i) = 1

        for time in range(observation.shape[0]-2, 0, -1):
            for state in range(self.n_states):
                product_term = (self.A[state, :] + self.b[:, time+1] + beta[:, time+1])
                sum_term = 0
                for iter in range(product_term.shape[0]):
                    sum_term = np.logaddexp(sum_term, product_term[iter])
            beta[state, time] = sum_term - self.norm_sum[0, time]
        self.beta = beta
        # return self.beta
    
    def GammaandXi(self, observation):

        #initialize empty gamma and xi arrays
        gamma = np.zeros((self.n_states, observation.shape[0]))
        xi = np.zeros((self.n_states, self.n_states, observation.shape[0]))

        for time in range(observation.shape[0]-1):
            for state_i in range(self.n_states):
                for state_j in range(self.n_states):
                    xi[state_i, state_j, time] = self.alpha[state_i, time] + self.A[state_i, state_j] + self.b[state_j, time+1] + self.beta[state_j, time+1]
                    gamma[state_i, time] = np.logaddexp(gamma[state_i, time], xi[state_i, state_j, time])
                
            
        gamma[:, -1] = self.alpha[:, -1]

        self.gamma= gamma
        self.xi = xi
        # return gamma, xi


    def EM(self, observations, thresh=0.01, maxIter=200):
        log_likelihood = []
        while len(log_likelihood)<maxIter:
            print("Iter number: %d"%(len(log_likelihood)))
            for observation in observations:
                #E step 
                #get forward, backward, gamma and xi
                self.Forward(observation=observation)
                self.Backward(observation=observation)
                self.GammaandXi(observation = observation)

                #M step
                #updating pi
                self.pi = self.gamma[:, 0]

                #re-estimating A
                for state_i in range(self.n_states):
                    denom = 0
                    for time in range(observation.shape[0]-1):
                        denom = np.logaddexp(denom, self.gamma[state_i, time])
                    
                    for state_j in range(self.n_states):
                        numer = 0
                        for time in range(observation.shape[0]-1):
                            numer = np.logaddexp(numer, self.xi[state_i,state_j, time])
                        self.A[state_i, state_j] = numer - denom
                
                #re-estimating mean and covariance for b
                #gamma shape : (n_states, T)
                gamma_sum = np.zeros((self.n_states))
                gamma_exp = np.exp(self.gamma)
                # for times in range(observation.shape[0]):
                gamma_sum = np.sum(gamma_exp, axis=1)

                
                for state_i in range(self.n_states):
                    self.mu[state_i, :] = np.zeros((1, 13))

                    for time in range(observation.shape[0]):
                        self.mu[state_i, :] += self.gamma[state_i, time] * observation[time, :]
                    self.mu[state_i, :] = self.mu[state_i, :]/gamma_sum[state_i]
                    self.sigma[state_i] = np.zeros((observation.shape[1], observation.shape[1]))

                    for time in range(observation.shape[0]):
                        obs_diff = observation[time, :] - self.mu[state_i]


                        self.sigma[state_i] += gamma_exp[state_i, time] * np.outer(obs_diff, obs_diff)

                    self.sigma[state_i] /=gamma_sum[state_i]
                
                    sigma_tied = self.sigma[0]
                    for i in range(1, self.n_states):
                        sigma_tied += self.sigma[i]
                    sigma_tied/=self.n_states
                    sigma_tied *=  0.02 * np.eye(13)
                    self.sigma = np.array([sigma_tied for _ in range(self.n_states)])

                log_like_i = 0
                for time in range(observation.shape[0]):
                    log_like_i = -self.norm_sum[0, time]
                log_likelihood.append(log_like_i)

            if len(log_likelihood)>2 and abs(log_like_i-log_likelihood[-2])<thresh:
                break
        return log_likelihood

# Function to extract MFCC features
def extract_mfcc(audio_path, sample_rate =16000, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_scaled = StandardScaler().fit_transform(mfcc_features.T)
    return mfcc_scaled

if __name__=='__main__':
    audio_files = []
    labels = []
    sample_rate = 22050
    directory = "OdessaRecordings"
    # filenames = ["play_music","stop_music","turn_off_the_lights","turn_on_the_lights","what_time_is_it","odessa"]
    filenames = ["odessa"]
    for filename in filenames:
        for i in range(29):
            audio_files.append(f"{directory}/{filename}_{i+1}.wav")
            labels.append(filename)

    # Extract MFCC features using librosa
    mfcc_features = []
    for audios in audio_files:
        y, sr = librosa.load(audios)
        # mfcc_features.append(compute_delta()._forward(y, sr).T)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T 
        # breakpoint()
        mfcc_features.append(mfccs)

    # breakpoint()
    mfcc_features = np.asarray(mfcc_features) #((samples, T, dim))
    # mfcc_features = np.expand_dims(mfcc_features[0], 0)
    
    # Initialize the HMM parameters
    n_states = 3 # Number of hidden states
    transmission_prob = np.log(np.full((n_states, n_states), 1 / (n_states)))
    pi = np.log(np.ones(n_states)/n_states)
    # flattened_features = mfcc_features.reshape(-1, mfcc_features.shape[-1])

    # Calculate global mean for each MFCC feature

    # global_mean = np.mean(np.concatenate(mfcc_features, axis=0), axis=0)
    # global_variance = np.var(np.concatenate(mfcc_features, axis=0), axis=0)
    # breakpoint()
    global_mean = np.mean(mfcc_features.reshape(-1, 13),axis=0)

    # Calculate global variance for each MFCC feature
    global_variance = np.var(mfcc_features.reshape(-1, 13),axis=0)
    # Print global mean and variance
    print("Global Mean:")
    print(global_mean)
    print("\nGlobal Variance:")
    print(global_variance)
    # breakpoint()
    emission_means = global_mean + np.asarray([np.random.randn(13) * 0.125 * global_variance for _ in range(n_states)])
    # breakpoint()
    # Initialize covariance matrices
    emission_covars = np.asarray([np.diag(global_variance) for _ in range(n_states)])
    # breakpoint()

        # return xi_values
    
    '''self.n_states = n_states
    self.A = A
    self.mu = mu
    self.sigma = sigma
    self.pi = pi'''
    print("starting hmm computation...")
    hmm = HMMGaus(n_states=n_states, A=transmission_prob, mu=emission_means, sigma=emission_covars, pi=pi)
    log_likelihood = hmm.EM(observations=mfcc_features)
    print(log_likelihood)
    breakpoint()

'''
if __name__=='__main__':
    audio_files = []
    labels = []
    sample_rate = 22050
    directory = "cleaned_audios_3005"
    # filenames = ["play_music","stop_music","turn_off_the_lights","turn_on_the_lights","what_time_is_it","odessa"]
    filenames = ["odessa"]
    for filename in filenames:
        for i in range(29):
            audio_files.append(f"{directory}/{filename}_{i+1}.wav")
            labels.append(filename)


    # Extract MFCC features using librosa
    mfcc_features = []
    for audios in audio_files:
        # y, sr = librosa.load(audios)
        # mfcc_features.append(compute_delta()._forward(y, sr).T)
        mfccs = extract_mfcc(audios, sample_rate)#librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26).T 
        # breakpoint()
        mfcc_features.append(mfccs)

    # breakpoint()
    mfcc_features = np.vstack(mfcc_features) #((samples, T, dim))
    # mfcc_features = np.expand_dims(mfcc_features[0], 0)
    
    # Initialize the HMM parameters
    n_states = 3 # Number of hidden states
    transmission_prob = np.log(np.full((n_states, n_states), 1 / (n_states)))
    pi = np.log(np.ones(n_states)/n_states)
    # flattened_features = mfcc_features.reshape(-1, mfcc_features.shape[-1])

    # Calculate global mean for each MFCC feature

    # global_mean = np.mean(np.concatenate(mfcc_features, axis=0), axis=0)
    # global_variance = np.var(np.concatenate(mfcc_features, axis=0), axis=0)
    # breakpoint()
    global_mean = np.mean(mfcc_features.reshape(-1, 26),axis=0)

    # Calculate global variance for each MFCC feature
    global_variance = np.var(mfcc_features.reshape(-1, 26),axis=0)
    # Print global mean and variance
    print("Global Mean:")
    print(global_mean)
    print("\nGlobal Variance:")
    print(global_variance)
    # breakpoint()
    emission_means = global_mean + np.asarray([np.random.randn(26) * 0.125 * global_variance for _ in range(n_states)])
    # breakpoint()
    # Initialize covariance matrices
    emission_covars = np.asarray([np.diag(global_variance) for _ in range(n_states)])
    # breakpoint()

        # return xi_values

    # self.n_states = n_states
    # self.A = A
    # self.mu = mu
    # self.sigma = sigma
    # self.pi = pi
    print("starting hmm computation...")
    hmm = HMMGaus(n_states=n_states, A=transmission_prob, mu=emission_means, sigma=emission_covars, pi=pi)
    log_likelihood = hmm.EM(observations=mfcc_features)
    breakpoint()
'''





            


            





        














    