import numpy as np
from scipy.stats import multivariate_normal
import mfcc
import os
import log_calc
import librosa
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


class HMM:
    def __init__(self, num_states, num_features):
        self.num_states = num_states
        self.num_features = num_features
        self.transitions = np.zeros((num_states, num_states))
        self.means = np.zeros((num_states, num_features))
        self.covariances = np.array([np.eye(num_features) for _ in range(num_states)])
        self.initial_probs = np.zeros(num_states)
    
    def initialize_parameters(self):
        np.fill_diagonal(self.transitions, 0.5)
        for i in range(self.num_states - 1):
            self.transitions[i, i + 1] = 0.5
        self.transitions[-1, -1] = 1.0
        self.initial_probs.fill(0.1)  # avoid zero probabilities
        self.initial_probs[0] = 0.9
        self.means = np.random.rand(self.num_states, self.num_features)
        self.covariances = np.array([np.eye(self.num_features) for _ in range(self.num_states)])

    def logsumexp(self, arr):
        sum = 0
        for elem in arr:
            sum += log_calc.log_add(elem,sum)
        return sum

    def log_forward(self, observations):
        T = len(observations)
        log_alpha = np.full((T, self.num_states), -np.inf)
        log_alpha[0, :] = np.log(self.initial_probs) + np.log(multivariate_normal.pdf(observations[0], self.means[0], self.covariances[0]))

        for t in range(1, T):
            for j in range(self.num_states):
                log_probs = log_calc.log_add(log_alpha[t-1],np.log(self.transitions[:, j]))
                log_alpha[t, j] = np.log(multivariate_normal.pdf(observations[t], self.means[j], self.covariances[j])) + self.logsumexp(log_probs)
        
        return log_alpha

    def log_backward(self, observations):
        T = len(observations)
        log_beta = np.zeros((T, self.num_states))
        log_beta[T-1, :] = 0  # log(1)

        for t in range(T-2, -1, -1):
            for i in range(self.num_states):
                log_pdf_values = np.log([multivariate_normal.pdf(observations[t+1], self.means[j], self.covariances[j]) for j in range(self.num_states)])
                log_probs = log_beta[t+1] + log_pdf_values + np.log(self.transitions[i, :])
                log_beta[t, i] = self.logsumexp(log_probs)

        return log_beta
 
    def log_e_step(self, observations):
        log_alpha = self.log_forward(observations)
        log_beta = self.log_backward(observations)
        T = len(observations)
        
        # Compute gamma using log_alpha and log_beta
        gamma = np.exp(log_alpha + log_beta - self.logsumexp(log_alpha[-1]))
        
        # Compute xi across all time steps and states
        xi = np.zeros((T-1, self.num_states, self.num_states))
        for t in range(T-1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    log_xi = (log_alpha[t, i] + np.log(self.transitions[i, j]) +
                            np.log(multivariate_normal.pdf(observations[t+1], self.means[j], self.covariances[j])) +
                            log_beta[t+1, j])
                    xi[t, i, j] = np.exp(log_xi - self.logsumexp(log_xi))
        
        return gamma, xi, self.logsumexp(log_alpha[-1])

    def log_m_step(self, observations, gamma, xi):
        T = len(observations)
        
        # Update initial probabilities
        self.initial_probs = np.exp(gamma[0] - self.logsumexp(gamma[0]))
        
        # Update transition probabilities
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.transitions[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
        
        # Update means and covariances
        for i in range(self.num_states):
            weighted_sum = np.zeros(self.num_features)
            for t in range(T):
                weighted_sum += gamma[t, i] * observations[t]
            self.means[i] = weighted_sum / np.sum(gamma[:, i])
            
            weighted_covar = np.zeros((self.num_features, self.num_features))
            for t in range(T):
                diff = observations[t] - self.means[i]
                weighted_covar += gamma[t, i] * np.outer(diff, diff)
            self.covariances[i] = weighted_covar / np.sum(gamma[:, i])

    def log_likelihood(self, log_alpha):
        return self.logsumexp(log_alpha[-1])

    def fit(self, observations, max_iterations=100, tolerance=1e-6):
        self.initialize_parameters()
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            gamma, xi, log_likelihood = self.log_e_step(observations)
            self.log_m_step(observations, gamma, xi)
            
            # Compute change in log likelihood for convergence check
            change = log_likelihood - prev_log_likelihood
            if abs(change) < tolerance:
                print("Convergence reached.")
                break
            prev_log_likelihood = log_likelihood
            print(f"Iteration {iteration}, Log Likelihood: {log_likelihood}")

        return self

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
                    self.mu[state_i, :] = np.zeros((1, 26))

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
                    sigma_tied *=  0.02 * np.eye(26)
                    self.sigma = np.array([sigma_tied for _ in range(self.n_states)])

                log_like_i = 0
                for time in range(observation.shape[0]):
                    log_like_i = -self.norm_sum[0, time]
                log_likelihood.append(log_like_i)

            if len(log_likelihood)>2 and abs(log_like_i-log_likelihood[-2])<thresh:
                break
        return log_likelihood

# Function to extract MFCC features
def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = StandardScaler().fit_transform(mfcc_features.T)
    return mfcc_scaled

# Example integration with a dataset
num_states = 3
num_features = 26
hmm = HMM(num_states, num_features)
hmm.initialize_parameters()

# List all files in the directory
directory = 'cleaned_audios_3005'
files_in_directory = os.listdir(directory)
sample_rate = 22050
features_arr = []
filename = "odessa"
for i in range(29):
    features = extract_mfcc(f"{directory}/{filename}_{i+1}.wav")#mfcc.calculations_for_onerecording(False, directory=directory,filename=f"{filename}_{i+1}",sample_rate=sample_rate)
    features_arr.append(features)

features_arr = np.array(features_arr)
transmission_prob = np.log(np.full((num_states, num_states), 1 / (num_states)))
pi = np.log(np.ones(num_states)/num_states)
global_mean = np.mean(features_arr.reshape(-1, 13),axis=0)

# Calculate global variance for each MFCC feature
global_variance = np.var(features_arr.reshape(-1, 13),axis=0)

emission_means = global_mean + np.asarray([np.random.randn(26) * 0.125 * global_variance for _ in range(n_states)])
# breakpoint()
# Initialize covariance matrices
emission_covars = np.asarray([np.diag(global_variance) for _ in range(num_states)])

hmm = HMMGaus(n_states=num_states, A=transmission_prob, mu=emission_means, sigma=emission_covars, pi=pi)


observations = np.array(features_arr[0])
log_likelihood = hmm.EM(observations=observations)

# print(observations[:,0].shape)
# Running one iteration of EM
gamma, xi, likelihood = hmm.log_e_step(observations[:,0])
hmm.log_m_step(observations, gamma, xi)
print("Log likelihood:", hmm.log_likelihood(hmm.log_forward(observations)))
