import numpy as np
from scipy.stats import multivariate_normal
import mfcc
import os
import log_calc
import librosa
from sklearn.preprocessing import StandardScaler

class HMM:
    def __init__(self, num_states, num_features):
        self.num_states = num_states
        self.num_features = num_features
        self.transitions = np.zeros((num_states, num_states))
        self.means = np.zeros((num_states, num_features))
        self.covariances = np.array([np.eye(num_features) for _ in range(num_states)])
        self.initial_probs = np.zeros(num_states)
        
    def initialize_parameters(self):
        # Transition probabilities with constraints for a left-to-right model
        for i in range(self.num_states - 1):
            self.transitions[i, i] = 0.5
            self.transitions[i, i + 1] = 0.5
        self.transitions[-1, -1] = 1.0
        print(self.transitions)
        self.initial_probs[0] = 1.0  # Start in the first state
        
        # Initialize means and covariances with dummy values 
        self.means = np.random.rand(self.num_states, self.num_features)
        self.covariances = np.array([np.eye(self.num_features) for _ in range(self.num_states)])
    
    def multivariate_normal_logpdf(self, x, mean, covar):
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


    def forward(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.num_states))
        alpha[0, :] = self.initial_probs * multivariate_normal.logpdf(observations[0], self.means[0], self.covariances[0])
        # alpha[0, :] = self.initial_probs * self.multivariate_normal_logpdf(observations[0], self.means[0], self.covariances[0])
        
        for t in range(1, T):
            for j in range(self.num_states):
                alpha[t, j] = multivariate_normal.logpdf(observations[t], self.means[j], self.covariances[j]) * np.dot(alpha[t-1, :], self.transitions[:, j])
                # alpha[t, j] = self.multivariate_normal_logpdf(observations[t], self.means[j], self.covariances[j]) * np.dot(alpha[t-1, :], self.transitions[:, j])
        
        return alpha
    
    def forward_log(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.num_states))
        alpha[0, :] = np.log(self.initial_probs) + multivariate_normal.logpdf(observations[0], self.means[0], self.covariances[0])
        # alpha[0, :] = self.initial_probs * self.multivariate_normal_logpdf(observations[0], self.means[0], self.covariances[0])
        
        for t in range(1, T):
            for j in range(self.num_states):
                log_sum = np.logaddexp.reduce(alpha[t-1, :] + np.log(self.transitions[:, j]))
                alpha[t, j] = multivariate_normal.logpdf(observations[t], self.means[j], self.covariances[j]) + log_sum
                # alpha[t, j] = self.multivariate_normal_logpdf(observations[t], self.means[j], self.covariances[j]) * np.dot(alpha[t-1, :], self.transitions[:, j])
        
        return alpha

    def backward(self, observations):
        T = len(observations)  # Total number of time steps
        beta = np.zeros((T, self.num_states))
        beta[T-1, :] = 1  # Initialize last step
        
        for t in range(T-2, -1, -1):
            for i in range(self.num_states):
                # Ensure correct mean and covariance are used
                # pdf_values = np.array([self.multivariate_normal_logpdf(observations[t+1], self.means[j], self.covariances[j]) for j in range(self.num_states)])
                pdf_values = np.array([multivariate_normal.logpdf(observations[t+1], self.means[j], self.covariances[j]) for j in range(self.num_states)])
                beta[t, i] = log_calc.log_add_arr(self.transitions[i, :] * pdf_values * beta[t+1, :])
        
        return beta

    def log_add_exp(self, log_probs):
        """Compute the log of the sum of exponentials of input elements."""
        max_log_prob = np.max(log_probs)
        return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

    def backward_log(self, observations):
        T = len(observations)  # Total number of time steps
        beta = np.zeros((T, self.num_states))
        beta[T-1, :] = 0  # Initialize last step with log(1) = 0

        for t in range(T-2, -1, -1):
            for i in range(self.num_states):
                # Compute log of the observation probabilities for each state
                log_pdf_values = np.array([multivariate_normal.logpdf(observations[t+1], self.means[j], self.covariances[j]) for j in range(self.num_states)])
                
                # Compute log(beta[t+1, j]) for each state
                log_beta_t1 = beta[t+1, :]
                
                # Compute log(alpha[t, i]) using log-space calculations
                log_transitions = np.log(self.transitions[i, :])
                log_terms = log_transitions + log_pdf_values + log_beta_t1
                
                # Use log-sum-exp to sum the terms in log space
                beta[t, i] = self.log_add_exp(log_terms)
        
        return beta
    
    def compute_gamma(self, alpha, beta):
        gamma = np.multiply(alpha, beta) / np.sum(np.multiply(alpha, beta), axis=0)
        return gamma

    def compute_xi(self, alpha, beta, observations):
        T = len(observations)
        xi = np.zeros((self.M, self.M, T-1))
        for t in range(T-1):
            denominator = np.dot(alpha[:, t], np.dot(self.A, np.multiply(beta[:, t+1], self.B[:, observations[t+1]])))
            for i in range(self.M):
                xi[i, :, t] = alpha[i, t] * self.A[i, :] * self.B[:, observations[t+1]] * beta[:, t+1] / denominator
        return xi

    def compute_gamma_log(self, log_alpha, log_beta):
        T, N = log_alpha.shape
        log_gamma = np.zeros((T, N))

        for t in range(T):
            log_gamma[t, :] = log_alpha[t, :] + log_beta[t, :]
            log_gamma[t, :] -= self.log_add_exp(log_gamma[t, :])

        return log_gamma

    def e_step_log(self, observations):
        log_alpha = self.forward_log(observations)
        log_beta = self.backward_log(observations)

        T = len(observations)
        log_gamma = self.compute_gamma_log(log_alpha, log_beta)
        log_xi = np.zeros((T-1, self.num_states, self.num_states))

        for t in range(T-1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    log_prob_obs = multivariate_normal.logpdf(observations[t+1], self.means[j], self.covariances[j])
                    log_xi[t, i, j] = log_alpha[t, i] + np.log(self.transitions[i, j]) + log_prob_obs + log_beta[t+1, j]
            log_xi[t, :, :] -= self.log_add_exp(log_xi[t, :, :])

        log_likelihood = self.log_add_exp(log_alpha[-1, :])

        return log_gamma, log_xi, log_likelihood

    def e_step(self, observations):
        alpha = self.forward(observations)
        beta = self.backward(observations)
        
        T = len(observations)
        gamma = np.zeros((T, self.num_states))
        xi = np.zeros((T-1, self.num_states, self.num_states))
        
        likelihood = self.log_add_exp(alpha[-1, :])  # P(O|λ)
        likelihood = np.logaddexp(alpha[-1, :])  # P(O|λ)
        for t in range(T):
            gamma[t, :] = (alpha[t, :] * beta[t, :]) / likelihood
        
        for t in range(T-1):
            xi[t, :, :] = (np.outer(alpha[t, :], beta[t+1, :] * multivariate_normal.logpdf(observations[t+1], self.means, self.covariances)) * self.transitions) / likelihood
            # xi[t, :, :] = (np.outer(alpha[t, :], beta[t+1, :] * self.multivariate_normal_logpdf(observations[t+1], self.means, self.covariances)) * self.transitions) / likelihood
        
        return gamma, xi, likelihood

    def m_step_log(self, observations, log_gamma, log_xi):
        T = len(observations)

        for i in range(self.num_states):
            # Update initial probabilities in log space
            self.initial_probs[i] = np.exp(log_gamma[0, i] - self.log_add_exp(log_gamma[0, :]))

            # Update transition probabilities in log space
            for j in range(self.num_states):
                log_numerator = self.log_add_exp(log_xi[:, i, j])
                log_denominator = self.log_add_exp(log_gamma[:-1, i])
                self.transitions[i, j] = np.exp(log_numerator - log_denominator)

            # Update means and covariances in log space
            mask = log_gamma[:, i] > np.log(1e-6)  # To prevent division by zero
            log_gamma_masked = log_gamma[mask, i]
            masked_observations = observations[mask]

            log_gamma_sum = self.log_add_exp(log_gamma_masked)
            # Update means
            log_weighted_sum = self.log_add_exp(log_gamma_masked[:, np.newaxis] + np.log(masked_observations))
            self.means[i] = np.exp(log_weighted_sum - log_gamma_sum)

            # Update covariances
            diffs = masked_observations - self.means[i]
            log_weighted_diff_sum = self.log_add_exp(log_gamma_masked[:, np.newaxis] + np.log(diffs**2))
            self.covariances[i] = np.exp(log_weighted_diff_sum - log_gamma_sum)

        # Ensure the transition matrix rows sum to 1
        self.transitions = self.transitions / np.sum(self.transitions, axis=1, keepdims=True)

    def m_step(self, observations, gamma, xi):
        T = len(observations)
        
        for i in range(self.num_states):
            self.initial_probs[i] = gamma[0, i]
            for j in range(self.num_states):
                self.transitions[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
                
            mask = gamma[:, i] > 1e-6  # To prevent division by zero
            self.means[i] = np.dot(gamma[mask, i], observations[mask]) / np.sum(gamma[mask, i])
            diffs = observations[mask] - self.means[i]
            self.covariances[i] = np.dot(gamma[mask, i] * diffs.T, diffs) / np.sum(gamma[mask, i])
    
    def log_likelihood(self, alpha):
        return self.log_add_exp(alpha[-1, :])
    
    def log_e_step(self, observations):
        log_alpha = self.log_forward(observations)
        log_beta = self.log_backward(observations)
        
        T = len(observations)
        gamma = np.exp(log_alpha + log_beta - log_calc.log_add(log_alpha[T-1]))
        
        xi = np.zeros((T-1, self.num_states, self.num_states))
        for t in range(T-1):
            log_xi_t = (log_alpha[t, :, np.newaxis] + np.log(self.transitions) +
                        np.log(multivariate_normal.logpdf(observations[t+1], self.means, self.covariances))[:, np.newaxis] +
                        log_beta[t+1])
            xi[t] = np.exp(log_xi_t - log_calc.log_add(log_xi_t))
        
        return gamma, xi, np.sum(log_alpha[T-1])

    def log_m_step(self, observations, gamma, xi):
        T = len(observations)
        
        self.initial_probs = np.exp(log_calc.log_add(np.log(gamma[0])) - np.log(T))
        
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.transitions[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
            
            mask = gamma[:, i] > 1e-6
            weighted_observations = np.dot(gamma[:, i], observations[mask])
            self.means[i] = weighted_observations / np.sum(gamma[:, i])
            diffs = observations[mask] - self.means[i]
            self.covariances[i] = np.dot(gamma[mask, i] * diffs.T, diffs) / np.sum(gamma[mask, i])

    def log_fit(self, observations, max_iterations=100, tolerance=1e-6):
        self.initialize_parameters()
        prev_log_likelihood = -np.inf

        for iteration in range(max_iterations):
            gamma, xi, likelihood = self.e_step_log(observations)
            self.m_step_log(observations, gamma, xi)

            current_log_likelihood = self.log_likelihood(self.forward_log(observations))
            print(f"Iteration {iteration}, Log Likelihood: {current_log_likelihood}")

            if np.abs(current_log_likelihood - prev_log_likelihood) < tolerance:
                print("Convergence reached.")
                break
            prev_log_likelihood = current_log_likelihood
        return self

    def fit(self, observations, max_iterations=100, tolerance=1e-6):
        self.initialize_parameters()  # Initialize parameters if not already initialized
        prev_log_likelihood = -np.inf

        for iteration in range(max_iterations):
            gamma, xi, likelihood = self.e_step(observations)
            self.m_step(observations, gamma, xi)

            # Compute log likelihood to check convergence
            current_log_likelihood = self.log_likelihood(self.forward(observations))
            print(f"Iteration {iteration}, Log Likelihood: {current_log_likelihood}")

            # Check for convergence
            if np.abs(current_log_likelihood - prev_log_likelihood) < tolerance:
                print("Convergence reached.")
                break
            prev_log_likelihood = current_log_likelihood
        return self

# # Example usage:
# num_states = 4
# num_features = 26  # Adjust based on your data, e.g., number of MFCC coefficients
# hmm = HMM(num_states, num_features)

# # Observations would be an array of observed feature vectors
# observations = [np.random.rand(num_features) for _ in range(100)]  # Replace with real data
# hmm.fit(observations)

# Example integration with a dataset
num_states = 3
num_features = 13
hmm = HMM(num_states, num_features)
hmm.initialize_parameters()

directory = 'OdessaRecordings'

# Function to extract MFCC features
def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = StandardScaler().fit_transform(mfcc_features.T)
    return mfcc_scaled

# List all files in the directory
files_in_directory = os.listdir(directory)
sample_rate = 16000
features_arr = []
filename = "odessa"
for i in range(29):
    # features = mfcc.calculations_for_onerecording(False, directory=directory,filename=f"{filename}_{i+1}",sample_rate=sample_rate)
    features = extract_mfcc(f"{directory}/{filename}_{i+1}.wav")
    features_arr.append(features)

observations = np.array(features_arr[0])
# print(observations[:,0].shape)
# Running one iteration of EM
gamma, xi, likelihood = hmm.e_step_log(observations[:,0])
hmm.m_step_log(observations, gamma, xi)
print("Log likelihood:", hmm.log_likelihood(hmm.forward_log(observations)))