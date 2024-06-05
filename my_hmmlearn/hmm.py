"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import logging

import numpy as np
from my_hmmlearn import utils
from collections import deque
import sys

class ConvergenceMonitor:

    _template = "{iter:>10d} {log_prob:>16.8f} {delta:>+16.8f}"

    def __init__(self, tol, n_iter, verbose):
        """
        Parameters
        ----------
        tol : double
            Convergence threshold.  EM has converged either if the maximum
            number of iterations is reached or the log probability improvement
            between the two consecutive iterations is less than threshold.
        n_iter : int
            Maximum number of iterations to perform.
        verbose : bool
            Whether per-iteration convergence reports are printed.
        """
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque()
        self.iter = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        params = sorted(dict(vars(self), history=list(self.history)).items())
        return ("{}(\n".format(class_name)
                + "".join(map("    {}={},\n".format, *zip(*params)))
                + ")")

    def _reset(self):
        """Reset the monitor's state."""
        self.iter = 0
        self.history.clear()

    def report(self, log_prob):
        if self.verbose:
            delta = log_prob - self.history[-1] if self.history else np.nan
            message = self._template.format(
                iter=self.iter + 1, log_prob=log_prob, delta=delta)
            print(message, file=sys.stderr)

        # Allow for some wiggleroom based on precision.
        precision = np.finfo(float).eps ** (1/2)
        if self.history and (log_prob - self.history[-1]) < -precision:
            delta = log_prob - self.history[-1]
            _log.warning(f"Model is not converging.  Current: {log_prob}"
                         f" is not greater than {self.history[-1]}."
                         f" Delta is {delta}")
        self.history.append(log_prob)
        self.iter += 1

    @property
    def converged(self):
        """Whether the EM algorithm converged."""
        return (self.iter == self.n_iter or
                (len(self.history) >= 2 and
                 self.history[-1] - self.history[-2] < self.tol))



class MyGaussianHMM():
   
    def __init__(self, n_components=1, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 implementation="log"):
        
        # super().__init__( n_components=1, random_state=1,n_iter=10, tol=1e-2, verbose=False,implementation="log")
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = True
        self.implementation = implementation
        self.random_state = random_state
        
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)

        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    @property
    def covars_(self):
        """Return covars as a full matrix."""
        return utils.fill_covars(self._covars_, self.covariance_type,
                           self.n_components, self.n_features)

    @covars_.setter
    def covars_(self, covars):
        covars = np.array(covars, copy=True)
        utils.validate_covars(covars, self.covariance_type,
                                self.n_components)
        self._covars_ = covars

    def init(self, X, lengths=None):

        self.check_and_set_n_features(X)
        init = 1. / self.n_components
        random_state = np.random.mtrand._rand
        self.startprob_ = random_state.dirichlet(
            np.full(self.n_components, init))
        self.transmat_ = random_state.dirichlet(
            np.full(self.n_components, init), size=self.n_components)

        np.random.seed(self.random_state) 
        indices = np.random.choice(X.shape[0], self.n_components, replace=False)
        self.means_ = X[indices]
        cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
        if not cv.shape:
            cv.shape = (1, 1)
        self.covars_ = \
            utils.distribute_covar_matrix_to_match_covariance_type(
                cv, self.covariance_type, self.n_components).copy()

    def compute_log_likelihood(self, X):
        likelihood = self._compute_likelihood(X)
        with np.errstate(divide="ignore"):
            return np.log(likelihood)

        # return log_multivariate_normal_density(
        #     X, self.means_, self._covars_, self.covariance_type)

    def compute_likelihood(self, X):
        """
        Compute per-component probability under the model.

        """
        sub_cll = self._log_multivariate_normal_density_diag(
            X, self.means_, self._covars_)
        with np.errstate(under="ignore"):
            return np.exp(sub_cll)

    def _log_multivariate_normal_density_diag(self, X, means, covars):
        """Compute Gaussian log-density at X for a diagonal model."""
        # X: (ns, nf); means: (nc, nf); covars: (nc, nf) -> (ns, nc)
        nc, nf = means.shape
        # Avoid 0 log 0 = nan in degenerate covariance case.
        covars = np.maximum(covars, np.finfo(float).tiny)
        with np.errstate(over="ignore"):
            return -0.5 * (nf * np.log(2 * np.pi)
                        + np.log(covars).sum(axis=-1)
                        + ((X[:, None, :] - means) ** 2 / covars).sum(axis=-1))

    def do_mstep(self, stats):
        self.do_em_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        denom = stats['post'][:, None]
        self.means_ = ((means_weight * means_prior + stats['obs'])
                        / (means_weight + denom))

        covars_prior = self.covars_prior
        covars_weight = self.covars_weight
        meandiff = self.means_ - means_prior

        c_n = (means_weight * meandiff**2
                + stats['obs**2']
                - 2 * self.means_ * stats['obs']
                + self.means_**2 * denom)
        c_d = max(covars_weight - 1, 0) + denom
        self._covars_ = (covars_prior + c_n) / np.maximum(c_d, 1e-5)
           
    def score(self, X, lengths=None, *, compute_posteriors=False):
        """
        Compute the log probability under the model
        """
        impl = {
            "scaling": self.score_scaling,
            "log": self.score_log,
        }[self.implementation]
        score = impl(
            X=X, lengths=lengths, compute_posteriors=compute_posteriors)
        return score[0]
    
    def fit(self, X, lengths=None):

        if lengths is None:
            lengths = np.asarray([X.shape[0]])

        self.init(X, lengths)
        self.monitor_._reset()

        for iter in range(self.n_iter):
            stats, curr_logprob = self.do_estep(X, lengths)
            # Compute lower bound before updating model parameters
            lower_bound = curr_logprob

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self.do_em_mstep(stats)
            self.monitor_.report(lower_bound)

            if self.monitor_.converged:
                break

            if (self.transmat_.sum(axis=1) == 0).any():
                _log.warning("Some rows of transmat_ have zero sum because no "
                             "transition from the state was ever observed.")
        return self

    def initialize_sufficient_statistics(self):
        """
        Initialize sufficient statistics required for M-step.
        """
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_components),
                 'trans': np.zeros((self.n_components, self.n_components)),
                 'post' : np.zeros(self.n_components),
                 'obs' : np.zeros((self.n_components, self.n_features)),
                 'obs**2' : np.zeros((self.n_components, self.n_features))
                 }
        return stats

    def accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        """
        Update sufficient statistics from a given sample.

        """
        stats['post'] += posteriors.sum(axis=0)
        stats['obs'] += posteriors.T @ X
        stats['obs**2'] += posteriors.T @ X**2

        impl = {
            "scaling": self.accumulate_sufficient_statistics_scaling,
            "log": self.accumulate_sufficient_statistics_log,
        }[self.implementation]

        return impl(stats=stats, X=X, lattice=lattice, posteriors=posteriors,
                    fwdlattice=fwdlattice, bwdlattice=bwdlattice)


    # Log Calculations
    def score_log(self, X, lengths=None, *, compute_posteriors):
       
        log_prob = 0
        sub_posteriors = [np.empty((0, self.n_components))]
        for sub_X in utils.split_X_lengths(X, lengths):
            log_frameprob = self.compute_log_likelihood(sub_X)
            log_probij, fwdlattice = utils.forward_log(
                self.startprob_, self.transmat_, log_frameprob)
            log_prob += log_probij
            if compute_posteriors:
                bwdlattice = utils.backward_log(
                    self.startprob_, self.transmat_, log_frameprob)
                sub_posteriors.append(
                    self.compute_posteriors_log(fwdlattice, bwdlattice))
        return log_prob, np.concatenate(sub_posteriors)

    def compute_posteriors_log(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by log_prob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        utils.log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    def fit_log(self, X):
        log_frameprob = self.compute_log_likelihood(X)
        log_prob, fwdlattice = utils.forward_log(
            self.startprob_, self.transmat_, log_frameprob)
        bwdlattice = utils.backward_log(
            self.startprob_, self.transmat_, log_frameprob)
        posteriors = self.compute_posteriors_log(fwdlattice, bwdlattice)
        return log_frameprob, log_prob, posteriors, fwdlattice, bwdlattice
   
    def accumulate_sufficient_statistics_log(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        
        stats['nobs'] += 1
        stats['start'] += posteriors[0]
        n_samples, n_components = lattice.shape
        # when the sample is of length 1, it contains no transitions
        # so there is no reason to update our trans. matrix estimate
        if n_samples <= 1:
            return
        log_xi_sum = utils.compute_log_xi_sum(
            fwdlattice, self.transmat_, bwdlattice, lattice)
        with np.errstate(under="ignore"):
            stats['trans'] += np.exp(log_xi_sum)

    # Scaling Calculations
    def score_scaling(self, X, lengths=None, *, compute_posteriors):
        log_prob = 0
        sub_posteriors = [np.empty((0, self.n_components))]
        for sub_X in utils.split_X_lengths(X, lengths):
            frameprob = self.compute_likelihood(sub_X)
            log_probij, fwdlattice, scaling_factors = utils.forward_scaling(
                self.startprob_, self.transmat_, frameprob)
            log_prob += log_probij
            if compute_posteriors:
                bwdlattice = utils.backward_scaling(
                    self.startprob_, self.transmat_,
                    frameprob, scaling_factors)
                sub_posteriors.append(
                    self.compute_posteriors_scaling(fwdlattice, bwdlattice))

        return log_prob, np.concatenate(sub_posteriors)

    def compute_posteriors_scaling(self, fwdlattice, bwdlattice):
        posteriors = fwdlattice * bwdlattice
        utils.normalize(posteriors, axis=1)
        return posteriors

    def fit_scaling(self, X):
        frameprob = self.compute_likelihood(X)
        log_prob, fwdlattice, scaling_factors = utils.forward_scaling(
            self.startprob_, self.transmat_, frameprob)
        bwdlattice = utils.backward_scaling(
            self.startprob_, self.transmat_, frameprob, scaling_factors)
        posteriors = self.compute_posteriors_scaling(fwdlattice, bwdlattice)
        return frameprob, log_prob, posteriors, fwdlattice, bwdlattice

    def accumulate_sufficient_statistics_scaling(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        """
        Implementation of `_accumulate_sufficient_statistics`
        for ``implementation = "log"``.
        """
        stats['nobs'] += 1
        stats['start'] += posteriors[0]
        n_samples, n_components = lattice.shape
        # when the sample is of length 1, it contains no transitions
        # so there is no reason to update our trans. matrix estimate
        if n_samples <= 1:
            return
        xi_sum = utils.compute_scaling_xi_sum(
            fwdlattice, self.transmat_, bwdlattice, lattice)
        stats['trans'] += xi_sum


    def decode(self, X, lengths=None):
        """
        Find most likely state sequence corresponding to ``X``.
        """

        log_frameprob = self.compute_log_likelihood(X)

        decoder = utils.viterbi(self.startprob_, self.transmat_, log_frameprob)
        log_prob = 0
        sub_state_sequences = []
        for sub_X in utils.split_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            sub_log_prob, sub_state_sequence = decoder(sub_X)
            log_prob += sub_log_prob
            sub_state_sequences.append(sub_state_sequence)

        return log_prob, np.concatenate(sub_state_sequences)

    def sample(self, n_samples=1, random_state=None, currstate=None):
        """
        Generate random samples from the model.
        """
        if random_state is None:
            random_state = self.random_state
        transmat_cdf = np.cumsum(self.transmat_, axis=1)

        if currstate is None:
            startprob_cdf = np.cumsum(self.startprob_)
            currstate = (startprob_cdf > random_state.rand()).argmax()

        state_sequence = [currstate]
        X = [self.generate_sample_from_state(
            currstate, random_state=random_state)]

        for t in range(n_samples - 1):
            currstate = (
                (transmat_cdf[currstate] > random_state.rand()).argmax())
            state_sequence.append(currstate)
            X.append(self.generate_sample_from_state(
                currstate, random_state=random_state))

        return np.atleast_2d(X), np.array(state_sequence, dtype=int)

    def check_and_set_n_features(self, X):
        _, n_features = X.shape
        if hasattr(self, "n_features"):
            if self.n_features != n_features:
                raise ValueError(
                    f"Unexpected number of dimensions, got {n_features} but "
                    f"expected {self.n_features}")
        else:
            self.n_features = n_features


    def generate_sample_from_state(self, state, random_state):
        """
        Generate a random sample from a given component.
        """
        return random_state.multivariate_normal(
            self.means_[state], self.covars_[state]
        )

    def do_estep(self, X, lengths):
        impl = {
            "scaling": self.fit_scaling,
            "log": self.fit_log,
        }[self.implementation]

        stats = self.initialize_sufficient_statistics()
        curr_logprob = 0
        for sub_X in utils.split_X_lengths(X, lengths):
            lattice, logprob, posteriors, fwdlattice, bwdlattice = impl(sub_X)
            self.accumulate_sufficient_statistics(
                stats, sub_X, lattice, posteriors, fwdlattice,
                bwdlattice)
            curr_logprob += logprob
        return stats, curr_logprob

    def do_em_mstep(self, stats):
        """
        Perform the M-step of EM algorithm.
        """
        # If a prior is < 1, `prior - 1 + starts['start']` can be negative.  In
        # that case maximization of (n1+e1) log p1 + ... + (ns+es) log ps under
        # the conditions sum(p) = 1 and all(p >= 0) show that the negative
        # terms can just be set to zero.
        # The ``np.where`` calls guard against updating forbidden states
        # or transitions in e.g. a left-right HMM.
        startprob_ = np.maximum(self.startprob_prior - 1 + stats['start'],
                                0)
        self.startprob_ = np.where(self.startprob_ == 0, 0, startprob_)
        utils.normalize(self.startprob_)
        transmat_ = np.maximum(self.transmat_prior - 1 + stats['trans'], 0)
        self.transmat_ = np.where(self.transmat_ == 0, 0, transmat_)
        utils.normalize(self.transmat_, axis=1)
