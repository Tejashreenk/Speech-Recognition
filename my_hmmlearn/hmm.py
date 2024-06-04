"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import logging

import numpy as np
from .base import BaseHMM
from my_hmmlearn import utils
from .stats import log_multivariate_normal_density
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
        # XXX we might want to check that ``log_prob`` is non-decreasing.
        return (self.iter == self.n_iter or
                (len(self.history) >= 2 and
                 self.history[-1] - self.history[-2] < self.tol))



class GaussianHMM(BaseHMM):
   
    def __init__(self, n_components=1, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 implementation="log"):
        
        self.n_components = n_components
        self.algorithm = algorithm
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
        utils._validate_covars(covars, self.covariance_type,
                                self.n_components)
        self._covars_ = covars

    def _init(self, X, lengths=None):
        super()._init(X, lengths)

        # if self._needs_init("m", "means_"):
            # kmeans = cluster.KMeans(n_clusters=self.n_components,
            #                         random_state=self.random_state,
            #                         n_init=10)  # sklearn <1.4 backcompat.
            # kmeans.fit(X)
            # self.means_ = kmeans.cluster_centers_
            # Randomly select n_components samples from X as initial means
        np.random.seed(self.random_state)  # Ensure reproducibility
        indices = np.random.choice(X.shape[0], self.n_components, replace=False)
        self.means_ = X[indices]

        # if self._needs_init("c", "covars_"):
        cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
        if not cv.shape:
            cv.shape = (1, 1)
        self.covars_ = \
            utils.distribute_covar_matrix_to_match_covariance_type(
                cv, self.covariance_type, self.n_components).copy()

    def _check(self):
        super()._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]


    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
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
           