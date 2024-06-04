"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import logging

import numpy as np
from .base import BaseHMM
from my_hmmlearn import utils
from .stats import log_multivariate_normal_density


class GaussianHMM(BaseHMM):
   
    def __init__(self, n_components=1, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 implementation="log"):
        
        super().__init__(n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior, algorithm=algorithm,
                         random_state=random_state, n_iter=n_iter,
                         tol=tol, verbose=verbose,
                         implementation=implementation)
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
           