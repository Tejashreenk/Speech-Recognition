import functools

import numpy as np

from .base import _AbstractHMM
from .stats import log_multivariate_normal_density

class BaseGaussianHMM(_AbstractHMM):

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
            "c": {
                "spherical": nc,
                "diag": nc * nf,
                "full": nc * nf * (nf + 1) // 2,
                "tied": nf * (nf + 1) // 2,
            }[self.covariance_type],
        }

    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(stats=stats, X=X,
                                                  lattice=lattice,
                                                  posteriors=posteriors,
                                                  fwdlattice=fwdlattice,
                                                  bwdlattice=bwdlattice)

        if self._needs_sufficient_statistics_for_mean():
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += posteriors.T @ X

        if self._needs_sufficient_statistics_for_covars():
            stats['obs**2'] += posteriors.T @ X**2

    def _needs_sufficient_statistics_for_mean(self):
        """
        Whether the sufficient statistics needed to update the means are
        updated during calls to `fit`.
        """
        raise NotImplementedError("Must be overriden in subclass")

    def _needs_sufficient_statistics_for_covars(self):
        """
        Whhether the sufficient statistics needed to update the covariances are
        updated during calls to `fit`.
        """
        raise NotImplementedError("Must be overriden in subclass")

    def _generate_sample_from_state(self, state, random_state):
        return random_state.multivariate_normal(
            self.means_[state], self.covars_[state]
        )

