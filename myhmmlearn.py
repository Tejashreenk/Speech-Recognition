import numpy as np
import hmm_utils as hu

class GaussianHMM(_emissions.BaseGaussianHMM, BaseHMM):
    """
    Hidden Markov Model with Gaussian emissions.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars_ : array
        Covariance parameters for each state.

        The shape depends on :attr:`covariance_type`:

        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)  #doctest: +ELLIPSIS
    GaussianHMM(algorithm='viterbi',...
    """

    def __init__(self, n_components=1, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc",
                 implementation="log"):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        covariance_type : {"spherical", "diag", "full", "tied"}, optional
            The type of covariance parameters to use:

            * "spherical" --- each state uses a single variance value that
              applies to all features (default).
            * "diag" --- each state uses a diagonal covariance matrix.
            * "full" --- each state uses a full (i.e. unrestricted)
              covariance matrix.
            * "tied" --- all states use **the same** full covariance matrix.

        min_covar : float, optional
            Floor on the diagonal of the covariance matrix to prevent
            overfitting. Defaults to 1e-3.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        means_prior, means_weight : array, shape (n_components, ), optional
            Mean and precision of the Normal prior distribtion for
            :attr:`means_`.

        covars_prior, covars_weight : array, shape (n_components, ), optional
            Parameters of the prior distribution for the covariance matrix
            :attr:`covars_`.

            If :attr:`covariance_type` is "spherical" or "diag" the prior is
            the inverse gamma distribution, otherwise --- the inverse Wishart
            distribution.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any combination
            of 's' for startprob, 't' for transmat, 'm' for means, and 'c' for
            covars.  Defaults to all parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.
        """
        super().__init__(n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior, algorithm=algorithm,
                         random_state=random_state, n_iter=n_iter,
                         tol=tol, params=params, verbose=verbose,
                         init_params=init_params,
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
        return hu.fill_covars(self._covars_, self.covariance_type,
                           self.n_components, self.n_features)

    @covars_.setter
    def covars_(self, covars):
        covars = np.array(covars, copy=True)
        hu._validate_covars(covars, self.covariance_type,
                                self.n_components)
        self._covars_ = covars

    def _init(self, X, lengths=None):
        super()._init(X, lengths)

        if self._needs_init("m", "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state,
                                    n_init=10)  # sklearn <1.4 backcompat.
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        if self._needs_init("c", "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = \
                hu.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy()

    def _check(self):
        super()._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {COVARIANCE_TYPES}")

    def _needs_sufficient_statistics_for_mean(self):
        return 'm' in self.params

    def _needs_sufficient_statistics_for_covars(self):
        return 'c' in self.params

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, None]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                c_n = (means_weight * meandiff**2
                       + stats['obs**2']
                       - 2 * self.means_ * stats['obs']
                       + self.means_**2 * denom)
                c_d = max(covars_weight - 1, 0) + denom
                self._covars_ = (covars_prior + c_n) / np.maximum(c_d, 1e-5)
                if self.covariance_type == 'spherical':
                    self._covars_ = np.tile(self._covars_.mean(1)[:, None],
                                            (1, self._covars_.shape[1]))
            elif self.covariance_type in ('tied', 'full'):
                c_n = np.empty((self.n_components, self.n_features,
                                self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self.means_[c])
                    c_n[c] = (means_weight * np.outer(meandiff[c],
                                                      meandiff[c])
                              + stats['obs*obs.T'][c]
                              - obsmean - obsmean.T
                              + np.outer(self.means_[c], self.means_[c])
                              * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == 'tied':
                    self._covars_ = ((covars_prior + c_n.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self.covariance_type == 'full':
                    self._covars_ = ((covars_prior + c_n) /
                                     (cvweight + stats['post'][:, None, None]))

