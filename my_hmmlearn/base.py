import logging
import string
import sys
from collections import deque

import numpy as np
from scipy import linalg, special
from my_hmmlearn import utils


_log = logging.getLogger(__name__)
#: Supported decoder algorithms.

class ConvergenceMonitor:

    _template = "{iter:>10d} {log_prob:>16.8f} {delta:>+16.8f}"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque()
        self.iter = 0

    # def __repr__(self):
    #     class_name = self.__class__.__name__
    #     params = sorted(dict(vars(self), history=list(self.history)).items())
    #     return ("{}(\n".format(class_name)
    #             + "".join(map("    {}={},\n".format, *zip(*params)))
    #             + ")")

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


# class _AbstractHMM():
class BaseHMM():
    """
    Base class for Hidden Markov Models learned via Expectation-Maximization
    and Variational Bayes.
    """
    def __init__(self, n_components, random_state, n_iter,
                 tol, verbose, implementation):
        """
            n_components : int
            Number of states in the model.
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
            of 's' for startprob, 't' for transmat, and other characters for
            subclass-specific emission parameters.  Defaults to all parameters.
        implementation: string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.  However, the
            scaling implementation is generally faster.
        """

        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = True
        self.implementation = implementation
        self.random_state = random_state

    def _init(self, X, lengths=None):
        """
        Initialize model parameters prior to fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        """
        self._check_and_set_n_features(X)
        init = 1. / self.n_components
        random_state = np.random.mtrand._rand
        self.startprob_ = random_state.dirichlet(
            np.full(self.n_components, init))
        self.transmat_ = random_state.dirichlet(
            np.full(self.n_components, init), size=self.n_components)
        n_fit_scalars_per_param = self._get_n_fit_scalars_per_param()
        if n_fit_scalars_per_param is not None:
            n_fit_scalars = sum(
                n_fit_scalars_per_param[p] for p in "stmc")
            if X.size < n_fit_scalars:
                _log.warning(
                    "Fitting a model with %d free scalar parameters with only "
                    "%d data points will result in a degenerate solution.",
                    n_fit_scalars, X.size)

    def score(self, X, lengths=None, *, compute_posteriors=False):
        """
        Helper for `score` and `score_samples`.

        Compute the log probability under the model, as well as posteriors if
        *compute_posteriors* is True (otherwise, an empty array is returned
        for the latter).
        """
        impl = {
            "scaling": self._score_scaling,
            "log": self._score_log,
        }[self.implementation]
        score = impl(
            X=X, lengths=lengths, compute_posteriors=compute_posteriors)
        return score[0]
    
    def fit(self, X, lengths=None):

        if lengths is None:
            lengths = np.asarray([X.shape[0]])

        self._init(X, lengths)
        self.monitor_._reset()

        for iter in range(self.n_iter):
            stats, curr_logprob = self._do_estep(X, lengths)
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

    def _initialize_sufficient_statistics(self):
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

    def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        """
        Update sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~.BaseHMM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features)
            Sample sequence.

        lattice : array, shape (n_samples, n_components)
            Probabilities OR Log Probabilities of each sample
            under each of the model states.  Depends on the choice
            of implementation of the Forward-Backward algorithm

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            forward and backward probabilities.
        """
        stats['post'] += posteriors.sum(axis=0)
        stats['obs'] += posteriors.T @ X
        stats['obs**2'] += posteriors.T @ X**2

        impl = {
            "scaling": self._accumulate_sufficient_statistics_scaling,
            "log": self._accumulate_sufficient_statistics_log,
        }[self.implementation]

        return impl(stats=stats, X=X, lattice=lattice, posteriors=posteriors,
                    fwdlattice=fwdlattice, bwdlattice=bwdlattice)


    # Log Calculations
    def _score_log(self, X, lengths=None, *, compute_posteriors):
        """
        Compute the log probability under the model, as well as posteriors if
        *compute_posteriors* is True (otherwise, an empty array is returned
        for the latter).
        """
        log_prob = 0
        sub_posteriors = [np.empty((0, self.n_components))]
        for sub_X in utils.split_X_lengths(X, lengths):
            log_frameprob = self._compute_log_likelihood(sub_X)
            log_probij, fwdlattice = utils.forward_log(
                self.startprob_, self.transmat_, log_frameprob)
            log_prob += log_probij
            if compute_posteriors:
                bwdlattice = utils.backward_log(
                    self.startprob_, self.transmat_, log_frameprob)
                sub_posteriors.append(
                    self._compute_posteriors_log(fwdlattice, bwdlattice))
        return log_prob, np.concatenate(sub_posteriors)

    def _compute_posteriors_log(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by log_prob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        utils.log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    def _fit_log(self, X):
        log_frameprob = self._compute_log_likelihood(X)
        log_prob, fwdlattice = utils.forward_log(
            self.startprob_, self.transmat_, log_frameprob)
        bwdlattice = utils.backward_log(
            self.startprob_, self.transmat_, log_frameprob)
        posteriors = self._compute_posteriors_log(fwdlattice, bwdlattice)
        return log_frameprob, log_prob, posteriors, fwdlattice, bwdlattice
   
    def _accumulate_sufficient_statistics_log(
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
        log_xi_sum = utils.compute_log_xi_sum(
            fwdlattice, self.transmat_, bwdlattice, lattice)
        with np.errstate(under="ignore"):
            stats['trans'] += np.exp(log_xi_sum)

# Scaling Calculations
    def _score_scaling(self, X, lengths=None, *, compute_posteriors):
        log_prob = 0
        sub_posteriors = [np.empty((0, self.n_components))]
        for sub_X in utils.split_X_lengths(X, lengths):
            frameprob = self._compute_likelihood(sub_X)
            log_probij, fwdlattice, scaling_factors = utils.forward_scaling(
                self.startprob_, self.transmat_, frameprob)
            log_prob += log_probij
            if compute_posteriors:
                bwdlattice = utils.backward_scaling(
                    self.startprob_, self.transmat_,
                    frameprob, scaling_factors)
                sub_posteriors.append(
                    self._compute_posteriors_scaling(fwdlattice, bwdlattice))

        return log_prob, np.concatenate(sub_posteriors)

    def _compute_posteriors_scaling(self, fwdlattice, bwdlattice):
        posteriors = fwdlattice * bwdlattice
        utils.normalize(posteriors, axis=1)
        return posteriors

    def _fit_scaling(self, X):
        frameprob = self._compute_likelihood(X)
        log_prob, fwdlattice, scaling_factors = utils.forward_scaling(
            self.startprob_, self.transmat_, frameprob)
        bwdlattice = utils.backward_scaling(
            self.startprob_, self.transmat_, frameprob, scaling_factors)
        posteriors = self._compute_posteriors_scaling(fwdlattice, bwdlattice)
        return frameprob, log_prob, posteriors, fwdlattice, bwdlattice

    def _accumulate_sufficient_statistics_scaling(
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


    def decode(self, X, lengths=None, algorithm=None):
        """
        Find most likely state sequence corresponding to ``X``.
        """

        log_frameprob = self._compute_log_likelihood(X)

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

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.
        currstate : int
            Current state, as the initial state of the samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix.
        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.

        Examples
        --------
        ::

            # generate samples continuously
            _, Z = model.sample(n_samples=10)
            X, Z = model.sample(n_samples=10, currstate=Z[-1])
        """
        if random_state is None:
            random_state = self.random_state
        transmat_cdf = np.cumsum(self.transmat_, axis=1)

        if currstate is None:
            startprob_cdf = np.cumsum(self.startprob_)
            currstate = (startprob_cdf > random_state.rand()).argmax()

        state_sequence = [currstate]
        X = [self._generate_sample_from_state(
            currstate, random_state=random_state)]

        for t in range(n_samples - 1):
            currstate = (
                (transmat_cdf[currstate] > random_state.rand()).argmax())
            state_sequence.append(currstate)
            X.append(self._generate_sample_from_state(
                currstate, random_state=random_state))

        return np.atleast_2d(X), np.array(state_sequence, dtype=int)

    def _check_and_set_n_features(self, X):
        _, n_features = X.shape
        if hasattr(self, "n_features"):
            if self.n_features != n_features:
                raise ValueError(
                    f"Unexpected number of dimensions, got {n_features} but "
                    f"expected {self.n_features}")
        else:
            self.n_features = n_features

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
            "c":  nc * nf,
        }

    def _compute_likelihood(self, X):
        """
        Compute per-component probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.
        """
        if (self._compute_log_likelihood  # prevent recursion
                != __class__._compute_log_likelihood.__get__(self)):
            # Probabilities equal to zero do occur, and exp(-LARGE) = 0 is OK.
            with np.errstate(under="ignore"):
                return np.exp(self._compute_log_likelihood(X))
        else:
            raise NotImplementedError("Must be overridden in subclass")

    def _compute_log_likelihood(self, X):
        """
        Compute per-component emission log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
            Emission log probability of each sample in ``X`` for each of the
            model states, i.e., ``log(p(X|state))``.
        """
        if (self._compute_likelihood  # prevent recursion
                != __class__._compute_likelihood.__get__(self)):
            # Probabilities equal to zero do occur, and log(0) = -inf is OK.
            likelihood = self._compute_likelihood(X)
            with np.errstate(divide="ignore"):
                return np.log(likelihood)
        else:
            raise NotImplementedError("Must be overridden in subclass")

    def _generate_sample_from_state(self, state, random_state):
        """
        Generate a random sample from a given component.

        Parameters
        ----------
        state : int
            Index of the component to condition on.
        random_state: RandomState
            A random number generator instance.  (`sample` is the only caller
            for this method and already normalizes *random_state*.)

        Returns
        -------
        X : array, shape (n_features, )
            A random sample from the emission distribution corresponding
            to a given component.
        """
        return random_state.multivariate_normal(
            self.means_[state], self.covars_[state]
        )

    def _do_estep(self, X, lengths):
        impl = {
            "scaling": self._fit_scaling,
            "log": self._fit_log,
        }[self.implementation]

        stats = self._initialize_sufficient_statistics()
        curr_logprob = 0
        for sub_X in utils.split_X_lengths(X, lengths):
            lattice, logprob, posteriors, fwdlattice, bwdlattice = impl(sub_X)
            self._accumulate_sufficient_statistics(
                stats, sub_X, lattice, posteriors, fwdlattice,
                bwdlattice)
            curr_logprob += logprob
        return stats, curr_logprob

    def do_em_mstep(self, stats):
        """
        Perform the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
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
