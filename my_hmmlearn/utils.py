import numpy as np
from scipy import special
import math

def normalize(a, axis=None):
    """
    Normalize the input array so that it sums to 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def log_normalize(a, axis=None):
    """
    Normalize the input array so that ``sum(exp(a)) == 1``.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    if axis is not None and a.shape[axis] == 1:
        # Handle single-state GMMHMM in the degenerate case normalizing a
        # single -inf to zero.
        a[:] = 0
    else:
        with np.errstate(under="ignore"):
            a_lse = special.logsumexp(a, axis, keepdims=True)
        a -= a_lse


def fill_covars(covars, covariance_type='full', n_components=1, n_features=1):
    if covariance_type == 'full':
        return covars
    elif covariance_type == 'diag':
        return np.array(list(map(np.diag, covars)))
    elif covariance_type == 'tied':
        return np.tile(covars, (n_components, 1, 1))
    elif covariance_type == 'spherical':
        # Regardless of what is passed in, we flatten in
        # and then expand it to the correct shape
        covars = np.ravel(covars)
        eye = np.eye(n_features)[np.newaxis, :, :]
        covars = covars[:, np.newaxis, np.newaxis]
        return eye * covars

def log_add(a, b):
    """
    Computes the log-add of two log-domain probabilities a and b using a stable numerical method.
    
    Steps:
    1. Sort a and b so that a <= b.
    2. Compute d = a - b.
    3. Check the condition on d:
       - If d < -log(-l0) (where l0 approximates -infinity), b is significantly larger than a.
    4. If the condition is met, return b. If not, proceed to compute the full log-add.
    
    """
    # Step 1: Ensure that a <= b
    if a > b:
        a, b = b, a
    
    # Step 2: Compute d = a - b (since we know a <= b, d will be non-positive)
    d = a - b
    # We use a very large negative value for l0, l0 is assumed to be large (e.g., -1 x 10^30)
    l0 = -745  # Approximating value where exp(x) underflows
    
    # Step 3. Check the condition on d:
    #  If d < −log(−l0) (where l0 approximates -infinity), b is significantly larger than a.
    if d < l0:
        # If d < −l0, return b because adding exp(a) to exp(b) does not change the value significantly
        return b
    else:
        # Step 5: Compute the full log-add operation (Equation 16)
        return b + math.log1p(math.exp(d))  # log1p(x) computes log(1 + x) more accurately for small x

def log_add_arr(ip_arr):
    """
    Computes the log-add of two log-domain probabilities a and b using a stable numerical method.
    
    Steps:
    1. Sort a and b so that a <= b.
    2. Compute d = a - b.
    3. Check the condition on d:
       - If d < -log(-l0) (where l0 approximates -infinity), b is significantly larger than a.
    4. If the condition is met, return b. If not, proceed to compute the full log-add.
    
    """
    summation = 0
    for value in ip_arr:
        summation =  log_add(summation,value)

    return  summation

def logaddexp(a, b):
    return np.logaddexp(a, b)

def logsumexp(v):
    max_val = np.max(v)
    if np.isinf(max_val):
        return max_val
    acc = np.sum(np.exp(v - max_val))
    return np.log(acc) + max_val


def logsumexp(a, axis=None):
    """Compute the log of the sum of exponentials of input elements."""
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(a - a_max), axis=axis)) + a_max.squeeze()

# def logsumexp(a):
#     a_max = np.max(a)
#     return np.log(np.sum(np.exp(a - a_max))) + a_max

def forward_scaling(startprob, transmat, frameprob):
    ns, nc = frameprob.shape
    if startprob.shape[0] != nc or transmat.shape[0] != nc or transmat.shape[1] != nc:
        raise ValueError("Shape mismatch")

    fwdlattice = np.zeros((ns, nc))
    scaling = np.zeros(ns)
    fwdlattice[0, :] = startprob * frameprob[0, :]
    scale = scaling[0] = 1.0 / np.sum(fwdlattice[0, :])
    fwdlattice[0, :] *= scale

    for t in range(1, ns):
        for j in range(nc):
            fwdlattice[t, j] = np.sum(fwdlattice[t-1, :] * transmat[:, j]) * frameprob[t, j]
        scale = scaling[t] = 1.0 / np.sum(fwdlattice[t, :])
        fwdlattice[t, :] *= scale

    log_prob = -np.sum(np.log(scaling))
    return log_prob, fwdlattice, scaling

def forward_log(startprob, transmat, log_frameprob):
    ns, nc = log_frameprob.shape
    if startprob.shape[0] != nc or transmat.shape[0] != nc or transmat.shape[1] != nc:
        raise ValueError("Shape mismatch")
    
    fwdlattice = np.full((ns, nc), -np.inf)  # Initialize with -inf for log space
    fwdlattice[0, :] = np.log(startprob) + log_frameprob[0, :]
    
    for t in range(1, ns):
        for j in range(nc):
            buffer = fwdlattice[t-1, :] + transmat[:, j]
            fwdlattice[t, j] = special.logsumexp(buffer) + log_frameprob[t, j]
    
    log_prob = special.logsumexp(fwdlattice[-1, :])
    return log_prob, fwdlattice


def backward_scaling(startprob, transmat, frameprob, scaling):
    ns, nc = frameprob.shape
    if startprob.shape[0] != nc or transmat.shape[0] != nc or transmat.shape[1] != nc or scaling.shape[0] != ns:
        raise ValueError("Shape mismatch")
    
    bwdlattice = np.zeros((ns, nc))
    bwdlattice[-1, :] = scaling[-1]

    for t in range(ns - 2, -1, -1):
        for i in range(nc):
            for j in range(nc):
                bwdlattice[t, i] += transmat[i, j] * frameprob[t + 1, j] * bwdlattice[t + 1, j]
            bwdlattice[t, i] *= scaling[t]

    return bwdlattice

def backward_log(startprob, transmat, log_frameprob):
    ns, nc = log_frameprob.shape
    log_transmat =  np.log(transmat)
    if startprob.shape[0] != nc or transmat.shape[0] != nc or transmat.shape[1] != nc:
        raise ValueError("Shape mismatch")
    
    bwdlattice = np.full((ns, nc), -np.inf)  # Initialize with -inf for log space
    bwdlattice[-1, :] = 0  # Terminal condition in log space is 0 (log(1))

    for t in range(ns - 2, -1, -1):
        for i in range(nc):
            buffer = np.array([log_transmat[i, j] + log_frameprob[t + 1, j] + bwdlattice[t + 1, j] for j in range(nc)])
            bwdlattice[t, i] = special.logsumexp(buffer)

    return bwdlattice

import numpy as np

def compute_scaling_xi_sum(fwdlattice, transmat, bwdlattice, frameprob):
    ns, nc = frameprob.shape
    if (fwdlattice.shape != (ns, nc) or 
        transmat.shape != (nc, nc) or 
        bwdlattice.shape != (ns, nc)):
        raise ValueError("Shape mismatch")

    xi_sum = np.zeros((nc, nc))

    for t in range(ns - 1):
        for i in range(nc):
            for j in range(nc):
                xi_sum[i, j] += (fwdlattice[t, i] *
                                 transmat[i, j] *
                                 frameprob[t + 1, j] *
                                 bwdlattice[t + 1, j])

    return xi_sum

def compute_log_xi_sum(fwdlattice, transmat, bwdlattice, log_frameprob):
    ns, nc = log_frameprob.shape
    if (fwdlattice.shape != (ns, nc) or
        transmat.shape != (nc, nc) or
        bwdlattice.shape != (ns, nc)):
        raise ValueError("Shape mismatch")

    log_transmat = np.log(transmat)
    log_prob = special.logsumexp(fwdlattice[-1, :])
    log_xi_sum = np.full((nc, nc), -np.inf)

    for t in range(ns - 1):
        for i in range(nc):
            for j in range(nc):
                log_xi = (fwdlattice[t, i] +
                          log_transmat[i, j] +
                          log_frameprob[t + 1, j] +
                          bwdlattice[t + 1, j] -
                          log_prob)
                log_xi_sum[i, j] = logaddexp(log_xi_sum[i, j], log_xi)

    return log_xi_sum

def viterbi(log_startprob, log_transmat, log_frameprob):
    ns, nc = log_frameprob.shape
    if log_startprob.shape[0] != nc or log_transmat.shape[0] != nc or log_transmat.shape[1] != nc:
        raise ValueError("Shape mismatch")

    viterbi_lattice = np.full((ns, nc), -np.inf)
    state_sequence = np.zeros(ns, dtype=int)

    # Initialize the first column of the viterbi lattice
    viterbi_lattice[0, :] = log_startprob + log_frameprob[0, :]

    # Fill in the viterbi lattice
    for t in range(1, ns):
        for i in range(nc):
            max_prob = np.max(viterbi_lattice[t-1, :] + log_transmat[:, i])
            viterbi_lattice[t, i] = max_prob + log_frameprob[t, i]

    # Backtrace to find the optimal state sequence
    state_sequence[-1] = np.argmax(viterbi_lattice[-1, :])
    for t in range(ns - 2, -1, -1):
        state_sequence[t] = np.argmax(viterbi_lattice[t, :] + log_transmat[:, state_sequence[t + 1]])

    log_prob = np.max(viterbi_lattice[-1, :])
    return log_prob, state_sequence
