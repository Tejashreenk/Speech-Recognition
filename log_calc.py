import math

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
