import numpy as np

def MC_integral(target, samples):
    """
    Monte Carlo integration using posterior samples already available.
        ∫p(x)q(x)dx ~ ∑p(x_i)/N with x_i ~ q(x)
    
    target [p(x)] must have a pdf() method
    Lists of targets are also accepted.
    
    Arguments:
        list or class instance target: the probability density to evaluate. Must have a pdf() method.
        np.ndarray samples:            posterior samples already drawn from q(x)
    
    Return:
        double: integral value
    """
    # Check that target is iterable or callable:
    if not ((hasattr(target, 'pdf') or np.iterable(target)):
        raise Exception("target must be list of callables or have pdf method")
    # Number of p draws and methods check
    if np.iterable(target):
        if not np.all([hasattr(pi, 'pdf') for pi in target]):
            raise Exception("target must have pdf method")
        n_p = len(target)
        np.random.shuffle(target)
        iter_target = True
    else:
        if not hasattr(target, 'pdf'):
            raise Exception("p must have pdf method")
        iter_target = False
    # Integrals
    if iter_target:
        probabilities = np.array([pi.pdf(samples) for pi in target])
    else:
        probabilities = np.atleast_2d(target.pdf(samples))
    means = probabilities.mean(axis = 1)
    I = means.mean()
    return np.log(I)
