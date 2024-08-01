import numpy as np
from scipy.stats import truncnorm

def truncated_normal(mu, sigma, min, max, samples):
    """Samples truncated normal distribution from scipy
    """
    a, b = (min - mu) / sigma, (max - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=samples)

def dust_index_function(dust2):
    return -0.095 + 0.111*dust2 - 0.0066*dust2*dust2

def dust_ratio_prior(nsamples):
    return truncated_normal(1.0, 0.3, 0.0, 2.0, nsamples)