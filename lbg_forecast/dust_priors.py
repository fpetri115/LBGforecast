import numpy as np
from scipy.stats import truncnorm

def truncated_normal(mu, sigma, min, max, samples):
    """Samples truncated normal distribution from scipy
    """
    a, b = (min - mu) / sigma, (max - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=samples)

def dust_index_function(dust2):
    dust_index_mean = -0.095 + 0.111*dust2 - 0.0066*dust2*dust2
    return truncated_normal(dust_index_mean, 0.4, -2.2, 0.4, dust_index_mean.shape[0])

def dust_ratio_prior(nsamples):
    return truncated_normal(1.0, 0.3, 0.0, 2.0, nsamples)

def dust2_function(sfr):
    """
    Parameters
    -----------
    sfr : ndarray of size (nsamples,) of recent sfr calculated 
    from sfh.calculate_recent_sfr() (CURRENTLY ONLY WORKS ONE SAMPLE AT A TIME)

    Returns
    ---------
    samples of dust2 sps parameter

    """
    dust2_mean = 0.2 + 0.5*np.log10(sfr)*np.heaviside(np.log10(sfr), 0.5)
    return truncated_normal(dust2_mean, 0.2, 0, 4.0, 1)[0]