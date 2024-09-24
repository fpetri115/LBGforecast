import numpy as np
import lbg_forecast.sfh as sfh
from scipy.stats import truncnorm

def truncated_normal(mu, sigma, min, max, samples):
    """Samples truncated normal distribution from scipy
    """
    a, b = (min - mu) / sigma, (max - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=samples)

def dust_index_function(dust2):
    dust_index_mean = -0.095 + 0.111*dust2 - 0.0066*dust2*dust2
    return truncated_normal(dust_index_mean, 0.4, -2.2, 0.4, len(dust_index_mean))

def dust_ratio_prior(nsamples):
    return truncated_normal(1.0, 0.3, 0.0, 2.0, nsamples)

def dust2_function(sfr):
    """
    Parameters
    -----------
    sfr : ndarray of size (nsamples,) of recent sfr calculated. Needs to be
    not logged, and not the sSFR, so use: sfh.calculate_recent_sfr(), 
    NOT sfh.calculate_recent_sfrs()!!

    Returns
    ---------
    samples of dust2 sps parameter

    """
    dust2_mean = 0.2 + 0.5*np.log10(sfr)*np.heaviside(np.log10(sfr), 0.5)
    dust2_mean = peturb_means(dust2_mean, 0.2)
    return truncated_normal(dust2_mean, 0.2, 0, 4.0, sfr.shape[0])

def peturb_means(means, pertubation):
    return means+np.random.uniform(-pertubation, pertubation)

def sample_dust_priors(redshift, mass, log_sfr_ratios):

    recent_sfrs = sfh.calculate_recent_sfr(redshift, 10**mass, log_sfr_ratios)
    dust2 = dust2_function(recent_sfrs)
    dust_index = dust_index_function(dust2)
    dust_ratio = dust_ratio_prior(dust2.shape[0])

    return dust2, dust_index, dust_ratio, recent_sfrs
