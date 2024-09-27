import numpy as np
import lbg_forecast.sfh as sfh
import lbg_forecast.priors_gp as gp
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
    mean = np.random.uniform(0.8, 1.2)
    sigma = np.random.uniform(0.1, 0.4)
    return truncated_normal(mean, sigma, 0.0, 2.0, nsamples)

def sample_dust1(dust2):
    """optical depth"""
    dust_ratio = dust_ratio_prior(dust2.shape[0])
    return dust_ratio*dust2

def a_to_tau(a):
    return 0.92103*a
def tau_to_a(tau):
    return 1.0857*tau

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

def sample_dust_model(redshift, logmass, logsfrratios, return_sfrs=False):

    diffuse_dust_prior = gp.DiffuseDustPrior()
    index_prior = gp.DustIndexPrior()

    recent_sfrs = np.log10(sfh.calculate_recent_sfr(redshift, 10**logmass, logsfrratios))
    dust2_av = a_to_tau(diffuse_dust_prior.sample_dust2(recent_sfrs))
    dust_index = index_prior.sample_dust_index(dust2_av)
    dust2 = a_to_tau(dust2_av)
    dust1 = sample_dust1(dust2)

    if(return_sfrs):
        return dust_index, dust1, dust2, recent_sfrs
    else:
        return dust_index, dust1, dust2