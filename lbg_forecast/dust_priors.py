import numpy as np
import lbg_forecast.sfh as sfh
import lbg_forecast.priors_gp as gp
from scipy.stats import truncnorm



class DustPriorPop():
    def __init__(self, path):

        self.path=path
        self.preloaded_popcosmos_samples = np.load(self.path+"/dust_data/popcosmos_parameters_rmag_lt_25.npy")
        self.preloaded_recent_sfrs = np.load(self.path+"/dust_data/popcosmos_recentsfrs.npy")
    
    def sample_prior(self, nsamples):

        dust2=self.get_dust2()
        dust_index=self.get_dust_index()
        dust1=self.get_dust1()

        indexes=np.random.randint(0, self.preloaded_recent_sfrs.shape[0], nsamples)

        dust2_samples=dust2[indexes]
        dust_index_samples=dust_index[indexes]
        dust1_samples=dust1[indexes]

        return [dust2_samples, dust_index_samples, dust1_samples]
    
    def sample_prior_and_sfrs(self, nsamples):

        dust2=self.get_dust2()
        dust_index=self.get_dust_index()
        dust1=self.get_dust1()

        indexes=np.random.randint(0, self.preloaded_recent_sfrs.shape[0], nsamples)

        dust2_samples=dust2[indexes]
        dust_index_samples=dust_index[indexes]
        dust1_samples=dust1[indexes]
        sfr=self.preloaded_recent_sfrs[indexes]

        return [dust2_samples, dust_index_samples, dust1_samples, sfr]

    def get_recent_sfrs_samples(self, nsamples):
        return self.preloaded_recent_sfrs[np.random.randint(0, self.preloaded_recent_sfrs.shape[0], nsamples)]

    def get_lsrs(self):
        return self.preloaded_popcosmos_samples[:, 2:8]
    def get_dust_parameters(self):
        """dust2, dust_index, dust1/dust2"""
        return self.preloaded_popcosmos_samples[:, 8:11]
    def get_dust2(self):
        dust_samples=self.get_dust_parameters()
        return dust_samples[:, 0]
    def get_dust_index(self):
        dust_samples=self.get_dust_parameters()
        return dust_samples[:, 1]
    def get_dust1(self):
        dust_samples=self.get_dust_parameters()
        dust1frac = dust_samples[:, 2]
        dust2 = self.get_dust2()
        return dust1frac*dust2
    
class DustPriorNag():
    def __init__(self, path):

        self.path=path
        self.n, self.tau, self.tau1, self.ne, self.taue, self.tau1e, self.sfr = np.load(self.path+"/dust_data/saved_nagaraj22samples.npy")
    
    def sample_prior(self, nsamples):

        dust2=self.get_dust2()
        dust_index=self.get_dust_index()
        dust1=self.get_dust1()

        indexes=np.random.randint(0, self.sfr.shape[0], nsamples)

        dust2_samples=dust2[indexes]
        dust_index_samples=dust_index[indexes]
        dust1_samples=dust1[indexes]

        return [dust2_samples, dust_index_samples, dust1_samples]
    
    def sample_prior_and_sfrs(self, nsamples):

        dust2=self.get_dust2()
        dust_index=self.get_dust_index()
        dust1=self.get_dust1()

        indexes=np.random.randint(0, self.sfr.shape[0], nsamples)

        dust2_samples=dust2[indexes]
        dust_index_samples=dust_index[indexes]
        dust1_samples=dust1[indexes]
        sfr=self.sfr[indexes]

        return [dust2_samples, dust_index_samples, dust1_samples, sfr]

    def get_recent_sfrs_samples(self, nsamples):
        return self.sfr[np.random.randint(0, self.sfr.shape[0], nsamples)]

    def get_dust2(self):
        return self.tau
    def get_dust_index(self):
        return self.n
    def get_dust1(self):
        return self.tau1

class DustPrior():
    def __init__(self, path, samples=9999999999):

        self.path = path
        self.popcosmos_samples = self.initialise_popcosmos_samples(samples)
        self.number_of_samples = self.popcosmos_samples.shape[0]

    def initialise_popcosmos_samples(self, nsamples=999999999):

        popcosmos_samples = np.load(self.path+"/dust_data/popcosmos_parameters_rmag_lt_25.npy")[:nsamples, :]

        dust_samples = popcosmos_samples[:, 8:11]
        logsfrratios = popcosmos_samples[:, 2:8]
        redshifts = popcosmos_samples[:, -1]
        logmasses = popcosmos_samples[:, 0]
        recent_sfrs = np.log10(sfh.calculate_recent_sfr(redshifts, 10**logmasses, logsfrratios))[:nsamples]

        dust2 = dust_samples[:, 0]
        dust_index = dust_samples[:, 1]
        dust1frac = dust_samples[:, 2]
        dust1 = dust1frac*dust2

        dustparams = np.vstack((recent_sfrs, dust2, dust_index, dust1)).T

        return dustparams

    def draw_dust2(self, sfrs):

        recent_sfrs, dust2, dust_index, dust1 = extract_samples(self.popcosmos_samples)

        sorted_inds = recent_sfrs.argsort()[:]
        sorted_sfrs = recent_sfrs[sorted_inds]
        sorted_dust2 = dust2[sorted_inds]

        return np.interp(sfrs, sorted_sfrs, sorted_dust2)
    
    def draw_dust_index(self, dust2_samples):

        recent_sfrs, dust2, dust_index, dust1 = extract_samples(self.popcosmos_samples)

        sorted_inds = dust2.argsort()[:]
        sorted_dust2 = dust2[sorted_inds]
        sorted_dust_index = dust_index[sorted_inds]

        return np.interp(dust2_samples, sorted_dust2, sorted_dust_index)
    
    def draw_dust1(self, dust2_samples):

        recent_sfrs, dust2, dust_index, dust1 = extract_samples(self.popcosmos_samples)

        sorted_inds = dust2.argsort()[:]
        sorted_dust2 = dust2[sorted_inds]
        sorted_dust1 = dust1[sorted_inds]

        return np.interp(dust2_samples, sorted_dust2, sorted_dust1)
    
    def sample_dust_model(self, sfrs):
        """is this factorised??"""

        dust2 = self.draw_dust2(sfrs)
        dust_index = self.draw_dust_index(dust2)
        dust1 = self.draw_dust1(dust2)

        return [dust2, dust_index, dust1]


def extract_samples(dust_params):

    recent_sfrs = dust_params[:, 0]
    dust2 = dust_params[:, 1]
    dust_index = dust_params[:, 2]
    dust1 = dust_params[:, 3]

    return [recent_sfrs, dust2, dust_index, dust1]

def truncated_normal(mu, sigma, min, max, samples):
    """Samples truncated normal distribution from scipy
    """
    a, b = (min - mu) / sigma, (max - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=samples)

def dust_index_function(dust2):
    dust_index_mean = -0.095 + 0.111*dust2 - 0.0066*dust2*dust2
    return truncated_normal(dust_index_mean, 0.4, -2.2, 0.4, len(dust_index_mean))

def dust_ratio_prior(nsamples):
    mean = np.random.uniform(0.7, 1.3)
    sigma = np.random.uniform(0.01, 0.4)
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