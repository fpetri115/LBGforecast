import numpy as np
from prospect.models import priors_beta as pb
from prospect.models.priors_beta import DymSFHfixZred
from prospect.models import priors
from prospect.models import transforms as ts
import lbg_forecast.priors_gp_csfrd as gp
from scipy.interpolate import UnivariateSpline
import lbg_forecast.cosmology as cosmology
import lbg_forecast.population_model as pop
from scipy.stats import t
import matplotlib.pyplot as plt


def sample_logsfrratios(csfrd_spline, redshift, logmass, sigma, alpha):

    logsfr_ratios = expe_logsfr_ratios_modified(csfrd_spline, this_z=redshift, this_m=logmass, nbins_sfh=7,
                                        logsfr_ratio_mini=-5,
                                        logsfr_ratio_maxi=5, alpha=alpha)
    
    logsfr_ratios_rvs = pop.continuity_prior(1, 2, logsfr_ratios, np.array([sigma]*logsfr_ratios.shape[0]))
    return logsfr_ratios_rvs

def get_csfrd_spline(lookback_times, csfrd_sample):
    csfrd_sample_spline = UnivariateSpline(lookback_times, csfrd_sample, s=0, ext=3)
    return lookback_times, csfrd_sample_spline

def expe_logsfr_ratios_modified(csfrd_spline, this_z, this_m, logsfr_ratio_mini, logsfr_ratio_maxi,
                    nbins_sfh=7, amin=7.1295, alpha=False):
    """expectation values of logsfr_ratios
    """

    age_shifted = np.log10(pb.cosmo.age(this_z).value) + pb.delta_t_dex(this_m)
    age_shifted = 10**age_shifted

    zmin_thres = 0.15
    zmax_thres = 10
    if age_shifted < pb.age[-1]:
        z_shifted = zmax_thres * 1
    elif age_shifted > pb.age[0]:
        z_shifted = zmin_thres * 1
    else:
        z_shifted = pb.f_age_z(age_shifted)
    if z_shifted > zmax_thres:
        z_shifted = zmax_thres * 1
    if z_shifted < zmin_thres:
        z_shifted = zmin_thres * 1

    agebins_shifted = pb.z_to_agebins_rescale(zstart=z_shifted, nbins_sfh=nbins_sfh, amin=amin)

    nsfrbins = agebins_shifted.shape[0]
    sfr_shifted = np.zeros(nsfrbins)
    for i in range(nsfrbins):
        a = agebins_shifted[i,0]
        b = agebins_shifted[i,1]
        sfr_shifted[i] = csfrd_spline.integral(a=a, b=b)/(b-a)

    logsfr_ratios_shifted = np.zeros(nsfrbins-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(nsfrbins-1):
            logsfr_ratios_shifted[i] = np.log10(sfr_shifted[i]/sfr_shifted[i+1])
    logsfr_ratios_shifted = np.clip(logsfr_ratios_shifted, logsfr_ratio_mini, logsfr_ratio_maxi)

    if not np.all(np.isfinite(logsfr_ratios_shifted)):
        # set nan accord. to its neighbor
        nan_idx = np.isnan(logsfr_ratios_shifted)
        finite_idx = np.min(np.where(nan_idx==True))-1
        neigh = logsfr_ratios_shifted[finite_idx]
        nan_idx = np.arange(6-finite_idx-1) + finite_idx + 1
        for i in range(len(nan_idx)):
            logsfr_ratios_shifted[nan_idx[i]] = neigh * 1.

    if(alpha):
        return logsfr_ratios_shifted*0.0
    else:
        return logsfr_ratios_shifted