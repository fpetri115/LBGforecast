import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from prospect.models import priors_beta as pb
from prospect.models import transforms as ts
import lbg_forecast.gaussian_priors as gpr
import lbg_forecast.sfh as sfh
import math
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.stats import t
import lbg_forecast.modified_prospector_beta as mpb


def generate_sps_parameters(nsamples, mass_function_prior, dust_prior, csfrd_prior, mean, return_sparams=False, uniform_redshift_mass=False, uniform_logf=False):
    """Sample sps parameters given some prior parameters.
    """

    #samples gaussian priors
    mu_bounds, sig_bounds = gpr.default_bounds()
    mu, sigma = gpr.sample_gaussian_prior_parameters(mean, mu_bounds, sig_bounds)
    logzsol_mu, igm_factor_mu, gas_logu_mu, gas_logz_mu, fagn_mu, agn_tau_mu = mu
    logzsol_sigma, igm_factor_sigma, gas_logu_sigma, gas_logz_sigma, fagn_sigma, agn_tau_sigma = sigma

    if(uniform_redshift_mass):
        z_samples = np.random.uniform(0.0, 7.0, nsamples)
        m_samples = np.random.uniform(7, 13, nsamples)
    else:
        z_samples, m_samples, sparams = mass_function_prior.sample_logpdf(nsamples)

    sps_parameters = []

    #Redshift - zred
    redshift = z_samples

    #Stellar Metallicity - logzsol
    logzsol_min = -2.5
    logzsol_max = 0.5
    logzsol = truncated_normal(logzsol_mu, logzsol_sigma, logzsol_min, logzsol_max, nsamples)

    #IGM dust attenuation fudge factor - igm_factor
    igm_factor_min = 0.0
    igm_factor_max = 2.0
    igm_factor = truncated_normal(igm_factor_mu, igm_factor_sigma, igm_factor_min, igm_factor_max, nsamples) 

    #Gas ionisation parameter - gas_logu
    gas_logu_min = -4.0
    gas_logu_max = -1.0
    gas_logu = truncated_normal(gas_logu_mu, gas_logu_sigma, gas_logu_min, gas_logu_max, nsamples) 

    #Gas ionisation parameter - gas_logz
    gas_logz_min = -2.0
    gas_logz_max = 0.5
    gas_logz = truncated_normal(gas_logz_mu, gas_logz_sigma, gas_logz_min, gas_logz_max, nsamples) 

    #AGN fraction to luminosity - fagn
    fagn_min = -5.0
    fagn_max = 1.0
    fagn = truncated_normal(fagn_mu, fagn_sigma, fagn_min, fagn_max, nsamples) 

    #Optical depth of AGN torus - agn_tau
    agn_tau_min = 5
    agn_tau_max = 150
    agn_tau = truncated_normal(agn_tau_mu, agn_tau_sigma, agn_tau_min, agn_tau_max, nsamples) 
    
    #Total stellar mass formed in solar masses - mass
    mass = m_samples

    print("Sampling SFHs ...")

    #Log SFR ratios
    if(uniform_logf):
        log_sfr_ratios = np.random.uniform(-5.0, 5.0, (nsamples, 6))
    else:
        log_sfr_ratios = modified_prospector_beta_sfh_prior(csfrd_prior, redshift, mass, 0.3, mean, alpha=False)
    
    #dust params
    #recent_sfrs = sfr_emulator.predict(np.hstack((np.reshape(redshift, (nsamples, 1)), np.reshape(mass, (nsamples, 1)), log_sfr_ratios)))

    print("Sampling Dust ...")

    recent_sfrs = np.log10(sfh.calculate_recent_sfr(redshift, 10**mass, log_sfr_ratios))
    #dust2, dust_index, dust1 = dust_prior.sample_dust_model_irac(recent_sfrs)
    dust2, dust_index, dust1 = dust_prior.sample_dust_model_nag(recent_sfrs)
    #dust2, dust_index, dust1 = dust_prior.sample_dust_model_cosmos(recent_sfrs)

    sps_parameters.append(redshift)
    sps_parameters.append(logzsol)
    sps_parameters.append(dust1)
    sps_parameters.append(dust2)
    sps_parameters.append(dust_index)
    sps_parameters.append(igm_factor)
    sps_parameters.append(gas_logu)
    sps_parameters.append(gas_logz)
    sps_parameters.append(10**fagn)
    sps_parameters.append(agn_tau)

    ncols = log_sfr_ratios.shape[1]
    for column in range(ncols):
        sps_parameters.append(log_sfr_ratios[:, column])
    
    sps_parameters.append(10**mass)

    if(return_sparams):
        return np.transpose(np.array(sps_parameters)), sparams

    else:
        return np.transpose(np.array(sps_parameters))

def truncated_normal(mu, sigma, min, max, samples):
    """Samples truncated normal distribution from scipy
    """
    a, b = (min - mu) / sigma, (max - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=samples)

def modified_prospector_beta_sfh_prior(csfrd_prior, redshift, logmass, sigma, mean, alpha):
    """Each call of this function will sample a different expected csfrd. Based
    off prospector-beta prior (Wang et al. 2023)

    redshift and logmass are arrays, sigma is float
    """
    
    #initialise
    logsfrratios_samples = np.empty((redshift.shape[0], 6))
    if(mean==0):
        csfrd_sample = 10**(csfrd_prior.sample_prior_corrected())
    if(mean==1):
        csfrd_sample = 10**(csfrd_prior.get_prior_mean_corrected())
    
    csfrd_spline = mpb.get_csfrd_spline(csfrd_prior.lookback_times, csfrd_sample)[1]

    indx = 0
    for z, logm in zip(redshift, logmass):

        logsfrratios = mpb.sample_logsfrratios(csfrd_spline, z, logm, sigma, alpha)
        logsfrratios_samples[indx, :] = logsfrratios
        indx+=1

    return logsfrratios_samples

def prospector_beta_sfh_prior(nsamples, redshift, logmass, sigma):
    """Samples log SFR ratios from prospector-beta prior (Wang et al. 2023).
    nbins restricted to nbins=7 for use with sfh.default_agebins().
    """
    logsfrratios_samples = np.empty((nsamples, 6))
    for n in range(nsamples):

        samples = pb.DymSFHfixZred(zred=redshift,
                mass_mini=logmass-1e-3, mass_maxi=logmass+1e-3,
                z_mini=-1.98, z_maxi=0.19,
                logsfr_ratio_mini=-5.0, logsfr_ratio_maxi=5.0,
                logsfr_ratio_tscale=sigma, nbins_sfh=7,
                const_phi=True).sample()
        
        logsfrratios = ts.nzsfh_to_logsfr_ratios(samples)
        logsfrratios_samples[n, :] = logsfrratios

    return logsfrratios_samples

def continuity_prior(nsamples, nu, mu, sigma):
    """Samples log sfr ratios from student's t distributions
    for continuity SFH, TRUNCATED at [min, max]
    
    :param nsamples:
        Number of samples to from prior (int)

    :param nu:
        Student's t degrees of freedom parameter (int). Controls heaviness of tails

    :param mu:
        (nbins,) shape array giving mean of student's t for each bin

    :param sigma:
        (nbins,) shape array giving width of student's t for each bin

    :param min:
        minumum value of logsfr allowed
    
    :param max:
        minumum value of logsfr allowed

    :returns log_sf_ratios:
        (nbins,) shape array containing log star formation
        ratios. These can be passed to sfh.continuity_sfh()
    """
    nsfrs = len(mu)
    all_log_sfr_ratios = []

    #inverse transform sampling for each logsfrratio parameter
    for sfrs in range(nsfrs):
        all_log_sfr_ratios.append(sample_truncated_t(nsamples, nu, mu[sfrs], sigma[sfrs]))

    all_log_sfr_ratios = np.hstack(all_log_sfr_ratios)

    return all_log_sfr_ratios

def sample_truncated_t(nsamples, nu, mu, sigma, min=-5.0, max=5.0):
        """Returns truncated students't distribution samples as column vector
        """
        cdf_samples = np.random.uniform(t.cdf(min, nu, loc=mu, scale=sigma), t.cdf(max, nu, loc=mu, scale=sigma), size=(nsamples,))
        log_sfr_ratios = t.ppf(cdf_samples, nu, loc=mu, scale=sigma)
        return np.reshape(log_sfr_ratios, (nsamples, 1))


def sps_parameter_names():
    """Returns array of strings containing names of sps parameters.
    """

    names = np.array(["zred", "logzsol", "dust1", "dust2", "dust_index", 
                    "igm_factor", "gas_logu", "gas_logz", "logfagn", "agn_tau",
                    "logf1", "logf2", "logf3", "logf4", "logf5","logf6", "logmass"])
    
    return names

def plot_galaxy_population(sps_parameters, rows=5, nbins=20):
    
    realisations = sps_parameters
    nparams = realisations.shape[1]

    names = sps_parameter_names()
    
    if(len(names) != nparams):
        raise Exception("Number of parameters and parameter labels don't match")

    columns = math.ceil(nparams/rows)
    total_plots = nparams
    grid = rows*columns

    fig1, axes1 = plt.subplots(rows, columns, figsize=(20,20), sharex=False, sharey=False)

    i = 0
    j = 0
    plot_no = 0
    name_count = 0
    col = 0
    while(col < nparams):

        if(i > rows - 1):
            j+=1
            i=0

        if(plot_no > total_plots):
            axes1[i, j].set_axis_off()

        else:
            if(names[name_count] == "logmass" or names[name_count] == "logfagn"):
                axes1[i, j].hist(np.log10(realisations[:, col]), density = True, bins=nbins)
                axes1[i, j].set_xlabel(names[name_count])
                axes1[i, j].set_ylabel("$p(z)$")
            else:
                axes1[i, j].hist(realisations[:, col], density = True, bins=nbins)
                axes1[i, j].set_xlabel(names[name_count])
                axes1[i, j].set_ylabel("$p(z)$")
        i+=1
        plot_no += 1
        name_count += 1
        col += 1

    #clear blank figures
    no_empty_plots = grid - nparams
    i = 0
    while(i < no_empty_plots):
        axes1[rows - i - 1, columns - 1].set_axis_off()
        i+=1
