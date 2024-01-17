import numpy as np
import lbg_forecast.distributions as dstr
import matplotlib.pyplot as plt
import math

from astropy.cosmology import WMAP9 as cosmo
import lbg_forecast.distributions as dstr

# sample hyperparameters given bounds from define_hyperparameter_bounds()
# minimum varaince given by sig_min
# returns: N x 3 nested array
#   first column contains (distr, a, b) for each SPS parameter passed from define_hyperparameter_bounds()
#   second and third contain sampled hyperparameters
#       if uniform this will just be a and b respectively
#       if gaussian this will be mu (between a and b) and a variance between sig_min and b-a respectively
def sample_hyper_parameters(bounds, sig_min=1e-6):

    i = 0
    hyperparams_list = []
    for bound in bounds:

        if(i == 0 or i == 14):
            hyperparams_list.append(dstr.sample_external_hyperparams(bound))
        else:
            hyperparams_list.append(dstr.sample_hyperparams(bound, sig_min))

        i+=1

    hyperparams = np.asarray(hyperparams_list)
    
    return hyperparams

def sample_nhyperparameters(bounds, nsamples):

    hyperparams_list = []
    i = 0
    while(i < nsamples):
        hyperparams_list.append(sample_hyper_parameters(bounds, sig_min=1e-6))
        i+=1
        
    hyperparams = np.asarray(hyperparams_list)
    return hyperparams

# bounds for hyperparams, each parameter has structure (distr, a, b)
# distr = 0 => uniform priors; distr = 1 => gaussian
# if distr = 0 (uniform) a and b are min and max of uniform distribution respectively
# if distr = 1 (gaussian) a and b are mu_min and mu_max of gaussian prior respectively
# returns: N x 3 array containing (distr, a, b) for each (N) SPS parameters
# REDSHIFT SHOULD ALWAYS BE FIRST, LOGMASS ALWAYS LAST
# IGM_FACTOR CURRENTLY NOT IN USE, HARDCODED IN SPS.PY
def define_hyperparameter_bounds( 
                           
    zred = np.array([0, 0, 7]),
    logtage = np.array([1, -3, 1]),
    logzsol = np.array([1, -2.5, 0.5]),
    dust1_frac = np.array([1, 0, 2]),
    dust2 = np.array([1, 0, 4]),
    igm_factor = np.array([0, 1, 1]), 
    gas_logu = np.array([1, -4, -1]),
    logfagn = np.array([0, 0, 0]),
    imf1 = np.array([0, 1.3, 1.3]), 
    imf2 = np.array([0, 2.3, 2.3]), 
    imf3 = np.array([0, 2.3, 2.3]), 
    logtau = np.array([1, -3, 1]),
    loga = np.array([1, -3, 1]),
    logb = np.array([1, -3, 1]),
    logmass = np.array([0, 7, 13])):

    bounds = np.array([zred, logtage, logzsol, dust1_frac, dust2, igm_factor,
                            gas_logu, logfagn, imf1, imf2, imf3,
                             logtau, loga, logb, logmass])

    return bounds

#for plotting hyperparameters given bounds (NO LONGER WORKS due to changes to zred and logmass and tau)
def _plot_hyperparameters(nsamples, bounds, sigmin=1e-6, rows=5, nbins=20):

    nparams = 2*len(bounds)
    columns = math.ceil(nparams/rows)
    total_plots = nparams
    grid = rows*columns

    i = 0
    tot_hyperparams = []
    while(i < nsamples):
        hyperparams = np.hstack(sample_hyper_parameters(bounds, sigmin))
        tot_hyperparams.append(hyperparams)
        i+=1

    tot_hyperparams = np.vstack(np.asarray(tot_hyperparams))
    
    #delete every third column to remove unneeded arrays
    tot_hyperparams = np.delete(tot_hyperparams, list(range(0, tot_hyperparams.shape[1], 3)), axis=1)

    names = np.array(["zred_mu", "zred_sig", "$\mathrm{log_{10}tage}$_mu", "$\mathrm{log_{10}tage}$_sig",
                      "logzsol_mu", "logzsol_sig", "dust1_mu", "dust1_sig", "dust2_mu", "dust2_sig",
                      "igm_factor_mu", "igm_factor_sig", "gas_logu_mu", "gas_logu_sig", "logfagn_mu", "logfagn_sig", "imf1_mu", "imf1_sig",
                      "imf2_mu", "imf2_sig", "imf3_mu", "imf3_sig", "$\mathrm{log_{10}}tau$_mu",
                      "$\mathrm{log_{10}}tau$_sig", "$\mathrm{log_{10}}a$_mu", "$\mathrm{log_{10}}a$_sig", 
                        "$\mathrm{log_{10}}b$_mu", "$\mathrm{log_{10}}b$_sig", "$\mathrm{log_{10}mass}$_mu",
                        "$\mathrm{log_{10}mass}$_sig"])
    
    names_uni = np.array(["zred_min", "zred_max", "$\mathrm{log_{10}tage}$_min", "$\mathrm{log_{10}tage}$_max",
                      "logzsol_min", "logzsol_max", "dust1_min", "dust1_max", "dust2_min", "dust2_max",
                      "igm_factor_min", "igm_factor_max", "gas_logu_min", "gas_logu_max", "logfagn_min", "logfagn_max", "imf1_min", "imf1_max",
                      "imf2_min", "imf2_max", "imf3_min", "imf3_max", "$\mathrm{log_{10}}tau$_min",
                      "$\mathrm{log_{10}}tau$_max", "$\mathrm{log_{10}}a$_min", "$\mathrm{log_{10}}a$_max", 
                        "$\mathrm{log_{10}}b$_min", "$\mathrm{log_{10}}b$_max", "$\mathrm{log_{10}mass}$_min",
                        "$\mathrm{log_{10}mass}$_max"])


    fig1, axes1 = plt.subplots(rows, columns, figsize=(20,20), sharex=False, sharey=False)

    i = 0
    j = 0
    plot_no = 0
    name_count = 0
    col = 0
    bounds = np.hstack(bounds)
    bounds = np.delete(bounds, list(range(1, bounds.size, 3)))
    bounds = np.delete(bounds, list(range(1, bounds.size, 2)))
    bounds = np.repeat(bounds, 2)
    while(col < nparams):

        if(i > rows - 1):
            j+=1
            i=0

        if(plot_no > total_plots):
            axes1[i, j].set_axis_off()

        else:
            axes1[i, j].hist(tot_hyperparams[:,col], density = True, bins=nbins)

            #change label depending on distribution
            if(int(bounds[col]) == 0):
                axes1[i, j].set_xlabel(names_uni[name_count])

            if(int(bounds[col]) == 1):
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
    
    plt.tight_layout()




