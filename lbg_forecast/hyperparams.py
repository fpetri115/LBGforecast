import numpy as np
import lbg_forecast.distributions as dstr
import matplotlib.pyplot as plt
import math


def sample_hyper_parameters(bounds, sig_min=0.1):
    
    ##################################### mu_min, mu_max, sig_min, sig_max
    zred = dstr.sample_normal_hyperparams(bounds[0][0], bounds[0][1], sig_min, bounds[0][1]-bounds[0][0])
    logtage = dstr.sample_normal_hyperparams(bounds[1][0], bounds[1][1], sig_min, bounds[1][1]-bounds[1][0])
    logzsol = dstr.sample_normal_hyperparams(bounds[2][0], bounds[2][1], sig_min, bounds[2][1]-bounds[2][0])
    dust1 = dstr.sample_normal_hyperparams(bounds[3][0], bounds[3][1], sig_min, bounds[3][1]-bounds[3][0])
    dust2 = dstr.sample_normal_hyperparams(bounds[4][0], bounds[4][1], sig_min, bounds[4][1]-bounds[4][0])
    igm_factor = dstr.sample_normal_hyperparams(bounds[5][0], bounds[5][0], bounds[5][1], bounds[5][1]) #fixed
    gas_logu = dstr.sample_normal_hyperparams(bounds[6][0], bounds[6][1], sig_min, bounds[6][1]-bounds[6][0])
    gas_logz = dstr.sample_normal_hyperparams(bounds[7][0], bounds[7][1], sig_min, bounds[7][1]-bounds[7][0]) 
    fagn = dstr.sample_normal_hyperparams(bounds[8][0], bounds[8][1], sig_min, bounds[8][1]-bounds[8][0])
    imf1 = dstr.sample_normal_hyperparams(bounds[9][0], bounds[9][0], bounds[9][1], bounds[9][1]) #fixed
    imf2 = dstr.sample_normal_hyperparams(bounds[10][0], bounds[10][0], bounds[10][1], bounds[10][1]) #fixed
    imf3 = dstr.sample_normal_hyperparams(bounds[11][0], bounds[11][0], bounds[11][1], bounds[11][1]) #fixed
    logtau = dstr.sample_normal_hyperparams(bounds[12][0], bounds[12][1], sig_min, bounds[12][1]-bounds[12][0])
    loga = dstr.sample_normal_hyperparams(bounds[13][0], bounds[13][1], sig_min, bounds[13][1]-bounds[13][0])
    logb = dstr.sample_normal_hyperparams(bounds[14][0], bounds[14][1], sig_min, bounds[14][1]-bounds[14][0])
    logmass = dstr.sample_normal_hyperparams(bounds[15][0], bounds[15][1], sig_min, bounds[15][1]-bounds[15][0])

    hyperparams = np.array([zred, logtage, logzsol, dust1, dust2, igm_factor,
                            gas_logu, gas_logz, fagn, imf1, imf2, imf3,
                             logtau, loga, logb, logmass])
    
    return hyperparams

#bounds for hyperparams(mu_min, mu_max), for fixed priors: (mu, sig)
def define_hyperparameter_bounds( 
                           
    zred = np.array([0, 7]),
    logtage = np.array([-3, 1]),
    logzsol = np.array([-2.5, 0.5]),
    dust1 = np.array([0, 2]),
    dust2 = np.array([0, 2]),
    igm_factor = np.array([1, 0.25]), #fixed (mu, sigma)
    gas_logu = np.array([-4, -1]),
    gas_logz = np.array([-2.5, 0.5]),
    fagn = np.array([0, 10]),
    imf1 = np.array([1.3, 0.1]), #fixed (mu, sigma)
    imf2 = np.array([2.3, 0.1]), #fixed (mu, sigma)
    imf3 = np.array([2.3, 0.1]), #fixed (mu, sigma)
    logtau = np.array([-4, 1]),
    loga = np.array([-3, 3]),
    logb = np.array([-3, 3]),
    logmass = np.array([7, 13])):

    bounds = np.array([zred, logtage, logzsol, dust1, dust2, igm_factor,
                            gas_logu, gas_logz, fagn, imf1, imf2, imf3,
                             logtau, loga, logb, logmass])

    return bounds

def plot_hyperparameters(nsamples, bounds, sigmin=0.1, rows=5, nbins=20):

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

    names = np.array(["zred_mu", "zred_sig", "$\mathrm{log_{10}tage}$_mu", "$\mathrm{log_{10}tage}$_sig",
                      "logzsol_mu", "logzsol_sig", "dust1_mu", "dust1_sig", "dust2_mu", "dust2_sig",
                      "igm_factor_mu", "igm_factor_sig", "gas_logu_mu", "gas_logu_sig", 
                      "gas_logz_mu", "gas_logz_sig", "fagn_mu", "fagn_sig", "imf1_mu", "imf1_sig",
                      "imf2_mu", "imf2_sig", "imf3_mu", "imf3_sig", "$\mathrm{log_{10}}tau$_mu",
                      "$\mathrm{log_{10}}tau$_sig", "$\mathrm{log_{10}}a$_mu", "$\mathrm{log_{10}}a$_sig", 
                        "$\mathrm{log_{10}}b$_mu", "$\mathrm{log_{10}}b$_sig", "$\mathrm{log_{10}mass}$_mu",
                        "$\mathrm{log_{10}mass}$_sig"])


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
            axes1[i, j].hist(tot_hyperparams[:,col], density = True, bins=nbins)
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




