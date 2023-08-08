import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
import lbg_forecast.distributions as dstr

def galaxy_population_model_dpl(hparams):

    i = 0
    realisation_list = []
    for hparam in hparams:

        if(i == 1): #sample age (dependent on redshift)
            logtage = dstr.sample_prior(hparam)
            zred = realisation_list[0]
            while(logtage >  cosmo.age(zred).value):
                logtage = dstr.sample_prior(hparam)

            realisation_list.append(logtage)

        else: #all other params
            realisation_list.append(dstr.sample_prior(hparam))

    realisation = np.asarray(realisation_list)

    return realisation

def draw_samples_from_population(nsamples, hparams):

    realisations = []
    i = 0
    while(i < nsamples):
        realisations.append(galaxy_population_model_dpl(hparams))
        i+=1

    realisations = np.array(realisations)
    realisations = np.vstack(realisations) #column for each parameter

    return realisations

def plot_galaxy_population(nsamples, hparams, rows=5, nbins=20):

    realisations = draw_samples_from_population(nsamples, hparams)
    nparams = realisations.shape[1]

    names = np.array(["zred", "$\mathrm{log_{10}tage}$", "logzsol", "dust1", "dust2", 
                      "igm_factor", "gas_logu", "gas_logz", "fagn", "imf1",
                        "imf2", "imf3", "$\mathrm{log_{10}}tau$", "$\mathrm{log_{10}}a$", 
                        "$\mathrm{log_{10}}b$", "$\mathrm{log_{10}mass}$"])
    
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
            axes1[i, j].hist(realisations[:,col], density = True, bins=nbins)
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

