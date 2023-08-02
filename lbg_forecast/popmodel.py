import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
import lbg_forecast.distributions as dstr

def galaxy_population_model_dpl():

    zred = dstr.sample_normal(0.5, 2, 0, 7)

    logtage = dstr.sample_normal(1, 1, -6, 1)
    if(logtage >  cosmo.age(zred).value):
        logtage = dstr.sample_normal(1, 1, -6, 1)
    
    logzsol = dstr.sample_normal(-1, 0.5, -2.5, 0.5)
    dust1 = dstr.sample_normal(0.1, 0.5, 0, 2)
    dust2 = dstr.sample_normal(0.5, 0.5, 0, 2)
    igm_factor = dstr.sample_normal(1, 0.25, 0, 99)
    gas_logu = dstr.sample_normal(-2, 0.25, -4, -1)
    gas_logz = dstr.sample_normal(-1, 0.25, -2.5, -0.5) #log(z/zsol)
    fagn = dstr.sample_normal(1, 0.25, 0, 10)
    imf1 = dstr.sample_normal(1.3, 0.1, 0.3, 2.3)
    imf2 = dstr.sample_normal(2.3, 0.1, 1.3, 3.3)
    imf3 = dstr.sample_normal(2.3, 0.1, 1.3, 3.3)
    logtau = dstr.sample_normal(0, 1, -4, 1)
    loga = dstr.sample_normal(0, 1, -3, 3)
    logb = dstr.sample_normal(0, 1, -3, 3)
    logmass = dstr.sample_normal(11, 2, 7, 13)

    realisation = np.array([zred, logtage, logzsol, dust1, dust2, igm_factor,
                            gas_logu, gas_logz, fagn, imf1, imf2, imf3,
                             logtau, loga, logb, logmass])

    return realisation

def draw_samples_from_population(nsamples):

    realisations = []
    i = 0
    while(i < nsamples):
        realisations.append(galaxy_population_model_dpl())
        i+=1

    realisations = np.array(realisations)
    realisations = np.vstack(realisations) #column for each parameter

    return realisations

def plot_galaxy_population(nsamples, rows=5, nbins=20):

    realisations = draw_samples_from_population(nsamples)
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





##########old
def galaxy_population_model(nsamples, pop_params):

    zmin = pop_params[0]

    tage = np.random.uniform(2, 2, nsamples)
    tau = np.random.uniform(10000, 10000, nsamples)
    const = np.random.uniform(1.0, 1.0, nsamples)
    zred = np.random.uniform(3, 3, nsamples)
    logzsol = np.random.uniform(-0.1, -0.1, nsamples)
    dust1 = np.random.uniform(0.0, 0.0, nsamples)
    dust2 = np.random.uniform(0.0, 0.0, nsamples)
    tburst = np.random.uniform(11, 11, nsamples)
    fburst = np.random.uniform(0.0, 0.0, nsamples)
    igm_factor = np.random.normal(1, 0.25, nsamples)
    gas_logu = np.random.uniform(-2.0, -2.0, nsamples)
    gas_logz = np.random.uniform(0.0, 0.0, nsamples)
    fagn = np.random.uniform(1, 1, nsamples)
    imf1 = np.random.uniform(1.3, 1.3, nsamples)
    imf2 = np.random.uniform(2.3, 2.3, nsamples)
    imf3 = np.random.uniform(2.3, 2.3, nsamples)
    mass = np.random.uniform(1e11, 1e11, nsamples)

    realisation = {
        'tage': tage,
        'tau': tau,
        'const': const,
        'zred': zred,
        'logzsol': logzsol,
        'dust1': dust1,
        'dust2': dust2,
        'tburst': tburst,
        'fburst': fburst,
        'igm_factor': igm_factor,
        'gas_logu': gas_logu,
        'gas_logz': gas_logz,
        'fagn': fagn,
        'imf1': imf1,
        'imf2': imf2,
        'imf3': imf3,
        'mass': mass
    }

    return realisation
