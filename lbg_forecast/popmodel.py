import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
import lbg_forecast.distributions as dstr
import lbg_forecast.priors as pr

#draws single galaxy sample given a set of hyper parmeters (hparams)
#hparams are found using sample_hyper_parameters() in hyperparams.py
#returns: 1D array with SPS parameters sampled from priors defined by hparams
def galaxy_population_model(hparams):
    
    i = 0
    realisation_list = []
    for hparam in hparams:

        if(i == 1): #sample age (dependent on redshift)
            tage = 10**dstr.sample_prior(hparam)
            zred = realisation_list[0]
            while(tage >  cosmo.age(zred).value):
                tage = 10**dstr.sample_prior(hparam)

            realisation_list.append(np.log10(tage))
        else: #all other params
            realisation_list.append(dstr.sample_prior(hparam))

        i+=1

    realisation = np.asarray(realisation_list)

    return realisation

# (VECTORISED)
def galaxy_population_model_vec(hparams, prior_params, nsamples):

    z_grid, logm_grid, priors, grid_params = prior_params
    z_samples, m_samples = pr.sample_priors(z_grid, logm_grid, priors, grid_params, nsamples, plotting=False)
    
    i = 0
    realisation_list = []
    for hparam in hparams:

        if(i == 0):
            realisation_list.append(np.vstack(z_samples))
            tuniv = []
            for z in z_samples:
                tuniv.append(cosmo.age(z).value)
            tuniv = np.reshape(np.log10(np.asarray(tuniv)), (nsamples,)) 
                
        elif(i == 1):
            realisation_list.append(np.transpose(dstr.sample_prior_vec(hparam, nsamples, vectorise_bounds=tuniv)))

        elif(i == 14):
            realisation_list.append(np.vstack(m_samples))
        
        else:
            realisation_list.append(np.vstack(dstr.sample_prior_vec(hparam, nsamples, vectorise_bounds=0)))

        i+=1

    realisation = np.reshape(np.transpose(np.asarray(realisation_list)), (nsamples, 15))

    return realisation
