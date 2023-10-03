import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
import lbg_forecast.distributions as dstr

#draws single galaxy sample given a set of hyper parmeters (hparams)
#hparams are found using sample_hyper_parameters() in hyperparams.py
#returns: 1D array with SPS parameters sampled from priors defined by hparams
def galaxy_population_model(hparams):

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
