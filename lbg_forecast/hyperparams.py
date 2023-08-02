import numpy as np
import lbg_forecast.distributions as dstr


def sample_hyper_parameters():
    
    ##################################### mu_min, mu_max, sig_min, sig_max
    zred = dstr.sample_normal_hyperparams(0, 7, 0.1, 7)
    logtage = dstr.sample_normal_hyperparams(-3, 1, 0.1, 4)
    logzsol = dstr.sample_normal_hyperparams(-2.5, 0.5, 0.1, 3)
    dust1 = dstr.sample_normal_hyperparams(0, 2, 0.1, 2)
    dust2 = dstr.sample_normal_hyperparams(0, 2, 0.1, 2)
    igm_factor = dstr.sample_normal_hyperparams(1, 1, 0.25, 0.25) #fixed
    gas_logu = dstr.sample_normal_hyperparams(-4, -1, 0.1, 3)
    gas_logz = dstr.sample_normal_hyperparams(-2.5, 0.5, 0.1, 3) 
    fagn = dstr.sample_normal_hyperparams(0, 10, 0.1, 10)
    imf1 = dstr.sample_normal_hyperparams(1.3, 1.3, 0.1, 0.1) #fixed
    imf2 = dstr.sample_normal_hyperparams(2.3, 2.3, 0.1, 0.1) #fixed
    imf3 = dstr.sample_normal_hyperparams(2.3, 2.3, 0.1, 0.1) #fixed
    logtau = dstr.sample_normal_hyperparams(-4, 1, 0.1, 5)
    loga = dstr.sample_normal_hyperparams(-3, 3, 0.1, 6)
    logb = dstr.sample_normal_hyperparams(-3, 3, 0.1, 6)
    logmass = dstr.sample_normal_hyperparams(7, 13, 0.1, 6)

    hyperparams = np.array([zred, logtage, logzsol, dust1, dust2, igm_factor,
                            gas_logu, gas_logz, fagn, imf1, imf2, imf3,
                             logtau, loga, logb, logmass])
    
    return hyperparams
