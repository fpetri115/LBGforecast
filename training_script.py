import sys

import numpy as np
import lbg_forecast.hyperparams as hyp
import lbg_forecast.tools as tools

ngals = int(sys.argv[1])

#uniform distribution for all
hyper_parameter_bounds = hyp.define_hyperparameter_bounds(
    
    zred = np.array([0, 0, 7]),
    logtage = np.array([0, -3, 1]),
    logzsol = np.array([0, -2.5, 0.5]),
    dust1 = np.array([0, 0, 2]),
    dust2 = np.array([0, 0, 2]),
    igm_factor = np.array([0, 0, 2]),#np.array([2, 0, 2]),
    gas_logu = np.array([0, -4, -1]),
    logfagn = np.array([0, -5, 1]),
    imf1 = np.array([0, 0.2, 2.6]), 
    imf2 = np.array([0, 1.0, 3.4]), 
    imf3 = np.array([0, 1.0, 3.4]), 
    logtau = np.array([0, -4, 1]),
    loga = np.array([0, -3, 3]),
    logb = np.array([0, -3, 3]),
    logmass = np.array([0, 7, 13])

)

hyperparameters = hyp.sample_hyper_parameters(hyper_parameter_bounds)
data = tools.simulate_photometry(ngals, hyperparameters, dust_type=0, imf_type=0, zhistory=True, nebem=True, filters='all')