import sys

import numpy as np
import lbg_forecast.hyperparams as hyp
import lbg_forecast.nz_sim as nz
import lbg_forecast.priors as pr
import lbg_forecast.popmodel as pop
import lbg_forecast.emulator as em

nrealisations = int(sys.argv[1])
ngals = int(sys.argv[2])

# initalise fsps emulator
model = em.fsps_emulator()
sps_params = model._sps_params

#setup grids
dz, dlogm = 0.1, 0.05
z_min, z_max = 0.0, 15.0
logm_min, logm_max = 7.0, 13.0
grid_params = np.array([dz, dlogm, z_min, z_max, logm_min, logm_max])
z_grid, logm_grid = pr.setup_grids(grid_params)

#load mass and redshift priors
n_priors_samples = 1000
priors = pr.load_priors(z_grid, logm_grid, n_priors_samples, init_sample=100000)
prior_params = [z_grid, logm_grid, priors, grid_params]

#initialise hyperparameter bounds
hyper_param_bounds = hyp.define_hyperparameter_bounds( 
                           
    zred = np.array([n_priors_samples, 0, 7]),
    logtage = np.array([1, -3, 1]),
    logzsol = np.array([1, -2.5, 0.5]),
    dust1_frac = np.array([1, 0, 1]),
    dust2 = np.array([1, 0, 2]),
    igm_factor = np.array([0, 0, 2]), 
    gas_logu = np.array([1, -4, -1]),
    logfagn = np.array([0, 0, 0]),
    imf1 = np.array([0, 1.3, 1.3]), 
    imf2 = np.array([0, 2.3, 2.3]), 
    imf3 = np.array([0, 2.3, 2.3]), 
    logtau = np.array([1, -4, 1]),
    loga = np.array([1, -3, 3]),
    logb = np.array([1, -3, 3]),
    logmass = np.array([n_priors_samples, 7, 13])

    )

#sample hyperparameters
hyperparameters = hyp.sample_nhyperparameters(hyper_param_bounds, nrealisations)

#sample realisations of galaxy population
nz_data = []
sps_params_list = []
i = 0
while(i<nrealisations):
    sps_params = pop.galaxy_population_model_vec(hyperparameters[i], prior_params, ngals)
    print("run ", i, ":", "SPS Parameters Generated")
    nzs = nz.simulate_nzs(sps_params, model)
    print("run ", i, ":", "N(z)'s Generated")
    sps_params_list.append(sps_params)
    nz_data.append(nzs)
    i+=1

#save
nz_data = np.asarray(nz_data)
sps_params = np.asarray(sps_params_list)
np.save("emulated_redshifts.npy", np.asarray(nz_data))
np.save("emulated_redshifts_spsparams.npy", np.asarray(sps_params_list))