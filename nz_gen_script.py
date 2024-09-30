import sys

from mpi4py import MPI
import os
import numpy as np
import lbg_forecast.nz as nz
import lbg_forecast.emulator as em
import lbg_forecast.hyperparameters as hyp
import lbg_forecast.population_model as pop
import lbg_forecast.priors_mass_func as pr
import matplotlib.pyplot as plt
import scipy as sc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if(rank == 0):
    print("nprocesses: ", size)

path = sys.argv[1]
nhypersamples = int(sys.argv[2])
ngals = int(sys.argv[3])
batch_size = int(sys.argv[4])

# initalise fsps emulator
model = em.fsps_emulator(path)

#hyperparams
bounds = hyp.hyperparameter_tighter()
hyperparameter_samples = np.vsplit(hyp.sample_prior_parameters(nhypersamples, bounds[0], bounds[1]), nhypersamples)

#load mass and redshift priors
prior_data = pr.preload_prior_data(zmax=7.0)

nzs = []
spsp = []
i = 0
for row in hyperparameter_samples:
    sps_params = pop.generate_sps_parameters(ngals, row[0], prior_data, uniform_redshift_mass=False, uniform_logf=False)
    print("run ", i, ":", "SPS Parameters Generated", flush=True)
    lbg_nzs = nz.simulate_nzs(sps_params, model, batch_size)
    print("run ", i, ":", "N(z)'s Generated", flush=True)
    nzs.append(lbg_nzs)
    spsp.append(sps_params)
    i+=1

#save
nz_data = np.asarray(nzs)
sps_params = np.asarray(sps_params)
np.save(path+"/redshifts/emulated_redshifts_"+str(rank)+".npy", nz_data)
np.save(path+"/redshifts/emulated_redshifts_spsparams_"+str(rank)+".npy", sps_params)
print("saved", flush=True)