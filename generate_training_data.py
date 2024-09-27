import sys
import os

###########
#FOR USE ON HPC MUST DELETE FIRST TWO LINES OF SPS.PY FILE
###########

import numpy as np
import lbg_forecast.hyperparameters as hyp
import lbg_forecast.population_model as pop
import lbg_forecast.sps as sps
import lbg_forecast.priors_mass_func as pr
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ngals = int(sys.argv[1])
path = sys.argv[2]
run_count = int(sys.argv[3])


if(rank == 0):
    print("nprocesses: ", size)

hyperparameter_mu_bounds, hyperparameter_sigma_max = hyp.uniform_hyperparameter_bounds()
prior_parameters = hyp.sample_prior_parameters(1, hyperparameter_mu_bounds, hyperparameter_sigma_max)
redshift_mass_prior_parameters = pr.preload_prior_data()
sps_parameters = pop.generate_sps_parameters(ngals, prior_parameters[0,:], redshift_mass_prior_parameters, uniform_redshift_mass=True, uniform_logf=True)

photometry = sps.simulate_photometry(sps_parameters, "lsst", imf=1, nebem=True, zhistory=False, enable_mpi=True, lya_uncertainity=False, mpi_rank=rank, save_spec=True, run_count=run_count, path=path)
