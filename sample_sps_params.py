import sys
import numpy as np
import lbg_forecast.population_model as pop
import lbg_forecast.priors_gp_massfunc as gpmf
import lbg_forecast.dust_priors as dpr
import lbg_forecast.priors_gp_csfrd as gpsf
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if(rank == 0):
    print("nprocesses: ", size)

ngals = int(sys.argv[1])
nrealisations = int(sys.argv[2])
run = int(sys.argv[3])
path = sys.argv[4]

if(rank == 0):
    print("Loading Priors ... ", flush=True)

mass_function_prior = gpmf.MassFunctionPrior()
dust_prior = dpr.DustPrior()
csfrd_prior = gpsf.CSFRDPrior()

if(rank == 0):
    print("Begin Sampling ... ", flush=True)

spsp = []
for n in range(nrealisations):
    sps_params = pop.generate_sps_parameters(ngals, mass_function_prior, dust_prior, csfrd_prior)
    spsp.append(sps_params)
    if(rank == 0):
        print("Realisation: ", n+1, flush=True)

if(rank == 0):
    print("Waiting For Other Processes ... ", flush=True)

all_realisations = comm.gather(spsp, root=0)

if(rank == 0):
    np.save(path+"/sps_parameter_samples/sps_"+str(run)+".npy",np.concatenate(np.array(all_realisations)))
    print("Complete.", flush=True)