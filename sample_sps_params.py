import sys
import numpy as np
import lbg_forecast.population_model as pop
import lbg_forecast.priors_gp_massfunc as gpmf
import lbg_forecast.dust_priors as dpr
import lbg_forecast.priors_gp_csfrd as gpsf
import joblib
from mpi4py import MPI
NSPS_PARAMS = 17

#initialise MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if(rank == 0):
    print("nprocesses: ", size)

#collect inputs
ngals = int(sys.argv[1])
nrealisations = int(sys.argv[2])
run = sys.argv[3]
path = sys.argv[4]
mean = int(sys.argv[5])

#load prior information
if(rank == 0):
    print("Loading Priors ... ", flush=True)
    if(mean==1):
        mass_function_prior = gpmf.MassFunctionPrior(path=path[:-1], mean=True)
        dust_prior = dpr.DustPriorNag(path=path[:-1])
        csfrd_prior = gpsf.CSFRDPrior(path=path[:-1])
    elif(mean==0):
        mass_function_prior = gpmf.MassFunctionPrior(path=path[:-1], mean=False)
        dust_prior = dpr.DustPriorNag(path=path[:-1])
        csfrd_prior = gpsf.CSFRDPrior(path=path[:-1])
    else:
        print(mean, type(mean))
        raise Exception("mean argument invalid")
else:
    mass_function_prior = None
    dust_prior = None
    csfrd_prior = None

#setup sfr emulator 
#sfr_emulator = joblib.load(path+'sfr_emulator/sfr_emulator.pkl')

#broadcast prior information to processes
mass_function_prior = comm.bcast(mass_function_prior, root=0)
dust_prior = comm.bcast(dust_prior, root=0)
csfrd_prior = comm.bcast(csfrd_prior, root=0)

#setup memory
sps_buf = np.zeros((nrealisations, ngals, NSPS_PARAMS))
recv_buf = None
if(rank == 0):
    recv_buf = np.zeros((nrealisations * size, ngals, NSPS_PARAMS))

#comm.barrier()

#sample SPS parameters
if(rank == 0):
    print("Begin Sampling ... ", flush=True)

for n in range(nrealisations):
    sps_params = pop.generate_sps_parameters(ngals, mass_function_prior, dust_prior, csfrd_prior, mean=mean, uniform_redshift_mass=False)
    sps_buf[n, :, :] = sps_params
    if(rank == 0):
        print("Realisation: ", n+1, flush=True)

if(rank == 0):
    print("Waiting For Other Processes ... ", flush=True)
#comm.barrier()

#gather arrays
comm.Gather(sps_buf, recv_buf, root=0)

if(rank == 0):
    print("Gather Finished ... ", flush=True)
#comm.barrier()

#save
if(rank == 0):
    np.save(path+"sps_parameter_samples/sps_"+run+".npy", recv_buf)
    print("Complete.", flush=True)