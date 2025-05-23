import lbg_forecast.sps as sps
import lbg_forecast.sfh as sfh
import lbg_forecast.cosmology as cosmo
import sys
import numpy as np

from mpi4py import MPI

NSPS_PARAMS = 17

#initialise MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if(rank == 0):
    print("nprocesses: ", size)

#collect inputs
run = sys.argv[1]
path = sys.argv[2]
filters = sys.argv[3]

if(filters=="lsst"):
    NBANDS=6
elif(filters=="suprimecam"):
    NBANDS=5
else:
    raise Exception("invalid filters")

if(rank==0):
    loaded_sps_params_train = np.vstack((np.load(path+"sps_parameter_samples/sps_"+run+".npy")))
    print(loaded_sps_params_train.shape)

    if(loaded_sps_params_train.shape[0]%size != 0):
        raise Exception("wrong size")
    
    split_sps_params_train = np.array_split(loaded_sps_params_train, size)
    flat_split_sps_params_train = [np.ravel(mat) for mat in split_sps_params_train]
    flat_split_sps_params_train = np.concatenate(flat_split_sps_params_train)

    data_len = split_sps_params_train[0].shape[0]

else:
    loaded_sps_params_train = None
    data_len = None
    split_sps_params_train = None
    flat_split_sps_params_train = None

data_len = comm.bcast(data_len, root=0)
sps_params_train = np.zeros((data_len, NSPS_PARAMS))

comm.Scatter(flat_split_sps_params_train, sps_params_train, root=0)

#setup memory
phot_buf = np.zeros((data_len*NBANDS))
recv_buf = None
if(rank == 0):
    recv_buf = np.zeros((data_len*NBANDS*size))#np.zeros((int(data_len*size), NBANDS))

if(rank == 0):
    print("Initialising SPS Model ...", flush=True)
sps_model=sps.initialise_sps_model(neb_em=True, imf_type=1, sfh_type=3, zcont=1)
if(rank == 0):
    print("Initialisation Complete.", flush=True)
    print("Calculating...", flush=True)
phot_true = []
for i in range(sps_params_train.shape[0]):

    sps.update_model(sps_model, sps_params_train[i, :], False, sfh.default_agebins())
    phot_sps = sps.get_magnitudes(sps_model, filters=filters, cosmology=cosmo.get_wmap9(), lya_uncertainity=False, path=path, return_spec=False, modify_igm=False)
    phot_true.append(phot_sps)
    if(i%1000 == 0):
        print(i)

phot_buf = np.ravel(np.vstack(phot_true))
comm.Gather(phot_buf, recv_buf, root=0)

if(rank==0):

    photometry_data = np.vstack(np.array_split(recv_buf, size*data_len))
    np.save(path+"photo_samples/sim_photo_"+run+filters+".npy", photometry_data)
    print("Complete.", flush=True)