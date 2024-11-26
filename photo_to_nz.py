import sys
import lbg_forecast.nz as nz
import numpy as np

path = sys.argv[1]
batch_size = int(sys.argv[2])
run = sys.argv[3]

sps_parameters = np.load(path[:-1]+"/sps_parameter_samples/sps_"+run+".npy")
photometry = np.load(path[:-1]+"/photo_samples/photo_"+run+".npy")

nzs = []
for n in range(photometry.shape[0]):
    lbg_nzs = nz.calculate_nzs_from_photometry(sps_parameters[n, :, :], photometry[n, :, :])
    nzs.append(lbg_nzs)
    print("Realisation: ", n+1, flush=True)

#save
nz_data = np.asarray(nzs)
np.save(path+"/nz_samples/nz_"+run+".npy", nz_data)
print("Complete.", flush=True)