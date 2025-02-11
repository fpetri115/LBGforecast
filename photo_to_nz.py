import sys
import lbg_forecast.nz as nz
import numpy as np

path = sys.argv[1]
run = sys.argv[2]

sps_parameters = np.load(path+"sps_parameter_samples/sps_"+run+".npy")
photometry = np.load(path+"photo_samples/photo_"+run+".npy")
sparams = np.load(path+"sps_parameter_samples/sparams_"+run+".npy")

nzs = []
n_detected = np.zeros((photometry.shape[0], 3))
ntot = photometry.shape[1]

cs = []
params = []
for n in range(photometry.shape[0]):
    lbg_nzs, lbg_colours, lbg_params = nz.calculate_nzs_from_photometry(sps_parameters[n, :, :], photometry[n, :, :], extra=1)
    nzs.append(lbg_nzs)
    cs.append(lbg_colours)
    params.append(lbg_params)

    n_detected[n, :] = nz.calculate_number_densities(n, sps_parameters, lbg_params, sparams)
    print("Realisation: ", n+1, flush=True)

c_data = np.array(cs)
params_data = np.array(params)
np.save(path+"nz_samples/c_"+run+".npy", c_data)
np.save(path+"sps_parameter_samples/selected_sps_"+run+".npy", params_data)

#save
nz_data = np.asarray(nzs)

np.save(path+"nz_samples/nz_"+run+".npy", nz_data)
np.save(path+"nz_samples/n_detected_"+run+".npy", n_detected)
print("Complete.", flush=True)
