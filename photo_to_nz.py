import sys
import lbg_forecast.nz as nz
import numpy as np
import lbg_forecast.utils as utils
import lbg_forecast.priors_gp_massfunc as gpmf

path = sys.argv[1]
run = sys.argv[2]
extra = 1# int(sys.argv[3])

mass_function_prior = gpmf.MassFunctionPrior('.', mean=False)

sps_parameters = np.load(path+"sps_parameter_samples/sps_"+run+".npy")
photometry = np.load(path+"photo_samples/photo_"+run+".npy")
sparams = np.load(path+"sps_parameter_samples/sparams_"+run+".npy")

nzs = []
n_detected = np.zeros((photometry.shape[0], 3))
ntot = photometry.shape[1]

cs = []
params = []
for n in range(photometry.shape[0]):
    lbg_nzs, lbg_colours, lbg_params = nz.calculate_nzs_from_photometry(sps_parameters[n, :, :], photometry[n, :, :], extra=extra)
    for i in range(3):
        pass
        #n_detected[n, i] = mass_function_prior.normalised_mass_function()/utils.FULL_SKY_DEG2
    nzs.append(lbg_nzs)
    cs.append(lbg_colours)
    params.append(lbg_params)
    print("Realisation: ", n+1, flush=True)
c_data = np.array(cs)
params_data = np.array(params)
np.save(path+"nz_samples/c_"+run+".npy", c_data)
np.save(path+"sps_parameter_samples/selected_sps_"+run+".npy", params_data)

#save
nz_data = np.asarray(nzs)

np.save(path+"nz_samples/nz_"+run+".npy", nz_data)
#np.save(path+"nz_samples/trans_cut_"+run+".npy", trans_cut)
#np.save(path+"nz_samples/trans_"+run+".npy", trans)
print("Complete.", flush=True)
