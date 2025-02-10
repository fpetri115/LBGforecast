import sys
import lbg_forecast.nz as nz
import numpy as np
import lbg_forecast.utils as utils

path = sys.argv[1]
run = sys.argv[2]
extra = 1# int(sys.argv[3])

sps_parameters = np.load(path+"sps_parameter_samples/sps_"+run+".npy")
photometry = np.load(path+"photo_samples/photo_"+run+".npy")
totn_cut = np.load(path+"sps_parameter_samples/totn_cut_"+run+".npy")
totn = np.load(path+"sps_parameter_samples/totn_"+run+".npy")

nzs = []
trans_cut = np.zeros((photometry.shape[0], 3))
trans = np.zeros((photometry.shape[0], 3))
ntot = photometry.shape[1]


if(extra==0):
    for n in range(photometry.shape[0]):
        lbg_nzs = nz.calculate_nzs_from_photometry(sps_parameters[n, :, :], photometry[n, :, :], extra=extra)
        for i in range(3):
            nlbgs = len(lbg_nzs[i])
            trans_cut[n, i] = (nlbgs/ntot)*totn_cut[n]
            trans[n, i] = (nlbgs/ntot)*totn[n]
        nzs.append(lbg_nzs)
        print("Realisation: ", n+1, flush=True)
elif(extra==1):
    cs = []
    params = []
    for n in range(photometry.shape[0]):
        lbg_nzs, lbg_colours, lbg_params = nz.calculate_nzs_from_photometry(sps_parameters[n, :, :], photometry[n, :, :], extra=extra)
        tot_m_cut = np.where(np.log10((np.squeeze(sps_parameters[n, :, -1]))) > 8)[0].shape[0]
        for i in range(3):
            nlbgs = len(lbg_nzs[i])
            lbgs_m = np.log10(np.squeeze(lbg_params[i])[:, -1])
            n_lbgs_cut = np.where(lbgs_m > 8)[0].shape[0]
            print(n_lbgs_cut, tot_m_cut, totn_cut[n], nlbgs, ntot, totn[n])
            trans_cut[n, i] = ((n_lbgs_cut/tot_m_cut)*totn_cut[n])/utils.FULL_SKY_DEG2
            trans[n, i] = ((nlbgs/ntot)*totn[n])/utils.FULL_SKY_DEG2
        nzs.append(lbg_nzs)
        cs.append(lbg_colours)
        params.append(lbg_params)
        print("Realisation: ", n+1, flush=True)
    c_data = np.array(cs)
    params_data = np.array(params)
    np.save(path+"nz_samples/c_"+run+".npy", c_data)
    np.save(path+"sps_parameter_samples/selected_sps_"+run+".npy", params_data)
else:
    raise Exception("extra value not valid")

#save
nz_data = np.asarray(nzs)

np.save(path+"nz_samples/nz_"+run+".npy", nz_data)
np.save(path+"nz_samples/trans_cut_"+run+".npy", trans_cut)
np.save(path+"nz_samples/trans_"+run+".npy", trans)
print("Complete.", flush=True)
