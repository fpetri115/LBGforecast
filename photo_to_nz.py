import sys
import lbg_forecast.nz as nz
import numpy as np

path = sys.argv[1]
run = sys.argv[2]
colours = int(sys.argv[3])

sps_parameters = np.load(path+"sps_parameter_samples/sps_"+run+".npy")
photometry = np.load(path+"photo_samples/photo_"+run+".npy")
nlssts = np.load(path+"sps_parameter_samples/nlsst_"+run+".npy")

nzs = []
trans = np.zeros((photometry.shape[0], 3))
ntot = photometry.shape[1]


if(colours==0):
    for n in range(photometry.shape[0]):
        lbg_nzs = nz.calculate_nzs_from_photometry(sps_parameters[n, :, :], photometry[n, :, :], colours=colours)
        for i in range(3):
            trans[n, i] = (len(lbg_nzs[i])/ntot)*nlssts[n]
        nzs.append(lbg_nzs)
        print("Realisation: ", n+1, flush=True)
elif(colours==1):
    cs = []
    for n in range(photometry.shape[0]):
        lbg_nzs, lbg_colours = nz.calculate_nzs_from_photometry(sps_parameters[n, :, :], photometry[n, :, :], colours=colours)
        for i in range(3):
            trans[n, i] = (len(lbg_nzs[i])/ntot)*nlssts[n]
        nzs.append(lbg_nzs)
        cs.append(lbg_colours)
        print("Realisation: ", n+1, flush=True)
    c_data = np.array(cs)
    np.save(path+"nz_samples/c_"+run+".npy", c_data)
else:
    raise Exception("colour value not valid")

#save
nz_data = np.asarray(nzs)

np.save(path+"nz_samples/nz_"+run+".npy", nz_data)
np.save(path+"nz_samples/trans_"+run+".npy", trans)
print("Complete.", flush=True)
