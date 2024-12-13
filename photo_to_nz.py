import sys
import lbg_forecast.nz as nz
import numpy as np

path = sys.argv[1]
run = sys.argv[2]
colours = int(sys.argv[3])

sps_parameters = np.load(path[:-1]+"/sps_parameter_samples/sps_"+run+".npy")
photometry = np.load(path[:-1]+"/photo_samples/photo_"+run+".npy")

nzs = []

if(colours==0):
    for n in range(photometry.shape[0]):
        lbg_nzs = nz.calculate_nzs_from_photometry(sps_parameters[n, :, :], photometry[n, :, :], colours=colours)
        nzs.append(lbg_nzs)
        print("Realisation: ", n+1, flush=True)
elif(colours==1):
    cs = []
    for n in range(photometry.shape[0]):
        lbg_nzs, lbg_colours = nz.calculate_nzs_from_photometry(sps_parameters[n, :, :], photometry[n, :, :], colours=colours)
        nzs.append(lbg_nzs)
        cs.append(lbg_colours)
        print("Realisation: ", n+1, flush=True)
    c_data = np.array(cs)
    np.save(path+"/nz_samples/c_"+run+".npy", c_data)
else:
    raise Exception("colour value not valid")

#save
nz_data = np.asarray(nzs)

np.save(path+"/nz_samples/nz_"+run+".npy", nz_data)
print("Complete.", flush=True)
