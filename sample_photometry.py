import sys
import numpy as np
import lbg_forecast.emulator as em

path = sys.argv[1]
run = sys.argv[2]

# initalise fsps emulator
model = em.fsps_emulator(path[:-1])

sps_parameters = np.load(path[:-1]+"/sps_parameter_samples/sps_"+run+".npy")

photometry = []
for n in range(sps_parameters.shape[0]):
    source_photometry = model.mimic_photometry(sps_parameters[n, :, :])    
    photometry.append(source_photometry)
    print("Realisation: ", n+1, flush=True)

#save
photometry_data = np.asarray(photometry)
np.save(path+"/photo_samples/photo_"+run+".npy", photometry_data)
print("Complete.", flush=True)