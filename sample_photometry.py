import sys
import numpy as np
import lbg_forecast.emulator as em

path = sys.argv[1]
run = sys.argv[2]
batch_size = int(sys.argv[3])

# initalise fsps emulator
model = em.fsps_emulator(path[:-1])

sps_parameters = np.load(path[:-1]+"/sps_parameter_samples/sps_"+run+".npy")

photometry = []
for n in range(sps_parameters.shape[0]):
    print("Realisation: ", n+1, flush=True)
    source_photometry = model.mimic_photometry(sps_parameters[n, :, :], batch_size=batch_size)    
    photometry.append(source_photometry)

#save
photometry_data = np.asarray(photometry)
np.save(path+"/photo_samples/photo_"+run+".npy", photometry_data)
print("Complete.", flush=True)