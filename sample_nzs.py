import sys
import numpy as np
import lbg_forecast.nz as nz
import lbg_forecast.emulator as em

path = sys.argv[1]
batch_size = int(sys.argv[2])
run = int(sys.argv[3])

# initalise fsps emulator
model = em.fsps_emulator(path)

sps_parameters = np.load(path+"/sps_parameter_samples/sps_"+str(run)+".npy")

nzs = []
batch_size=100
for n in range(sps_parameters.shape[0]):    
    lbg_nzs = nz.simulate_nzs(sps_parameters[n, :, :], model, batch_size)
    nzs.append(lbg_nzs)

#save
nz_data = np.asarray(nzs)
np.save(path+"/nz_samples/nz_"+str(run)+".npy", nz_data)
print("Complete.", flush=True)