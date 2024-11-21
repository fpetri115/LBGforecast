import sys
import numpy as np
import lbg_forecast.nz as nz
import lbg_forecast.emulator as em

path = sys.argv[1]
batch_size = int(sys.argv[2])
run = sys.argv[3]

# initalise fsps emulator
model = em.fsps_emulator(path[:-1])

sps_parameters = np.load(path[:-1]+"/sps_parameter_samples/sps_"+run+".npy")

nzs = []
for n in range(sps_parameters.shape[0]):    
    lbg_nzs = nz.simulate_nzs(sps_parameters[n, :, :], model, batch_size)
    nzs.append(lbg_nzs)
    print("Realisation: ", n+1, flush=True)

#save
nz_data = np.asarray(nzs)
np.save(path+"/nz_samples/nz_"+run+".npy", nz_data)
print("Complete.", flush=True)