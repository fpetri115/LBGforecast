
import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.selection as sel
import lbg_forecast.colour_cuts as cuts
import lbg_forecast.noise as noise

def default_nz_bins():
    
    dz = 0.1
    minz = 0.0
    maxz = 7.0

    return np.arange(minz, maxz, dz)

def simulate_nzs(sps_params, model, emulator_batch_size):

    source_photometry = model.mimic_photometry(sps_params, emulator_batch_size)

    #apply detection limits and calculate colours
    all_dropouts = noise.get_noisy_magnitudes(sps_params, source_photometry, random_state=np.random.randint(0, 100000))
    all_dropouts = cuts.colours(all_dropouts)
    
    nzs = cuts.apply_cuts(all_dropouts)

    return nzs

