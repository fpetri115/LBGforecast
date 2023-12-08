
import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.emulator as em
import lbg_forecast.selection as sel
import lbg_forecast.colour_cuts as cuts
import lbg_forecast.noise as noise

def simulate_nzs(sps_params, model, bins):

    source_photometry = model.mimic_photometry(sps_params)

    #guess
    brightness_cut = 19


    #apply detection limits and calculate colours
    all_dropouts = noise.get_noisy_magnitudes(sps_params, source_photometry, brightness_cut, random_state=42)
    all_dropouts = sel.colours(all_dropouts)
    
    nzs = apply_cuts(all_dropouts, bins)

    return nzs

def apply_cuts(dropout_data, bins):

    u_dropouts, g_dropouts, r_dropouts = dropout_data

    #Select dropout sources
    selected_u_dropouts = cuts.SelectDropouts('u', u_dropouts)
    selected_g_dropouts = cuts.SelectDropouts('g', g_dropouts)
    selected_r_dropouts = cuts.SelectDropouts('r', r_dropouts)

    u_redshifts = selected_u_dropouts[:, 0]
    g_redshifts = selected_g_dropouts[:, 0]
    r_redshifts = selected_r_dropouts[:, 0]

    nz_u = np.histogram(u_redshifts, bins=bins, density=True)[0]
    nz_g = np.histogram(g_redshifts, bins=bins, density=True)[0]
    nz_r = np.histogram(r_redshifts, bins=bins, density=True)[0]
    
    return [nz_u, nz_g, nz_r]