
import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.emulator as em
import lbg_forecast.selection as sel
import lbg_forecast.colour_cuts as cuts

def simulate_nzs(sps_params, model, bins):

    source_index = 0
    source_photometry = []
    n_sources = len(sps_params)
    #while(source_index < n_sources):

    #    source_photometry.append(model.mimic_photometry(sps_params[source_index])) #this should work if I give all sps params at once? avoid loop?
 
    #    source_index +=1

    #source_photometry = np.vstack((np.asarray(source_photometry)))

    source_photometry = model.mimic_photometry(sps_params)
    
    #https://www.lsst.org/scientists/keynumbers    (y10, 5sigma)
    #detection_limits = np.array([26.1, 27.4, 27.5, 26.8, 26.1, 24.8]) possibly from w&w? source in lsa?
    detection_limits = np.array([25.6, 26.9, 26.9, 26.4, 25.6, 24.8])

    #guess
    brightness_cut = 15 #NOT IN USE (!)

    #########ADD NOISE MODEL HERE############
    
    #apply detection limits and calculate colours
    all_dropouts = sel.select_magnitudes(sps_params, source_photometry, detection_limits)
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