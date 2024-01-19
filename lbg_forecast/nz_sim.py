
import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.selection as sel
import lbg_forecast.colour_cuts as cuts
import lbg_forecast.noise as noise

def simulate_nzs(sps_params, model):

    source_photometry = model.mimic_photometry(sps_params)

    #apply detection limits and calculate colours
    all_dropouts = noise.get_noisy_magnitudes(sps_params, source_photometry, random_state=np.random.randint(0, 100000))
    all_dropouts = sel.colours(all_dropouts)
    
    nzs = apply_cuts(all_dropouts)

    return nzs

def apply_cuts(dropout_data):

    u_dropouts, g_dropouts, r_dropouts = dropout_data

    #Select dropout sources
    selected_u_dropouts = cuts.SelectDropouts('u', u_dropouts)
    selected_g_dropouts = cuts.SelectDropouts('g', g_dropouts)
    selected_r_dropouts = cuts.SelectDropouts('r', r_dropouts)

    u_redshifts = selected_u_dropouts[:, 0]
    g_redshifts = selected_g_dropouts[:, 0]
    r_redshifts = selected_r_dropouts[:, 0]
    
    redshift_array = np.empty(3, object)
    redshift_array[:] = [u_redshifts, g_redshifts, r_redshifts]     

    return redshift_array

def visualise_redshifts(redshift_array, minz=0, maxz=7, alpha=0.5, density=True):

    dz = 0.1
    minz = 0.0
    maxz = 7.0
    bins = np.arange(minz, maxz, dz)

    u_redshifts = redshift_array[0]
    g_redshifts = redshift_array[1]
    r_redshifts = redshift_array[2]

    plt.hist(u_redshifts, bins=bins, alpha=alpha, color = 'blue', density=density)
    plt.hist(g_redshifts, bins=bins, alpha=alpha, color = 'red', density=density)
    plt.hist(r_redshifts, bins=bins, alpha=alpha, color = 'green', density=density)
