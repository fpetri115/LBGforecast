
import numpy as np
import lbg_forecast.colour_cuts as cuts
import lbg_forecast.noise as noise

def default_nz_bins():
    
    dz = 0.1
    minz = 0.0
    maxz = 7.0

    return np.arange(minz, maxz, dz)

def simulate_nzs(sps_params, model, emulator_batch_size):

    #emulate photometry given sps parameters
    source_photometry = model.mimic_photometry(sps_params, emulator_batch_size)

    #apply noise, the perform SNR, brightness and faintness cuts
    all_dropouts_mags = noise.get_noisy_magnitudes(sps_params, source_photometry, random_state=np.random.randint(0, 100000))

    #convert magnitudes to colours
    all_dropouts_colours = cuts.colours(all_dropouts_mags)
    
    #apply LBG colour cuts
    u_data, g_data, r_data = cuts.apply_cuts_to_colours(all_dropouts_colours)

    #get selected redshift samples and combine into object array
    nzs = build_redshift_distribution_samples_object(u_data, g_data, r_data)

    return nzs

def build_redshift_distribution_samples_object(u_data, g_data, r_data):

    u_redshifts = cuts.get_zs(u_data)
    g_redshifts = cuts.get_zs(g_data)
    r_redshifts = cuts.get_zs(r_data)

    redshift_array = np.empty(3, object)
    redshift_array[:] = [u_redshifts, g_redshifts, r_redshifts]     

    return redshift_array

def build_colour_samples_object(u_data, g_data, r_data):

    u_colours = cuts.get_colours(u_data)
    g_colours = cuts.get_colours(g_data)
    r_colours = cuts.get_colours(r_data)

    colour_array = np.empty(3, object)
    colour_array[:] = [u_colours, g_colours, r_colours]     

    return colour_array

def calculate_nzs_from_photometry(sps_params, source_photometry, colours=False):

    #apply noise, the perform SNR, brightness and faintness cuts
    all_dropouts_mags = noise.get_noisy_magnitudes(sps_params, source_photometry)

    #convert magnitudes to colours
    all_dropouts_colours = cuts.colours(all_dropouts_mags)
    
    #apply LBG colour cuts
    u_data, g_data, r_data = cuts.apply_cuts_to_colours(all_dropouts_colours)

    #get selected redshift samples and combine into object array
    nzs = build_redshift_distribution_samples_object(u_data, g_data, r_data)

    if(colours==False):
        return nzs
    else:
        colours = build_colour_samples_object(u_data, g_data, r_data)
        return nzs, colours


