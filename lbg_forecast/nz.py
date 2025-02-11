
import numpy as np
import lbg_forecast.colour_cuts as cuts
import lbg_forecast.noise as noise
import lbg_forecast.priors_gp_massfunc as gpmf
import lbg_forecast.utils as utils

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

def build_sps_samples_object(u_data, g_data, r_data):

    u_params = cuts.get_params(u_data)
    g_params = cuts.get_params(g_data)
    r_params = cuts.get_params(r_data)

    params_array = np.empty(3, object)
    params_array[:] = [u_params, g_params, r_params]     

    return params_array

def build_colour_samples_object(u_data, g_data, r_data):

    u_colours = cuts.get_colours(u_data)
    g_colours = cuts.get_colours(g_data)
    r_colours = cuts.get_colours(r_data)

    colour_array = np.empty(3, object)
    colour_array[:] = [u_colours, g_colours, r_colours]     

    return colour_array

def calculate_nzs_from_photometry(sps_params, source_photometry, extra=False):

    #apply noise, the perform SNR, brightness and faintness cuts
    all_dropouts_mags = noise.get_noisy_magnitudes(sps_params, source_photometry)

    #convert magnitudes to colours
    all_dropouts_colours = cuts.colours(all_dropouts_mags)
    
    #apply LBG colour cuts
    u_data, g_data, r_data = cuts.apply_cuts_to_colours(all_dropouts_colours)

    #get selected redshift samples and combine into object array
    nzs = build_redshift_distribution_samples_object(u_data, g_data, r_data)

    if(extra==False):
        return nzs
    else:
        colours = build_colour_samples_object(u_data, g_data, r_data)
        params = build_sps_samples_object(u_data, g_data, r_data)
        return nzs, colours, params

def calculate_number_densities(real, spsp, selected_spsp, sparams):

    tot_m = np.log10(spsp[real, :, -1])
    tot_z = spsp[real, :][:, 0]
    sparams_real = sparams[real, :, :]

    mass_function_prior = gpmf.MassFunctionPrior('.', mean=False)

    dz=0.05
    dlogm=0.05

    start_z = 0
    end_z = 7
    z_bin = np.linspace(start_z, end_z, int((end_z-start_z)/dz))
    z_midpoint = (z_bin[1:]+z_bin[:-1])/2

    start_logm = 7
    end_logm = 13
    m_bin = np.linspace(start_logm, end_logm, int((end_logm-start_logm)/dlogm))
    m_midpoint = (m_bin[1:]+m_bin[:-1])/2

    u_sel = np.squeeze(selected_spsp[0])
    u_sel_m = np.log10(u_sel[:, -1])
    u_sel_z = u_sel[:, 0]

    g_sel = np.squeeze(selected_spsp[1])
    g_sel_m = np.log10(g_sel[:, -1])
    g_sel_z = g_sel[:, 0]

    r_sel = np.squeeze(selected_spsp[2])
    r_sel_m = np.log10(r_sel[:, -1])
    r_sel_z = r_sel[:, 0]

    u_counts_2d = np.histogram2d(u_sel_z, u_sel_m, bins=(z_bin, m_bin), density=False)[0]
    g_counts_2d = np.histogram2d(g_sel_z, g_sel_m, bins=(z_bin, m_bin), density=False)[0]
    r_counts_2d = np.histogram2d(r_sel_z, r_sel_m, bins=(z_bin, m_bin), density=False)[0]
    tot_counts_2d = np.histogram2d(tot_z, tot_m, bins=(z_bin, m_bin), density=False)[0]

    phi_z_m_2d = np.zeros_like(u_counts_2d)
    da_2d = np.zeros_like(phi_z_m_2d)

    i = 0
    for z in z_midpoint:
        da = dlogm*mass_function_prior.volume_element(z, dz)
        phi_z_m = mass_function_prior.mass_function(z, m_midpoint, sparams_real)
        da_2d[i, :] = da
        phi_z_m_2d[i, :] = phi_z_m
        i+=1

    p_detection_2d_u = np.divide(u_counts_2d, tot_counts_2d, out=np.zeros_like(u_counts_2d), where=(tot_counts_2d!=0))
    n_gal_u = np.sum(p_detection_2d_u*phi_z_m_2d*da_2d)
    n_den_u = n_gal_u/utils.FULL_SKY_DEG2

    p_detection_2d_g = np.divide(g_counts_2d, tot_counts_2d, out=np.zeros_like(g_counts_2d), where=(tot_counts_2d!=0))
    n_gal_g = np.sum(p_detection_2d_g*phi_z_m_2d*da_2d)
    n_den_g = n_gal_g/utils.FULL_SKY_DEG2

    p_detection_2d_r = np.divide(r_counts_2d, tot_counts_2d, out=np.zeros_like(r_counts_2d), where=(tot_counts_2d!=0))
    n_gal_r = np.sum(p_detection_2d_r*phi_z_m_2d*da_2d)
    n_den_r = n_gal_r/utils.FULL_SKY_DEG2

    return np.array([n_den_u, n_den_g, n_den_r])