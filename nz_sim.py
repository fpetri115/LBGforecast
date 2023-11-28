import fsps
import numpy as np
import matplotlib.pyplot as plt
from sedpy import observate

from useful_functions import get_filters
from useful_functions import get_magnitudes
from useful_functions import select_magnitudes
from useful_functions import select_colours
from useful_functions import plot_colours
from useful_functions import get_nz

from astropy.cosmology import WMAP9 as cosmo

from scipy.stats import loguniform

from cl_functions import pdf


def nz_sim(sim_no, prior_params):
    
    age_mu, age_sigma, mass_mu, mass_sigma, tau_mu, tau_sigma, const_mu, const_sigma, red_mu, red_sigma, met_mu, met_sigma, dust_mu, dust_sigma, tburst_mu, tburst_sigma, fburst_mu, fburst_sigma, imf = prior_params
    
    experiment = "lsst"
    filters = get_filters(experiment)
    
    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, sfh=1, dust_type=2)
    sp.libraries
    
    sp.params['add_neb_emission'] = True 
    sp.params['add_igm_absorption'] = True
    sp.params['imf_type'] = 2
    
    sources = []

    source_index = 0
    #np.random.seed(1)

    sf_start = 0.0
    sf_trunc = 0.0
    tburst = 11
    fburst = 0

    t_univ=cosmo.age(0).value

    ########################
    name = 'sim'+ str(sim_no)

    n_sources = 10000
    imf3_value = imf
    error = 0.1
    ########################

    imf3 = [imf3_value]*n_sources
    imf3_sorted = np.sort(imf3)

    while(source_index < n_sources):

        age = np.random.lognormal(np.log(10**(age_mu)), np.log(10**(age_sigma)))
        redshift = np.random.normal(red_mu, red_sigma)
        while(age<1e-4 or age>13 or redshift<0.1 or age>cosmo.age(redshift).value or redshift > 7):
            age = np.random.lognormal(np.log(10**(age_mu)), np.log(10**(age_sigma)))
            redshift = np.random.normal(red_mu, red_sigma)

        mass = np.random.lognormal(np.log(10**(mass_mu)), np.log(10**(mass_sigma)))

        tau = np.random.lognormal(np.log(10**(tau_mu)), np.log(10**(tau_sigma))) 
        while(tau<0.1 or tau>100):
            tau =  np.random.lognormal(np.log(10**(tau_mu)), np.log(10**(tau_sigma)))  

        const = np.random.normal(const_mu, const_sigma)
        fburst = np.random.normal(fburst_mu, fburst_sigma)
        while(const<0.0 or const>1.0 or fburst>1.0-const or fburst < 0.0):
            const = np.random.normal(const_mu, const_sigma)
            fburst = np.random.normal(fburst_mu, fburst_sigma)

        metal = np.random.normal(met_mu, met_sigma)
        while(metal<-2.5 or metal>0.5):
            metal  = np.random.normal(met_mu, met_sigma)

        dust  = np.random.normal(dust_mu, dust_sigma)
        while(dust<0.0):
            dust  = np.random.normal(dust_mu, dust_sigma)

        tburst = np.random.lognormal(np.log(10**(tburst_mu)), np.log(10**(tburst_sigma)))
        while(tburst>age):
            tburst = np.random.lognormal(np.log(10**(tburst_mu)), np.log(10**(tburst_sigma)))
            while(age<1e-4 or age>13 or redshift<0.1 or age>cosmo.age(redshift).value or redshift > 7):
                age = np.random.lognormal(np.log(10**(age_mu)), np.log(10**(age_sigma)))
                redshift = np.random.normal(red_mu, red_sigma)

        imf3 = imf3_sorted[source_index]

        igm_fudge = np.random.normal(1.0, 0.4)

        ###fagn - maybe have a param which is p(agn) - then if agn => fagn > 0 (what are the bounds here...) else agn = 0

        sources.append(np.array([age, mass, tau, const, redshift, metal, dust, tburst, fburst, imf3, igm_fudge, source_index]))
        source_index +=1

    sources = np.array(sources)
    
    source_data, source_indexes = get_magnitudes(sp, sources, filters, error, experiment)
    
    if(experiment == "suprimecam"):
        #g,r,i,z
        detection_limits = np.array([0, 26, 26, 25, 25])
    
    #https://www.lsst.org/scientists/keynumbers    (y10, 5sigma)
    elif(experiment == "lsst"):
        #u,g,r,i,z
        detection_limits = np.array([26.1, 27.4, 27.5, 26.8, 26.1])

    #guess
    brightness_cut = 15 #NOT IN USE (!)
    
    u_dropout_sources, u_detected_indexes, g_dropout_sources, g_detected_indexes, r_dropout_sources, r_detected_indexes = select_magnitudes(source_data, source_indexes, experiment, detection_limits)
    
    selected_colours, selected_redshifts = select_colours(experiment, sources, u_dropout_sources, u_detected_indexes, g_dropout_sources, g_detected_indexes, r_dropout_sources, r_detected_indexes)

    u_umg_data, u_gmr_data, g_gmr_data, g_rmi_data, r_rmi_data, r_imz_data = selected_colours
    u_redshifts, g_redshifts, r_redshifts = selected_redshifts
    
    dz = 0.1
    minz = 0.0
    maxz = 7.0

    u_colour_data_list = u_umg_data, u_gmr_data, u_redshifts, u_detected_indexes 
    g_colour_data_list = g_gmr_data, g_rmi_data, g_redshifts, g_detected_indexes 
    r_colour_data_list = r_rmi_data, r_imz_data, r_redshifts, r_detected_indexes 
    
    nzs = get_nz(experiment, dz, minz, maxz, u_colour_data_list, g_colour_data_list, r_colour_data_list, source_data)
                 
    np.save('data_cosmo/sources_'+name+'.npy', sources)
    np.save('data_cosmo/source_data_'+name+'.npy', source_data)
    np.save('nzu.npy', nzs[0])
    np.save('nzg.npy', nzs[1])
    np.save('nzr.npy', nzs[2])
    np.save('bins.npy', nzs[3])