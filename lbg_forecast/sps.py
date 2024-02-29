import os
os.environ["SPS_HOME"] = "/Users/fpetri/packages/fsps" 

import fsps
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import lbg_forecast.sfh as sfh
import lbg_forecast.zhistory as zh
import lbg_forecast.population_model as pop

from astropy.cosmology import WMAP7 as cosmo
#from astropy.coordinates import Distance
from astropy.constants import L_sun
from astropy.constants import c

from sedpy import observate

def initialise_sps_model(neb_em, sfh_type=3, zcont=1, imf_type=2, dust_type=0, igm=True):
    
    sps_model = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=zcont, imf_type=imf_type, sfh=sfh_type, dust_type=dust_type)

    sps_model.params['add_neb_emission'] = neb_em
    sps_model.params['add_neb_continuum'] = neb_em
    sps_model.params['nebemlineinspec'] = neb_em
    
    sps_model.params['add_igm_absorption'] = igm
    sps_model.params['imf_type'] = imf_type

    return sps_model 

def update_model(sps_model, sps_parameters, z_history, agebins):
    
    sps_model.params['zred'] = sps_parameters[0]
    sps_model.params['logzsol'] = sps_parameters[1]
    sps_model.params['dust1'] = sps_parameters[2]
    sps_model.params['dust2'] = sps_parameters[3]
    sps_model.params['dust_index'] = sps_parameters[4]
    sps_model.params['igm_factor'] = sps_parameters[5]
    sps_model.params['gas_logu'] = sps_parameters[6]
    sps_model.params['fagn'] = sps_parameters[7]
    sps_model.params['agn_tau'] = sps_parameters[8]

    log_sfr_ratios = sps_parameters[9:-1]

    total_mass_formed = sps_parameters[-1]
    
    shifted_age_bins = sfh.zred_to_agebins(sps_model.params['zred'], agebins)
    time, star_formation_history, tage = sfh.continuity_sfh(shifted_age_bins, log_sfr_ratios, total_mass_formed)[0]
    sps_model.params['tage'] = tage

    if(z_history):
        Z_MIST = 0.0142 #solar metallicity for MIST
        sps_model.params['logzsol'] = 0.0
        sps_model.params['gas_logz'] = 0.0
        metallicity_history = zh.sfr_to_zh(star_formation_history, time,
                                            (10**sps_model.params['logzsol'])*Z_MIST, sps_parameters[-1])
        sps_model.set_tabular_sfh(time, star_formation_history, metallicity_history)
    else:
        sps_model.params['gas_logz'] = sps_model.params['logzsol']
        sps_model.set_tabular_sfh(time, star_formation_history) 


def simulate_photometry(sps_parameters, filters, imf, dust, nebem=True, zhistory=True, agebins=None, enable_mpi=False, mpi_rank=0):

    ngalaxies = sps_parameters.shape[0]

    if(nebem == False and zhistory == True):
        raise Exception("nebular emission cannot be turned off with zhistory enabled at present")
    
    if agebins is None:
        agebins = sfh.default_agebins()

    #Generate photometry with Nebular emmision###################
    print("Starting Run 1/3")
    sps_model = initialise_sps_model(neb_em=nebem, sfh_type=3, zcont=1, dust_type=dust, imf_type=imf)
    print("libraries: ", sps_model.libraries)

    i = 0
    photometry_neb = []
    while(i < ngalaxies):

        source = sps_parameters[i, :]
        update_model(sps_model, source, z_history=False, agebins=agebins)

        #generate photometry for source
        photometry_neb.append(get_magnitudes(sps_model, filters=filters))

        i+=1
        if(i%100 == 0 and mpi_rank==0):
            print(i)

    photometry_neb = np.vstack(np.asarray(photometry_neb))
    print("Run 1/3 Complete")

    if(zhistory):

        #Generate photometry without nebular emmision################
        print("Starting Run 2/3")
        sps_model = initialise_sps_model(sfh_type=3, neb_em=False, zcont=1, dust_type=dust, imf_type=imf)
        i = 0
        photometry_no_neb = []
        while(i < ngalaxies):

            source = sps_parameters[i, :]
            update_model(sps_model, source, z_history=False, agebins=agebins)

            #generate photometry for source
            photometry_no_neb.append(get_magnitudes(sps_model, filters=filters))

            i+=1
            if(i%100 == 0 and mpi_rank==0):
                print(i)

        photometry_no_neb = np.vstack(np.asarray(photometry_no_neb))
        print("Run 2/3 Complete")

        #Nebular emmission contribution
        photometric_contribution_from_neb = photometry_neb - photometry_no_neb
        
        #Define SPS Model without Nebular emmision BUT with zhistory############
        print("Starting Run 3/3")
        sps_model = initialise_sps_model(neb_em=False, sfh_type=3, zcont=3, dust_type=dust, imf_type=imf)

        i = 0
        photometry_zhis = []
        while(i < ngalaxies):

            source = sps_parameters[i, :]
            update_model(sps_model, source, z_history=True, agebins=agebins)

            #generate photometry for source
            photometry_zhis.append(get_magnitudes(sps_model, filters=filters))

            i+=1
            if(i%100 == 0 and mpi_rank==0):
                print(i)

        photometry_zhis = np.vstack(np.asarray(photometry_zhis))
        ###################################################
        
        print("Run 3/3 Complete")
        photometry_final = photometry_zhis + photometric_contribution_from_neb
    
    else:
        photometry_final = photometry_neb

    print("Complete")

    if(enable_mpi):
        np.save("simulation_data/simulated_photometry_"+str(mpi_rank)+".npy", photometry_final)
        np.save("simulation_data/sps_parameters_"+str(mpi_rank)+".npy", sps_parameters)
    else:
        np.save("simulation_data/simulated_photometry.npy", photometry_final)
        np.save("simulation_data/sps_parameters.npy", sps_parameters)

    return photometry_final


def fsps_simulate_sed(sps_model):
    
    tage = sps_model.params['tage']

    #get SED
    angstroms, spectrum = sps_model.get_spectrum(tage=tage, peraa=False)
    #spectrum_cgs_redshifted, aa_redshifted = redshift_fsps_spectrum(spectrum, angstroms, zred)

    return spectrum, angstroms

def fsps_get_magnitudes(sps_model, filters):

    if(filters=='lsst'):
        bands = fsps.find_filter('lsst')
    if(filters=='suprimecam'):
        bands = fsps.find_filter('suprimecam')[1:2]+fsps.find_filter('suprimecam')[3:]
    if(filters=='all'):
        bands = fsps.find_filter('lsst') + fsps.find_filter('suprimecam')[1:2]+fsps.find_filter('suprimecam')[3:]

    mags = sps_model.get_mags(tage=sps_model.params['tage'], bands=bands)
    return mags# - 2.5*logmass

def get_magnitudes(sps_model, filters):

    spectrum = sps_model.get_spectrum(tage=sps_model.params['tage'], peraa=False)
    redshifted_spectrum = redshift_fsps_spectrum(spectrum[1], spectrum[0], sps_model.params['zred'])
    photometry = get_photometry(redshifted_spectrum, filters)
    return photometry

def get_sed(sps_model, filters):

    if(filters=='lsst'):
        spectrum = sps_model.get_spectrum(tage=sps_model.params['tage'], peraa=False)
        redshifted_spectrum = redshift_fsps_spectrum(spectrum[1], spectrum[0], sps_model.params['zred'])
        return redshifted_spectrum
    if(filters=='suprimecam'):
        raise NotImplementedError()

#my own version of fsps get_mags using sedpy
def get_photometry(redshifted_spectrum, filters):

    redshifted_spectrum_cgs, aa_redshifted = redshifted_spectrum

    if(filters == 'lsst'):
        bands = get_lsst_filters()
    if(filters == 'suprimecam'):
        bands = get_suprimecam_filters()
    
    mags = observate.getSED(aa_redshifted, redshifted_spectrum_cgs, filterlist=bands, linear_flux=False)

    return mags

def redshift_fsps_spectrum(spectrum, angstroms, redshift):

    L_sol_cgs = L_sun.cgs.value
    luminosity_distance = cosmo.luminosity_distance(z=redshift).cgs.value
    c_light = c.value*1e10
    aa_red = angstroms*(1+redshift)

    spec_cgs = spectrum*(1+redshift)*L_sol_cgs
    f_cgs = spec_cgs/(4*np.pi*(luminosity_distance)**2)
    f_cgs = f_cgs*((c_light)/((aa_red)**2))
    f_cgs = np.interp(angstroms, aa_red, f_cgs)
    return [f_cgs, angstroms]

def plot_sed(spectrum, scaley, xmin, xmax, ymin, ymax, xsize=10, ysize=5, fontsize=32, log=False, **kwargs):
    
    plt.figure(figsize=(xsize,ysize))
    plt.plot(spectrum[1], spectrum[0]*(10**scaley), **kwargs)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    mscaley = -1*scaley
    plt.ylabel("Flux Density $f_{\lambda}$ $[$" + r"$10^{{{}}}$".format(mscaley) + "$\mathrm{ergs}^{-1}\mathrm{cm}^{-3}]$", fontsize=fontsize)
    plt.xlabel("Wavelength $\lambda$ $[\mathrm{\AA}]$", fontsize=fontsize)
            
    plt.tick_params(axis="x", width = 2, labelsize=fontsize*0.8)
    plt.tick_params(axis="y", width = 2, labelsize=fontsize*0.8)

    if(log):
        plt.xscale("log")

#for homebrew get_mags
def get_lsst_filters():
            
    ufltr = fsps.filters.Filter(144, 'lsst_u', 'lsst')
    gfltr = fsps.filters.Filter(145, 'lsst_g', 'lsst')
    rfltr = fsps.filters.Filter(146, 'lsst_r', 'lsst')
    ifltr = fsps.filters.Filter(147, 'lsst_i', 'lsst')
    zfltr = fsps.filters.Filter(148, 'lsst_z', 'lsst')
    yfltr = fsps.filters.Filter(149, 'lsst_y', 'lsst')
    
    u_filt = observate.Filter("lsst_u", data=(ufltr.transmission[0], ufltr.transmission[1]))
    g_filt = observate.Filter("lsst_g", data=(gfltr.transmission[0], gfltr.transmission[1]))
    r_filt = observate.Filter("lsst_r", data=(rfltr.transmission[0], rfltr.transmission[1]))
    i_filt = observate.Filter("lsst_i", data=(ifltr.transmission[0], ifltr.transmission[1]))
    z_filt = observate.Filter("lsst_z", data=(zfltr.transmission[0], zfltr.transmission[1]))
    y_filt = observate.Filter("lsst_z", data=(yfltr.transmission[0], yfltr.transmission[1]))

    filters = np.array([u_filt, g_filt, r_filt, i_filt, z_filt, y_filt])
    
    return filters

#for homebrew get_mags
def get_suprimecam_filters():

    gfltr = observate.Filter("hsc_g")
    rfltr = observate.Filter("hsc_r")
    ifltr = observate.Filter("hsc_i")
    zfltr = observate.Filter("hsc_z")
    #yfltr = observate.Filter("hsc_z")

    filters = np.array([gfltr, rfltr, ifltr, zfltr])

    return filters

def plot_lsst_filters(factor):

    ufltr = fsps.filters.Filter(144, 'lsst_u', 'lsst')
    gfltr = fsps.filters.Filter(145, 'lsst_g', 'lsst')
    rfltr = fsps.filters.Filter(146, 'lsst_r', 'lsst')
    ifltr = fsps.filters.Filter(147, 'lsst_i', 'lsst')
    zfltr = fsps.filters.Filter(148, 'lsst_z', 'lsst')
    yfltr = fsps.filters.Filter(149, 'lsst_y', 'lsst')

    plt.plot(ufltr.transmission[0], ufltr.transmission[1]*factor)
    plt.plot(gfltr.transmission[0], gfltr.transmission[1]*factor)
    plt.plot(rfltr.transmission[0], rfltr.transmission[1]*factor)
    plt.plot(ifltr.transmission[0], ifltr.transmission[1]*factor)
    plt.plot(zfltr.transmission[0], zfltr.transmission[1]*factor)
    plt.plot(yfltr.transmission[0], yfltr.transmission[1]*factor)
