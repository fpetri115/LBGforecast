import os
os.environ["SPS_HOME"] = "/Users/fpetri/packages/fsps" 

import fsps
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import lbg_forecast.sfh as sfh
import lbg_forecast.zhistory as zh
import lbg_forecast.population_model as pop
import lbg_forecast.lyalpha as ly

from astropy.cosmology import WMAP1 as cosmo
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


def simulate_photometry(sps_parameters, filters, imf, dust, nebem=True, zhistory=True, agebins=None, enable_mpi=False, lya_uncertainity=False, mpi_rank=0):

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
        photometry_neb.append(get_magnitudes(sps_model, filters=filters, lya_uncertainity=lya_uncertainity))

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
            photometry_no_neb.append(get_magnitudes(sps_model, filters=filters, lya_uncertainity=lya_uncertainity))

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
            photometry_zhis.append(get_magnitudes(sps_model, filters=filters, lya_uncertainity=lya_uncertainity))

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


def fsps_get_sed(sps_model):
    
    tage = sps_model.params['tage']
    angstroms, spectrum = sps_model.get_spectrum(tage=tage, peraa=True)

    return angstroms, spectrum

def fsps_get_magnitudes(sps_model, filters):

    if(filters=='lsst'):
        bands = fsps.find_filter('lsst')
    if(filters=='suprimecam'):
        bands = fsps.find_filter('suprimecam')[1:2]+fsps.find_filter('suprimecam')[3:]
    if(filters=='all'):
        bands = fsps.find_filter('lsst') + fsps.find_filter('suprimecam')[1:2]+fsps.find_filter('suprimecam')[3:]

    mags = sps_model.get_mags(tage=sps_model.params['tage'], bands=bands)
    return mags# - 2.5*logmass


def get_magnitudes(sps_model, filters, lya_uncertainity=False):
    
    lsun = L_sun.cgs.value

    lambdas, spectrum = sps_model.get_spectrum(tage=sps_model.params['tage'], peraa=True)
    redshift = sps_model.params['zred']

    if(lya_uncertainity):
        lya_width = np.random.uniform(30, 100)
        lya_bias = 0
        spectrum = ly.modify_peak(lambdas, spectrum, redshift, 100, lya_width, lya_bias, diagnostics=False)

    luminosity_distance = cosmo.luminosity_distance(redshift).cgs.value

    redshifted_spectrum = redshift_fsps_spectrum(sps_model, (lambdas, spectrum)) #lsun aa-1
    redshifted_spectrum_cgs = redshifted_spectrum*(lsun/(4.0*np.pi*(luminosity_distance**2))) #erg s-1 cm-2 aa-1
    redshifted_spectrum_sedpy = redshifted_spectrum_cgs #erg s-1 cm-2 aa-1

    if(filters == 'lsst'):
        bands = get_lsst_filters()
    if(filters == 'suprimecam'):
        bands = get_suprimecam_filters()

    magnitudes = observate.getSED(lambdas, redshifted_spectrum_sedpy, filterlist=bands, linear_flux=False)
    
    return magnitudes

def redshift_fsps_spectrum(sps_model, spec):
    """Takes output of fsps_get_sed() and redshifts spectrum.
    Used in get_magnitudes() too. Wavelength grid is kept the same.
    Spectrum must be in Lsun/AA. Wavelength in AA.
    
    """

    lambdas, spectrum = spec
    redshift = sps_model.params['zred']
    redshifted_spectrum = np.interp(lambdas, lambdas*(1+redshift), spectrum)/(1+redshift)

    return redshifted_spectrum 




def get_lsst_filters():
            
    ufltr = fsps.filters.Filter(144, 'lsst_u', 'lsst')
    gfltr = fsps.filters.Filter(145, 'lsst_g', 'lsst')
    rfltr = fsps.filters.Filter(146, 'lsst_r', 'lsst')
    ifltr = fsps.filters.Filter(147, 'lsst_i', 'lsst')
    zfltr = fsps.filters.Filter(148, 'lsst_z', 'lsst')
    yfltr = fsps.filters.Filter(149, 'lsst_y', 'lsst')

    filters = []
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        filter_data = np.genfromtxt('./lbg_forecast/lsst_filters/total_'+band+'.dat', skip_header=7, delimiter=' ')
        filter_data[:, 0] = filter_data[:, 0]*10 #covert to angstroms
        filters.append(observate.Filter("lsst_"+band, data=(filter_data[:, 0], filter_data[:, 1])))
    
    return filters

#for homebrew get_mags
def get_lsst_filters_fsps():
            
    ufltr = fsps.filters.Filter(144, 'lsst_u', 'lsst')
    gfltr = fsps.filters.Filter(145, 'lsst_g', 'lsst')
    rfltr = fsps.filters.Filter(146, 'lsst_r', 'lsst')
    ifltr = fsps.filters.Filter(147, 'lsst_i', 'lsst')
    zfltr = fsps.filters.Filter(148, 'lsst_z', 'lsst')
    yfltr = fsps.filters.Filter(149, 'lsst_y', 'lsst')

    filters = []
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        filter_data = np.genfromtxt('./lbg_forecast/lsst_filters_fsps/total_'+band+'.dat', skip_header=1, delimiter=' ')
        filters.append([filter_data[:, 0], filter_data[:, 1]])
    
    return filters

#for homebrew get_mags
def get_suprimecam_filters():

    gfltr = observate.Filter("hsc_g")
    rfltr = observate.Filter("hsc_r")
    ifltr = observate.Filter("hsc_i")
    zfltr = observate.Filter("hsc_z")
    yfltr = observate.Filter("hsc_y")

    filters = np.array([gfltr, rfltr, ifltr, zfltr, yfltr])

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

    return [ufltr, gfltr, rfltr, ifltr, zfltr, yfltr]

def plot_suprimecam_filters(factor):

    gfltr = fsps.filters.Filter(134, 'suprimecam_g', 'suprimecam')
    rfltr = fsps.filters.Filter(136, 'suprimecam_r', 'suprimecam')
    ifltr = fsps.filters.Filter(137, 'suprimecam_i', 'suprimecam')
    zfltr = fsps.filters.Filter(138, 'suprimecam_z', 'suprimecam')

    plt.plot(gfltr.transmission[0], gfltr.transmission[1]*factor)
    plt.plot(rfltr.transmission[0], rfltr.transmission[1]*factor)
    plt.plot(ifltr.transmission[0], ifltr.transmission[1]*factor)
    plt.plot(zfltr.transmission[0], zfltr.transmission[1]*factor)

def _fsps_lumdist(z):
    c_light = 2.9979E18
    dhub = c_light/1E13/72*1E6
    size = 10000
    i = 1
    zz = []
    while(i < size+1):
        zz.append((i/size)*z)
        i+=1
    zz = np.array(zz)
    hub = np.sqrt(0.27*(np.ones_like(zz)+zz)**3+0.73)
    luminosity_distance = np.trapz(1/hub, zz) * (1+z) * dhub
    return luminosity_distance

def fsps_cloned_get_magnitudes(sps_model, filters):
    lambdas, spectrum = sps_model.get_spectrum(tage=sps_model.params['tage'], peraa=False)
    redshift = sps_model.params['zred']
    redshifted_spectrum = np.interp(lambdas, lambdas*(1+redshift), spectrum)

    lsun = 3.839E33
    pc2cm = 3.08568E18
    mypi = 3.14159265
    c_light = 2.9979E18


    a = (np.linspace(1, 500, 500)-1)/499.*(1-1/1001.)+1/1001
    redshift_grid = 1/a-1

    luminosity_distances = []
    for red in redshift_grid:
        luminosity_distances.append(_fsps_lumdist(red))

    luminosity_distances = np.flip(np.array(luminosity_distances))

    luminosity_distance = _fsps_lumdist(redshift)
    #luminosity_distance = cosmo.luminosity_distance(redshift).cgs.value

    dm = 5.0*np.log10(luminosity_distance/10)
    const = dm - 2.5*np.log10(1+redshift)
    mag2cgs = np.log10(lsun/(4.0*mypi*pc2cm*pc2cm*100))

    if(filters == 'lsst'):
        bands = get_lsst_filters()
    if(filters == 'fsps'):
        bands = get_lsst_filters_fsps()
    if(filters == 'suprimecam'):
        bands = get_suprimecam_filters()

    magnitudes = []
    for band in bands:

        if(filters == 'fsps'):
            i = 1
            imax = 50000
            trans_lambdas = np.zeros(imax)
            trans = np.zeros(imax)
            j=0
            
            while(i < imax+1):
                trans_lambdas[i-1] = i
                trans[i-1] = 0.0
                if(j < len(band[0])):
                    if(i == band[0][j]):
                        trans[i-1] = band[1][j]
                        j+=1
                i+=1

            trans_v = trans#/(c_light/trans_lambdas)
            trans_v = np.maximum(trans_v, np.zeros_like(trans_v))
            trans_v = np.interp(lambdas, trans_lambdas, trans_v)

            #trans = trans[1:]
            #trans_lambdas = trans_lambdas[1:]
        else:
            trans = band[1]
            trans_lambdas = band[0]

            trans_v = trans#/(c_light/trans_lambdas)
            trans_v = np.maximum(trans_v, np.zeros_like(trans_v))
            trans_v = np.interp(lambdas, trans_lambdas, trans_v)

        trans_v = trans_v/np.trapz(trans_v/lambdas, lambdas)
        trans_v = np.maximum(trans_v, np.zeros_like(trans_v))

        mags = np.trapz(redshifted_spectrum*trans_v/lambdas, lambdas)
        mags = -2.5*np.log10(mags) - 48.60 - 2.5*mag2cgs + const
        magnitudes.append(mags)
    
    return np.array(magnitudes)
