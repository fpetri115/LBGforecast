import fsps
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from astropy.cosmology import WMAP9
from astropy.coordinates import Distance
from astropy.constants import L_sun

from sedpy import observate

def initialise_sps_model():

    sps_model = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, sfh=1, dust_type=0)

    sps_model.params['add_neb_emission'] = True 
    sps_model.params['add_igm_absorption'] = True
    sps_model.params['imf_type'] = 2

    return sps_model 

def update_sps_model(sps_model, sps_parameters):

    age, mass, tau, const, redshift, metal, dustesc, dust1, dust2, tburst, fburst, igm, gas_ion, gas_z, fagn, imf1, imf2, imf3 = sps_parameters

    #need to reset these two every loop otherwise FSPS will break
    sps_model.params['const'] = 0
    sps_model.params['fburst'] = 0
    #############################################################

    #set parameters
    sps_model.params['tage'] = age
    sps_model.params['tau'] = tau
    sps_model.params['const'] = const
    sps_model.params['zred'] = redshift
    sps_model.params['logzsol'] = metal
    sps_model.params['dust_tesc'] = dustesc
    sps_model.params['dust1'] = dust1
    sps_model.params['dust2'] = dust2
    sps_model.params['tburst'] = tburst
    sps_model.params['fburst'] = fburst
    sps_model.params['igm_factor'] = igm
    sps_model.params['gas_logu'] = gas_ion
    sps_model.params['gas_logz'] = gas_z
    sps_model.params['fagn'] = fagn
    sps_model.params['imf1'] = imf1
    sps_model.params['imf2'] = imf2
    sps_model.params['imf3'] = imf3

    #############################################################

def simulate_sed(sps_model, sps_parameters):
    
    age, mass = sps_parameters[0], sps_parameters[1]

    #get SED
    angstroms, spectrum = sps_model.get_spectrum(tage=age, peraa=True)
    spectrum_cgs_redshifted, aa_redshifted = redshift_fsps_spectrum(spectrum, angstroms, mass, sps_model.params['zred'])

    return spectrum_cgs_redshifted, aa_redshifted

def simulate_photometry_lsst_fsps(sps_model, mass):

    lsst_filters = fsps.find_filter('lsst')
    mags = sps_model.get_mags(tage=sps_model.params['tage'], bands=lsst_filters)

    return mags - 2.5*np.log10(mass)

#my own version of fsps get_mags using sedpy
def simulate_photometry_lsst(aa_redshifted, redshifted_spectrum_cgs):

    lsst_filters = get_lsst_filters()

    mag_u = lsst_filters[0].ab_mag(aa_redshifted, redshifted_spectrum_cgs)
    mag_g = lsst_filters[1].ab_mag(aa_redshifted, redshifted_spectrum_cgs)
    mag_r = lsst_filters[2].ab_mag(aa_redshifted, redshifted_spectrum_cgs)
    mag_i = lsst_filters[3].ab_mag(aa_redshifted, redshifted_spectrum_cgs)
    mag_z = lsst_filters[4].ab_mag(aa_redshifted, redshifted_spectrum_cgs)
    mag_y = lsst_filters[5].ab_mag(aa_redshifted, redshifted_spectrum_cgs)

    mags = np.array([mag_u, mag_g, mag_r, mag_i, mag_z, mag_y])

    return mags

def redshift_fsps_spectrum(spectrum, angstroms, mass, redshift):

    L_sol_cgs = L_sun.cgs.value
    spectrum = spectrum*mass
    DL = Distance(z=redshift, cosmology=WMAP9).cgs.value

    f_cgs_aa = spectrum/(4*(1+redshift)*np.pi*DL**2) * (L_sol_cgs)
    aa_red = angstroms*(1+redshift)
    
    return  f_cgs_aa, aa_red

def update_sps_parameters(age, mass, tau, const, redshift, metal, dustesc, dust1, dust2, tburst, fburst, igm, gas_ion, gas_z, fagn, imf1, imf2, imf3, sps_model):

    sps_params = np.array([age, mass, tau, const, redshift, metal, 
                           dustesc, dust1, dust2, tburst, 
                           fburst, igm, gas_ion, gas_z, fagn, 
                           imf1, imf2, imf3])

    update_sps_model(sps_model, sps_params)

    return sps_params

def plot_sed(spectrum):
    
    plt.plot(spectrum[1], spectrum[0])
    plt.xlim(2000, 9000)

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

def plot_lsst_filters():

    ufltr = fsps.filters.Filter(144, 'lsst_u', 'lsst')
    gfltr = fsps.filters.Filter(145, 'lsst_g', 'lsst')
    rfltr = fsps.filters.Filter(146, 'lsst_r', 'lsst')
    ifltr = fsps.filters.Filter(147, 'lsst_i', 'lsst')
    zfltr = fsps.filters.Filter(148, 'lsst_z', 'lsst')
    yfltr = fsps.filters.Filter(149, 'lsst_y', 'lsst')

    plt.plot(ufltr.transmission[0], ufltr.transmission[1])
    plt.plot(gfltr.transmission[0], gfltr.transmission[1])
    plt.plot(rfltr.transmission[0], rfltr.transmission[1])
    plt.plot(ifltr.transmission[0], ifltr.transmission[1])
    plt.plot(zfltr.transmission[0], zfltr.transmission[1])
    plt.plot(yfltr.transmission[0], yfltr.transmission[1])