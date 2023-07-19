import fsps
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import lbg_forecast.sfh as sfh

from astropy.cosmology import WMAP9
from astropy.coordinates import Distance
from astropy.constants import L_sun

from sedpy import observate

#If dust_type=2, dust1 must be zero!
def initialise_sps_model(sfh_type=1, dust_type=2):

    sps_model = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, sfh=sfh_type, dust_type=dust_type)

    sps_model.params['add_neb_emission'] = True 
    sps_model.params['add_igm_absorption'] = True
    sps_model.params['imf_type'] = 2

    return sps_model 

def update_sps_model(sps_model, sps_parameters):

    #need to reset these two every loop otherwise FSPS will break
    sps_model.params['const'] = 0
    sps_model.params['fburst'] = 0
    #############################################################

    #set parameters
    sps_model.params['tage'] = sps_parameters['tage']
    sps_model.params['tau'] = sps_parameters['tau']
    sps_model.params['const'] = sps_parameters['const']
    sps_model.params['zred'] = sps_parameters['zred']
    sps_model.params['logzsol'] = sps_parameters['logzsol']
    sps_model.params['dust_tesc'] = sps_parameters['dust_tesc']
    sps_model.params['dust1'] = sps_parameters['dust1']
    sps_model.params['dust2'] = sps_parameters['dust2']
    sps_model.params['tburst'] = sps_parameters['tburst']
    sps_model.params['fburst'] = sps_parameters['fburst']
    sps_model.params['igm_factor'] = sps_parameters['igm_factor']
    sps_model.params['gas_logu'] = sps_parameters['gas_logu']
    sps_model.params['gas_logz'] = sps_parameters['gas_logz']
    sps_model.params['fagn'] = sps_parameters['fagn']
    sps_model.params['imf1'] = sps_parameters['imf1']
    sps_model.params['imf2'] = sps_parameters['imf2']
    sps_model.params['imf3'] = sps_parameters['imf3']

    #############################################################

def update_sps_model_dpl(sps_model, sps_parameters):

    #############################################################

    #set parameters
    sps_model.params['tage'] = sps_parameters['tage']
    sps_model.params['zred'] = sps_parameters['zred']
    sps_model.params['logzsol'] = sps_parameters['logzsol']
    sps_model.params['dust_tesc'] = sps_parameters['dust_tesc']
    sps_model.params['dust1'] = sps_parameters['dust1']
    sps_model.params['dust2'] = sps_parameters['dust2']
    sps_model.params['igm_factor'] = sps_parameters['igm_factor']
    sps_model.params['gas_logu'] = sps_parameters['gas_logu']
    sps_model.params['gas_logz'] = sps_parameters['gas_logz']
    sps_model.params['fagn'] = sps_parameters['fagn']
    sps_model.params['imf1'] = sps_parameters['imf1']
    sps_model.params['imf2'] = sps_parameters['imf2']
    sps_model.params['imf3'] = sps_parameters['imf3']

    time_grid = np.logspace(-5, np.log10(sps_parameters['tage'][0]), 10000)

    sfr = sfh.dpl(sps_parameters['a'][0], sps_parameters['b'][0],
                                sps_parameters['tau'][0], time_grid)
    
    normed_sfr = sfr/np.trapz((10**9)*sfr, time_grid)
    sps_model.set_tabular_sfh(time_grid, normed_sfr)

    sfh.plot_sfh(normed_sfr, time_grid)

    #############################################################

def simulate_sed(sps_model, sps_parameters):
    
    tage, mass, zred = sps_model.params['tage'], sps_parameters['mass'], sps_model.params['zred']

    #get SED
    angstroms, spectrum = sps_model.get_spectrum(tage=tage, peraa=True)
    spectrum_cgs_redshifted, aa_redshifted = redshift_fsps_spectrum(spectrum, angstroms, mass, zred)

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