import fsps
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import lbg_forecast.sfh as sfh
import lbg_forecast.zhistory as zh
import lbg_forecast.popmodel as pop

from astropy.cosmology import WMAP9
from astropy.coordinates import Distance
from astropy.constants import L_sun

from sedpy import observate

#If dust_type=2, dust1 must be zero!
def initialise_sps_model(sfh_type=1, neb_em=True, zcont=1, imf_type=2, dust_type=2):

    sps_model = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=zcont, imf_type=imf_type, sfh=sfh_type, dust_type=dust_type)

    sps_model.params['add_neb_emission'] = neb_em 
    sps_model.params['add_igm_absorption'] = True
    sps_model.params['imf_type'] = imf_type

    return sps_model 

def update_sps_model_dpl(sps_model, sps_parameters, zhis=False, plot=False):

    #############################################################
    #set parameters
    sps_model.params['zred'] = sps_parameters[0]
    sps_model.params['tage'] = 10**(sps_parameters[1])
    sps_model.params['logzsol'] = sps_parameters[2]
    sps_model.params['dust1'] = sps_parameters[3]
    sps_model.params['dust2'] = sps_parameters[4]
    sps_model.params['igm_factor'] = sps_parameters[5]
    sps_model.params['gas_logu'] = sps_parameters[6]
    sps_model.params['gas_logz'] = sps_parameters[7]
    sps_model.params['fagn'] = sps_parameters[8]
    sps_model.params['imf1'] = sps_parameters[9]
    sps_model.params['imf2'] = sps_parameters[10]
    sps_model.params['imf3'] = sps_parameters[11]

    time_grid = np.logspace(-7, np.log10(sps_model.params['tage']), 1000)
    sfr = sfh.normed_sfh(sps_parameters[12], sps_parameters[13], sps_parameters[14], time_grid)

    if(zhis):
        Z_MIST = 0.0142 #solar metallicity for MIST
        sps_model.params['logzsol'] = 0.0
        zhis = zh.sfr_to_zh(sfr, time_grid, 10**sps_parameters[-1], (10**sps_parameters[2])*Z_MIST)
        sps_model.set_tabular_sfh(time_grid, sfr, zhis)
    else:
        sps_model.set_tabular_sfh(time_grid, sfr)

    if(plot):
        sfh.plot_sfh(sfr*10**sps_parameters[-1], time_grid)

        plt.figure(figsize=(10,5))
        plt.plot(time_grid, zhis)
        

    #############################################################

def simulate_sed(sps_model, sps_parameters):
    
    tage, logmass, zred = sps_model.params['tage'], sps_parameters[-1], sps_model.params['zred']

    #get SED
    angstroms, spectrum = sps_model.get_spectrum(tage=tage, peraa=True)
    spectrum_cgs_redshifted, aa_redshifted = redshift_fsps_spectrum(spectrum, angstroms, logmass, zred)

    return spectrum_cgs_redshifted, aa_redshifted

def simulate_photometry_fsps(sps_model, logmass, filters='lsst'):

    if(filters=='lsst'):
        bands = fsps.find_filter('lsst')
    if(filters=='suprimecam'):
        bands = fsps.find_filter('suprimecam')[1:2]+fsps.find_filter('suprimecam')[3:]
    if(filters=='all'):
        bands = fsps.find_filter('lsst') + fsps.find_filter('suprimecam')[1:2]+fsps.find_filter('suprimecam')[3:]

    mags = sps_model.get_mags(tage=sps_model.params['tage'], bands=bands)
    return mags - 2.5*logmass

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

def redshift_fsps_spectrum(spectrum, angstroms, logmass, redshift):

    L_sol_cgs = L_sun.cgs.value
    spectrum = spectrum*10**(logmass)
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
