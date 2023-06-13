import fsps
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from astropy.cosmology import WMAP9
from astropy.coordinates import Distance
from astropy.constants import L_sun

def initialise_sps_model():

    sps_model = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, sfh=1, dust_type=0)

    sps_model.params['add_neb_emission'] = True 
    sps_model.params['add_igm_absorption'] = True
    sps_model.params['imf_type'] = 2

    return sps_model 

def simulate_sed(sps_model, sps_parameters):
    
    age, mass, tau, const, redshift, metal, dustesc, dust1, dust2, tburst, fburst, igm, gas_ion, gas_z, fagn, imf1, imf2, imf3 = sps_parameters

    #need to reset these two every loop otherwise FSPS will break
    sps_model.params['const'] = 0
    sps_model.params['fburst'] = 0
    #############################################################

    #set parameters
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

    #get SED
    angstroms, spectrum = sps_model.get_spectrum(tage=age, peraa=True)
    spectrum_cgs_redshifted, aa_redshifted = redshift_fsps_spectrum(spectrum, angstroms, mass, redshift)

    return spectrum_cgs_redshifted, aa_redshifted

def redshift_fsps_spectrum(spectrum, angstroms, mass, redshift):

    L_sol_cgs = L_sun.cgs.value
    spectrum = spectrum*mass
    DL = Distance(z=redshift, cosmology=WMAP9).cgs.value

    f_cgs_aa = spectrum/(4*(1+redshift)*np.pi*DL**2) * (L_sol_cgs)
    aa_red = angstroms*(1+redshift)
    
    return  f_cgs_aa, aa_red

def define_sps_parameters(age, mass, tau, const, redshift, metal, dustesc, dust1, dust2, tburst, fburst, igm, gas_ion, gas_z, fagn, imf1, imf2, imf3):
    
    return np.array([age, mass, tau, const, redshift, metal, dustesc, dust1, dust2, tburst, fburst, igm, gas_ion, gas_z, fagn, imf1, imf2, imf3])

def plot_sed(spectrum):
    
    plt.plot(spectrum[1], spectrum[0])
    plt.xlim(2000, 9000)
