import numpy as np
import matplotlib.pyplot as plt

def find_wave_range(wave, l0, dlambda):
    """returns indexes of spectra at wavelengths of interest
    """

    indx1 = np.where(wave>l0-dlambda)[0]
    indx2 = np.where(wave<l0+dlambda)[0]
    indx = np.intersect1d(indx1, indx2)

    return indx

def estimate_continuum(wave, sed, l0, dlambda):
    """lineraly interpolates around l0 to find continuum
    """

    l1 = l0-dlambda
    l2 = l0+dlambda

    em1 = np.interp(l1, wave, sed)
    em2 = np.interp(l2, wave, sed)

    return (em2-em1)/(l2-l1)*(l0-l1) + em1

def lin_interp(wave, sed, l0, dlambda):
    """returns grid of points in a straight line around l0
    """

    l1 = l0-dlambda
    l2 = l0+dlambda

    em1 = np.interp(l1, wave, sed)
    em2 = np.interp(l2, wave, sed)

    return (em2-em1)/(l2-l1)*(wave-l1) + em1

def modify_peak(wave, sed, dlambda, sig, bias, a, diagnostics=False):
    """Replaces Lyman alpha absorbtion or emission with a gaussian.

    :param wave:
        Grid on which parameter sed is given (angstroms).

    :param sed:
        The sed

    :param dlambda:
        Estimation of width of lyman-alpha region. E.g. if dlambda = 60, 
        the lyman alpha peak will be assumed to be in the region of 
        1215.16*(1+redshift) +/- 60
    
    :param sig:
        Width of gaussian to replace peak with
    
    :param bias:
        Optionally shift peak by an amount given by this parameter in angstroms

    :param a:
        Flux retained in peak. Takes values between 0 and 1. A value of 1 means flux
        is conserved after modifying peak.
    
    :param diagnostics:
        If true, shows plots for diagnostics
    
    :returns modified_sed:
        New sed with replaced peak evaluated on the same grid in angstroms
    
    """

    #setup
    lyalpha = 1215.67#*(1+redshift)
    gaussian = (1/(np.sqrt(2*np.pi*sig**2)))*np.exp(-0.5*(wave-(lyalpha+bias))**2/(sig**2))

    #find peak
    indx = find_wave_range(wave, lyalpha, dlambda)
    flattened_peak = lin_interp(wave[indx], sed[indx], lyalpha, dlambda)

    #Find strength of emmission/absorbtion
    peak = np.copy(sed)
    peak[indx] = sed[indx] - flattened_peak
    area = np.trapz(peak[indx], wave[indx])
    sign = np.sign(area)
    area = abs(area)

    #modifying sed
    flattened_sed = np.copy(sed)
    modified_sed = np.copy(sed)

    if(sign == 1):
        flattened_sed[indx] = flattened_peak
        modified_sed = flattened_sed + gaussian*area*sign*a
        modified_sed = np.clip(modified_sed, 0 , None)

    #plotting
    if(diagnostics):
        indx = find_wave_range(wave, lyalpha, 500)
        plt.plot(wave[indx], sed[indx])
        plt.plot(wave[indx], flattened_sed[indx])
        plt.plot(wave[indx], modified_sed[indx])

    return modified_sed


