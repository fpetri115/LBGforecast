from astropy.cosmology import WMAP1 as wmap1
from astropy.cosmology import WMAP9 as wmap9
import numpy as np


def get_cosmology():
    return wmap9

def get_wmap1():
    return wmap1

def get_wmap9():
    return wmap9

def scale_to_z(scale):
    """ 1+z = 1/a
    """
    return 1/scale - 1.0

def wmap1_to_9(new_redshifts):
    """convert wmap1 photometry to wmap9
    (all bands)

    params
    ---------------- 
    :new_redshifts: redshifts at which to evaluate photometry at


    returns
    ----------------
    photometry (any band) at wmap9 for given redshifts
    """
    
    wmap1_to_9 = np.loadtxt("corrections/wmap1_to_9.txt")
    redshifts = wmap1_to_9[0,:]
    phot_corrections = wmap1_to_9[1,:]

    return np.interp(new_redshifts, redshifts, phot_corrections)