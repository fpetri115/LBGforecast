from astropy.cosmology import WMAP1 as wmap1
from astropy.cosmology import WMAP9 as wmap9


def get_cosmology():
    return wmap1

def get_wmap1():
    return wmap1

def get_wmap9():
    return wmap9

def scale_to_z(scale):
    """ 1+z = 1/a
    """
    return 1/scale - 1.0