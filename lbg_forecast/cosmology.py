from astropy.cosmology import WMAP1 as cosmo


def get_cosmology():
    return cosmo

def scale_to_z(scale):
    """ 1+z = 1/a
    """
    return 1/scale - 1.0