import numpy as np

def sources_inside_u_cuts(umg, gmr):
    """Given two (ndim,) arrays of u-g and g-r
    colours, return indexes of lbgs
    """
    
    return [np.where((umg > 1.5) & (gmr > -1.0) & (gmr < 1.2) & (umg > 0.75 + 1.5*gmr))[0]]

def sources_inside_g_cuts(gmr, rmi):
    """Given two (ndim,) arrays of g-r and r-i
    colours, return indexes of lbgs
    """
    
    return [np.where((gmr > 1.0) & (rmi < 1.0) & (gmr > 0.8 + 1.5*rmi))[0]]

def sources_inside_r_cuts(rmi, imz):
    """Given two (ndim,) arrays of r-i and i-z
    colours, return indexes of lbgs
    """
    
    return [np.where((rmi > 1.2) & (imz < 0.7) & (rmi > 1.0 + 1.5*imz))[0]]


def select_dropouts(dropout, dropout_colour_data):
    """Takes element of output from colour_cuts.colours
     (output[0] = udrop, output[1] =gdrop, output[2] = rdrop) and 
    returns selected redshifts for given dropout
    """
  
    sps_params, colours = dropout_colour_data

    umg = colours[:, 0]
    gmr = colours[:, 1]
    rmi = colours[:, 2]
    imz = colours[:, 3]

    if(dropout == 'u'):
        inds = sources_inside_u_cuts(umg, gmr)
        
    if(dropout == 'g'):
        inds = sources_inside_g_cuts(gmr, rmi)
        
    if(dropout == 'r'):
        inds = sources_inside_r_cuts(rmi, imz)

    return sps_params[:, 0][inds]



def apply_cuts(dropout_data):
    """takes output from colours() and returns
    redshift samples for each dropout class in list
    """

    u_dropouts, g_dropouts, r_dropouts = dropout_data

    #Select dropout sources
    u_redshifts = select_dropouts('u', u_dropouts)
    g_redshifts = select_dropouts('g', g_dropouts)
    r_redshifts = select_dropouts('r', r_dropouts)

    redshift_array = np.empty(3, object)
    redshift_array[:] = [u_redshifts, g_redshifts, r_redshifts]     

    return redshift_array


def colours(dropout_data):
    """Gets output from noise.get_noiseymagnitudes and converts
    magnitudes to colours
    """

    u_photometry = dropout_data[0][1]
    u_colours = calculate_colours(u_photometry)
    u_colour_dropout_data = [dropout_data[0][0], u_colours]

    g_photometry = dropout_data[1][1]
    g_colours = calculate_colours(g_photometry)
    g_colour_dropout_data = [dropout_data[1][0], g_colours]

    r_photometry = dropout_data[2][1]
    r_colours = calculate_colours(r_photometry)
    r_colour_dropout_data = [dropout_data[2][0], r_colours]

    return [u_colour_dropout_data, g_colour_dropout_data, r_colour_dropout_data]

# calculate colours of a set of photometry
def calculate_colours(photometry):
    
    photo1 = photometry[:,:-1]
    photo2 = photometry[:,1:]
    colours = photo1 - photo2

    return colours