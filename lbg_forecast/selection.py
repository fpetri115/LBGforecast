import numpy as np

# select sources given some detection limits
def select_magnitudes(sps_params, source_photometry, detection_limits):
    
    u_dropouts = []
    u_dropout_params = []
    g_dropouts = []
    g_dropout_params = []
    r_dropouts = []
    r_dropout_params =[]

    #See goldrush I and IV
    source_index = 0
    n_sources = len(sps_params)
    while(source_index < n_sources):

        source = source_photometry[source_index, :]

        #u-dropout - require detection in g,r,i
        if(source[3]<detection_limits[3] and source[2]<detection_limits[2] and source[1]<detection_limits[1]):
            u_dropouts.append(source)
            u_dropout_params.append(sps_params[source_index, :])

        #g-dropout - require detection in r,i; no detection in u
        if(source[3]<detection_limits[3] and source[2]<detection_limits[2] and source[0]>detection_limits[0]):
            g_dropouts.append(source)
            g_dropout_params.append(sps_params[source_index, :])

        #r_dropout - require detection in i,z; no detection in g    
        if(source[3]<detection_limits[3] and source[4]<detection_limits[4] and source[1]>detection_limits[1]):
            r_dropouts.append(source)
            r_dropout_params.append(sps_params[source_index, :])

        source_index+=1

    u_dropouts = np.hstack((np.asarray([u_dropouts])))
    g_dropouts = np.hstack((np.asarray([g_dropouts])))
    r_dropouts = np.hstack((np.asarray([r_dropouts])))

    u_dropout_params = np.hstack((np.asarray([u_dropout_params])))
    g_dropout_params = np.hstack((np.asarray([g_dropout_params])))
    r_dropout_params = np.hstack((np.asarray([r_dropout_params])))

    u_data = [u_dropout_params, u_dropouts]
    g_data = [g_dropout_params, g_dropouts]
    r_data = [r_dropout_params, r_dropouts]
        
    return (u_data, g_data, r_data)

#takes dropout_data from select_magnitudes() and coverts magnitudes to colours for a given dropout
def colours(dropout_data):

    u_photometry = dropout_data[0][1]
    u_colours = calculate_colours(u_photometry)
    dropout_data[0][1] = u_colours

    g_photometry = dropout_data[1][1]
    g_colours = calculate_colours(g_photometry)
    dropout_data[1][1] = g_colours

    r_photometry = dropout_data[2][1]
    r_colours = calculate_colours(r_photometry)
    dropout_data[2][1] = r_colours

    return dropout_data

# calculate colours of a set of photometry
def calculate_colours(photometry):
    
    photo1 = photometry[:,:-1]
    photo2 = photometry[:,1:]
    colours = photo1 - photo2

    return colours