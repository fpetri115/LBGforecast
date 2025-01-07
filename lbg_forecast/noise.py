from photerr import LsstErrorModel
import pandas as pd
import numpy as np

def select_u_dropouts(observed_catalog, depth):
    
    udrop = observed_catalog.copy(deep=True)

    #udrop = udrop.dropna(axis=0, subset=['g5'])
    #udrop = udrop.dropna(axis=0, subset=['r5'])
    #udrop = udrop.dropna(axis=0, subset=['u5']) 

    #udrop = udrop.drop(udrop[np.isnan(udrop.u2) == False].index)
    #udrop['u'].replace(np.nan, 30.0, inplace=True)

    udrop = udrop.drop(udrop[udrop.r < 23].index)
    udrop = udrop.drop(udrop[udrop.r > depth].index)

    #udrop = udrop.drop(udrop[np.isnan(udrop.r5) == True].index)

    return udrop.filter(['u','g','r','i','z'])

def select_g_dropouts(observed_catalog, depth):
    
    gdrop = observed_catalog.copy(deep=True)

    #gdrop = gdrop.dropna(axis=0, subset=['r5'])
    #gdrop = gdrop.dropna(axis=0, subset=['i5'])
    #gdrop = gdrop.dropna(axis=0, subset=['g']) 

    gdrop = gdrop.drop(gdrop[gdrop.i < 23].index)
    #gdrop = gdrop.drop(gdrop[gdrop.i > depth].index)

    gdrop = gdrop.drop(gdrop[np.isnan(gdrop.i5) == True].index)
    #gdrop = gdrop.drop(gdrop[np.isnan(gdrop.u2) == False].index)

    return gdrop.filter(['u','g','r','i','z'])

def select_r_dropouts(observed_catalog, depth):

    rdrop = observed_catalog.copy(deep=True)

    #rdrop = rdrop.dropna(axis=0, subset=['i5'])
    #rdrop = rdrop.dropna(axis=0, subset=['z5'])
    #rdrop = rdrop.dropna(axis=0, subset=['r']) 

    rdrop = rdrop.drop(rdrop[rdrop.z < 23].index)
    #rdrop = rdrop.drop(rdrop[rdrop.z > depth].index)
    
    rdrop = rdrop.drop(rdrop[np.isnan(rdrop.z5) == True].index)
    rdrop = rdrop.drop(rdrop[np.isnan(rdrop.g2) == False].index)

    return rdrop.filter(['u','g','r','i','z'])

def setup_catalog(noiseless_photometry):

    random_state = np.random.randint(0, 1000000000)
    catalog = pd.DataFrame(noiseless_photometry, columns=['u', 'g', 'r', 'i', 'z'])

    errModel = LsstErrorModel(sigLim=0, absFlux=True)
    sig5detections = LsstErrorModel(sigLim=5, absFlux=True)
    sig2detections = LsstErrorModel(sigLim=2, absFlux=True)

    observed_catalog = errModel(catalog, random_state=random_state).filter(['u', 'g', 'r', 'i', 'z']).replace([np.inf, -np.inf], np.nan, inplace=False)
    #observed_catalog = observed_catalog.dropna(axis=0)
    catalog_sig5 = sig5detections(observed_catalog, random_state=random_state).filter(['u', 'g', 'r', 'i', 'z']).replace([np.inf, -np.inf], np.nan, inplace=False)
    catalog_sig5.rename(columns={"u": "u5", "g": "g5", "r": "r5", "i": "i5", "z": "z5"}, inplace=True)

    catalog_sig2 = sig2detections(observed_catalog, random_state=random_state).filter(['u', 'g', 'r', 'i', 'z']).replace([np.inf, -np.inf], np.nan, inplace=False)
    catalog_sig2.rename(columns={"u": "u2", "g": "g2", "r": "r2", "i": "i2", "z": "z2"}, inplace=True)

    observed_catalog = observed_catalog.join(catalog_sig5).join(catalog_sig2)

    return observed_catalog

def pack_dropout_data(dropouts, sps_params):

    dropout_mags_numpy = dropouts.to_numpy()
    dropout_index = dropouts.index.to_numpy()
    dropout_params = sps_params[dropout_index,:]
    dropout_data = [dropout_params, dropout_mags_numpy]

    return dropout_data

def unpack_z(dropout_data):

    dropout_params = dropout_data[0]

    return dropout_params[:, 0]

def unpack_spsparams(dropout_data):

    dropout_params = dropout_data[0]

    return dropout_params

def unpack_mags(dropout_data, band):

    dropout_mags = dropout_data[1]
    
    return dropout_mags[:, band]

def unpack_mags_all(dropout_data):

    dropout_mags = dropout_data[1]
    
    return dropout_mags

def get_noisy_magnitudes(sps_params, noiseless_photometry, udepth=25.3, gdepth=25, rdepth=25):

    observed_catalog = setup_catalog(noiseless_photometry)

    u_dropouts = select_u_dropouts(observed_catalog, udepth)
    g_dropouts = select_g_dropouts(observed_catalog, gdepth)
    r_dropouts = select_r_dropouts(observed_catalog, rdepth)

    u_data = pack_dropout_data(u_dropouts, sps_params)
    g_data = pack_dropout_data(g_dropouts, sps_params)
    r_data = pack_dropout_data(r_dropouts, sps_params)


    return (u_data, g_data, r_data)