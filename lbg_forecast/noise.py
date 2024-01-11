from photerr import LsstErrorModel
import pandas as pd
import numpy as np

def select_u_dropouts(observed_catalog):
    
    udrop = observed_catalog.copy(deep=True)
    udrop = udrop.dropna(axis=0, subset=['i5'])
    #udrop = udrop.dropna(axis=0, subset=['r5'])
    #udrop = udrop.dropna(axis=0, subset=['g5'])
    #udrop = udrop.drop(udrop[np.isnan(udrop.u2) == False].index)

    return udrop.filter(['u','g','r','i','z','y'])

def select_g_dropouts(observed_catalog):
    
    gdrop = observed_catalog.copy(deep=True)
    gdrop = gdrop.dropna(axis=0, subset=['i5'])
    #gdrop = gdrop.dropna(axis=0, subset=['r5'])
    #gdrop = gdrop.drop(gdrop[np.isnan(gdrop.g2) == False].index)
    gdrop = gdrop.drop(gdrop[np.isnan(gdrop.u2) == False].index)

    
    return gdrop.filter(['u','g','r','i','z','y'])

def select_r_dropouts(observed_catalog):

    rdrop = observed_catalog.copy(deep=True)
    #rdrop = rdrop.dropna(axis=0, subset=['z5'])
    rdrop = rdrop.dropna(axis=0, subset=['i5'])
    #rdrop = rdrop.drop(rdrop[np.isnan(rdrop.r2) == False].index)
    rdrop = rdrop.drop(rdrop[np.isnan(rdrop.g2) == False].index)
    
    return rdrop.filter(['u','g','r','i','z','y'])

def get_noisy_magnitudes(sps_params, noiseless_photometry, random_state=42):

    #noiseless_photometry = np.load("/Users/fpetri/repos/LBGforecast/data/data/training_data.npy")[:1000000, :6]
    #sps_params = np.load("/Users/fpetri/repos/LBGforecast/data/data/training_params.npy")[:1000000, :]
    catalog = pd.DataFrame(noiseless_photometry, columns=['u', 'g', 'r', 'i', 'z', 'y'])

    # remove photometry brighter than brightness cut
    brightness_cut = 18
    for column in ['u','g','r','i','z','y']:
        catalog = catalog.drop(catalog[catalog[column] < brightness_cut].index)


    errModel = LsstErrorModel(sigLim=0, absFlux=True)
    sig5detections = LsstErrorModel(sigLim=5)
    sig2detections = LsstErrorModel(sigLim=2)

    observed_catalog = errModel(catalog, random_state=random_state).filter(['u', 'g', 'r', 'i', 'z', 'y']).replace([np.inf, -np.inf], np.nan, inplace=False)

    catalog_sig5 = sig5detections(observed_catalog, random_state=random_state).filter(['u', 'g', 'r', 'i', 'z', 'y']).replace([np.inf, -np.inf], np.nan, inplace=False)
    catalog_sig5.rename(columns={"u": "u5", "g": "g5", "r": "r5", "i": "i5", "z": "z5", "y": "y5" }, inplace=True)

    catalog_sig2 = sig2detections(observed_catalog, random_state=random_state).filter(['u', 'g', 'r', 'i', 'z', 'y']).replace([np.inf, -np.inf], np.nan, inplace=False)
    catalog_sig2.rename(columns={"u": "u2", "g": "g2", "r": "r2", "i": "i2", "z": "z2", "y": "y2" }, inplace=True)

    observed_catalog = observed_catalog.join(catalog_sig5).join(catalog_sig2)
    
    #observed_catalog.dropna(axis=0, subset=['i5'], inplace=True) #require 5sigma detection in i band

    udrop = select_u_dropouts(observed_catalog)
    gdrop = select_g_dropouts(observed_catalog)
    rdrop = select_r_dropouts(observed_catalog)

    u_dropouts = udrop.to_numpy()
    g_dropouts = gdrop.to_numpy()
    r_dropouts = rdrop.to_numpy()

    u_index = udrop.index.to_numpy().tolist()
    g_index = gdrop.index.to_numpy().tolist()
    r_index = rdrop.index.to_numpy().tolist()

    u_params = sps_params[u_index,:]
    g_params = sps_params[g_index,:]
    r_params = sps_params[r_index,:]

    u_data = [u_params, u_dropouts]
    g_data = [g_params, g_dropouts]
    r_data = [r_params, r_dropouts]

    return (u_data, g_data, r_data)