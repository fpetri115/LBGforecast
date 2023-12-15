from photerr import LsstErrorModel
import pandas as pd
import numpy as np


def get_noisy_magnitudes(sps_params, noiseless_photometry, brightness_cut, random_state=42):

    df = pd.DataFrame(noiseless_photometry, columns=['u', 'g', 'r', 'i', 'z', 'y'])

    #generate two error models, one at 5sigma detection, one at 2
    errModel5sig = LsstErrorModel(sigLim=5)
    errModel2sig = LsstErrorModel(sigLim=2)

    siglim5 = errModel5sig.getLimitingMags()
    siglim2 = errModel2sig.getLimitingMags()

    catalog_with_errors5sig = errModel5sig(df, random_state=random_state).filter(['u', 'g', 'r', 'i', 'z', 'y'])

    catalog_with_errors2sig = errModel2sig(df, random_state=random_state).filter(['u', 'g', 'r', 'i', 'z', 'y'])
    catalog_with_errors2sig.rename(columns={"u": "u2", "g": "g2", "r": "r2", "i": "i2", "z": "z2", "y": "y2" }, inplace=True)

    #join catalogues
    catalog = catalog_with_errors5sig.join(catalog_with_errors2sig)

    # remove photometry brighter than brightness cut
    for column in ['u','g','r','i','z','y']:
        catalog = catalog.drop(catalog[catalog[column] < brightness_cut].index)

    udrop = catalog.replace([np.inf, -np.inf], np.nan, inplace=False).dropna(axis=0, subset=['g', 'r']).filter(['u','u2','g','r','i','z','y'])
    udrop = udrop.drop(udrop[np.isnan(udrop.u) == False].index).filter(['u2','g','r','i','z','y'])
    #udrop = udrop.drop(udrop[np.isnan(udrop.u2) == False].index)
    udrop['u2'].replace([np.nan], siglim2['u'], inplace=True)

    gdrop = catalog.replace([np.inf, -np.inf], np.nan, inplace=False).dropna(axis=0, subset=['r', 'i']).filter(['u2','g2','r','i','z','y'])
    gdrop = gdrop.drop(gdrop[np.isnan(gdrop.u2) == False].index) #require detections greater than 2sigma to be dropped in u
    #gdrop = gdrop.drop(gdrop[np.isnan(gdrop.g2) == False].index)
    gdrop['g2'].replace([np.nan], siglim2['g'], inplace=True)

    rdrop = catalog.replace([np.inf, -np.inf], np.nan, inplace=False).dropna(axis=0, subset=['i', 'z']).filter(['u','g2','r2','i','z','y'])
    rdrop = rdrop.drop(rdrop[np.isnan(rdrop.g2) == False].index) #require detections greater than 2sigma to be dropped in g
    #rdrop = rdrop.drop(rdrop[np.isnan(rdrop.r2) == False].index)
    rdrop['r2'].replace([np.nan], siglim2['r'], inplace=True)

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