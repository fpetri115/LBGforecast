import numpy as np
from speculator import Photulator
import lbg_forecast.cosmology as cosmo

class fsps_emulator:
    
    def __init__(self, path):

        #Define attrributes
        self._models = []
        self._filters = ['u','g','r','i','z']

        #load photulator
        for f in self._filters:
            self._models.append(Photulator(restore=True, restore_filename = path+"/trained_models/model_0x0lsst_"+f))

    #forward pass for all filters
    def mimic_photometry_wmap1(self, sps_params, batch_size):

        photometry_all = []

        data_size = sps_params.shape[0]

        if(data_size%batch_size != 0):
            raise Exception("batch sizes do not fit")
        
        nbatches = int(data_size/batch_size)

        i = 0
        for f in self._filters:
            photometry_band = []
            for n in range(nbatches):
                photometry_band.append(self._models[i].magnitudes_(sps_params[n*batch_size:(n+1)*batch_size]))
            i+=1
            photometry_all.append(np.reshape(np.asarray(photometry_band), (data_size, 1)))

        return np.hstack((np.asarray(photometry_all)))

    #forward pass for all filters
    def mimic_photometry(self, sps_params, batch_size):

        photometry_all = []

        data_size = sps_params.shape[0]
        redshifts = sps_params[:, 0]
        photo_corrections = cosmo.wmap1_to_9(redshifts)

        if(data_size%batch_size != 0):
            raise Exception("batch sizes do not fit")
        
        nbatches = int(data_size/batch_size)

        i = 0
        for f in self._filters:
            photometry_band = []
            for n in range(nbatches):
                photometry_band.append(self._models[i].magnitudes_(sps_params[n*batch_size:(n+1)*batch_size]))
            i+=1
            photometry_all.append(np.reshape(np.asarray(photometry_band), (data_size, 1)) + np.reshape(photo_corrections, (data_size, 1)))

        return np.hstack((np.asarray(photometry_all)))