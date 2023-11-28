import numpy as np
import tensorflow as tf
import pickle
from speculator import Photulator
import matplotlib.pyplot as plt

import os

class fsps_emulator:
    
    def __init__(self):

        #Define attrributes
        self._models = []
        self._ndata = 1000000
        self._filters = ['u', 'g', 'r', 'i', 'z', 'y']

        # change directory
        os.chdir('/Users/fpetri/repos/LBGforecast/models')

        # load training data
        self._sps_params = np.load("/Users/fpetri/repos/LBGforecast/data/data/training_params.npy")[:self._ndata]
        self._training_photometry = np.load("/Users/fpetri/repos/LBGforecast/data/data/training_data.npy")[:self._ndata]

        #load photulator
        for f in self._filters:
            self._models.append(Photulator(restore=True, restore_filename = "model_4x128lsst_"+f))

    #forward pass for all filters
    def mimic_photometry(self, sps_params):

        photometry = []

        i = 0
        for f in self._filters:
            photometry.append(self._models[i].magnitudes_(sps_params))
            i+=1

        return np.hstack((np.asarray(photometry)))






