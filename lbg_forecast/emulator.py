import numpy as np
import tensorflow as tf
import pickle
from speculator import Photulator
import matplotlib.pyplot as plt

class fsps_emulator:
    
    def __init__(self, path):

        #Define attrributes
        self._models = []
        self._filters = ['u', 'g', 'r', 'i', 'z', 'y']

        #load photulator
        for f in self._filters:
            self._models.append(Photulator(restore=True, restore_filename = path+"/trained_models/model_4x128lsst_"+f))

    #forward pass for all filters
    def mimic_photometry(self, sps_params):

        photometry = []

        i = 0
        for f in self._filters:
            photometry.append(self._models[i].magnitudes_(sps_params))
            i+=1

        return np.hstack((np.asarray(photometry)))






