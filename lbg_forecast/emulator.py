import numpy as np
from speculator import Photulator
import tensorflow as tf
import lbg_forecast.cosmology as cosmo

class fsps_emulator:
    
    def __init__(self, path, rand=True):

        #Define attrributes
        self._models = []
        self._filters = ['u']#,'g','r','i','z']
        self.path = path
        print("Physical Devices:", tf.config.list_physical_devices())

        if(rand==True):
            tf.random.set_seed(42)
            np.random.seed(42)

        #load photulator
        for f in self._filters:
            self._models.append(Photulator(restore=True, restore_filename = self.path+"/trained_models/model_0x0lsst_"+f))

        self.model_params = [self._models[0].W, self._models[0].b, self._models[0].alphas, self._models[0].betas]

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
    def mimic_photometry(self, sps_params):

        photometry_all = []

        data_size = sps_params.shape[0]
        redshifts = sps_params[:, 0]
        photo_corrections = cosmo.wmap1_to_9(redshifts, path=self.path)

        sps_params_tensor = tf.cast(tf.convert_to_tensor(sps_params), tf.float32)
        self.diag(sps_params_tensor, self._models[0])

        photometry_bands = []
        for f in range(len(self._filters)):
            emulated_magnitudes = self._models[f].call(sps_params_tensor)
            photometry_bands.append(emulated_magnitudes.numpy())

        photometry_bands_array = np.asarray(photometry_bands)[:, :, 0].T
        photometry_all.append(photometry_bands_array + np.reshape(photo_corrections, (data_size, 1)))

        return np.hstack((np.asarray(photometry_all)))
    

        #forward pass for all filters
    def mimic_photometry_cpu(self, sps_params):

        photometry_all = []

        data_size = sps_params.shape[0]
        redshifts = sps_params[:, 0]
        photo_corrections = cosmo.wmap1_to_9(redshifts, path=self.path)

        photometry_bands = []
        for f in range(len(self._filters)):
            emulated_magnitudes = self._models[f].magnitudes_(sps_params)
            photometry_bands.append(emulated_magnitudes)

        photometry_bands_array = np.asarray(photometry_bands)[:, :, 0].T
        photometry_all.append(photometry_bands_array + np.reshape(photo_corrections, (data_size, 1)))

        return np.hstack((np.asarray(photometry_all)))
    
    def diag(self, parameters, model):

        outputs = []
        layers = [tf.divide(tf.subtract(parameters, model.parameters_shift), model.parameters_scale)]
        #no diff
        for i in range(model.n_layers - 1):
            
            # linear network operation
            outputs.append(tf.add(tf.matmul(layers[-1], model.W[i]), model.b[i]))
        
            if(i == 0):
                print("outputs i=0 :", tf.matmul(layers[-1], model.W[i]))
            
            if(i == model.n_layers - 2):
                print("outputs i=-1 :", tf.matmul(layers[-1], model.W[i]))
            
            # non-linear activation function
            layers.append(model.activation(outputs[-1], model.alphas[i], model.betas[i]))
            #print("layers :", layers[i])

        # linear output layer
        layers.append(tf.add(tf.matmul(layers[-1], model.W[-1]), model.b[-1]))

        #print("Layers :", layers)
        #diff here
            
        # rescale the output and return
        return tf.add(tf.multiply(layers[-1], model.magnitudes_scale), model.magnitudes_shift)