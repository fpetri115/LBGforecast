import sys
import numpy as np
from speculator import Photulator
import fsps
import matplotlib.pyplot as plt
import tensorflow as tf

dir = sys.argv[1]
ndata = sys.argv[4]

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

filter_list = fsps.find_filter('lsst') + fsps.find_filter('suprimecam')[1:2]+fsps.find_filter('suprimecam')[3:]
print(filter_list)

#training data
spsparams = np.load(dir+"/training_params.npy")[:ndata].astype(np.float32)
photometry = np.load(dir+"/training_data.npy")[:ndata].astype(np.float32)
print(spsparams.shape, photometry.shape)

# parameters shift and scale
parameters_shift = np.mean(spsparams, axis=0)
parameters_scale = np.std(spsparams, axis=0)
magnitudes_shift = np.mean(photometry, axis=0)
magnitudes_scale = np.std(photometry, axis=0)

#select filters
select = int(sys.argv[3])
filters = filter_list[select:select+1]
training_theta = tf.convert_to_tensor(spsparams)
training_mag = tf.convert_to_tensor(photometry[:,select:select+1])
print(filter)

# training set up
validation_split = 0.1
lr = [1e-3, 1e-4, 1e-5, 1e-5]
batch_size = [1000, 10000, 30000, int((1-validation_split) * training_theta.shape[0])]
gradient_accumulation_steps = [1, 1, 1, 10]

# early stopping set up
patience = int(sys.argv[2])

# architecture
n_layers = 4
n_units = 128

#extra params
verbose = True
disable_early_stopping=True

#optimiser
optimiser = tf.keras.optimizers.legacy.Adam()

#running loss
running_loss = []
running_val_loss = []
running_lr = []

# architecture
n_hidden = [n_units]*n_layers

# train each band in turn
for f in range(len(filters)):

    if verbose is True:
        print('filter ' + filters[f] + '...')

    # construct the PHOTULATOR model
    photulator = Photulator(n_parameters=training_theta.shape[-1], 
                        filters=[filters[f]], 
                        parameters_shift=parameters_shift, 
                        parameters_scale=parameters_scale, 
                        magnitudes_shift=magnitudes_shift[f], 
                        magnitudes_scale=magnitudes_scale[f], 
                        n_hidden=[128, 128, 128, 128], 
                        restore=False, 
                        restore_filename=None,
                        optimizer=optimiser)

    # train using cooling/heating schedule for lr/batch-size
    for i in range(len(lr)):
        
        if verbose is True:
            print('learning rate = ' + str(lr[i]) + ', batch size = ' + str(batch_size[i]))

        # set learning rate
        photulator.optimizer.lr = lr[i]

        # split into validation and training sub-sets
        n_validation = int(training_theta.shape[0] * validation_split)
        n_training = training_theta.shape[0] - n_validation
        training_selection = tf.random.shuffle([True] * n_training + [False] * n_validation)

        # create iterable dataset (given batch size)
        train_mag = tf.expand_dims(training_mag[:,f],-1)
        training_data = tf.data.Dataset.from_tensor_slices((training_theta[training_selection], train_mag[training_selection])).shuffle(n_training).batch(batch_size[i])

        # set up training loss
        training_loss = [np.infty]
        validation_loss = [np.infty]
        best_loss = np.infty
        early_stopping_counter = 0

        # loop over epochs
        while early_stopping_counter < patience:

            # loop over batches for a single epoch
            for theta, mag in training_data:

                # training step: check whether to accumulate gradients or not (only worth doing this for very large batch sizes)
                if gradient_accumulation_steps[i] == 1:
                    loss = photulator.training_step(theta, mag)
                else:
                    loss = photulator.training_step_with_accumulated_gradients(theta, mag, accumulation_steps=gradient_accumulation_steps[i])

                running_loss.append(loss)
                running_lr.append(photulator.optimizer.lr)

            # compute total loss and validation loss
            validation_loss.append(photulator.compute_loss(training_theta[~training_selection], train_mag[~training_selection]).numpy())

            # early stopping condition
            if validation_loss[-1] < best_loss:
                best_loss = validation_loss[-1]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= patience:
                photulator.update_emulator_parameters()
                photulator.save('model_{}x{}'.format(n_layers, n_units) + filters[f])
                if verbose is True:
                    print('Validation loss = ' + str(best_loss))
        
        for v in validation_loss:
            running_val_loss.append(v)

np.save("loss_"+filters[f]+".npy", running_loss)
np.save("valloss_"+filters[f]+".npy", running_val_loss)
np.save("lr_"+filters[f]+".npy", running_lr)