import sys

import numpy as np
from speculator import  Photulator
import fsps
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


filter = int(sys.argv[1])
device = sys.argv[2]
nbatchs = int(sys.argv[3])
epochs = int(sys.argc[4])


filter_list = fsps.find_filter('lsst') + fsps.find_filter('suprimecam')[1:2]+fsps.find_filter('suprimecam')[3:]

spsparams = np.load("data/data/training_params.npy")
photometry = np.load("data/data/training_data.npy")
print(spsparams.shape, photometry.shape)

# parameters shift and scale
parameters_shift = np.mean(spsparams, axis=0)
parameters_scale = np.std(spsparams, axis=0)
magnitudes_shift = np.mean(photometry, axis=0)
magnitudes_scale = np.std(photometry, axis=0)

training_theta = torch.from_numpy(spsparams).float().to(device)
training_mags = torch.from_numpy(photometry[:,filter]).float().to(device)

n_parameters = spsparams.shape[-1]
n_layers=4
n_units=128
n_hidden = [n_units]*n_layers

photulator = Photulator(n_parameters=n_parameters,
                           filters=[filter_list[filter]],
                           parameters_shift=parameters_shift,
                           parameters_scale=parameters_scale,
                           magnitudes_shift=magnitudes_shift[filter],
                           magnitudes_scale=magnitudes_scale[filter],
                           n_hidden=n_hidden,
                           optimizer=lambda x: torch.optim.Adam(x, lr=1e-3),
                           device=device)

#create dataset
training_dataset = torch.utils.data.TensorDataset(training_theta, training_mags)
training_data, validation_data = torch.utils.data.random_split(training_dataset, [0.90, 0.1])
validation_theta, validation_mag = validation_data[:]

trainloader = torch.utils.data.DataLoader(training_data, batch_size=nbatchs,
                                          shuffle=True)

optimiser = torch.optim.Adam(photulator.parameters(), lr=1e-3)

j = 0
loss_list = []
validation_loss = []
#loop over dataset multiple times
for epoch in range(epochs):
    print("epoch ", j)
    #loops over batches
    i = 0
    for th, mgs in trainloader:

        # zero the parameter gradients
        optimiser.zero_grad()

        th.to(device)
        mgs.to(device)

        loss = photulator.compute_loss(th, mgs)
        loss.backward()
        optimiser.step()
        
        loss_list.append(loss.item())

        if(i%100 == 0):
            print(i)
        i+=1
    
    validation_loss.append(photulator.compute_loss(validation_theta, torch.reshape(validation_mag, (validation_mag.shape[0], 1))).cpu().detach().numpy())
    j+=1

np.save(loss_list)
np.save(validation_loss)

torch.save(photulator, 'model'+ filter_list[filter])