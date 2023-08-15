import numpy as np
import matplotlib.pyplot as plt

def tau_model(tau, t):
    return np.exp(-t/tau)

def dpl(tau, a, b, t):
    return ((t/tau)**(a) + (t/tau)**(-b))**(-1)

def normed_sfh(logtau, loga, logb, t):

    sfh = dpl(10**logtau, 10**loga, 10**logb, t)
    normed_sfh = sfh/np.trapz((10**9)*sfh, t)

    return normed_sfh

def plot_sfh(sfh, t):

    plt.figure(figsize=(10,5))
    plt.plot(t, sfh)
    plt.xlabel("Time Since the Beginning of the Universe [$\mathrm{Gyr}$]",
               fontsize=12)
    plt.ylabel("Star Formation Rate [$\mathrm{M}_{\odot}\mathrm{yr}^{-1}$]",
               fontsize=12)
    
    plt.tick_params(axis="x", width = 2, labelsize=12*0.8)
    plt.tick_params(axis="y", width = 2, labelsize=12*0.8)
