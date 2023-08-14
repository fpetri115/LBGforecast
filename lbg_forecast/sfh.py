import numpy as np
import matplotlib.pyplot as plt

def tau_model(tau, t):
    return np.exp(-t/tau)

def dpl(tau, a, b, t):
    return ((t/tau)**(a) + (t/tau)**(-b))**(-1)

# metallicity at max(t)
def z(t, normedsfr, zgas, zmin, mass):
    
    f_mass_formed = np.trapz(normedsfr, t) #does this work? tgird != sfr grid?

    return (zgas - zmin)*f_mass_formed/mass + zmin


def plot_sfh(sfh, t):

    plt.figure(figsize=(10,5))
    plt.plot(t, sfh)
    plt.xlabel("Time Since the Beginning of the Universe [$\mathrm{Gyr}$]",
               fontsize=12)
    plt.ylabel("Star Formation Rate [$\mathrm{M}_{\odot}\mathrm{yr}^{-1}$]",
               fontsize=12)
    
    plt.tick_params(axis="x", width = 2, labelsize=12*0.8)
    plt.tick_params(axis="y", width = 2, labelsize=12*0.8)
