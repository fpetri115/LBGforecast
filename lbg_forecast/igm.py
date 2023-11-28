import numpy as np
import matplotlib.pyplot as plt

#(Wavelengths shortward of Lyman Limit)
def lyman_continuum_tau(l_obs, z_em):

    l_lylim = 912

    x_c = l_obs/l_lylim
    x_em = 1+z_em

    tau_eff = (0.25*(x_c**3)*(x_em**0.46 - x_c**0.46) \
                +9.4*(x_c**1.5)*(x_em**0.18 - x_c**0.18) \
                -0.7*(x_c**3)*(x_c**(-1.32) - x_em**(-1.32)) \
                -0.023*(x_em**1.68 - x_c**1.68))
    
    tau_eff = np.where(l_obs < l_lylim*(1+z_em), tau_eff, 0)
    
    return tau_eff

def lyman_series_line_tau(l_obs, z_em):

    ly_series = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803,
          930.748, 926.226, 923.150, 920.963, 919.352,
          918.129, 917.181, 916.429, 915.824, 915.329,
          914.919, 914.576])
    
    ly_series = np.reshape(ly_series, (17, 1))
    
    coeffs = np.array([0.0036,0.0017,0.0011846,0.0009410,0.0007960,
       0.0006967,0.0006236,0.0005665,0.0005200,0.0004817,
       0.0004487,0.0004200,0.0003947,0.000372,0.000352,
       0.0003334,0.00031644])
    
    coeffs = np.reshape(coeffs, (17, 1))

    no_lines = len(ly_series)
    l_rest = l_obs/(1+z_em)
    l_rest = np.tile(l_rest, (no_lines, 1)) #tiles into rows
    l_obs = np.tile(l_obs, (no_lines, 1))

    bool = (l_rest < ly_series)

    tau_eff_all = coeffs*((l_obs/ly_series)**(3.46))
    tau_eff_reduced = tau_eff_all*bool #true, false turns to 1, 0

    return np.sum(tau_eff_reduced, axis=0) #sum columns (optical depths are additive)

def lyalpha_transmission(z, tau=False):

    if(tau):
        return -np.log(np.exp(-3.6e-3*(1+z)**3.46))
    else:
        return np.exp(-3.6e-3*(1+z)**3.46)
    
def plot_lya_transmission(zmin, zmax, tau=False):

    z_space = np.linspace(zmin, zmax, 1000)
    return plt.plot(z_space, lyalpha_transmission(z_space, tau))

def plot_transmission_curves(z):

    l_min = 750*(1+z)
    l_max = 1216*(1+z)
    l_obs = np.linspace(l_min, l_max, int(l_max-l_min)*10)
    tau_eff = lyman_continuum_tau(l_obs, z)
    tau_eff_series = lyman_series_line_tau(l_obs, z)

    plt.figure(figsize=(10,5))
    plt.plot(l_obs, np.exp(-1*(tau_eff + tau_eff_series)))
    plt.plot(l_obs, np.exp(-1*(tau_eff + tau_eff_series))*1.5)
    plt.plot(l_obs, np.exp(-1*(tau_eff + tau_eff_series))*0.5)

