import numpy as np
import matplotlib.pyplot as plt

def tau_model(tau, t):
    return np.exp(-t/tau)

def dpl(a, b, tau, t):
    return ((t/tau)**(a) + (t/tau)**(-b))**(-1)

def plot_sfh(sfh, t):
    return plt.plot(t, sfh)
