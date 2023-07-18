import numpy as np
import matplotlib.pyplot as plt

def tau_model(tau, t):
    return np.exp(-t/tau)

def plot_sfh(sfh, t):
    return plt.plot(t, sfh)
