import numpy as np

def tau_model(tau, t):
    return np.exp(-t/tau)


