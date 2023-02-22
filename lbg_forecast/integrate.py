import jax.numpy as np

def histogram_integrator(bin_heights, bin_edges):
    return np.sum(bin_heights*np.diff(bin_edges))

