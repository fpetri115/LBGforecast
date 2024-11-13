import numpy as np
import matplotlib.pyplot as plt

def sample_gaussian_prior_parameters():
        
    mu_bounds = np.array([[-2.5, 0.5],    #logzsol
                            [0.7, 1.3],   #igm_factor 
                            [-4.0, -1.0], #gas_logu
                            [-2.0, 0.5],  #gas_logz
                            [-5.0, 1.0],  #log10(fagn)
                            [5, 150]])    #agn_tau  

    sig_bounds = np.array([[0.01, 3.0],   #logzsol
                            [0.1, 0.1],   #igm_factor 
                            [0.01, 3.0],  #gas_logu
                            [0.01, 2.5],  #gas_logz
                            [0.01, 6.0],  #log10(fagn)
                            [0.1, 145]])  #agn_tau
    
    mu = []
    sigma = []

    for bounds in mu_bounds:
        mu.append(np.random.uniform(bounds[0], bounds[1]))
    for bounds in sig_bounds:
        sigma.append(np.random.uniform(bounds[0], bounds[1]))

    return [np.array(mu), np.array(sigma)]

def gaussian_parameter_names():

    return np.array(["logzsol", "igm_factor", "gas_logu", "gas_logz",
                     "log10fagn", "agntau"])

def plot_gaussian_prior_parameters(nsamples):

    mus = []
    sigs = []

    for i in range(nsamples):
        mu, sig = sample_gaussian_prior_parameters()
        mus.append(mu)
        sigs.append(sig)

    mu_arr = np.vstack(np.array(mus))
    sig_arr = np.vstack(np.array(sigs))
    names = gaussian_parameter_names()

    nparams = mu_arr.shape[1]
    f, ax = plt.subplots(nparams, 2, figsize=(15, 20))
    for i in range(nparams):
        ax[i, 0].hist(mu_arr[:, i], density=True)
        ax[i, 1].hist(sig_arr[:, i], density=True)

        ax[i, 0].set_xlabel(names[i])
        ax[i, 1].set_xlabel(names[i])




