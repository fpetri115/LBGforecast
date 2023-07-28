import numpy as np

def sample_lognormal(mu, sig, min, max):

    param = np.random.lognormal(np.log(10**mu), np.log(10**sig))
    while(param < min or param > max):
         param = np.random.lognormal(np.log(10**mu), np.log(10**sig))

    return param

def sample_normal(mu, sig, min, max):

    param = np.random.normal(mu, sig)
    while(param < min or param > max):
         param = np.random.normal(mu, sig)

    return param