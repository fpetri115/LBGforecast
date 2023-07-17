import numpy as np
import matplotlib.pyplot as plt

def galaxy_population_model(pop_params):

    zmin = pop_params[0]

    tage = 1e-1
    tau = 1
    const = 0.2
    zred = np.random.uniform(zmin, 3)
    logzsol = -0.1
    dust_tesc = 8.0
    dust1 = 0.0
    dust2 = 0.1
    tburst = 1e-2
    fburst = 0.1
    igm_factor = 1
    gas_logu = -2.0
    gas_logz = -0.0
    fagn = 1
    imf1 = 1.3
    imf2 = 2.3
    imf3 = 2.3
    mass = 1e11

    realisation = {
        'tage': tage,
        'tau': tau,
        'const': const,
        'zred': zred,
        'logzsol': logzsol,
        'dust_tesc': dust_tesc,
        'dust1': dust1,
        'dust2': dust2,
        'tburst': tburst,
        'fburst': fburst,
        'igm_factor': igm_factor,
        'gas_logu': gas_logu,
        'gas_logz': gas_logz,
        'fagn': fagn,
        'imf1': imf1,
        'imf2': imf2,
        'imf3': imf3,
        'mass': mass
    }

    return realisation

def plot_galaxy_population():

    return 0

