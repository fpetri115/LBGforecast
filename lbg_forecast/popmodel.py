import numpy as np
import matplotlib.pyplot as plt

def galaxy_population_model(pop_params):

    zmin = pop_params[0]

    tage = np.random.uniform(1e-1, 1e-1)
    tau = np.random.uniform(1, 1)
    const = np.random.uniform(0.2, 0.2)
    zred = np.random.uniform(zmin, 3)
    logzsol = np.random.uniform(-0.1, -0.1)
    dust_tesc = np.random.uniform(8.0, 8.0)
    dust1 = np.random.uniform(0.0, 0.0)
    dust2 = np.random.uniform(0.1, 0.1)
    tburst = np.random.uniform(1e-2, 1e-2)
    fburst = np.random.uniform(0.1, 0.1)
    igm_factor = np.random.uniform(1, 1)
    gas_logu = np.random.uniform(-2.0, -2.0)
    gas_logz = np.random.uniform(0.0, 0.0)
    fagn = np.random.uniform(1, 1)
    imf1 = np.random.uniform(1.3, 1.3)
    imf2 = np.random.uniform(2.3, 2.3)
    imf3 = np.random.uniform(2.3, 2.3)
    mass = np.random.uniform(1e11, 1e11)

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

