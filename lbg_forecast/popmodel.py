import numpy as np
from numpy import random

def galaxy_population_model():

    age = 1e-2
    mass = 1e10
    tau = 1
    const = 0.2
    redshift = 3
    metal = -0.1
    dustesc = 8.0
    dust1 = 0.0
    dust2 = 0.1
    tburst = 1e-2
    fburst = 0.1
    igm = 1
    gas_ion = -2.0
    gas_z = -0.0
    fagn = 1
    imf1 = 1.3
    imf2 = 2.3
    imf3 = 2.3

    realisation = np.array([age, mass, tau, const, redshift, metal,
                            dustesc, dust1, dust2, tburst, fburst,
                            igm, gas_ion, gas_z, fagn, imf1, imf2, imf3])

    return realisation

