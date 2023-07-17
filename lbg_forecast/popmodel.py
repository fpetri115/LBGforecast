import numpy as np
import matplotlib.pyplot as plt

def galaxy_population_model(nsamples, pop_params):

    zmin = pop_params[0]

    tage = np.random.uniform(1e-1, 1e-1, nsamples)
    tau = np.random.uniform(1, 1, nsamples)
    const = np.random.uniform(0.2, 0.2, nsamples)
    zred = np.random.uniform(zmin, 3, nsamples)
    logzsol = np.random.uniform(-0.1, -0.1, nsamples)
    dust_tesc = np.random.uniform(8.0, 8.0, nsamples)
    dust1 = np.random.uniform(0.0, 0.0, nsamples)
    dust2 = np.random.uniform(0.1, 0.1, nsamples)
    tburst = np.random.uniform(1e-2, 1e-2, nsamples)
    fburst = np.random.uniform(0.1, 0.1, nsamples)
    igm_factor = np.random.uniform(1, 1, nsamples)
    gas_logu = np.random.uniform(-2.0, -2.0, nsamples)
    gas_logz = np.random.uniform(0.0, 0.0, nsamples)
    fagn = np.random.uniform(1, 1, nsamples)
    imf1 = np.random.uniform(1.3, 1.3, nsamples)
    imf2 = np.random.uniform(2.3, 2.3, nsamples)
    imf3 = np.random.uniform(2.3, 2.3, nsamples)
    mass = np.random.uniform(1e11, 1e11, nsamples)

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

def plot_galaxy_population(nsamples):

    realisation = galaxy_population_model(nsamples, np.array([3]))
    parameters = realisation.values()
    names = realisation.keys()

    columns = 3
    nparams = len(parameters)

    fig1, axes1 = plt.subplots(int(nparams/columns), columns, figsize=(20,20), sharex=False, sharey=False)

    i = 0
    j = 0
    for name in names:
        
        if(i > nparams/columns - 1):
            j+=1
            i=0

        axes1[i, j].hist(realisation[name], density = True)
        axes1[i, j].set_xlabel(name)
        axes1[i, j].set_ylabel("$p(z)$")
        i+=1

