import numpy as np

def sfr_to_zh(sfr, time_grid, mass, zgas):

    zmin = 0.2
    zh = []
    nsfr = len(sfr)
    sfr = sfr
    for i in range(0, nsfr):

        t = time_grid[:i+1]
        mass_formed_fraction = np.trapz((10**9)*sfr[:i+1], t)
        z_at_t = (zgas - zmin)*mass_formed_fraction + zmin
        zh.append(z_at_t)

    return np.asarray(zh)

