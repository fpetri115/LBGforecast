import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.sfh as sfh

def sps_parameters_to_zhistory(sps_parameters, agebins):
    """Plots metallicity history given sps parameters and 
    prints absolute metallicity of observed galaxy and total mass formed
    
    """
    tabulated_sfh, masses = sfh.continuity_sfh(sfh.zred_to_agebins(sps_parameters[0], agebins), 
                                        sps_parameters[9:-1], sps_parameters[-1])
    
    Z_MIST = 0.0142 #solar metallicity for MIST
    metallicity_history = sfr_to_zh(tabulated_sfh[1], tabulated_sfh[0], (10**sps_parameters[1]*Z_MIST), sps_parameters[-1])
    plt.plot(tabulated_sfh[0], metallicity_history)
    plt.xlabel("Time Since the Beginning of the Universe [$\mathrm{Gyr}$]",
            fontsize=12)
    plt.ylabel("Chemical Evolution [$\mathrm{Absolute \  Metallicity}$]",
            fontsize=12)
    
    plt.tick_params(axis="y", width = 2, labelsize=12*0.8)
    plt.tick_params(axis="x", width = 2, labelsize=12*0.8)
    print(Z_MIST*10**sps_parameters[1], metallicity_history[-1], np.sum(masses), sps_parameters[-1])
    

def sfr_to_zh(sfr, time_grid, zgas, total_mass_formed):
    """Calculates chemical evolution in absolute metallicity given
    a SFH
    
    """
    zmin = 10**(-2.5)*0.0142 #converting min z in isochrone from logzsol to absolute metallicity
    zh = []
    nsfr = len(sfr)
    sfr = sfr
    for i in range(0, nsfr):
        t = time_grid[:i+1]
        mass_formed_fraction = np.trapz((10**9)*sfr[:i+1], t)/total_mass_formed
        z_at_t = (zgas - zmin)*mass_formed_fraction + zmin
        zh.append(z_at_t)

    return np.asarray(zh)

