import numpy as np

def cal00(lam):

    rv = 4.05
    lam=lam/1e4
    return np.where(lam < 0.63,
                     2.659*(-2.156 + (1.509/lam) - (0.198/(lam**2)) + (0.011/(lam**3))) + rv,
                     2.659*(-1.857 + (1.04/lam)) + rv)

def drude(lam, index):
    eb = 0.85 - 1.9*index
    dlam = 350

    return (eb*(lam*dlam)**2)/(((lam**2)-(2175**2))**2+(lam*dlam)**2)

def bc_attenuation(lam, a1):
    return a1*(lam/5500)**(-1.0)

def dif_attenuation(lam, a2, index):
    return (a2/4.05)*(cal00(lam)+drude(lam, index))*(lam/5500)**(index)

def total_attenuation(lam, a1, a2, index):
    return bc_attenuation(lam, a1) + dif_attenuation(lam, a2, index)

def tau_to_a(tau):
    return 2.5*np.log10(np.e)*tau

def a_to_tau(a):
    return a/(2.5*np.log10(np.e))

def sps_to_tauuv(tau1, tau2, index):
    auv = total_attenuation(1500, tau_to_a(tau1), tau_to_a(tau2), index)
    return a_to_tau(auv)

def sps_to_tauv(tau1, tau2, index):
    auv = total_attenuation(5500, tau_to_a(tau1), tau_to_a(tau2), index)
    return a_to_tau(auv)

def uv_slope(tau1, tau2, index):
    av = total_attenuation(5500, tau_to_a(tau1), tau_to_a(tau2), index)
    a_fuv = total_attenuation(1500, tau_to_a(tau1), tau_to_a(tau2), index)

    return a_fuv/av