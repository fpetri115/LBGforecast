import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.distributions as dstr

def sample_lognormal(mu, sig, min, max):

    param = np.random.lognormal(dstr.mu_lognorm(mu, sig), dstr.sig_lognorm(mu, sig))
    while(param < min or param > max):
         param = np.random.lognormal(dstr.mu_lognorm(mu, sig), dstr.sig_lognorm(mu, sig))

    return param

def sample_normal(mu, sig, min, max):

    param = np.random.normal(mu, sig)
    while(param < min or param > max):
         param = np.random.normal(mu, sig)

    return param

def lognormal_pdf(x, mu, sig, log=False):

     pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sig**2))
       / (x * sig * np.sqrt(2 * np.pi)))
     
     if(log):
          plt.xscale("log")
     
     return plt.plot(x, pdf)

def lognormal_mean(mu, sig):
     return np.exp(mu + (sig*sig)/2)

def mu_lognorm(mu, sig):
     return np.log((mu*mu)/np.sqrt(mu*mu+sig*sig))

def sig_lognorm(mu, sig):
     return np.sqrt(np.log(1+(sig*sig)/(mu*mu)))