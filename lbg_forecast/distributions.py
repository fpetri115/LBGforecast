import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.distributions as dstr

#sample uniform distribution or truncated normal
def sample_prior(hparams):

     bound, p1, p2 = hparams
     distribution = bound[0]

     if(distribution == 0):
          param = np.random.uniform(p1, p2)
     
     if(distribution == 1):
          param = np.random.normal(p1, p2)
          bmin = bound[1]
          bmax = bound[2]
          while(param < bmin or param > bmax):
               param = np.random.normal(p1, p2)

     if(distribution != 1 and distribution != 0):
          raise Exception("Unknown Distribution, bound[0] must be int < 2")

     return param

# sample/set prior parameters depending on distribution specified
def sample_hyperparams(bound, sig_min):

     distribution, bmin, bmax = bound

     #uniform - fixed
     if(distribution == 0):

          uniform_min = bmin
          uniform_max = bmax

          return np.array([bound, uniform_min, uniform_max], dtype=object)
     
     #gaussian - sample mu and sigma
     if(distribution == 1):

          mu_min = bmin
          mu_max = bmax

          mu = np.random.uniform(mu_min, mu_max)

          #minimum gaussian width = sig_min, max given by range of means allowed
          sig = np.random.uniform(sig_min, (mu_max-mu_min)+sig_min)

          return np.array([bound, mu, sig], dtype=object)
     
     if(distribution != 1 and distribution != 0):
          raise Exception("Unknown Distribution, bound[0] must be int < 2")