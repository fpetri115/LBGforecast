import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.distributions as dstr
from scipy.stats import truncnorm

#sample uniform distribution or truncated normal
def sample_prior(hparams):

     bound, p1, p2 = hparams
     distribution = bound[0]

     #uniform
     if(distribution == 0):
          param = np.random.uniform(p1, p2)
     
     #truncated gaussian with free parameters
     if(distribution == 1):
          bmin = bound[1]
          bmax = bound[2]

          a, b = (bmin - p1) / p2, (bmax - p1) / p2
          param = truncnorm.rvs(a, b, loc=p1, scale=p2)

     #(for igm_factor)
     if(distribution == 2):
          param = np.random.normal(1, 0.5)
          bmin = bound[1]
          bmax = bound[2]

          while(param < bmin or param > bmax):
               param = np.random.normal(p1, p2)

     if(distribution != 1 and distribution != 0 and distribution != 2):
          raise Exception("Unknown Distribution, bound[0] must be int < 3")

     return param

#vectorised
def sample_prior_vec(hparams, nsamples, vectorise_bounds=0):

     bound, p1, p2 = hparams
     distribution = bound[0]

     #uniform
     if(distribution == 0):

          if(isinstance(vectorise_bounds, int) == False):
               p1 = [p1]*nsamples
               p2 = vectorise_bounds
               nsamples = [1, nsamples]

          params = np.random.uniform(p1, p2, nsamples)
     
     #truncated gaussian with free parameters
     if(distribution == 1):
          bmin = bound[1]
          bmax = bound[2]

          if(isinstance(vectorise_bounds, int) == False):
               bmin = np.asarray([bmin]*nsamples)
               bmax = vectorise_bounds
               p1 = np.asarray([p1]*nsamples)
               p2 = np.asarray([p2]*nsamples)
               nsamples = [1, nsamples]

          a, b = (bmin - p1) / p2, (bmax - p1) / p2
          params = truncnorm.rvs(a, b, loc=p1, scale=p2, size=nsamples)

     if(distribution != 1 and distribution != 0):
          raise Exception("Unknown Distribution, bound[0] must be int < 2")

     return params

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
     
     #(for igm_factor)
     if(distribution == 2):
           return np.array([bound, 1, 0.5], dtype=object)
     
     if(distribution != 1 and distribution != 0 and distribution != 2):
          raise Exception("Unknown Distribution, bound[0] must be int < 3")
