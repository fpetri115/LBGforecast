import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy as sc

LSST_AREA_DEG2 = 18000
DEG2_TO_ARCMIN2 = 3600
RAD_TO_DEG = 180/np.pi

FULL_SKY_STERAD = 4*np.pi
FULL_SKY_ARCMIN2 = FULL_SKY_STERAD*(RAD_TO_DEG**2)*DEG2_TO_ARCMIN2
FULL_SKY_DEG2 = FULL_SKY_STERAD*(RAD_TO_DEG**2)

def interlopers(samples, cutoff):
    nint = len(np.where(samples<cutoff)[0])
    ntot = len(samples)
    return (nint/ntot)*100

def plot_contours(fisher, pos, i, j, nstd=2., K=5.991, ax=None, **kwargs):
  """
  Plot 2D parameter contours given a Hessian matrix of the likelihood
  """
  inds = [i, j]
  
  def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]

  mat = fisher
  cov = np.linalg.inv(mat)
  sigma_marg = lambda i: np.sqrt(cov[i, i])

  if ax is None:
      ax = plt.gca()

  # Extracts the block we are interested in
  cov = cov[inds][::,inds]
  vals, vecs = eigsorted(cov)
  theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

  # Width and height are "full" widths, not radius
  #width, height = 2 * nstd * np.sqrt(vals)
  width, height = 2 * np.sqrt(vals*K)
  ellip = Ellipse(xy=pos[inds], width=width,
                  height=height, angle=theta, **kwargs)

  ax.add_artist(ellip)
  sz = max(width, height)
  s1 = 1.5*nstd*sigma_marg(0)
  s2 = 1.5*nstd*sigma_marg(1)
  ax.set_xlim(pos[inds[0]] - s1, pos[inds[0]] + s1)
  ax.set_ylim(pos[inds[1]] - s2, pos[inds[1]] + s2)

  xticks = np.delete(np.around(np.linspace(pos[inds[0]] - s1, pos[inds[0]] + s1, 7), 3)[1:6], [1, 3])

  ax.set_xticks(np.around(xticks, 3))

  plt.draw()
  return ellip

def quantile975(x):
    return np.quantile(x, 0.975)

def quantile025(x):
    return np.quantile(x, 0.025)

def quantile84(x):
    return np.quantile(x, 0.84)

def quantile16(x):
    return np.quantile(x, 0.16)

def process_samples_median(x, y, xl, xh, ngrid=15):

    bin_median, bin_edges, binnumber = sc.stats.binned_statistic(x, y, 'median', np.linspace(xl, xh, ngrid))
    bin_975, bin_edges, binnumber = sc.stats.binned_statistic(x, y, quantile975, np.linspace(xl, xh, ngrid))
    bin_025, bin_edges, binnumber = sc.stats.binned_statistic(x, y, quantile025, np.linspace(xl, xh, ngrid))
    bin_84, bin_edges, binnumber = sc.stats.binned_statistic(x, y, quantile84, np.linspace(xl, xh, ngrid))
    bin_16, bin_edges, binnumber = sc.stats.binned_statistic(x, y, quantile16, np.linspace(xl, xh, ngrid))

    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    return bin_centers, bin_median, bin_025, bin_975, bin_16, bin_84