import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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