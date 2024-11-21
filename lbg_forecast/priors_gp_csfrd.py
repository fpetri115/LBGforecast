import gpytorch
import torch
import numpy as np
import lbg_forecast.cosmology as cosmo
from astropy.io import ascii
import matplotlib.pyplot as plt
import lbg_forecast.population_model as pop
from uncertainties import unumpy as upy

class CSFRDModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, lengthscale, scale, likelihood):
        super(CSFRDModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(lengthscale[0], lengthscale[1])), outputscale_prior=gpytorch.priors.SmoothedBoxPrior(scale[0], scale[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class CSFRDPrior():

    def __init__(self, path):

        self.path = path
        self.train_data = get_training_data(path=self.path, plot=False)
        state_dict = torch.load(path+'/gp_models/csfrd.pth', weights_only=True)
        self.model = create_gp_model(self.train_data[0], self.train_data[2], self.train_data[3], [1.0, 7.0], [0.5, 100])[0]
        self.model.load_state_dict(state_dict)
        self.test_redshift = torch.linspace(0, 30, 500).to(torch.double)
        self.prior = gp_evaluate_model(self.model, self.test_redshift)
        self.lookback_times = cosmo.get_cosmology().lookback_time(self.test_redshift.numpy()).value*1e9

    def sample_prior_corrected(self):
        return self.prior.sample().numpy() + mean_obs_behroozi(self.test_redshift.numpy(), log=True, path=self.path) - systematic_shift(self.test_redshift.numpy(), path=self.path)
    
    def get_prior_mean(self):
        return self.prior.mean.numpy() + mean_obs_behroozi(self.test_redshift.numpy(), log=True, path=self.path)
    
    def get_prior_mean_corrected(self):
        return self.prior.mean.numpy() + mean_obs_behroozi(self.test_redshift.numpy(), log=True, path=self.path) - systematic_shift(self.test_redshift.numpy(), path=self.path)
    
    def get_prior_confidence_region(self):
        lower, upper = self.prior.confidence_region()
        lower = lower + mean_obs_behroozi(self.test_redshift.numpy(), log=True, path=self.path)
        upper = upper + mean_obs_behroozi(self.test_redshift.numpy(), log=True, path=self.path)
        return [lower, upper]
    
    def get_prior_confidence_region_corrected(self):
        lower, upper = self.prior.confidence_region()
        lower = lower + mean_obs_behroozi(self.test_redshift.numpy(), log=True, path=self.path) - systematic_shift(self.test_redshift.numpy(), path=self.path)
        upper = upper + mean_obs_behroozi(self.test_redshift.numpy(), log=True, path=self.path) - systematic_shift(self.test_redshift.numpy(), path=self.path)
        return [lower, upper]
    
    def plot_combined(self):

        train_redshift, train_log_csfrd, train_log_shifted_csfrd, train_log_csfrd_errors = self.train_data
        behroozi19 = get_behroozi19_curves(path=self.path)
        with torch.no_grad():

            f, ax = plt.subplots(1, 1, figsize=(22, 16))

            ax.errorbar(train_redshift, train_log_csfrd, yerr=train_log_csfrd_errors, fmt='o', mfc='k', ecolor='k', mec='k', alpha=1.0, elinewidth=2.5, capsize=5, ms=12, lw=3, label="Observed CSFRD (Behroozi et al. 2019)")
            
            ax.plot(self.test_redshift.numpy(), self.get_prior_mean(), lw=7, zorder=1200, c="k", label="Gaussian Process Mean")
            lower, upper = self.get_prior_confidence_region()
            ax.fill_between(self.test_redshift.numpy(), lower, upper, alpha=0.4, lw=0, color="k", label="$2\sigma $ Confidence")
            ax.plot(behroozi19[0], behroozi19[1], zorder=1100, ls="--", c="k", lw=7, label="Behroozi et al. (2019) Fit")


            ax.plot(self.test_redshift.numpy(), self.get_prior_mean_corrected(), lw=7, zorder=1200, c='red', ls="-", label="Gaussian Process Mean (Corrected)")
            lower_corrected, upper_corrected = self.get_prior_confidence_region_corrected()
            ax.fill_between(self.test_redshift.numpy(), lower_corrected, upper_corrected, alpha=0.4, lw=0, color="red", label="$2\sigma $ Confidence (Corrected)")
            ax.plot(behroozi19[0], behroozi19[2], zorder=1100, c="red", ls="--", lw=7, label="Behroozi et al. (2019) Fit (Corrected)")

            ax.set_xlabel("Redshift", fontsize=32)
            ax.set_ylabel("Cosmic Star Formation Rate Density [$\mathrm{M}_{\odot} \mathrm{yr}^{-1} \mathrm{Mpc}^{-3}$]", fontsize=32)
            ax.set_xlim(0, 10)
            ax.tick_params(axis='y', labelsize=32)
            ax.tick_params(axis='x', labelsize=32)
            ax.set_ylim(-2.7, -0.5)
            ax.set_xscale('function', functions=(forward, inverse))
            ax.legend(frameon=False, fontsize=22)

def create_gp_model(train_redshift, train_log_csfrd_shifted, train_log_csfrd_errors, lengthscale, scale):
        
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_log_csfrd_errors)
    model = CSFRDModel(train_redshift, train_log_csfrd_shifted, lengthscale, scale, likelihood).to(torch.double)

    return model, likelihood

def gp_training_loop(model, likelihood, train_x, train_y, training_iter=5000, lr=0.1):

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #i + 1, training_iter, loss.item(),
        print(loss.item())
            #model.likelihood.noise.item()
        #))
        optimizer.step()

    return model, likelihood

def gp_evaluate_model(model, test_x):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    f_preds = model(test_x.to(torch.double))

    return f_preds

def get_training_data(path, plot=False):

    data = ascii.read(path+"/csfr_data/obs.txt")
    csfr_rows = data['Type']=="csfr"
    csfr_uv_rows = data['Type']=="csfr_(uv)"
    rows = np.logical_or(csfr_rows, csfr_uv_rows)

    z1 = np.array(data["Z1"])[rows]
    z2 = np.array(data["Z2"])[rows]
    redshift = (z2+z1)/2

    log_val = np.array(data["Val"])[rows]
    log_err_h = np.array(data["Err_h"])[rows]
    log_err_l = np.array(data["Err_l"])[rows]
    log_err = np.maximum(log_err_l, log_err_h)

    train_redshift = redshift
    train_csfrd = log_val
    train_csfrd_errors = log_err

    if(plot):
        plt.errorbar(train_redshift, train_csfrd, yerr=train_csfrd_errors, fmt='ko', capsize=2, alpha=1.0)
        plt.xlabel("redshift")
        plt.ylabel("csfrd")

    train_redshift = torch.from_numpy(train_redshift)
    train_log_csfrd = torch.from_numpy(train_csfrd)
    train_log_shifted_csfrd = train_log_csfrd - torch.from_numpy(mean_obs_behroozi(train_redshift.numpy(), log=True, path=path))
    train_log_csfrd_errors = torch.from_numpy(train_csfrd_errors)

    return [train_redshift, train_log_csfrd, train_log_shifted_csfrd, train_log_csfrd_errors]

def log_to_lin(train_log_csfrd, train_log_csfrd_errors):

    log_csfrd_arr = upy.uarray(train_log_csfrd, train_log_csfrd_errors)
    csfrd_arr = 10**log_csfrd_arr
    train_csfrd = upy.nominal_values(csfrd_arr)
    train_csfrd_errors = upy.std_devs(csfrd_arr)

    return train_csfrd, train_csfrd_errors

def mean_obs_behroozi(zgrid, log, path):

    data = ascii.read(path+"/csfr_data/csfrs.dat")  
    scale = np.array(data["Scale"])
    redshift = cosmo.scale_to_z(scale)

    if(log):
        total_obs_csfr = np.log10(np.array(data["Total_Obs_CSFR"]))
    else:
        total_obs_csfr = np.array(data["Total_Obs_CSFR"])

    return np.interp(zgrid, np.flip(redshift), np.flip(total_obs_csfr))

def systematic_shift(zgrid, path):

    data = ascii.read(path+"/csfr_data/csfrs.dat")  

    scale = np.array(data["Scale"])
    redshift = cosmo.scale_to_z(scale)
    total_obs_csfr = np.log10(np.array(data["Total_Obs_CSFR"]))
    true_csfr = np.log10(np.array(data["True_CSFR"]))

    sys_shift = total_obs_csfr - true_csfr

    return np.interp(zgrid, np.flip(redshift), np.flip(sys_shift))

def get_behroozi19_curves(path):

        data = ascii.read(path+"/csfr_data/csfrs.dat")  

        scale = np.array(data["Scale"])
        redshift = cosmo.scale_to_z(scale)
        total_obs_csfr = np.log10(np.array(data["Total_Obs_CSFR"]))
        true_csfr = np.log10(np.array(data["True_CSFR"]))

        return redshift, total_obs_csfr, true_csfr

def inverse(x):
    return x**2
def forward(x):
    return np.sqrt(x)