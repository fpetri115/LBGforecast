import gpytorch
import torch
import numpy as np
import lbg_forecast.cosmology as cosmo
from astropy.io import ascii
import matplotlib.pyplot as plt
import lbg_forecast.population_model as pop


def create_gp_model(train_redshift, train_log_csfrd_shifted, train_log_csfrd_errors, lengthscale, scale):

    class CSFRDModel(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, likelihood):
            super(CSFRDModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(lengthscale[0], lengthscale[1])), outputscale_prior=gpytorch.priors.SmoothedBoxPrior(scale[0], scale[1]))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_log_csfrd_errors)
    model = CSFRDModel(train_redshift, train_log_csfrd_shifted, likelihood).to(torch.double)

    for param_name, param in model.named_parameters():
        print(f'Parameter name: {param_name:42} value = {param.item()}')

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

def shift_csfrd(new_redshift, csfrd, redshift, mean):
    return csfrd - np.interp(new_redshift, redshift, mean)

def shift_csfrd_inverse(new_redshift, csfrd, redshift, mean):
    return csfrd + np.interp(new_redshift, redshift, mean)

def process_training_data_csfrd(training_data):

    data = ascii.read("csfr_data/csfrs.dat")  
    scale = np.array(data["Scale"])
    redshift = cosmo.scale_to_z(scale)
    total_obs_csfr = np.log10(np.array(data["Total_Obs_CSFR"]))

    train_redshift = torch.from_numpy(training_data[0])
    log_train_csfrd = torch.from_numpy(training_data[1])
    log_train_csfrd_errors = torch.from_numpy(training_data[2])

    log_train_shifted_csfrd = shift_csfrd(train_redshift, log_train_csfrd, np.flip(redshift), np.flip(total_obs_csfr))
    return train_redshift, log_train_shifted_csfrd, log_train_csfrd_errors, log_train_csfrd

def get_training_data(plot=False):

    data = ascii.read("csfr_data/obs.txt")
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

    return train_redshift, train_csfrd, train_csfrd_errors