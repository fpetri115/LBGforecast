import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.sfh as sfh
import torchvision
import gpytorch
import torch
from duste.DustAttnCalc import DustAttnCalc
import duste.DustAttnCalc as dd
import lbg_forecast.dust_priors as dp
import scipy as sc

class DustPrior():
    def __init__(self):
        
        print("Loading Models")
        self.n, self.tau, self.tau1, self.ne, self.taue, self.tau1e, self.sfr = get_nagaraj22_samples(ngal=20000)
        self.recent_sfrs, self.dust2, self.dust_index, self.dust1 = get_pop_cosmos_samples(nsamples=500000)
        print("Loading Complete")

        #load saved models
        state_dict_dust2 = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/dust2.pth', weights_only=True)
        state_dict_dust_index = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/dust_index.pth', weights_only=True)
        state_dict_dust1 = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/dust1.pth', weights_only=True)

        #load dust2 model
        self.dust2_training_data = process_training_data_dust2(self.tau, self.sfr, self.recent_sfrs, self.dust2)
        self.model_dust2 = create_gp_model_obs([4.0, 10.0], self.dust2_training_data[0], self.dust2_training_data[1], self.dust2_training_data[2])[0]
        self.model_dust2.load_state_dict(state_dict_dust2)

        #load dust index model
        self.dust_index_training_data = process_training_data_dust_index(self.n, self.tau, self.dust2, self.dust_index)
        self.model_dust_index = create_gp_model_obs([2.0, 6.0], self.dust_index_training_data[0], self.dust_index_training_data[1], self.dust_index_training_data[2])[0]
        self.model_dust_index.load_state_dict(state_dict_dust_index)

        #load dust1 model
        self.dust1_training_data = process_training_data_dust1(self.tau1, self.tau, self.dust2, self.dust1)
        self.model_dust1 = create_gp_model_obs([4.0, 10.0], self.dust1_training_data[0], self.dust1_training_data[1], self.dust1_training_data[2])[0]
        self.model_dust1.load_state_dict(state_dict_dust1)

    def evaluate_model(self, model, test_x, mul, muh, sigl, sigh):
        f_preds = gp_evaluate_model(model, torch.from_numpy(test_x))
        mean = f_preds.sample().numpy()
        scatter = np.random.uniform(sigl, sigh)
        return dp.truncated_normal(mean, scatter, mul, muh, len(test_x))

def train_gp_model(train_x, train_y, train_yerrs, lengthscales, lr=0.1, training_iter=20000):

    model, likelihood = create_gp_model_obs([lengthscales[0], lengthscales[1]], train_x, train_y, train_yerrs)
    trained_model, trained_likelihood = gp_training_loop(model, likelihood, train_x, train_y, training_iter=training_iter, lr=lr)

    return trained_model, trained_likelihood

def process_training_data_dust2(tau, sfr, recent_sfrs, dust2):

    bin_centers_de, bin_means_de, bin_std_de = proccess_nagaraj22_samples(sfr, tau, -8, 3)
    bin_centers, bin_means, bin_std = process_popcosmos_samples(recent_sfrs, dust2)
    train_sfr, train_dust2, train_dust2errs = training_data_to_torch(bin_centers, bin_means, bin_std, bin_centers_de, bin_means_de, bin_std_de)
    return [train_sfr, train_dust2, train_dust2errs]

def process_training_data_dust_index(n, tau, dust2, dust_index):

    bin_centers_de, bin_means_de, bin_std_de = process_popcosmos_samples(tau, n)
    bin_centers, bin_means, bin_std = process_popcosmos_samples(dust2, dust_index)
    train_dust2, train_dust_index, train_dust_index_errs = training_data_to_torch(bin_centers, bin_means, bin_std, bin_centers_de, bin_means_de, bin_std_de)
    return [train_dust2, train_dust_index, train_dust_index_errs]

def process_training_data_dust1(tau1, tau, dust2, dust1):

    bin_centers_de, bin_means_de, bin_std_de = process_popcosmos_samples(tau, tau1)
    bin_centers, bin_means, bin_std = process_popcosmos_samples(dust2, dust1)
    train_dust2, train_dust1, train_dust1_errs = training_data_to_torch(bin_centers, bin_means, bin_std, bin_centers_de, bin_means_de, bin_std_de)
    return [train_dust2, train_dust1, train_dust1_errs]

def training_data_to_torch(x1, y1, y1err, x2, y2, y2err):

    train_x = torch.from_numpy(np.hstack((x1, x2)))
    train_y = torch.from_numpy(np.hstack((y1, y2)))
    train_y_errs = torch.from_numpy(np.hstack((y1err, y2err)))

    train_x = torch.from_numpy(np.delete(train_x.numpy(), np.where(np.isnan(train_y)==1)[0]))
    train_y_errs = torch.from_numpy(np.delete(train_y_errs.numpy(), np.where(np.isnan(train_y)==1)[0]))
    train_y = torch.from_numpy(np.delete(train_y.numpy(), np.where(np.isnan(train_y)==1)[0]))

    return train_x, train_y, train_y_errs

def get_pop_cosmos_samples(nsamples):

    popcosmos_samples = np.load("dust_data/popcosmos_parameters_rmag_lt_25.npy")[:nsamples, :]

    dust_samples = popcosmos_samples[:, 8:11]
    logsfrratios = popcosmos_samples[:, 2:8]
    redshifts = popcosmos_samples[:, -1]
    logmasses = popcosmos_samples[:, 0]
    recent_sfrs = np.log10(sfh.calculate_recent_sfr(redshifts, 10**logmasses, logsfrratios))

    dust2 = dust_samples[:, 0]
    dust_index = dust_samples[:, 1]
    dust1frac = dust_samples[:, 2]
    dust1 = dust1frac*dust2

    return recent_sfrs, dust2, dust_index, dust1

def process_popcosmos_samples(x, y, ngrid=15):

    bin_means, bin_edges, binnumber = sc.stats.binned_statistic(x, y,'mean', ngrid)
    bin_std, bin_edges, binnumber = sc.stats.binned_statistic(x, y,'std', ngrid)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    return bin_centers, bin_means, bin_std

def get_nagaraj22_samples(ngal):

    logM = np.random.uniform(8.74,11.30,ngal)
    sfr = np.random.uniform(-2.06,2.11,ngal)
    logZ = np.random.uniform(-1.70,0.18,ngal)
    dobj = DustAttnCalc(sfr=sfr, logM=logM, logZ=logZ, bv=True, eff=False)
    dac, dac1, n, tau, tau1, ne, taue, tau1e = dobj.calcDust(max_num_plot=0)
    return n, tau, tau1, ne, taue, tau1e, sfr

def proccess_nagaraj22_samples(x, y, xl, xh, ngrid=15):

    bin_means_de, bin_edges_de, binnumber_de = sc.stats.binned_statistic(x, y, 'mean', np.linspace(xl, xh, ngrid))
    bin_std_de, bin_edges_de, binnumber_de = sc.stats.binned_statistic(x, y, 'std', np.linspace(xl, xh, ngrid))
    bin_width_de = (bin_edges_de[1] - bin_edges_de[0])
    bin_centers_de = bin_edges_de[1:] - bin_width_de/2

    return bin_centers_de, bin_means_de, bin_std_de

def create_gp_model_obs(lengthscale, train_x, train_y, noise):

    class GPModel(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(lengthscale[0], lengthscale[1])))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise)
    model = GPModel(train_x, train_y, likelihood).to(torch.double)

    return model, likelihood

def gp_training_loop(model, likelihood, train_x, train_y, training_iter, lr=1e-4):

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

def gp_plot_confidence(f_preds, test_x, train_x, train_y, labelx, labely, alpha=0.5):

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Get upper and lower confidence bounds
        lower, upper = f_preds.confidence_region()
        # Plot training data as black stars
        ax.scatter(train_x.numpy(), train_y.numpy(), c='k', alpha=alpha)
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), f_preds.mean, 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower, upper, alpha=0.5)
        #ax.legend(['Observed Data', 'Mean', 'Confidence'])

        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)


def gp_plot_confidence_obs(f_preds, test_x, train_x, train_y, train_yerrs, labelx, labely, alpha=0.5):

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Get upper and lower confidence bounds
        lower, upper = f_preds.confidence_region()
        # Plot training data as black stars
        ax.errorbar(train_x.numpy(), train_y.numpy(), train_yerrs.numpy(), fmt='ko', capsize=2)
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), f_preds.mean, 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower, upper, alpha=0.5)
        #ax.legend(['Observed Data', 'Mean', 'Confidence'])

        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)

def binned_weighted_mean_std(x, y, yerr, bins):

    bin_midpoints = (bins[:-1]+bins[1:])/2
    nbins = len(bins)-1
    bin_indexes = np.digitize(x, bins)-1

    weighted_averages = []
    error = []
    non_empty_bins = []
    for bin in range(nbins):

        y_indexes = np.where(bin_indexes == bin)[0]

        if(len(y_indexes)> 0):
            y_in_bin = y[y_indexes]
            yerrs_in_bin = yerr[y_indexes]
            weights_in_bin = (1/yerrs_in_bin)**2
            weighted_average_in_bin = np.average(y_in_bin, weights=weights_in_bin)
            #error_on_wmean = np.sqrt(1/np.sum(weights_in_bin))
            std_in_bin = np.std(y_in_bin)
            avg_err_in_bin = np.mean(yerrs_in_bin)
            weighted_averages.append(weighted_average_in_bin)
            #set error floor
            if(std_in_bin<avg_err_in_bin):
                error.append(avg_err_in_bin)
            else:
                error.append(std_in_bin)
            non_empty_bins.append(bin_midpoints[bin])

    return np.array(non_empty_bins), np.array(weighted_averages), np.array(error)