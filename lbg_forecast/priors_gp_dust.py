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

class GPModel(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, lengthscale, scale, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ConstantKernel() + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(lengthscale[0], lengthscale[1])), outputscale_prior=gpytorch.priors.SmoothedBoxPrior(scale[0], scale[1]))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DustPrior():
    def __init__(self, path):
        
        self.path = path
        print("Loading Models")
        self.preloaded_popcosmos_samples = np.load(self.path+"/dust_data/popcosmos_parameters_rmag_lt_25.npy")
        self.preloaded_recent_sfrs = np.load(self.path+"/dust_data/popcosmos_recentsfrs.npy")
        self.n, self.tau, self.tau1, self.ne, self.taue, self.tau1e, self.sfr = self.get_nagaraj22_samples()
        self.recent_sfrs, self.dust2, self.dust_index, self.dust1 = self.get_pop_cosmos_samples(nsamples=500000)
        print("Loading Complete")

        #load saved models
        state_dict_dust2 = torch.load(path+'/gp_models/dust2.pth', weights_only=True)
        state_dict_dust2nag = torch.load(path+'/gp_models/dust2nag.pth', weights_only=True)
        state_dict_dust2sig = torch.load(path+'/gp_models/dust2sig.pth', weights_only=True)
        state_dict_dust_index = torch.load(path+'/gp_models/dust_index.pth', weights_only=True)
        state_dict_dust_indexsig = torch.load(path+'/gp_models/dust_indexsig.pth', weights_only=True)
        state_dict_dust1 = torch.load(path+'/gp_models/dust1.pth', weights_only=True)
        state_dict_dust1sig = torch.load(path+'/gp_models/dust1sig.pth', weights_only=True)

        #load dust2 model
        self.dust2_training_data = process_training_data_dust2(self.tau, self.sfr, self.recent_sfrs, self.dust2)
        self.dust2_training_data_nag = process_training_data_dust2_nag(self.tau, self.sfr)
        self.model_dust2 = create_gp_model_noerr([1.0, 15.0], self.dust2_training_data[0], self.dust2_training_data[1], [-100, 100])[0]
        self.model_dust2_nag = create_gp_model_noerr([1.0, 15.0], self.dust2_training_data_nag[0], self.dust2_training_data_nag[1], [-100, 100])[0]
        self.model_dust2_sig = create_gp_model_noerr([5.0, 15.0], self.dust2_training_data[0], self.dust2_training_data[2], [-100, 100])[0]
        self.model_dust2.load_state_dict(state_dict_dust2)
        self.model_dust2_nag.load_state_dict(state_dict_dust2nag)
        self.model_dust2_sig.load_state_dict(state_dict_dust2sig)
        self.dust2_grid = np.linspace(-5, 3, 1000)

        #load dust index model
        self.dust_index_training_data = process_training_data_dust_index(self.n, self.ne, self.tau, self.dust2, self.dust_index)
        self.model_dust_index = create_gp_model_noerr([2.0, 10.0], self.dust_index_training_data[0], self.dust_index_training_data[1], [-100, 100])[0]
        self.model_dust_index_sig = create_gp_model_noerr([2.0, 10.0], self.dust_index_training_data[0], self.dust_index_training_data[2], [-100, 100])[0]
        self.model_dust_index.load_state_dict(state_dict_dust_index)
        self.model_dust_index_sig.load_state_dict(state_dict_dust_indexsig)
        self.dust_index_grid = np.linspace(0.0, 4.0, 1000)

        #load dust1 model
        self.dust1_training_data = process_training_data_dust1(self.tau1, self.tau1e, self.tau, self.dust2, self.dust1)
        self.model_dust1 = create_gp_model_noerr([2.0, 10.0], self.dust1_training_data[0], self.dust1_training_data[1], [-100, 100])[0]
        self.model_dust1_sig = create_gp_model_noerr([2.0, 10.0], self.dust1_training_data[0], self.dust1_training_data[2], [-100, 100])[0]
        self.model_dust1.load_state_dict(state_dict_dust1)
        self.model_dust1_sig.load_state_dict(state_dict_dust1sig)
        self.dust1_grid = np.linspace(0.0, 4.0, 1000)

    def sample_dust_model(self, sfrs):

        dust2 = self.sample_dust2(sfrs)
        dust_index = self.sample_dust_index(dust2)
        dust1 = self.sample_dust1(dust2)

        return [dust2, dust_index, dust1]
    
    def dust2_mean(self):
        
        f_preds_mu = gp_evaluate_model(self.model_dust2, torch.from_numpy(self.dust2_grid))
        mean_dust2 = f_preds_mu.mean.detach().numpy()
        return mean_dust2
    
    def dust2_mean_nagaraj(self):
        
        f_preds_mu = gp_evaluate_model(self.model_dust2_nag, torch.from_numpy(self.dust2_grid))
        mean_dust2 = f_preds_mu.mean.detach().numpy()
        return mean_dust2
   
    def interpolate_models(self, f):

        return (1-f)*self.dust2_mean() + (f)*self.dust2_mean_nagaraj()

    def sample_dust2(self, sfrs):
        
        f_preds_mu = gp_evaluate_model(self.model_dust2, torch.from_numpy(self.dust2_grid))
        mean_dust2_sample = f_preds_mu.sample().numpy()
        mean_dust2 = f_preds_mu.mean.detach().numpy()
        delta_dust2 = mean_dust2_sample - mean_dust2
        delta = np.interp(self.recent_sfrs, self.dust2_grid, delta_dust2)

        f = 0#np.random.uniform(0, 1)
        sys_shift = self.interpolate_models(f) - mean_dust2
        sys_shift = np.interp(self.recent_sfrs, self.dust2_grid, sys_shift)

        sorted_inds = self.recent_sfrs.argsort()[:]
        sorted_sfrs = self.recent_sfrs[sorted_inds]
        sorted_dust2 = self.dust2[sorted_inds]
        sorted_delta = delta[sorted_inds]
        sorted_shift = sys_shift[sorted_inds]

        dust2 = np.interp(sfrs, sorted_sfrs, sorted_dust2)
        delta = np.interp(sfrs, sorted_sfrs, sorted_delta)
        sys_shift = np.interp(sfrs, sorted_sfrs, sorted_shift)

        return np.clip(abs(dust2 + delta + sys_shift), 0.0, 4.0)

    def sample_dust_index(self, dust2s):

        f_preds_mu = gp_evaluate_model(self.model_dust_index, torch.from_numpy(self.dust_index_grid))
        mean_dust_index_sample = f_preds_mu.sample().numpy()
        mean_dust_index = f_preds_mu.mean.detach().numpy()
        delta_dust_index = mean_dust_index_sample - mean_dust_index
        delta = np.interp(self.dust2, self.dust_index_grid, delta_dust_index)

        sorted_inds = self.dust2.argsort()[:]
        sorted_dust2 = self.dust2[sorted_inds]
        sorted_dust_index = self.dust_index[sorted_inds]
        sorted_delta = delta[sorted_inds]

        dust_index = np.interp(dust2s, sorted_dust2, sorted_dust_index)
        delta = np.interp(dust2s, sorted_dust2, sorted_delta)

        return np.clip(dust_index + delta, -2.2, 0.4)
    
    def sample_dust1(self, dust2s):

        f_preds_mu = gp_evaluate_model(self.model_dust1, torch.from_numpy(self.dust1_grid))
        mean_dust1_sample = f_preds_mu.sample().numpy()
        mean_dust1 = f_preds_mu.mean.detach().numpy()
        delta_dust1 = mean_dust1_sample - mean_dust1
        delta = np.interp(self.dust2, self.dust1_grid, delta_dust1)

        sorted_inds = self.dust2.argsort()[:]
        sorted_dust2 = self.dust2[sorted_inds]
        sorted_dust1 = self.dust1[sorted_inds]
        sorted_delta = delta[sorted_inds]

        dust1 = np.interp(dust2s, sorted_dust2, sorted_dust1)
        delta = np.interp(dust2s, sorted_dust2, sorted_delta)

        return np.clip(dust1 + delta, 0.0, 4.0)
    
    def get_pop_cosmos_samples(self, nsamples):

        popcosmos_samples = self.preloaded_popcosmos_samples[:nsamples, :]

        dust_samples = popcosmos_samples[:, 8:11]
        recent_sfrs = self.preloaded_recent_sfrs[:nsamples]
        #np.log10(sfh.calculate_recent_sfr(redshifts, 10**logmasses, logsfrratios))

        dust2 = dust_samples[:, 0]
        dust_index = dust_samples[:, 1]
        dust1frac = dust_samples[:, 2]
        dust1 = dust1frac*dust2

        return recent_sfrs, dust2, dust_index, dust1
    
    def draw_popcosmos_samples(self, nsamples):

        recent_sfrs, dust2, dust_index, dust1 = self.get_pop_cosmos_samples(nsamples)

        indexes = np.random.uniform(0, nsamples, nsamples)
        drawn_sfrs = recent_sfrs[indexes]
        drawn_dust2 = dust2[indexes]
        drawn_dust_index = dust_index[indexes]
        drawn_dust1 = dust1[indexes]

        return drawn_sfrs, drawn_dust2, drawn_dust_index, drawn_dust1
    
    def get_nagaraj22_samples(self):

        #logM = np.random.uniform(8.74,11.30,ngal)
        #sfr = np.random.uniform(-5,2.5,ngal)
        #logZ = np.random.uniform(-1.70,0.18,ngal)
        #z=np.random.uniform(0.51,2.83,ngal)
        #i=np.random.uniform(0.09,0.97,ngal)
        #dobj = DustAttnCalc(sfr=sfr, logM=logM, logZ=logZ, z=z, i=i, bv=True, eff=False)
        #dac, dac1, n, tau, tau1, ne, taue, tau1e = dobj.calcDust(max_num_plot=0)
        #return n, tau, tau1, ne, taue, tau1e, sfr
        return np.load(self.path+"/dust_data/saved_nagaraj22samples.npy")

def train_gp_model_noerr(train_x, train_y, lengthscales, scales, lr=0.1, training_iter=20000):

    model, likelihood = create_gp_model_noerr([lengthscales[0], lengthscales[1]], train_x, train_y, scales)
    trained_model, trained_likelihood = gp_training_loop(model, likelihood, train_x, train_y, training_iter=training_iter, lr=lr)

    return trained_model, trained_likelihood

def process_training_data_dust2(tau, sfr, recent_sfrs, dust2):

    bin_centers_de, bin_means_de, bin_std_de = process_samples(sfr, tau, -10, -5, 5)
    bin_centers, bin_means, bin_std = process_samples(recent_sfrs, dust2, -5, 3, 50)
    train_sfr, train_dust2, train_dust2errs = training_data_to_torch(bin_centers, bin_means, bin_std, bin_centers_de, bin_means_de, bin_std_de)
    return [train_sfr, train_dust2, train_dust2errs]

def process_training_data_dust2_nag(tau, sfr):

    bin_centers_de, bin_means_de, bin_std_de = process_samples(sfr, tau, -5, 3, 15)
    train_sfrs, train_tau, train_tau_errs = torch.from_numpy(bin_centers_de), torch.from_numpy(bin_means_de), torch.from_numpy(bin_std_de)

    return [train_sfrs, train_tau, train_tau_errs]

def process_training_data_dust_index(n, ne, tau, dust2, dust_index):

    bin_centers_de, bin_means_de, bin_std_de = process_samples_modified(tau, n, ne, -10, -5, 2)
    bin_centers, bin_means, bin_std = process_samples(dust2, dust_index, 0.0, 3.0, 50)
    train_dust2, train_dust_index, train_dust_index_errs = training_data_to_torch(bin_centers, bin_means, bin_std, bin_centers_de, bin_means_de, bin_std_de)
    return [train_dust2, train_dust_index, train_dust_index_errs]

def process_training_data_dust1(tau1, tau1e, tau, dust2, dust1):

    bin_centers_de, bin_means_de, bin_std_de = process_samples_modified(tau, tau1, tau1e, -10, -5, 2)
    bin_centers, bin_means, bin_std = process_samples(dust2, dust1, 0.0, 3.0, 50)
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

def process_samples(x, y, xl, xh, ngrid=15):

    bin_means, bin_edges, binnumber = sc.stats.binned_statistic(x, y, 'mean', np.linspace(xl, xh, ngrid))
    bin_std, bin_edges, binnumber = sc.stats.binned_statistic(x, y, 'std', np.linspace(xl, xh, ngrid))
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    return bin_centers, bin_means, bin_std

def process_samples_modified(x, y, yerr, xl, xh, ngrid=15):

    bins = np.linspace(xl, xh, ngrid)
    x, y, yerr = binned_weighted_mean_std(x, y, yerr, bins)

    return x, y, yerr

def process_popcosmos_samples(x, y, ngrid=15):

    bin_means, bin_edges, binnumber = sc.stats.binned_statistic(x, y,'mean', ngrid)
    bin_std, bin_edges, binnumber = sc.stats.binned_statistic(x, y,'std', ngrid)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    return bin_centers, bin_means, bin_std

def proccess_nagaraj22_samples(x, y, xl, xh, ngrid=15):

    bin_means_de, bin_edges_de, binnumber_de = sc.stats.binned_statistic(x, y, 'mean', np.linspace(xl, xh, ngrid))
    bin_std_de, bin_edges_de, binnumber_de = sc.stats.binned_statistic(x, y, 'std', np.linspace(xl, xh, ngrid))
    bin_width_de = (bin_edges_de[1] - bin_edges_de[0])
    bin_centers_de = bin_edges_de[1:] - bin_width_de/2

    return bin_centers_de, bin_means_de, bin_std_de

def create_gp_model_noerr(lengthscale, train_x, train_y, scale):

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, lengthscale, scale, likelihood).to(torch.double)

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