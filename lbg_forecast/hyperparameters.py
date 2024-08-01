import numpy as np
import matplotlib.pyplot as plt
import math


def sample_prior_parameters(nsamples, bounds, sigma_bounds):
    """Sample prior parameters (hyperparameters) for
    SPS model. 

    :param nsamples:
        Number of samples

    :param bounds:
        (no. priors, 2) shape array with minimum and maximum
        mean for gaussian distribution for a given prior in 
        each row. The last rows are for uniform priors, where
        each row is then the minimum and maximum of that uniform
        distribution. The number of gaussian priors is determined
        by the length of the sigma_bounds parameter.

    :param sigma_bounds:
        (no. gaussian priors, 2) shape array with minimum and maximum
        standard deviation for gaussian priors. len(sigma_bounds) gives
        number of gaussian priors. So for any rows of parameter bounds
        with index greater then len(sigma_bounds) will be treated as
        uniform priors

    :returns prior_parameters:
        (nsamples, len(bounds)) shape array of hyperparameters. Each row is a
        different reaslisation of the hyperparameters.
        Number of rows = nsamples. Columns are parameters,
        in order given by (mu1, sig1, mu2, sig2, ... ) up
        to uniform prior parameters which then go as
        (min1, max1, min2, max2 ... )
    
    """
    n_priors = len(bounds)
    n_gaussian_priors = len(sigma_bounds)
    n_uniform_priors = n_priors - n_gaussian_priors
    n_prior_parameters = int(2*n_gaussian_priors + n_uniform_priors)
    prior_parameters = np.empty((nsamples, n_prior_parameters))

    indx = 0
    #loop over gaussian priors
    while(indx < n_gaussian_priors):

        parameter_bounds = bounds[indx, :]
        parameter_sigma_bounds = sigma_bounds[indx, :]
        prior_parameters[:, int(2*indx)] = np.random.uniform(parameter_bounds[0], parameter_bounds[1], (nsamples,))
        prior_parameters[:, int(2*indx+1)] = np.random.uniform(parameter_sigma_bounds[0], parameter_sigma_bounds[1], (nsamples,)) 

        indx+=1

    #loop over uniform priors
    indx2 = 0
    while(indx2 < n_uniform_priors):

        parameter_bounds = bounds[indx+indx2, :]
        prior_parameters[:, int(2*indx)+indx2] = np.random.uniform(parameter_bounds[0], parameter_bounds[1], (nsamples,))
        indx2+=1

    return prior_parameters

def hyperparameter_bounds_broad():
    """Some default parameters for sample_prior_parameters()
    """

    hyperparameter_mu_bounds = np.array([[-2.5, 0.5],  #logzsol
                                    [0.0, 4.0],      #dust1
                                    [0.0, 4.0],      #dust2
                                    [-2.2, 0.4],     #dust_index
                                    [1.0, 1.0],      #igm_factor 
                                    [-4.0, -1.0],    #gas_logu
                                    [-2.0, 0.5],     #gas_logz
                                    [-5.0, 1.0],     #log10(fagn)
                                    [5, 150],        #agn_tau
                                    [-2, 2],         #logsfmu1
                                    [-2, 2],         #logsfmu2
                                    [-2, 2],         #logsfmu3
                                    [-2, 2],         #logsfmu4
                                    [-2, 2],         #logsfmu5
                                    [-2, 2],         #logsfmu6
                                    [0.3, 0.3],      #logsfsig1
                                    [0.3, 0.3],      #logsfsig2
                                    [0.3, 0.3],      #logsfsig3
                                    [0.3, 0.3],      #logsfsig4
                                    [0.3, 0.3],      #logsfsig5
                                    [0.3, 0.3],      #logsfsig6
                                    [2, 2]           #nu   
    ])

    hyperparameter_sigma_max = np.array([[0.01, 3.0], #logzsol
                                        [0.01, 4.0], #dust1
                                        [0.01, 4.0], #dust2
                                        [0.01, 2.6], #dust_index
                                        [0.1, 0.1], #igm_factor 
                                        [0.01, 3.0], #gas_logu
                                        [0.01, 2.5], #gas_logz
                                        [0.01, 6.0], #log10(fagn)
                                        [0.1, 145]   #agn_tau
    ])

    return hyperparameter_mu_bounds, hyperparameter_sigma_max

def hyperparameter_bounds_low_dust():
    """Some default parameters for sample_prior_parameters()
    """

    hyperparameter_mu_bounds = np.array([[-2.5, 0.5],  #logzsol
                                    [0.0, 0.2],      #dust1
                                    [0.0, 0.2],      #dust2
                                    [-2.2, 0.4],     #dust_index
                                    [1.0, 1.0],      #igm_factor 
                                    [-4.0, -1.0],    #gas_logu
                                    [-2.0, 0.5],     #gas_logz
                                    [-5.0, 1.0],     #log10(fagn)
                                    [5, 150],        #agn_tau
                                    [-2, 2],         #logsfmu1
                                    [-2, 2],         #logsfmu2
                                    [-2, 2],         #logsfmu3
                                    [-2, 2],         #logsfmu4
                                    [-2, 2],         #logsfmu5
                                    [-2, 2],         #logsfmu6
                                    [0.3, 0.3],      #logsfsig1
                                    [0.3, 0.3],      #logsfsig2
                                    [0.3, 0.3],      #logsfsig3
                                    [0.3, 0.3],      #logsfsig4
                                    [0.3, 0.3],      #logsfsig5
                                    [0.3, 0.3],      #logsfsig6
                                    [2, 2]           #nu   
    ])

    hyperparameter_sigma_max = np.array([[0.01, 3.0], #logzsol
                                        [0.01, 0.5], #dust1
                                        [0.01, 0.5], #dust2
                                        [0.01, 2.6], #dust_index
                                        [0.1, 0.1], #igm_factor 
                                        [0.01, 3.0], #gas_logu
                                        [0.01, 2.5], #gas_logz
                                        [0.01, 6.0], #log10(fagn)
                                        [0.1, 145]   #agn_tau
    ])

def hyperparameter_tighter():
    """Some default parameters for sample_prior_parameters()
    """

    hyperparameter_mu_bounds = np.array([[-2.5, 0.5],  #logzsol
                                    [0.0, 2.0],      #dust1
                                    [0.0, 2.0],      #dust2
                                    [-1.0, 0.4],     #dust_index
                                    [1.0, 1.0],      #igm_factor 
                                    [-4.0, -1.0],    #gas_logu
                                    [-1.5, 0.5],     #gas_logz
                                    [-5.0, 1.0],     #log10(fagn)
                                    [5, 150],        #agn_tau
                                    [-2, 2],         #logsfmu1
                                    [-2, 2],         #logsfmu2
                                    [-2, 2],         #logsfmu3
                                    [-2, 2],         #logsfmu4
                                    [-2, 2],         #logsfmu5
                                    [-2, 2],         #logsfmu6
                                    [0.3, 0.3],      #logsfsig1
                                    [0.3, 0.3],      #logsfsig2
                                    [0.3, 0.3],      #logsfsig3
                                    [0.3, 0.3],      #logsfsig4
                                    [0.3, 0.3],      #logsfsig5
                                    [0.3, 0.3],      #logsfsig6
                                    [2, 2]           #nu   
    ])

    hyperparameter_sigma_max = np.array([[0.5, 3.0], #logzsol
                                        [0.1, 2.0], #dust1
                                        [0.1, 2.0], #dust2
                                        [0.1, 2.6], #dust_index
                                        [0.1, 0.1], #igm_factor 
                                        [0.5, 3.0], #gas_logu
                                        [0.5, 2.5], #gas_logz
                                        [0.5, 6.0], #log10(fagn)
                                        [1, 145]   #agn_tau
    ])

    return hyperparameter_mu_bounds, hyperparameter_sigma_max

def default_hyperparameter_bounds():
    """Some default parameters for sample_prior_parameters()
    """

    hyperparameter_mu_bounds = np.array([[-2.5, 0.5],  #logzsol
                                    [0.0, 4.0],      #dust1
                                    [0.0, 4.0],      #dust2
                                    [-2.2, 0.4],     #dust_index
                                    [1.0, 1.0],      #igm_factor 
                                    [-4.0, -1.0],    #gas_logu
                                    [-2.0, 0.5],     #gas_logz
                                    [-5.0, 1.0],     #log10(fagn)
                                    [5, 150],        #agn_tau
                                    [-2, 2],         #logsfmu1
                                    [-2, 2],         #logsfmu2
                                    [-2, 2],         #logsfmu3
                                    [-2, 2],         #logsfmu4
                                    [-2, 2],         #logsfmu5
                                    [-2, 2],         #logsfmu6
                                    [0.3, 0.3],      #logsfsig1
                                    [0.3, 0.3],      #logsfsig2
                                    [0.3, 0.3],      #logsfsig3
                                    [0.3, 0.3],      #logsfsig4
                                    [0.3, 0.3],      #logsfsig5
                                    [0.3, 0.3],      #logsfsig6
                                    [2, 2]           #nu   
    ])

    hyperparameter_sigma_max = np.array([[0.01, 3.0], #logzsol
                                        [0.01, 4.0], #dust1
                                        [0.01, 4.0], #dust2
                                        [0.01, 2.6], #dust_index
                                        [0.1, 0.1], #igm_factor 
                                        [0.01, 3.0], #gas_logu
                                        [0.01, 2.5], #gas_logz
                                        [0.01, 6.0], #log10(fagn)
                                        [0.1, 145]   #agn_tau
    ])

    return hyperparameter_mu_bounds, hyperparameter_sigma_max

def uniform_hyperparameter_bounds():
    """Why is this called uniform?? uniform parameters for sample_prior_parameters()
    """

    hyperparameter_mu_bounds = np.array([[-1.0, -1.0],  #logzsol
                                    [2.0, 2.0],      #dust1
                                    [2.0, 2.0],      #dust2
                                    [-0.7, -0.7],    #dust_index
                                    [1.0, 1.0],      #igm_factor 
                                    [-2.0, -2.0],    #gas_logu
                                    [0.0, 0.0],      #gas_logz
                                    [-3.0, -3.0],    #log10(fagn)
                                    [50, 50],        #agn_tau
                                    [-2, 2],         #logsfmu1
                                    [-2, 2],         #logsfmu2
                                    [-2, 2],         #logsfmu3
                                    [-2, 2],         #logsfmu4
                                    [-2, 2],         #logsfmu5
                                    [-2, 2],         #logsfmu6
                                    [0.3, 0.3],      #logsfsig1
                                    [0.3, 0.3],      #logsfsig2
                                    [0.3, 0.3],      #logsfsig3
                                    [0.3, 0.3],      #logsfsig4
                                    [0.3, 0.3],      #logsfsig5
                                    [0.3, 0.3],      #logsfsig6
                                    [2, 2]           #nu   
    ])

    hyperparameter_sigma_max = np.array([[3.0, 3.0], #logzsol
                                        [4.0, 4.0], #dust1
                                        [4.0, 4.0], #dust2
                                        [2.6, 2.6], #dust_index
                                        [2.0, 2.0], #igm_factor 
                                        [3.0, 3.0], #gas_logu
                                        [2.5, 2.5], #gas_logz
                                        [6.0, 6.0], #log10(fagn)
                                        [145, 145]   #agn_tau
    ])

    return hyperparameter_mu_bounds, hyperparameter_sigma_max

def plot_hyperparameters(prior_parameters, rows=5, nbins=20):
    """Plots output of sample_prior_parameters()
    
    """
    nparams = prior_parameters.shape[1]
    columns = math.ceil(nparams/rows)
    total_plots = nparams
    grid = rows*columns

    names = np.array(["logzsol_mu", "logzsol_sig", "dust1_mu", "dust1_sig", "dust2_mu", "dust2_sig", "dust_index_mu", "dust_index_sig",
                      "igm_factor_mu", "igm_factor_sig", "gas_logu_mu", "gas_logu_sig", "gas_logz_mu", "gas_logz_sig", "logfagn_mu", "logfagn_sig", "agn_tau_mu", "agn_tau_sig", 
                        "logsfmu1", "logsfmu2", "logsfmu3", "logsfmu4", "logsfmu5", "logsfmu6", "logsfsig1",
                          "logsfsig2", "logsfsig3", "logsfsig4", "logsfsig5", "logsfsig6", "nu"])


    fig1, axes1 = plt.subplots(rows, columns, figsize=(20,20), sharex=False, sharey=False)

    i = 0
    j = 0
    plot_no = 0
    name_count = 0
    col = 0
    while(col < nparams):

        if(i > rows - 1):
            j+=1
            i=0

        if(plot_no > total_plots):
            axes1[i, j].set_axis_off()

        else:
            axes1[i, j].hist(prior_parameters[:,col], density = True, bins=nbins)
            axes1[i, j].set_xlabel(names[name_count])
            axes1[i, j].set_ylabel("$p(z)$")
        
        i+=1
        plot_no += 1
        name_count += 1
        col += 1

    #clear blank figures
    no_empty_plots = grid - nparams
    i = 0
    while(i < no_empty_plots):
        axes1[rows - i - 1, columns - 1].set_axis_off()
        i+=1
    
    plt.tight_layout()




