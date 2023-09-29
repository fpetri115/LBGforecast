import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.sps as sps
import lbg_forecast.hyperparams as hyp
import lbg_forecast.popmodel as pop
import lbg_forecast.sfh as sfh

# simulate photometry for ngalaxies given some hyperparameters
# returns tuple of 2 arrays
# first element is a ngalaxies x number of filters array containing photometry for the sample
# second element is a ngalaxies x number of SPS parameters containing SPS parameters for each source/galaxy
def simulate_photometry(ngalaxies, hyperparams, dust_type=2, imf_type=2, filters='lsst', show_sfh=False):

    #Define SPS Model
    sps_model = sps.initialise_sps_model(sfh_type=3, dust_type=dust_type, imf_type=imf_type)
    
    #draw parameters from priors
    sps_parameters = draw_sps_parameters(ngalaxies, hyperparams)

    #sort and discretise imf parameters for ease of computation
    #this reduces the amount of times fsps needs to recompile
    #it should be okay to shuffle around imfs as imf priors are uncorrelated to other parameters
    #i.e. drawing zred, shouldn't affect the values drawnfor imf1 2 and 3
    # hence imf1, 2 and 3 can be shuffled for ease of computation
    imf_params = np.vstack(sps_parameters[:, 9:12])
    imf_params = np.round_(imf_params, decimals = 2)
    imf_params = np.sort(imf_params, axis=0, kind='quicksort')
    sps_parameters[:, 9:12] = imf_params

    print("SPS Parameters Generated")
    ###################################################

    #GENERATE PHOTOMETRY FOR GIVEN SPS PARAMETERS
    ###################################################
    i = 0
    photometry = []
    while(i < ngalaxies):

        source = sps_parameters[i]
        sps.update_sps_model_dpl(sps_model, source, plot=show_sfh)

        #generate photometry for source
        photometry.append(sps.simulate_photometry_fsps(sps_model, logmass=source[-1], filters=filters))

        i+=1
        if(i%100 == 0):
            print(i)

    photometry = np.vstack(np.asarray(photometry))
    ###################################################
    
    print("Complete")

    return photometry, sps_parameters


def draw_sps_parameters(ngalaxies, hyperparams):

    i = 0
    sps_parameters = []
    while(i < ngalaxies):

        source = pop.galaxy_population_model(hyperparams)
        sps_parameters.append(source)

        i+=1
    
    sps_parameters = np.vstack(np.asarray(sps_parameters))

    return sps_parameters

#calculate sfh at index
def calculate_sfh(sps_parameters, index, show_plot=False):

    sfh_params = np.vstack(sps_parameters[:, 12:15])
    logages = sps_parameters[:, 1]

    time_grid = np.logspace(-7, logages[index], 10000)

    tau = sfh_params[index, 0]
    a = sfh_params[index, 1]
    b = sfh_params[index, 2]

    if(show_plot):
        plt.figure(figsize=(10,5))
        plt.plot(time_grid, sfh.normed_sfh(tau, a, b, time_grid))
        plt.xlabel("Time Since the Beginning of the Universe [$\mathrm{Gyr}$]",
                fontsize=12)
        plt.ylabel("Star Formation Rate [$\mathrm{M}_{\odot}\mathrm{yr}^{-1}$]",
                fontsize=12)
        
        plt.tick_params(axis="x", width = 2, labelsize=12*0.8)
        plt.tick_params(axis="y", width = 2, labelsize=12*0.8)
    else:
        return sfh.normed_sfh(tau, a, b, time_grid)

def calculate_colours(photometry):
    
    photo1 = photometry[:,:-1]
    photo2 = photometry[:,1:]
    colours = photo1 - photo2

    return colours
