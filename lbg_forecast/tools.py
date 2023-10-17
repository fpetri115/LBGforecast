import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.sps as sps
import lbg_forecast.hyperparams as hyp
import lbg_forecast.popmodel as pop
import lbg_forecast.sfh as sfh
import lbg_forecast.zhistory as zh
from astropy.cosmology import WMAP9 as cosmo
import math

# simulate photometry for ngalaxies given some hyperparameters
# returns tuple of 2 arrays
# first element is a ngalaxies x number of filters array containing photometry for the sample
# second element is a ngalaxies x number of SPS parameters containing SPS parameters for each source/galaxy
def simulate_photometry(ngalaxies, hyperparams, dust_type, imf_type, zhistory, nebem, filters):
    
    if(nebem == False and zhistory == True):
        raise Exception("nebular emission cannot be turned off with zhistory enabled at present")

    #draw parameters from priors
    sps_parameters = draw_sps_parameters(ngalaxies, hyperparams)
    np.save('generated_spsparams', sps_parameters)

    print("SPS Parameters Generated")
    ###################################################

    #Define SPS Model with Nebular emmision
    print("Starting Run 1/3")
    sps_model = sps.initialise_sps_model(sfh_type=3, neb_em=nebem, zcont=1, dust_type=dust_type, imf_type=imf_type)

    print("libraries: ", sps_model.libraries)

    #GENERATE PHOTOMETRY FOR GIVEN SPS PARAMETERS WITH NEBULAR EMMISION
    ###################################################
    i = 0
    photometry_neb = []
    while(i < ngalaxies):

        source = sps_parameters[i]
        sps.update_sps_model_dpl(sps_model, source, zhis=False)

        #generate photometry for source
        photometry_neb.append(sps.simulate_photometry_fsps(sps_model, logmass=source[-1], filters=filters))

        i+=1
        if(i%1000 == 0):
            print(i)

    photometry_neb = np.vstack(np.asarray(photometry_neb))
    #np.save('generated_photo_neb', photometry_neb)

    print("Run 1/3 Complete")
    ###################################################

    if(zhistory):

        print("Starting Run 2/3")
        #Define SPS Model without Nebular emmision
        sps_model = sps.initialise_sps_model(sfh_type=3, neb_em=False, zcont=1, dust_type=dust_type, imf_type=imf_type)

        #GENERATE PHOTOMETRY FOR GIVEN SPS PARAMETERS WITHOUT NEBULAR EMMISION
        ###################################################
        i = 0
        photometry_no_neb = []
        while(i < ngalaxies):

            source = sps_parameters[i]
            sps.update_sps_model_dpl(sps_model, source, zhis=False)

            #generate photometry for source
            photometry_no_neb.append(sps.simulate_photometry_fsps(sps_model, logmass=source[-1], filters=filters))

            i+=1
            if(i%1000 == 0):
                print(i)

        photometry_no_neb = np.vstack(np.asarray(photometry_no_neb))
        #np.save('generated_photo_no_neb', photometry_no_neb)

        print("Run 2/3 Complete")
        ###################################################

        photometric_contribution_from_neb = photometry_neb - photometry_no_neb
        
        print("Starting Run 3/3")
        #Define SPS Model without Nebular emmision BUT with zhistory
        sps_model = sps.initialise_sps_model(sfh_type=3, neb_em=False, zcont=3, dust_type=dust_type, imf_type=imf_type)

        #GENERATE PHOTOMETRY FOR GIVEN SPS PARAMETERS WITHOUT NEBULAR EMMISION BUT WITH ZHISTORY
        ###################################################
        i = 0
        photometry_zhis = []
        while(i < ngalaxies):

            source = sps_parameters[i]
            sps.update_sps_model_dpl(sps_model, source, zhis=True)

            #generate photometry for source
            photometry_zhis.append(sps.simulate_photometry_fsps(sps_model, logmass=source[-1], filters=filters))

            i+=1
            if(i%1000 == 0):
                print(i)

        photometry_zhis = np.vstack(np.asarray(photometry_zhis))
        ###################################################
        
        print("Run 3/3 Complete")
        photometry_final = photometry_zhis + photometric_contribution_from_neb
        #np.save('generated_photo_zhis_no_neb', photometry_zhis)
        np.save('generated_photo_final', photometry_final)
    
    else:
        photometry_final = photometry_neb
        np.save('generated_photo_final', photometry_final)

    print("Complete")

    return photometry_final, sps_parameters

#draw population of sps parameters given priors/hyperparameters
#extra sorting of imf paramters for faster computation using large ngalaxies
def draw_sps_parameters(ngalaxies, hyperparams, imf_spacing=4):

    #main loop over parameters
    i = 0
    sps_parameters = []
    while(i < ngalaxies):

        source = pop.galaxy_population_model(hyperparams)
        sps_parameters.append(source)

        i+=1
    
    sps_parameters = np.vstack(np.asarray(sps_parameters))

    #round imf parameters to nearest decimal
    #if imf_spacing =4 and decimals =1, then this will be to every 0.4
    imf_params = np.vstack(sps_parameters[:, 8:11])
    imf_params = np.round(imf_params*(1/imf_spacing), decimals = 1)*imf_spacing
    sps_parameters[:, 8:11] = imf_params

    #do a weighted sum of IMF parameters
    #add these sums to in a column to sps params
    #this allows for the sps parameters to be ordered given the weighted sum of imf parameters
    #this should allow for faster computation of photometry in large simulations
    imf_params = imf_params*np.array([1.02, 1.01, 1.00]) #weights
    sums = np.sum(imf_params, axis=1)
    sums = np.reshape(sums, (len(sums), 1))

    #add to column
    sps_parameters = np.append(sps_parameters, sums, axis=1)

    #sort rows by sums
    sps_parameters = sps_parameters[sps_parameters[:, -1].argsort()]

    #remove sum column after sorting
    sps_parameters = sps_parameters[:, :-1]

    if(all(cosmo.age(sps_parameters[:, 0][:]).value > 10**sps_parameters[:, 1][:]) == False):
        raise Exception("Age of galaxy > Age of universe at given redshift!!")

    return sps_parameters

#calculate sfh at index for a given set of sps parameters from draw_sps_parameters()
def calculate_sfh(sps_parameters, index, show_plot=True):

    sfh_params = np.vstack(sps_parameters[:, 11:14])
    logages = sps_parameters[:, 1]

    time_grid = np.logspace(-7, logages[index], 1000)

    tau = sfh_params[index, 0]
    a = sfh_params[index, 1]
    b = sfh_params[index, 2]

    sfhis = sfh.normed_sfh(tau, a, b, time_grid)

    if(show_plot):
        plt.figure(figsize=(10,5))
        plt.plot(time_grid, sfhis)
        plt.xlabel("Time Since the Beginning of the Universe [$\mathrm{Gyr}$]",
                fontsize=12)
        plt.ylabel("Star Formation Rate [$\mathrm{M}_{\odot}\mathrm{yr}^{-1}$]",
                fontsize=12)
        
        plt.tick_params(axis="x", width = 2, labelsize=12*0.8)
        plt.tick_params(axis="y", width = 2, labelsize=12*0.8)

        return sfhis
    else:
        return sfhis

#calculate sfh at index for a given set of sps parameters from draw_sps_parameters()
def calculate_zhis(sps_parameters, index, show_plot=True):

    Z_MIST = 0.0142
    sfh_params = np.vstack(sps_parameters[:, 11:14])
    logages = sps_parameters[:, 1]
    z_gases = (10**sps_parameters[:, 2])*Z_MIST

    time_grid = np.logspace(-7, logages[index], 1000)

    tau = sfh_params[index, 0]
    a = sfh_params[index, 1]
    b = sfh_params[index, 2]
    z_gas = z_gases[index]

    z_history = zh.sfr_to_zh(sfh.normed_sfh(tau, a, b, time_grid), time_grid, z_gas)
    if(show_plot):
        plt.figure(figsize=(10,5))
        plt.plot(time_grid, z_history)
        plt.xlabel("Time Since the Beginning of the Universe [$\mathrm{Gyr}$]",
                fontsize=12)
        plt.ylabel("Chemical Evolution [$\mathrm{Absolute \  Metallicity}$]",
                fontsize=12)
        
        plt.tick_params(axis="y", width = 2, labelsize=12*0.8)
        plt.tick_params(axis="x", width = 2, labelsize=12*0.8)

        return z_history
    else:
        return z_history
    
def sfh_zhis_diag(sps_parameters, index):

    print("Galaxy Age (Gyr):", 10**sps_parameters[index, 1])
    print("Age of the universe at given redshift (Gyr):", cosmo.age(sps_parameters[index, 0]).value)
    print("Redshift:", sps_parameters[index, 0])
    print("Observed Metallicity (Absolute Metallicity):", 10**sps_parameters[index, 2]*0.0142)
    sfh = calculate_sfh(sps_parameters, index)
    zhis = calculate_zhis(sps_parameters, index)

    return (zhis, sfh)


#plot galaxy population given by draw_sps_parameters()/simulate_photo..()
def plot_galaxy_population(sps_parameters, rows=5, nbins=20):
    
    realisations = sps_parameters
    nparams = realisations.shape[1]

    names = np.array(["zred", "$\mathrm{log_{10}tage}$", "logzsol", "dust1", "dust2", 
                      "igm_factor", "gas_logu", "fagn", "imf1",
                        "imf2", "imf3", "$\mathrm{log_{10}}tau$", "$\mathrm{log_{10}}a$", 
                        "$\mathrm{log_{10}}b$", "$\mathrm{log_{10}mass}$"])
    
    if(len(names) != nparams):
        raise Exception("Number of parameters and parameter labels don't match")

    columns = math.ceil(nparams/rows)
    total_plots = nparams
    grid = rows*columns

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
            axes1[i, j].hist(realisations[:,col], density = True, bins=nbins)
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

def calculate_colours(photometry):
    
    photo1 = photometry[:,:-1]
    photo2 = photometry[:,1:]
    colours = photo1 - photo2

    return colours
