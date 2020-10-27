"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved physics-informed deep learning of the intravoxel-incoherent motion model: accurate, unique and consistent. MRM 2021)
If this code was useful, please cite:
http://arxiv.org/abs/1903.00095

requirements:
numpy
torch
tqdm
matplotlib
scipy
joblib
"""

# import libraries
import numpy as np
import deep as deep
import fitting_algorithms as fit
import time
import scipy.stats as scipy
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import os


def sim(SNR, arg):
    """ This function defines how well the different fit approaches perform on simulated data. Data is simulated by
    randomly selecting a value of D, f and D* from within the predefined range. The script calculates the random,
    systematic, root-mean-squared error (RMSE) and Spearman Rank correlation coefficient for each of the IVIM parameters.
    Furthermore, it calculates the stability of the neural network (when trained multiple times).

    input:
    :param SNR: SNR of the simulated data. If SNR is set to 0, no noise is added
    :param arg: an object with simulation options. hyperparams.py gives most details on the object (and defines it),
    Relevant attributes are:
    arg.sim.sims = number of simulations to be performed (need a large amount for training)
    arg.sim.num_samples_eval = number of samples to evaluate (save time for lsq fitting)
    arg.sim.repeats = number of times to repeat the training and evaluation of the network (to assess stability)
    arg.sim.bvalues: 1D Array of b-values used
    arg.fit contains the parameters regarding lsq fitting
    arg.train_pars and arg.net_pars contain the parameters regarding the neural network
    arg.sim.range gives the simulated range of D, f and D* in a 2D array

    :return matlsq: 2D array containing the performance of the lsq fit (if enabled). The rows indicate D, f (Fp), D*
    (Dp), whereas the colums give the mean input value, the random error and the systematic error
    :return matNN: 2D array containing the performance of the NN. The rows indicate D, f (Fp), D*
    (Dp), whereas the colums give the mean input value, the random error and the systematic error
    :return stability: a 1D array with the stability of D, f and D* as a fraction of their mean value.
    Stability is only relevant for neural networks and is calculated from the repeated network training.
    """
    arg = deep.checkarg(arg)
    # this simulated the signal
    IVIM_signal_noisy, D, f, Dp = sim_signal(SNR, arg.sim.bvalues, sims=arg.sim.sims, Dmin=arg.sim.range[0][0],
                                             Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                             fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                             Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)

    # only remember the D, Dp and f needed for evaluation
    D = D[:arg.sim.num_samples_eval]
    Dp = Dp[:arg.sim.num_samples_eval]
    f = f[:arg.sim.num_samples_eval]

    # prepare a larger array in case we repeat training
    if arg.sim.repeats > 1:
        paramsNN = np.zeros([arg.sim.repeats, 4, arg.sim.num_samples_eval])
    else:
        paramsNN = np.zeros([4, arg.sim.num_samples_eval])

    # if we are not skipping the network for evaluation
    if not arg.train_pars.skip_net:
        # loop over repeats
        for aa in range(arg.sim.repeats):
            start_time = time.time()
            # train network
            print('\nRepeat: {repeat}\n'.format(repeat=aa))
            net = deep.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg)
            elapsed_time = time.time() - start_time
            print('\ntime elapsed for training: {}\n'.format(elapsed_time))
            start_time = time.time()
            # predict parameters
            if arg.sim.repeats > 1:
                paramsNN[aa] = deep.predict_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net,
                                                 arg)
            else:
                paramsNN = deep.predict_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net,
                                             arg)
            elapsed_time = time.time() - start_time
            print('\ntime elapsed for inference: {}\n'.format(elapsed_time))
            # remove network to save memory
            del net
            if arg.train_pars.use_cuda:
               torch.cuda.empty_cache()
        print('results for NN')
        # if we repeat training, then evaluate stability
        if arg.sim.repeats > 1:
            matNN = np.zeros([arg.sim.repeats, 3, 3])
            for aa in range(arg.sim.repeats):
                # determine Errors and Spearman Rank
                matNN[aa] = print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsNN[aa])
            matNN = np.mean(matNN, axis=0)
            # calculate Stability Factor
            stability = np.sqrt(np.mean(np.square(np.std(paramsNN, axis=0)), axis=1))
            stability = stability[[0, 1, 2]] / [np.mean(D), np.mean(f), np.mean(Dp)]
            # set paramsNN for the plots
            paramsNN_0 = paramsNN[0]
        else:
            matNN = print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsNN)
            stability = np.zeros(3)
            paramsNN_0 = paramsNN
        # del paramsNN
        # show figures if requested
        plots(arg, D, Dp, f, paramsNN)
    else:
        # if network is skipped
        stability = np.zeros(3)
        matNN = np.zeros([3, 5])
    if arg.fit.do_fit:
        start_time = time.time()
        # all fitting is done in the fit.fit_dats for the other fitting algorithms (lsq, segmented and Baysesian)
        paramsf = fit.fit_dats(arg.sim.bvalues, IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.fit)
        elapsed_time = time.time() - start_time
        print('\ntime elapsed for fit: {}\n'.format(elapsed_time))
        print('results for fit')
        # calculate and print the errors and Spearman Rank
        matlsq = print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsf)
        # del paramsf, IVIM_signal_noisy
        # Show figures if requested
        plots(arg, D, Dp, f, paramsf)
        return matlsq, matNN, stability
    else:
        # if lsq fit is skipped, don't export lsq results
        return matNN, stability


def plots(arg,D,Dp,f,params):
    if arg.fig:
        dummy = np.array(params)
        # plot correlations
        plt.figure()
        plt.plot(D[:1000], Dp[:1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.3)
        plt.xlabel('Dt')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('inputDtDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(f[:1000], Dp[:1000], 'rx', markersize=5)
        plt.xlim(0, 0.6)
        plt.ylim(0, 0.3)
        plt.xlabel('f')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('inputfDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(D[:1000], f[:1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.6)
        plt.xlabel('Dt')
        plt.ylabel('f')
        plt.gcf()
        # plt.savefig('inputDtf.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(dummy[0, :1000], dummy[2, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.3)
        plt.xlabel('Dt')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('NNDtDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(dummy[1, :1000], dummy[2, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.6)
        plt.ylim(0, 0.3)
        plt.xlabel('f')
        plt.ylabel('Dp')
        plt.gcf()
        # plt.savefig('NNfDp.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(dummy[0, :1000], dummy[1, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.6)
        plt.xlabel('Dt')
        plt.ylabel('f')
        plt.gcf()
        # plt.savefig('NNDtf.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(Dp[:1000], dummy[2, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.3)
        plt.ylim(0, 0.3)
        plt.ylabel('DpNN')
        plt.xlabel('Dpin')
        plt.gcf()
        # plt.savefig('DpNNDpin.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(D[:1000], dummy[0, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.005)
        plt.ylim(0, 0.005)
        plt.ylabel('DtNN')
        plt.xlabel('Dtin')
        plt.gcf()
        # plt.savefig('DtNNDtin.png')
        plt.ion()
        plt.show()
        plt.figure()
        plt.plot(f[:1000], dummy[1, :1000], 'rx', markersize=5)
        plt.xlim(0, 0.6)
        plt.ylim(0, 0.6)
        plt.ylabel('fNN')
        plt.xlabel('fin')
        plt.gcf()
        # plt.savefig('fNNfin.png')
        plt.ion()
        plt.show()
        plt.close('all')


def augmented_signal(data, bvalues, arg, fraction=0.3, Dmin=0.3 / 1000, Dmax=4.0 / 1000, fmin=0.0, fmax=0.8, Dsmin=0.01,
                     Dsmax=0.2):
    """
    This simulates IVIM curves with real noise from actual data. This can be used as augmented training data.
    Data is simulated by randomly selecting a value of D, f and D* from within the predefined range.

    input:
    :param data: real data from experiment
    :param bvalues: 1D Array of b-values used
    :param sims: number of simulations to be performed (need a large amount for training)

    optional:
    :param fraction: fraction of data to augment
    :param Dmin: minimal simulated D. Default = 0.0003
    :param Dmax: maximal simulated D. Default = 0.004
    :param fmin: minimal simulated f. Default = 0.0
    :param Dmax: minimal simulated f. Default = 0.8
    :param Dpmin: minimal simulated D*. Default = 0.01
    :param Dpmax: minimal simulated D*. Default = 0.2

    :return IVIM_signal_noisy: 2D array with augmented noisy IVIM signal (x-axis is sims long, y-axis is len(b-values) long)
    """
    arg = deep.checkarg(arg)
    indx = np.argsort(bvalues.copy())
    sdata = np.shape(data)[0]
    # train
    net = deep.learn_IVIM(data[:, indx], bvalues[indx], arg)
    # select random samples to augment
    sels = random.sample(range(sdata), round(sdata * fraction))
    # predict noise
    noise = deep.predict_noise(data[sels][:, indx], bvalues[indx], net, arg)
    # augment data
    IVIM_signal_noisy, D, f, Dp = sim_signal(0, bvalues[indx], sims=len(noise), Dmin=Dmin, Dmax=Dmax, fmin=fmin,
                                             fmax=fmax,
                                             Dsmin=Dsmin, Dsmax=Dsmax)
    # add noise to augmented data
    IVIM_signal_noisy = IVIM_signal_noisy + noise
    indx_back = np.argsort(indx)
    del net
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    return IVIM_signal_noisy[indx_back]


def sim_signal(SNR, bvalues, sims=100000, Dmin=0.5 / 1000, Dmax=2.0 / 1000, fmin=0.1, fmax=0.5, Dsmin=0.05, Dsmax=0.2,
               rician=False, state=123):
    """
    This simulates IVIM curves. Data is simulated by randomly selecting a value of D, f and D* from within the
    predefined range.

    input:
    :param SNR: SNR of the simulated data. If SNR is set to 0, no noise is added
    :param bvalues: 1D Array of b-values used
    :param sims: number of simulations to be performed (need a large amount for training)

    optional:
    :param Dmin: minimal simulated D. Default = 0.0005
    :param Dmax: maximal simulated D. Default = 0.002
    :param fmin: minimal simulated f. Default = 0.1
    :param Dmax: minimal simulated f. Default = 0.5
    :param Dpmin: minimal simulated D*. Default = 0.05
    :param Dpmax: minimal simulated D*. Default = 0.2
    :param rician: boolean giving whether Rician noise is used; default = False

    :return data_sim: 2D array with noisy IVIM signal (x-axis is sims long, y-axis is len(b-values) long)
    :return D: 1D array with the used D for simulations, sims long
    :return f: 1D array with the used f for simulations, sims long
    :return Dp: 1D array with the used D* for simulations, sims long
    """

    # randomly select parameters from predefined range
    rg = np.random.RandomState(state)
    test = rg.uniform(0, 1, (sims, 1))
    D = Dmin + (test * (Dmax - Dmin))
    test = rg.uniform(0, 1, (sims, 1))
    f = fmin + (test * (fmax - fmin))
    test = rg.uniform(0, 1, (sims, 1))
    Dp = Dsmin + (test * (Dsmax - Dsmin))

    # initialise data array
    data_sim = np.zeros([len(D), len(bvalues)])
    bvalues = np.array(bvalues)

    # loop over array to fill with simulated IVIM data
    for aa in range(len(D)):
        data_sim[aa, :] = fit.ivim(bvalues, D[aa][0], f[aa][0], Dp[aa][0], 1)

    # if SNR is set to zero, don't add noise
    if SNR > 0:
        # initialise noise arrays
        noise_imag = np.zeros([sims, len(bvalues)])
        noise_real = np.zeros([sims, len(bvalues)])
        # fill arrays
        for i in range(0, sims - 1):
            noise_real[i,] = rg.normal(0, 1 / SNR,
                                       (1, len(bvalues)))  # wrong! need a SD per input. Might need to loop to maD noise
            noise_imag[i,] = rg.normal(0, 1 / SNR, (1, len(bvalues)))
        if rician:
            # add Rician noise as the square root of squared gaussian distributed real signal + noise and imaginary noise
            data_sim = np.sqrt(np.power(data_sim + noise_real, 2) + np.power(noise_imag, 2))
        else:
            # or add Gaussian noise
            data_sim = data_sim + noise_imag
    else:
        data_sim = data_sim

    # normalise signal
    S0_noisy = np.mean(data_sim[:, bvalues == 0], axis=1)
    data_sim = data_sim / S0_noisy[:, None]
    return data_sim, D, f, Dp


def print_errors(D, f, Dp, params):
    # this function calculates and prints the random, systematic, root-mean-squared (RMSE) errors and Spearman Rank correlation coefficient

    rmse_D = np.sqrt(np.square(np.subtract(D, params[0])).mean())

    rmse_f = np.sqrt(np.square(np.subtract(f, params[1])).mean())

    rmse_Dp = np.sqrt(np.square(np.subtract(Dp, params[2])).mean())

    # Initialise Spearman Rank matrix
    Spearman = np.zeros([3, 2])
    # Calculate Spearman Rank correlation coefficient and p-value
    Spearman[0, 0], Spearman[0, 1] = scipy.spearmanr(params[0], params[2])  # DvDp
    Spearman[1, 0], Spearman[1, 1] = scipy.spearmanr(params[0], params[1])  # Dvf
    Spearman[2, 0], Spearman[2, 1] = scipy.spearmanr(params[1], params[2])  # fvDp
    # If spearman is nan, set as 1 (because of constant estimated IVIM parameters)
    Spearman[np.isnan(Spearman)] = 1
    # Take absolute Spearman
    Spearman = np.absolute(Spearman)
    del params

    normD_lsq = np.mean(D)
    normf_lsq = np.mean(f)
    normDp_lsq = np.mean(Dp)

    print('\nresults from NN: columns show themean, the SD/mean, the systematic error/mean, the RMSE/mean and the Spearman coef [DvDp,Dvf,fvDp] \n'
          'the rows show D, f and D*\n')
    print([normD_lsq, '  ', rmse_D / normD_lsq, ' ', Spearman[0, 0]])
    print([normf_lsq, '  ', rmse_f / normf_lsq, ' ', Spearman[1, 0]])
    print([normDp_lsq, '  ', rmse_Dp / normDp_lsq,' ', Spearman[2, 0]])

    mats = [[normD_lsq, rmse_D / normD_lsq, Spearman[0, 0]],
            [normf_lsq, rmse_f / normf_lsq, Spearman[1, 0]],
            [normDp_lsq, rmse_Dp / normDp_lsq, Spearman[2, 0]]]

    return mats


def sim_signal_predict(arg, SNR):
    # Init Randomstate
    rg = np.random.RandomState(123)
    ## define parameter values in the three regions
    S0_region0, S0_region1, S0_region2 = 1, 1, 1
    Dp_region0, Dp_region1, Dp_region2 = 0.03, 0.05, 0.07
    Dt_region0, Dt_region1, Dt_region2 = 0.0020, 0.0015, 0.0010
    Fp_region0, Fp_region1, Fp_region2 = 0.15, 0.3, 0.45
    # image size
    sx, sy, sb = 100, 100, len(arg.sim.bvalues)
    # create image
    dwi_image = np.zeros((sx, sy, sb))
    Dp_truth = np.zeros((sx, sy))
    Dt_truth = np.zeros((sx, sy))
    Fp_truth = np.zeros((sx, sy))

    # fill image with simulated values
    for i in range(sx):
        for j in range(sy):
            if (40 < i < 60) and (40 < j < 60):
                # region 0
                dwi_image[i, j, :] = fit.ivim(arg.sim.bvalues, Dt_region0, Fp_region0, Dp_region0, S0_region0)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region0, Dt_region0, Fp_region0
            elif (20 < i < 80) and (20 < j < 80):
                # region 1
                dwi_image[i, j, :] = fit.ivim(arg.sim.bvalues, Dt_region1, Fp_region1, Dp_region1, S0_region1)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region1, Dt_region1, Fp_region1
            else:
                # region 2
                dwi_image[i, j, :] = fit.ivim(arg.sim.bvalues, Dt_region2, Fp_region2, Dp_region2, S0_region2)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region2, Dt_region2, Fp_region2

    # plot simulated diffusion weighted image
    fig, ax = plt.subplots(2, int(np.round(arg.sim.bvalues.shape[0] / 2)), figsize=(20, 20))
    b_id = 0
    for i in range(2):
        for j in range(int(np.round(arg.sim.bvalues.shape[0] / 2))):
            if not b_id == arg.sim.bvalues.shape[0]:
                ax[i, j].imshow(dwi_image[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('b = ' + str(arg.sim.bvalues[b_id]))
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            else:
                # ax[i, j].imshow(dwi_image[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('End of b-values')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            b_id += 1
    plt.subplots_adjust(hspace=0)
    plt.show()
    if not os.path.isdir('plots'):
        os.makedirs('plots')
    plt.savefig('plots/plot_dwi_without_noise_param_{snr}_{method}.png'.format(snr=SNR, method=arg.save_name))

    # Initialise dwi noise image
    dwi_noise_imag = np.zeros((sx, sy, sb))
    # fill dwi noise image with Gaussian noise
    for i in range(sx):
        for j in range(sy):
            dwi_noise_imag[i, j, :] = rg.normal(0, 1 / SNR, (1, len(arg.sim.bvalues)))
    # Add Gaussian noise to dwi image
    dwi_image_noise = dwi_image + dwi_noise_imag
    # normalise signal
    S0_dwi_noisy = np.mean(dwi_image_noise[:, :, arg.sim.bvalues == 0], axis=1)
    dwi_image_noise_norm = dwi_image_noise / S0_dwi_noisy[:, None]

    # plot simulated diffusion weighted image with noise
    fig, ax = plt.subplots(2, int(np.round(arg.sim.bvalues.shape[0] / 2)), figsize=(20, 20))
    b_id = 0
    for i in range(2):
        for j in range(int(np.round(arg.sim.bvalues.shape[0] / 2))):
            if not b_id == arg.sim.bvalues.shape[0]:
                ax[i, j].imshow(dwi_image_noise_norm[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('b = ' + str(arg.sim.bvalues[b_id]))
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            else:
                # ax[i, j].imshow(dwi_image[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('End of b-values')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            b_id += 1
    plt.subplots_adjust(hspace=0)
    plt.show()
    plt.savefig('plots/plot_dwi_with_noise_param_{snr}_{method}.png'.format(snr=SNR, method=arg.save_name))

    # Reshape image
    dwi_image_long = np.reshape(dwi_image_noise_norm, (sx * sy, sb))
    return dwi_image_long, Dt_truth, Fp_truth, Dp_truth


def plot_example1(paramsNN, paramsf, Dt_truth, Fp_truth, Dp_truth, arg, SNR):
    # Initialise figure
    sx, sy, sb = 100, 100, len(arg.sim.bvalues)
    if arg.fit.do_fit:
        fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    else:
        fig, ax = plt.subplots(2, 3, figsize=(20, 20))
    # Fill Figure with values
    Dt_t_plot = ax[0, 0].imshow(Dt_truth, cmap='gray', clim=(0, 0.003))
    ax[0, 0].set_title('Dt, ground truth')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    fig.colorbar(Dt_t_plot, ax=ax[0, 0], fraction=0.046, pad=0.04)

    Dt_plot = ax[1, 0].imshow(np.reshape(paramsNN[0], (sx, sy)), cmap='gray', clim=(0, 0.003))
    ax[1, 0].set_title('Dt, estimate')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    fig.colorbar(Dt_plot, ax=ax[1, 0], fraction=0.046, pad=0.04)

    Fp_t_plot = ax[0, 1].imshow(Fp_truth, cmap='gray', clim=(0, 0.5))
    ax[0, 1].set_title('Fp, ground truth')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    fig.colorbar(Fp_t_plot, ax=ax[0, 1], fraction=0.046, pad=0.04)

    Fp_plot = ax[1, 1].imshow(np.reshape(paramsNN[1], (sx, sy)), cmap='gray', clim=(0, 0.5))
    ax[1, 1].set_title('Fp, estimate')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    fig.colorbar(Fp_plot, ax=ax[1, 1], fraction=0.046, pad=0.04)

    Dp_t_plot = ax[0, 2].imshow(Dp_truth, cmap='gray', clim=(0.01, 0.1))
    ax[0, 2].set_title('Dp, ground truth')
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    fig.colorbar(Dp_t_plot, ax=ax[0, 2], fraction=0.046, pad=0.04)

    Dp_plot = ax[1, 2].imshow(np.reshape(paramsNN[2], (sx, sy)), cmap='gray', clim=(0.01, 0.1))
    ax[1, 2].set_title('Dp, estimate')
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    fig.colorbar(Dp_plot, ax=ax[1, 2], fraction=0.046, pad=0.04)
    plt.savefig('plots/plot_imshow_IVIM_param_{snr}_nn_{save}.png'.format(snr=SNR, save=arg.save_name))

    if arg.fit.do_fit:
        Dt_fit_plot = ax[2, 0].imshow(np.reshape(paramsf[0], (sx, sy)), cmap='gray', clim=(0, 0.003))
        ax[2, 0].set_title('Dt, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[2, 0].set_xticks([])
        ax[2, 0].set_yticks([])
        fig.colorbar(Dt_fit_plot, ax=ax[2, 0], fraction=0.046, pad=0.04)

        Fp_fit_plot = ax[2, 1].imshow(np.reshape(paramsf[1], (sx, sy)), cmap='gray', clim=(0, 0.5))
        ax[2, 1].set_title('f, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[2, 1].set_xticks([])
        ax[2, 1].set_yticks([])
        fig.colorbar(Fp_fit_plot, ax=ax[2, 2], fraction=0.046, pad=0.04)

        Dp_fit_plot = ax[2, 2].imshow(np.reshape(paramsf[2], (sx, sy)), cmap='gray', clim=(0.01, 0.1))
        ax[2, 2].set_title('Dp, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[2, 2].set_xticks([])
        ax[2, 2].set_yticks([])
        fig.colorbar(Dp_fit_plot, ax=ax[2, 2], fraction=0.046, pad=0.04)

        plt.subplots_adjust(hspace=0.2)
        plt.show()
        plt.savefig('plots/plot_imshow_IVIM_param_{snr}_lsq_{save}.png'.format(snr=SNR, save=arg.save_name))