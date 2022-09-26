"""
September 2022 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

This code particularly contains the simulations by Troelstra et al. Self-supervised neural network improves tri-exponential intravoxel incoherent motion model fitting compared to least-squares fitting in non-alcoholic fatty liver disease. Front. Physiol., 06 September 2022
Sec. Computational Physiology and Medicine
(https://doi.org/10.3389/fphys.2022.942495)

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients. MRM 2021)

requirements:
numpy
torch
tqdm
matplotlib
scipy
joblib
"""
import os
print(os.getcwd())
# import
import numpy as np
from hyperparams import hyperparams as hp_example
from sys import path
path.append('../')
import IVIMNET.simulations as sim
import IVIMNET.deep as deep
import time
import torch
import IVIMNET.fitting_algorithms as fit
# load hyperparameter
arg = hp_example()
arg = deep.checkarg(arg)

arg = deep.checkarg(arg)
# this simulated the signal
if arg.fit.model == 'bi-exp':
    dims = 4
else:
    dims = 6
# prepare a larger array in case we repeat training
lowest_loss=True
lowest_loss_val=10000
if arg.sim.repeats > 1 and not lowest_loss:
    paramsNN = np.zeros([arg.sim.repeats, len(arg.sim.SNR_eval), dims, arg.sim.num_samples_eval])
    matNN = np.zeros([arg.sim.repeats, len(arg.sim.SNR_eval), dims - 1, 3])
else:
    paramsNN = np.zeros([len(arg.sim.SNR_eval), dims, arg.sim.num_samples_eval])
    matNN = np.zeros([len(arg.sim.SNR_eval), dims - 1, 3])

matlsq = np.zeros([len(arg.sim.SNR_eval),dims - 1, 3])
stability = np.sqrt(np.mean(np.square(np.std(paramsNN, axis=0)), axis=1))

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
        bb = 0
        print('results for NN')
        for SNR in arg.sim.SNR_eval:
            print('\nSNR is {}\n'.format(SNR))
            if arg.fit.model == 'bi-exp':
                IVIM_signal_noisy, D, f, Dp = sim.sim_signal(SNR, arg.sim.bvalues, sims=arg.sim.num_samples_eval,
                                                         Dmin=arg.sim.range[0][0],
                                                         Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                                         fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                                         Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)
            else:
                IVIM_signal_noisy, D, f, Dp, f2, Dp2 = sim.sim_signal(SNR, arg.sim.bvalues, sims=arg.sim.num_samples_eval,
                                                                  Dmin=arg.sim.range[0][0],
                                                                  Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                                                  fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                                                  Dsmax=arg.sim.range[1][2], fp2min=arg.sim.range[0][3],
                                                                  fp2max=arg.sim.range[1][3],
                                                                  Ds2min=arg.sim.range[0][4],
                                                                  Ds2max=arg.sim.range[1][4], rician=arg.sim.rician,
                                                                  bi_exp=False)
            print(IVIM_signal_noisy[0])
            if arg.sim.repeats > 1:
                if lowest_loss:
                    if lowest_loss_val>net.best_loss:
                        paramsNN[bb] = deep.predict_IVIM(IVIM_signal_noisy, arg.sim.bvalues, net,
                                                             arg)
                        elapsed_time = time.time() - start_time
                        print('\ntime elapsed for inference: {}\n'.format(elapsed_time))
                        if arg.fit.model == 'bi-exp':
                            matNN[bb] = sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp),
                                                             paramsNN[bb])
                        else:
                            matNN[bb] = sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp),
                                                             paramsNN[bb], np.squeeze(f2),
                                                             np.squeeze(Dp2))
                else:
                    paramsNN[aa, bb] = deep.predict_IVIM(IVIM_signal_noisy, arg.sim.bvalues, net,
                                                         arg)
                    elapsed_time = time.time() - start_time
                    print('\ntime elapsed for inference: {}\n'.format(elapsed_time))
                    if arg.fit.model == 'bi-exp':
                        matNN[aa,bb] = sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsNN[aa,bb])
                        stability = stability[[0, 1, 2]] / [np.mean(D), np.mean(f), np.mean(Dp)]
                    else:
                        matNN[aa,bb] = sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsNN[aa,bb], np.squeeze(f2),
                                                 np.squeeze(Dp2))
                        #stability = stability[[0, 1, 2, 3, 4]] / [np.mean(D), np.mean(f), np.mean(Dp), np.mean(f2),
                        #                                          np.mean(Dp2)]
            else:
                paramsNN[bb] = deep.predict_IVIM(IVIM_signal_noisy, arg.sim.bvalues, net,
                                             arg)
                elapsed_time = time.time() - start_time
                print('\ntime elapsed for inference: {}\n'.format(elapsed_time))
                if arg.fit.model == 'bi-exp':
                    matNN[bb] = sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsNN[bb])
                else:
                    matNN[bb] = sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsNN[bb], np.squeeze(f2),
                                         np.squeeze(Dp2))
                stability = np.zeros(dims - 1)

            bb = bb + 1
        # remove network to save memory
        del net
        if arg.train_pars.use_cuda:
            torch.cuda.empty_cache()
    if arg.sim.repeats > 1 and not lowest_loss:
        matNN = np.mean(matNN, axis=0)
else:
    # if network is skipped
    stability = np.zeros(dims - 1)
    matNN = np.zeros([3, 5])

if arg.fit.do_fit:
    bb=0
    for SNR in arg.sim.SNR_eval:
        if arg.fit.model == 'bi-exp':
            IVIM_signal_noisy, D, f, Dp = sim.sim_signal(SNR, arg.sim.bvalues, sims=arg.sim.num_samples_eval,
                                                     Dmin=arg.sim.range[0][0],
                                                     Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                                     fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                                     Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)
        else:
            IVIM_signal_noisy, D, f, Dp, f2, Dp2 = sim.sim_signal(SNR, arg.sim.bvalues, sims=arg.sim.num_samples_eval,
                                                              Dmin=arg.sim.range[0][0],
                                                              Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                                              fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                                              Dsmax=arg.sim.range[1][2], fp2min=arg.sim.range[0][3],
                                                              fp2max=arg.sim.range[1][3],
                                                              Ds2min=arg.sim.range[0][4],
                                                              Ds2max=arg.sim.range[1][4], rician=arg.sim.rician,
                                                              bi_exp=False)
        start_time = time.time()
        # all fitting is done in the fit.fit_dats for the other fitting algorithms (lsq, segmented and Baysesian)
        paramsf = fit.fit_dats(arg.sim.bvalues, IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.fit)
        elapsed_time = time.time() - start_time
        print('\ntime elapsed for fit: {}\n'.format(elapsed_time))
        print('results for fit')
        # determine errors and Spearman Rank
        if arg.fit.model == 'bi-exp':
            matlsq[bb] = sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsf)
        else:
            matlsq[bb] = sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsf, np.squeeze(f2), np.squeeze(Dp2))
        # del paramsf, IVIM_signal_noisy
        # show figures if requested
        sim.plots(arg, D, Dp, f, paramsf)
        bb=bb+1

print(matlsq[:,:,1])
print(matNN[:,:,1])
