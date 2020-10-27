"""
September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients. MRM 2021)
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

# import
import numpy as np
import simulations as sim
import hyperparams as hp
from hyperparams import hyperparams as hp_example_2

# load hyperparameters
arg = hp_example_2()

matlsq = np.zeros([len(arg.sim.SNR), 3, 3])
matNN = np.zeros([len(arg.sim.SNR), 3, 3])
stability = np.zeros([len(arg.sim.SNR), 3])
a = 0

for SNR in arg.sim.SNR:
    print('\n simulation at SNR of {snr}\n'.format(snr=SNR))
    if arg.fit.do_fit:
        matlsq[a, :, :], matNN[a, :, :], stability[a, :] = sim.sim(SNR, arg)
        print('\nresults from lsq:')
        print(matlsq)
    else:
        matNN[a, :, :], stability[a, :] = sim.sim(SNR, arg)
    a = a + 1
    print('\nresults from NN: columns show the mean, the RMSE/mean and the Spearman coef [DvDp,Dvf,fvDp] \n'
          'the rows show D, f and D*\n'
          'and the different matixes represent the different SNR levels:')
    print(matNN)
    # if repeat is higher than 1, then print stability (CVNET)
    if arg.sim.repeats > 1:
        print('\nstability of NN for D, f and D* at different SNR levels:')
        print(stability)
