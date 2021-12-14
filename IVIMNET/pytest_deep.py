import pytest
import numpy as np
import simulations as sim
from hyperparams import hyperparams as hp_example
import deep
import time


def test_NN():
    arg = hp_example()
    arg = deep.checkarg(arg)
    arg.sim.repeats = 1
    arg.sim.sims=5000000
    SNR = (10,120)
    IVIM_signal_noisy, D, f, Dp = sim.sim_signal(SNR, arg.sim.bvalues, sims=arg.sim.sims, Dmin=arg.sim.range[0][0],
                                             Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                             fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                             Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)

    start_time = time.time()
    # train network
    net = deep.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg)
    elapsed_time = time.time() - start_time
    print('\ntime elapsed for training: {}\n'.format(elapsed_time))
    matNN = np.zeros([5, 3, 3])
    aa=0
    for SNR in [15, 20, 25, 50, 100]:
        IVIM_signal_noisy, D, f, Dp = sim.sim_signal(SNR, arg.sim.bvalues, sims=30000, Dmin=arg.sim.range[0][0],
                                                 Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                                 fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                                 Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)
        paramsNN = deep.predict_IVIM(IVIM_signal_noisy, arg.sim.bvalues, net,
                                                 arg)
        matNN[aa] = sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsNN)
        aa=aa+1

    ref = 1.1 * np.array([[[0.21870159],[0.27667806],[0.46173239]],[[0.18687773],[0.2458038],[0.40430529]],[[0.16957438],[0.22985733],[0.37065742]],[[0.14261743],[0.20649752],[0.31245713]],[[0.13493688],[0.20026352],[0.29356715]]])

    np.testing.assert_array_less(matNN[:,:,1:2], ref)
    assert elapsed_time < 600