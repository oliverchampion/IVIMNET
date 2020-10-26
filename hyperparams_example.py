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

import torch
import numpy as np
import warnings


class train_pars:
    def __init__(self, nets):
        self.optim='adam' #these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
        self.lr = 0.00001 # this is the learning rate. adam needs order of 0.001; others order of 0.05? sgdr can do 0.5
        self.patience= 10 # this is the number of epochs without improvement that the network waits untill determining it found its optimum
        self.batch_size= 128 # number of datasets taken along per iteration
        self.maxit = 500 # max iterations per epoch
        self.split = 0.9 # split of test and validation data
        self.load_nn= False # load the neural network instead of retraining
        self.loss_fun = 'rms' # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
        self.skip_net = False # skip the network training and evaluation
        self.scheduler = False # as discussed in the article, LR is important. This approach allows to reduce the LR itteratively when there is no improvement throughout an 5 consecutive epochs
        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = False


class net_pars:
    def __init__(self,nets):
        # select a network setting
        if (nets == 'optim') or (nets == 'optim_adsig') :
            # the optimized network settings
            self.dropout = 0.1 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
            self.batch_norm = True # False/True turns on batch normalistion
            self.parallel = True # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'sigmoid' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
            #### only if sigmoid constraint is used!
            self.cons_min = [-0.0001, -0.05, -0.05, 0.7, -0.05, 0.06]  # Dt, Fp, Ds, S0 F2p, D2*
            self.cons_max = [0.005, 0.7, 0.3, 1.3, 0.3, 0.3]  # Dt, Fp, Ds, S0
            ####
            self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
            self.depth = 4 # number of layers
        elif nets == 'orig':
            # as summarized in Table 1 from the main article for the original network
            self.dropout = 0.0 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
            self.batch_norm = False # False/True turns on batch normalistion
            self.parallel = False  # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'abs' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
            #### only if sigmoid constraint is used!
            self.cons_min = [-0.0001, -0.05, -0.05, 0.7, -0.05, 0.06]  # Dt, Fp, Ds, S0 F2p, D2*
            self.cons_max = [0.005, 0.7, 0.3, 1.3, 0.3, 0.3]  # Dt, Fp, Ds, S0
            ####
            self.fitS0 = False # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
            self.depth = 3 # number of layers
        elif nets == 'custom':
            # chose wisely :)
            self.dropout = 0.3 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
            self.batch_norm = True # False/True turns on batch normalistion
            self.parallel = True # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'sigmoid' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
            #### only if sigmoid constraint is used!
            self.cons_min = [-0.0001, -0.15, -0.05, 0.7, -0.15, 0.06]  # Dt, Fp, Ds, S0 F2p, D2*
            self.cons_max = [0.005, 1.15, 0.12, 1.3, 1.15, 0.5]  # Dt, Fp, Ds, S0
            ####
            self.fitS0 = False # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
            self.depth = 4 # number of layers
            self.width = 500 # new option that determines network width. Putting to 0 makes it as wide as the number of b-values
        else:
            raise Exception('Choose the correct network. either ''optim'', ''orig'' or ''custom''')


class lsqfit:
    def __init__(self):
        self.method = 'lsq' #seg, bayes or lsq
        self.do_fit = False # skip lsq fitting
        self.load_lsq = False # load the last results for lsq fit
        self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 in the least squares fit.
        self.jobs = 2 # number of parallel jobs. If set to 1, no parallel computing is used
        self.bounds = ([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3]) #Dt, Fp, Ds, S0


class sim:
    def __init__(self):
        self.bvalues = np.array([0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700])  # array of b-values
        self.SNR = [10, 20, 30, 40, 50, 100]  # the SNRs to simulate at
        self.sims = 100000  # number of simulations to run
        self.num_samples_eval = 10000  # number of simualtiosn te evaluate. This can be lower than the number run. Particularly to save time when fitting. More simulations help with generating sufficient data for the neural network
        self.repeats = 1  # this is the number of repeats for simulations
        self.rician = False  # add rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0005, 0.05, 0.01],
                  [0.003, 0.55, 0.1])

class hyperparams:
    def __init__(self, net_example):
        self.fig = True # plot results and intermediate steps
        self.save_name = 'optim' # orig or optim (or optim_adsig for in vivo)
        self.net_pars = net_pars(self.save_name)
        self.train_pars = train_pars(self.save_name)
        self.fit = lsqfit()
        self.reps = 50
        self.sim = sim()


def checkarg_train_pars(arg):
    if not hasattr(arg,'optim'):
        warnings.warn('arg.train.optim not defined. Using default ''adam''')
        arg.optim = 'adam'  # these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
    if not hasattr(arg,'lr'):
        warnings.warn('arg.train.lr not defined. Using default value 0.0001')
        arg.lr = 0.0001  # this is the learning rate. adam needs order of 0.001; others order of 0.05? sgdr can do 0.5
    if not hasattr(arg, 'patience'):
        warnings.warn('arg.train.patience not defined. Using default value 10')
        arg.patience = 10  # this is the number of epochs without improvement that the network waits untill determining it found its optimum
    if not hasattr(arg,'batch_size'):
        warnings.warn('arg.train.batch_size not defined. Using default value 128')
        arg.batch_size = 128  # number of datasets taken along per iteration
    if not hasattr(arg,'maxit'):
        warnings.warn('arg.train.maxit not defined. Using default value 500')
        arg.maxit = 500  # max iterations per epoch
    if not hasattr(arg,'split'):
        warnings.warn('arg.train.split not defined. Using default value 0.9')
        arg.split = 0.9  # split of test and validation data
    if not hasattr(arg,'load_nn'):
        warnings.warn('arg.train.load_nn not defined. Using default of False')
        arg.load_nn = False
    if not hasattr(arg,'loss_fun'):
        warnings.warn('arg.train.loss_fun not defined. Using default of ''rms''')
        arg.loss_fun = 'rms'  # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
    if not hasattr(arg,'skip_net'):
        warnings.warn('arg.train.skip_net not defined. Using default of False')
        arg.skip_net = False
    if not hasattr(arg,'use_cuda'):
        arg.use_cuda = torch.cuda.is_available()
    if not hasattr(arg, 'device'):
        arg.device = torch.device("cuda:0" if arg.use_cuda else "cpu")
    return arg

def checkarg_net_pars(arg):
    if not hasattr(arg,'dropout'):
        warnings.warn('arg.net_pars.dropout not defined. Using default value of 0.1')
        arg.dropout = 0.1  # 0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
    if not hasattr(arg,'batch_norm'):
        warnings.warn('arg.net_pars.batch_norm not defined. Using default of True')
        arg.batch_norm = True  # False/True turns on batch normalistion
    if not hasattr(arg,'parallel'):
        warnings.warn('arg.net_pars.parallel not defined. Using default of True')
        arg.parallel = True  # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
    if not hasattr(arg,'con'):
        warnings.warn('arg.net_pars.con not defined. Using default of ''sigmoid''')
        arg.con = 'sigmoid'  # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
    if not hasattr(arg,'cons_min'):
        warnings.warn('arg.net_pars.cons_min not defined. Using default values of  [-0.0001, -0.05, -0.05, 0.7]')
        arg.cons_min = [-0.0001, -0.05, -0.05, 0.7, -0.05, 0.06]  # Dt, Fp, Ds, S0 F2p, D2*
    if not hasattr(arg,'cons_max'):
        warnings.warn('arg.net_pars.cons_max not defined. Using default values of  [-0.0001, -0.05, -0.05, 0.7]')
        arg.cons_max = [0.005, 0.7, 0.3, 1.3, 0.3, 0.3]  # Dt, Fp, Ds, S0
    if not hasattr(arg,'fitS0'):
        warnings.warn('arg.net_pars.parallel not defined. Using default of False')
        arg.fitS0 = False  # indicates whether to fix S0 to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
    if not hasattr(arg,'depth'):
        warnings.warn('arg.net_pars.depth not defined. Using default value of 4')
        arg.depth = 4  # number of layers
    if not hasattr(arg, 'width'):
        warnings.warn('arg.net_pars.width not defined. Using default of number of v-balues')
        arg.width = 0
    return arg

def checkarg_sim(arg):
    if not hasattr(arg, 'bvalues'):
        warnings.warn('arg.sim.bvalues not defined. Using default value of [0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]')
        arg.bvalues = [0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]
    if not hasattr(arg, 'repeats'):
        warnings.warn('arg.sim.repeats not defined. Using default value of 1')
        arg.repeats = 1  # this is the number of repeats for simulations
    if not hasattr(arg, 'rician'):
        warnings.warn('arg.sim.rician not defined. Using default of False')
        arg.rician = False
    if not hasattr(arg, 'SNR'):
        warnings.warn('arg.sim.SNR not defined. Using default of [20]')
        arg.SNR = [20]
    if not hasattr(arg, 'sims'):
        warnings.warn('arg.sim.sims not defined. Using default of 100000')
        arg.sims = 100000
    if not hasattr(arg, 'num_samples_eval'):
        warnings.warn('arg.sim.num_samples_eval not defined. Using default of 100000')
        arg.num_samples_eval = 100000
    if not hasattr(arg, 'range'):
        warnings.warn('arg.sim.range not defined. Using default of ([0.0005, 0.05, 0.01],[0.003, 0.4, 0.1])')
        arg.range = ([0.0005, 0.05, 0.01],
                  [0.003, 0.4, 0.1])
    return arg

def checkarg_lsq(arg):
    if not hasattr(arg, 'method'):
        warnings.warn('arg.fit.method not defined. Using default of ''lsq''')
        arg.method='lsq'
    if not hasattr(arg, 'do_fit'):
        warnings.warn('arg.fit.do_fit not defined. Using default of True')
        arg.do_fit=True
    if not hasattr(arg, 'load_lsq'):
        warnings.warn('arg.fit.load_lsq not defined. Using default of False')
        arg.load_lsq=False
    if not hasattr(arg, 'fitS0'):
        warnings.warn('arg.fit.fitS0 not defined. Using default of False')
        arg.fitS0=False
    if not hasattr(arg, 'jobs'):
        warnings.warn('arg.fit.jobs not defined. Using default of 4')
        arg.jobs = 4
    if not hasattr(arg, 'bounds'):
        warnings.warn('arg.fit.bounds not defined. Using default of ([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3])')
        arg.bounds = ([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3]) #Dt, Fp, Ds, S0
    return arg

def checkarg(arg):
    if not hasattr(arg, 'fig'):
        arg.fig = False
        warnings.warn('arg.fig not defined. Using default of False')
    if not hasattr(arg, 'save_name'):
        warnings.warn('arg.save_name not defined. Using default of ''default''')
        arg.save_name = 'default'
    if not hasattr(arg,'net_pars'):
        warnings.warn('arg no net_pars. Using default initialisation')
        arg.net_pars=net_pars()
    if not hasattr(arg, 'train_pars'):
        warnings.warn('arg no train_pars. Using default initialisation')
        arg.train_pars = train_pars()
    if not hasattr(arg, 'sim'):
        warnings.warn('arg no sim. Using default initialisation')
        arg.sim = sim()
    if not hasattr(arg, 'fit'):
        warnings.warn('arg no lsq. Using default initialisation')
        arg.fit = lsqfit()
    arg.net_pars=checkarg_net_pars(arg.net_pars)
    arg.train_pars = checkarg_train_pars(arg.train_pars)
    arg.sim = checkarg_sim(arg.sim)
    arg.fit = checkarg_lsq(arg.fit)
    return arg