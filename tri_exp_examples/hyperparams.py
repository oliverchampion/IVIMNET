"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients. MRM 2021)

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


#most of these are options from the article and explained in the M&M.
class train_pars(object):
    def __init__(self,nets):
        self.optim='adam' #these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
        if nets == 'orig':
            self.lr = 0.0001
        elif nets == 'optim':
            self.lr = 0.00003
        else:
            self.lr = 0.00003
        self.patience= 20 # this is the number of epochs without improvement that the network waits untill determining it found its optimum
        self.plateau_size = 10 # this is the number of epochs without improvement that the network waits untill determining it found its optimum
        self.batch_size= 128 # number of datasets taken along per iteration
        self.maxit = 500 # max iterations per epoch
        self.split = 0.9 # split of test and validation data
        self.load_nn= False # load the neural network instead of retraining
        self.loss_fun = 'rms' # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
        self.skip_net = False # skip the network training and evaluation
        self.scheduler = True # was false # as discussed in the article, LR is important. This approach allows to reduce the LR itteratively when there is no improvement throughout an 5 consecutive epochs
        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.select_best = True
        self.regul = 0
        self.forcepos = False
        if self.scheduler:
            self.disturb = [1000000] #100, 250, 500
        else:
            self.disturb = [1000000]

class net_pars:
    def __init__(self,nets):
        # chose wisely :)
        self.dropout = 0.1 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
        self.batch_norm = True # False/True turns on batch normalistion
        self.parallel = 'semi_parallel' # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
        self.con = 'sigmoid' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
        #### only if sigmoid constraint is used!
        #self.cons_min = [-0.0001, 0, 0.003, 0.3, 0, 0.06]  # Dt, Fp, Ds, S0 F2p, D2*
        #self.cons_max = [0.003, 0.5, 0.08, 2.5, 1, 0.5]  # Dt, Fp, Ds, S0
        #self.cons_min = [-0.02, -1, -0.2, -1, -1, -1]  # Dt, Fp, Ds, S0 F2p, D2*
        #self.cons_max = [0.02, 2, 0.3, 4, 2, 2]  # Dt, Fp, Ds, S0
        ####
        self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
        self.depth = 4 # number of layers #was 3
        self.neighbours = False # new option that takes all neighbouring voxels into account too
        self.tri_exp = True
        self.train_seperately = 'no' # 'no', 'full' or 'initialize'
        self.width = 0 # new option that determines network width. Putting to 0 makes it as wide as the number of b-values
        self.repeat = 10
        if self.tri_exp:
            self.cons_min = [0., 0.000, 0.0, 0.008, 0.0, 0.06]  # Dt, F1, D1, F0 F2, D2
            self.cons_max = [2.5, 0.008, 1, 0.08, 1, 5]  # Dt, F1, D1, F0 F2, D2
        else:
            self.cons_min = [0.0003, 0.0, 0.003, 0.0, 0.0, 0.08]  # Dt, F1, D1, F0 F2, D2
            self.cons_max = [0.003, 0.5, 0.08, 2.5, 1, 0.4]  # Dt, F1, D1, F0 F2, D2
        boundsrange = 0.15 * (np.array(self.cons_max) - np.array(
            self.cons_min))  # ensure that we are on the most lineair bit of the sigmoid function #stond fout haakje
        self.cons_min = np.array(self.cons_min) - boundsrange
        self.cons_max = np.array(self.cons_max) + boundsrange


class fit(object):
    def __init__(self):
        self.method = 'lsq'  # seg, bayes or lsq, tri-exp, seg_tri-exp
        self.model = 'tri-exp' #seg, bayes or lsq, tri-exp, seg_tri-exp
        self.do_fit = True # skip lsq fitting
        self.load_lsq = False # load the last results for lsq fit
        self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 in the least squares fit.
        self.jobs = 8 # False of parallel jobs. If set to 1, no parallel computing is used
        if self.model == 'tri-exp':
            self.bounds = ([0, 0, 0, 0.008, 0, 0.06], [2.5, 0.008, 1, 0.08, 1, 5])
        else:
            self.bounds = ([0.0003, 0, 0.005, 0.5], [0.005, 0.7, 0.3, 2.5]) #Dt, Fp, Ds, S0

class sim:
    def __init__(self):
        #self.bvalues = np.sort(np.array([0, 0, 0, 0, 700, 700, 700, 700, 1, 1, 1, 1, 5,  5,  5,  5,  100, 100, 100, 100, 300, 300, 300, 300, 10, 10, 10, 10, 0, 0, 0, 0, 20, 20, 20, 20, 500, 500, 500, 500, 50, 50, 50, 50, 40, 40, 40, 40, 30, 30, 30, 30, 150, 150, 150, 150, 75,  75,  75,  75,  0, 0, 0, 0, 600, 600, 600, 600, 200, 200, 200, 200, 400, 400, 400, 400, 2, 2, 2, 2]))
        self.bvalues = np.sort(np.unique(np.array([0, 0, 0, 0, 700, 700, 700, 700, 1, 1, 1, 1, 5,  5,  5,  5,  100, 100, 100, 100, 300, 300, 300, 300, 10, 10, 10, 10, 0, 0, 0, 0, 20, 20, 20, 20, 500, 500, 500, 500, 50, 50, 50, 50, 40, 40, 40, 40, 30, 30, 30, 30, 150, 150, 150, 150, 75,  75,  75,  75,  0, 0, 0, 0, 600, 600, 600, 600, 200, 200, 200, 200, 400, 400, 400, 400, 2, 2, 2, 2])))
        self.SNR = (10, 100) # the SNRs to simulate at
        self.SNR_eval = [15, 20, 30, 50] # the SNRs to simulate at
        self.sims = 5000000 # number of simulations to run
        self.num_samples_eval = 300000 # number of simualtiosn te evaluate. This can be lower than the number run. Particularly to save time when fitting. More simulations help with generating sufficient data for the neural network
        self.repeats = 10 # this is the number of repeats for simulations
        self.rician = False # add rician noise to simulations; if false, gaussian noise is added instead
        self.method = 'tri-exp' #bi-exp or tri-exp
        if self.method == 'bi-exp':
            self.range = ([0.0005, 0.05, 0.01], [0.003, 0.55, 0.1])
        else:
            self.range = ([0.0005, 0.05, 0.01, 0.05, 0.2], [0.003, 0.3, 0.05, 0.3, 4]) # D0, F1', D1, F2', D2

class hyperparams(object):
    def __init__(self):
        self.fig = False # plot results and intermediate steps
        self.save_name = 'bi_Exp_build_lib' # orig or optim (or optim_adsig for in vivo)  torch18_015bounds_force_pos_false
        self.net_pars = net_pars(self.save_name)
        self.train_pars = train_pars(self.save_name)
        self.fit = fit()
        self.sharp = True # options for in vivo. Not relevant to upload
        self.sim = sim()
        if self.sharp:
            self.dats='_not_blurred'
        else:
            self.dats=''
        self.reps = 10