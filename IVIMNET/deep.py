"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Built on code by Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved physics-informed deep learning of the intravoxel-incoherent motion model: accurate, unique and consistent. MRM 2021)
If this code was useful, please cite:
https://doi.org/10.1002/mrm.27910

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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
import IVIMNET.fitting_algorithms as fit
from joblib import Parallel, delayed
import copy
import warnings

# Define the neural network.

class Net(nn.Module):
    def __init__(self, bvalues, net_pars):
        """
        this defines the Net class which is the network we want to train.
        :param bvalues: a 1D array with the b-values
        :param net_pars: an object with network design options, as explained in the publication, with attributes:
        fitS0 --> Boolean determining whether S0 is fixed to 1 (False) or fitted (True)
        times len(bvalues), with data sorted per voxel. This option was not explored in the publication
        dropout --> Number between 0 and 1 indicating the amount of dropout regularisation
        batch_norm --> Boolean determining whether to use batch normalisation
        parallel --> Boolean determining whether to use separate networks for estimating the different IVIM parameters
        (True), or have them all estimated by a single network (False)
        con --> string which determines what type of constraint is used for the parameters. Options are:
        'sigmoid' allowing a sigmoid constraint
        'abs' having the absolute of the estimated values to constrain parameters to be positive
        'none' giving no constraints
        cons_min --> 1D array, if sigmoid is the constraint, these values give [Dmin, fmin, D*min, S0min]
        cons_max --> 1D array, if sigmoid is the constraint, these values give [Dmax, fmax, D*max, S0max]
        depth --> integer giving the network depth (number of layers)
        """
        super(Net, self).__init__()
        self.bvalues = bvalues
        self.net_pars = net_pars
        if self.net_pars.width is 0:
            self.net_pars.width = len(bvalues)
        # define number of parameters being estimated
        if self.net_pars.tri_exp:
            self.est_pars = 5
        else:
            self.est_pars = 3
        if self.net_pars.fitS0:
            self.est_pars += 1
        # define module lists. If network is not parallel, we can do with 1 list, otherwise we need a list per parameter
        self.fc_layers0 = nn.ModuleList()
        if self.net_pars.parallel != 'single':
            self.fc_layers1 = nn.ModuleList()
            self.fc_layers2 = nn.ModuleList()
            self.fc_layers3 = nn.ModuleList()
            if self.net_pars.tri_exp:
                self.fc_layers4 = nn.ModuleList()
                self.fc_layers5 = nn.ModuleList()

        # loop over the layers
        width = len(bvalues)
        for i in range(self.net_pars.depth):
            # extend with a fully-connected linear layer
            self.fc_layers0.extend([nn.Linear(width, self.net_pars.width)])
            if self.net_pars.parallel != 'single':
                self.fc_layers1.extend([nn.Linear(width, self.net_pars.width)])
                self.fc_layers2.extend([nn.Linear(width, self.net_pars.width)])
                self.fc_layers3.extend([nn.Linear(width, self.net_pars.width)])
                if self.net_pars.tri_exp:
                    self.fc_layers4.extend([nn.Linear(width, self.net_pars.width)])
                    self.fc_layers5.extend([nn.Linear(width, self.net_pars.width)])
            width = self.net_pars.width
            # if desired, add batch normalisation
            if self.net_pars.batch_norm:
                self.fc_layers0.extend([nn.BatchNorm1d(self.net_pars.width)])
                if self.net_pars.parallel != 'single':
                    self.fc_layers1.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers2.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers3.extend([nn.BatchNorm1d(self.net_pars.width)])
                    if self.net_pars.tri_exp:
                        self.fc_layers4.extend([nn.BatchNorm1d(self.net_pars.width)])
                        self.fc_layers5.extend([nn.BatchNorm1d(self.net_pars.width)])
            # add ELU units for non-linearity
            self.fc_layers0.extend([nn.ELU()])
            if self.net_pars.parallel != 'single':
                self.fc_layers1.extend([nn.ELU()])
                self.fc_layers2.extend([nn.ELU()])
                self.fc_layers3.extend([nn.ELU()])
                if self.net_pars.tri_exp:
                    self.fc_layers4.extend([nn.ELU()])
                    self.fc_layers5.extend([nn.ELU()])
            # if dropout is desired, add dropout regularisation
            if self.net_pars.dropout is not 0 and i is not (self.net_pars.depth - 1):
                self.fc_layers0.extend([nn.Dropout(self.net_pars.dropout)])
                if self.net_pars.parallel != 'single':
                    self.fc_layers1.extend([nn.Dropout(self.net_pars.dropout)])
                    self.fc_layers2.extend([nn.Dropout(self.net_pars.dropout)])
                    self.fc_layers3.extend([nn.Dropout(self.net_pars.dropout)])
                    if self.net_pars.tri_exp:
                        self.fc_layers4.extend([nn.Dropout(self.net_pars.dropout)])
                        self.fc_layers5.extend([nn.Dropout(self.net_pars.dropout)])
        # Final layer yielding output, with either 3 (fix S0) or 4 outputs of a single network, or 1 output
        # per network in case of parallel networks.
        if self.net_pars.parallel == 'parallel':
            self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, 1))
            self.encoder1 = nn.Sequential(*self.fc_layers1, nn.Linear(self.net_pars.width, 1))
            self.encoder2 = nn.Sequential(*self.fc_layers2, nn.Linear(self.net_pars.width, 1))
            if self.net_pars.tri_exp:
                self.encoder3 = nn.Sequential(*self.fc_layers3, nn.Linear(self.net_pars.width, 1))
                self.encoder4 = nn.Sequential(*self.fc_layers4, nn.Linear(self.net_pars.width, 1))
                if self.net_pars.fitS0:
                    self.encoder5 = nn.Sequential(*self.fc_layers5, nn.Linear(self.net_pars.width, 1))
            if self.net_pars.fitS0:
                self.encoder3 = nn.Sequential(*self.fc_layers3, nn.Linear(self.net_pars.width, 1))
        elif self.net_pars.parallel == 'semi_parallel':
            self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, 2 * 1))
            self.encoder1 = nn.Sequential(*self.fc_layers1, nn.Linear(self.net_pars.width, 2 * 1))
            self.encoder2 = nn.Sequential(*self.fc_layers2, nn.Linear(self.net_pars.width, 2 * 1))
            if self.net_pars.tri_exp:
                del self.fc_layers3
                del self.fc_layers4
                if self.net_pars.fitS0:
                    del self.fc_layers5
            else:
                del self.encoder2
                if self.net_pars.fitS0:
                    self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, 2 * 1))
                    self.encoder1 = nn.Sequential(*self.fc_layers1, nn.Linear(self.net_pars.width, 2 * 1))
                else:
                    self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, 2 * 1))
                    self.encoder1 = nn.Sequential(*self.fc_layers1, nn.Linear(self.net_pars.width, 1))
        else:
            self.encoder = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, self.est_pars * 1))

    def forward(self, X):
        # select constraint method
        if self.net_pars.con == 'sigmoid':
            # define constraints
            if self.net_pars.tri_exp:
                f0min = self.net_pars.cons_min[0]
                f0max = self.net_pars.cons_max[0]
                Dmin = self.net_pars.cons_min[1]
                Dmax = self.net_pars.cons_max[1]
                fmin = self.net_pars.cons_min[2]
                fmax = self.net_pars.cons_max[2]
                Dpmin = self.net_pars.cons_min[3]
                Dpmax = self.net_pars.cons_max[3]
                f2min = self.net_pars.cons_min[4]
                f2max = self.net_pars.cons_max[4]
                Dp2min = self.net_pars.cons_min[5]
                Dp2max = self.net_pars.cons_max[5]
            else:
                Dmin = self.net_pars.cons_min[0]
                Dmax = self.net_pars.cons_max[0]
                fmin = self.net_pars.cons_min[1]
                fmax = self.net_pars.cons_max[1]
                Dpmin = self.net_pars.cons_min[2]
                Dpmax = self.net_pars.cons_max[2]
                f0min = self.net_pars.cons_min[3]
                f0max = self.net_pars.cons_max[3]
            # this network constrains the estimated parameters between two values by taking the sigmoid.
            # Advantage is that the parameters are constrained and that the mapping is unique.
            # Disadvantage is that the gradients go to zero close to the prameter bounds.
            params0 = self.encoder0(X)
            # if parallel again use each param comes from a different output
            if self.net_pars.parallel == 'parallel':
                params1 = self.encoder1(X)
                params2 = self.encoder2(X)
                if self.net_pars.tri_exp:
                    params3 = self.encoder3(X)
                    params4 = self.encoder4(X)
                    if self.net_pars.fitS0:
                        params5 = self.encoder5(X)
                elif self.net_pars.fitS0:
                    params3 = self.encoder3(X)
            elif self.net_pars.parallel == 'semi_parallel':
                params2 = self.encoder1(X)[:, 0].unsqueeze(1)
                params1 = params0[:, 1].unsqueeze(1)
                params0 = params0[:, 0].unsqueeze(1)
                if self.net_pars.tri_exp:
                    params3 = self.encoder2(X)[:, 0].unsqueeze(1)
                    params4 = self.encoder2(X)[:, 1].unsqueeze(1)
                    if self.net_pars.fitS0:
                        params5 = self.encoder1(X)[:, 1].unsqueeze(1)
                elif self.net_pars.fitS0:
                    params3 = self.encoder1(X)[:, 1].unsqueeze(1)

        elif self.net_pars.con == 'none' or self.net_pars.con == 'abs':
            if self.net_pars.con == 'abs':
                # this network constrains the estimated parameters to be positive by taking the absolute.
                # Advantage is that the parameters are constrained and that the derrivative of the function remains
                # constant. Disadvantage is that -x=x, so could become unstable.
                params0 = torch.abs(self.encoder0(X))
                if self.net_pars.parallel == 'parallel':
                    params1 = torch.abs(self.encoder1(X))
                    params2 = torch.abs(self.encoder2(X))
                    if self.net_pars.tri_exp:
                        params3 = torch.abs(self.encoder3(X))
                        params4 = torch.abs(self.encoder4(X))
                        if self.net_pars.fitS0:
                            params5 = torch.abs(self.encoder5(X))
                    elif self.net_pars.fitS0:
                        params3 = torch.abs(self.encoder3(X))
                elif self.net_pars.parallel == 'semi_parallel':
                    params2 = torch.abs(self.encoder1(X)[:, 0].unsqueeze(1))
                    params1 = torch.abs(params0[:, 1].unsqueeze(1))
                    params0 = torch.abs(params0[:, 0].unsqueeze(1))
                    if self.net_pars.tri_exp:
                        params3 = torch.abs(self.encoder2(X)[:,  0].unsqueeze(1))
                        params4 = torch.abs(self.encoder2(X)[:,  1].unsqueeze(1))
                        if self.net_pars.fitS0:
                            params5 = torch.abs(self.encoder1(X)[:,  1].unsqueeze(1))
                    elif self.net_pars.fitS0:
                        params3 = torch.abs(self.encoder1(X)[:, 1].unsqueeze(1))
            else:
                # this network is not constraint
                params0 = self.encoder0(X)
                if self.net_pars.parallel == 'parallel':
                    params1 = self.encoder1(X)
                    params2 = self.encoder2(X)
                    if self.net_pars.tri_exp:
                        params3 = self.encoder3(X)
                        params4 = self.encoder4(X)
                        if self.net_pars.fitS0:
                            params5 = self.encoder5(X)
                    elif self.net_pars.fitS0:
                        params3 = self.encoder3(X)
                elif self.net_pars.parallel == 'semi_parallel':
                    params2 = self.encoder1(X)[:,  0].unsqueeze(1)
                    params1 = params0[:,  1].unsqueeze(1)
                    params0 = params0[:,  0].unsqueeze(1)
                    if self.net_pars.tri_exp:
                        params3 = self.encoder2(X)[:,  0].unsqueeze(1)
                        params4 = self.encoder2(X)[:,  1].unsqueeze(1)
                        if self.net_pars.fitS0:
                            params5 = self.encoder1(X)[:,  1].unsqueeze(1)
                    elif self.net_pars.fitS0:
                        params3 = self.encoder1(X)[:,  1].unsqueeze(1)
        else:
            raise Exception('the chose parameter constraint is not implemented. Try ''sigmoid'', ''none'' or ''abs''')
        X_temp=[]
        if self.net_pars.con == 'sigmoid':
            # applying constraints
            if self.net_pars.parallel == 'parallel' or self.net_pars.parallel == 'semi_parallel':
                Dt = Dmin + torch.sigmoid(params2[:, 0].unsqueeze(1)) * (Dmax - Dmin)
                Fp = fmin + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (fmax - fmin)
                Dp = Dpmin + torch.sigmoid(params1[:, 0].unsqueeze(1)) * (Dpmax - Dpmin)
                if self.net_pars.tri_exp:
                    Dp2 = Dp2min + torch.sigmoid(params3[:, 0].unsqueeze(1)) * (Dp2max - Dp2min)
                    Fp2 = f2min + torch.sigmoid(params4[:, 0].unsqueeze(1)) * (f2max - f2min)
                    if self.net_pars.fitS0:
                        f0 = f0min + torch.sigmoid(params5[:, 0].unsqueeze(1)) * (f0max - f0min)
                elif self.net_pars.fitS0:
                    f0 = f0min + torch.sigmoid(params3[:, 0].unsqueeze(1)) * (f0max - f0min)
            else:
                Dp = Dpmin + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dpmax - Dpmin)
                Dt = Dmin + torch.sigmoid(params0[:, 1].unsqueeze(1)) * (Dmax - Dmin)
                Fp = fmin + torch.sigmoid(params0[:, 2].unsqueeze(1)) * (fmax - fmin)
                if self.net_pars.tri_exp:
                    Fp2 = fmin + torch.sigmoid(params0[:, 3].unsqueeze(1)) * (f2max - f2min)
                    Dp2 = Dpmin + torch.sigmoid(params0[:, 4].unsqueeze(1)) * (Dp2max - Dp2min)
                    if self.net_pars.fitS0:
                        f0 = f0min + torch.sigmoid(params0[:, 5].unsqueeze(1)) * (
                                f0max - f0min)
                elif self.net_pars.fitS0:
                    f0 = f0min + torch.sigmoid(params0[:, 3].unsqueeze(1)) * (f0max - f0min)
        elif self.net_pars.con == 'none' or self.net_pars.con == 'abs':
            if self.net_pars.parallel == 'parallel' or self.net_pars.parallel == 'semi_parallel':
                Dt = params2[:, 0].unsqueeze(1)
                Fp = params0[:, 0].unsqueeze(1)
                Dp = params1[:, 0].unsqueeze(1)
                if self.net_pars.tri_exp:
                    Dp2 = params3[:, 0].unsqueeze(1)
                    Fp2 = params4[:, 0].unsqueeze(1)
                    if self.net_pars.fitS0:
                        f0 = params5[:, 0].unsqueeze(1)
                elif self.net_pars.fitS0:
                    f0 = params3[:, 0].unsqueeze(1)
            else:
                Dp = params0[:, 0].unsqueeze(1)
                Dt = params0[:, 1].unsqueeze(1)
                Fp = params0[:, 2].unsqueeze(1)
                if self.net_pars.tri_exp:
                    Dp2 = params0[:, 3].unsqueeze(1)
                    Fp2 = params0[:, 4].unsqueeze(1)
                    if self.net_pars.fitS0:
                        f0 = params0[:, 5].unsqueeze(1)
                elif self.net_pars.fitS0:
                    f0 = params0[:, 3].unsqueeze(1)
        # here we estimate X, the signal as function of b-values given the predicted IVIM parameters. Although
        # this parameter is not interesting for prediction, it is used in the loss function
        # in this a>0 case, we fill up the predicted signal of the neighbouring voxels too, as these are used in
        # the loss function.
        if self.net_pars.tri_exp:
            if self.net_pars.fitS0:
                X_temp.append((Fp * torch.exp(-self.bvalues * Dp) + Fp2 * torch.exp(-self.bvalues * Dp2) +
                                f0 * torch.exp(-self.bvalues * Dt)))
            else:
                X_temp.append((Fp * torch.exp(-self.bvalues * Dp) + Fp2 * torch.exp(-self.bvalues * Dp2) +
                                    (1 - Fp - Fp2) * torch.exp(-self.bvalues * Dt)))
        else:
            if self.net_pars.fitS0:
                X_temp.append((Fp * torch.exp(-self.bvalues * Dp) + f0 * torch.exp(-self.bvalues * Dt)))
            else:
                X_temp.append((Fp * torch.exp(-self.bvalues * Dp) + (1 - Fp) * torch.exp(-self.bvalues * Dt)))
        X = torch.cat(X_temp,dim=1)
        if self.net_pars.tri_exp:
            if self.net_pars.fitS0:
                return X, f0+Fp2+Fp, Dt, Fp/(f0+Fp2+Fp), Dp, Fp2/(f0+Fp2+Fp), Dp2
            else:
                return X, torch.ones(len(Dt)), Dt, Fp, Dp, Fp2, Dp2
        else:
            if self.net_pars.fitS0:
                return X, Dt, Fp/(f0+Fp), Dp, f0+Fp
            else:
                return X, Dt, Fp, Dp, torch.ones(len(Dt))


def learn_IVIM(X_train, bvalues, arg, net=None):
    """
    This program builds a IVIM-NET network and trains it.
    :param X_train: 2D array of IVIM data we use for training. First axis are the voxels and second axis are the b-values
    :param bvalues: a 1D array with the b-values
    :param arg: an object with network design options, as explained in the publication check hyperparameters.py for
    options
    :param net: an optional input pre-trained network with initialized weights for e.g. transfer learning or warm start
    :return net: returns a trained network
    """

    torch.backends.cudnn.benchmark = True
    arg = checkarg(arg)

    ## normalise the signal to b=0 and remove data with nans
    X_train = normalise(X_train, bvalues, arg)
    # removing non-IVIM-like data; this often gets through when background data is not correctly masked
    # Estimating IVIM parameters in these data is meaningless anyways.
    if not arg.norm_data_full:
        X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
        X_train[X_train > 1.5] = 1.5

    # initialising the network of choice using the input argument arg
    if net is None:
        bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
        net = Net(bvalues, arg.net_pars).to(arg.train_pars.device)
    else:
        # if a network was used as input parameter, work with that network instead (transfer learning/warm start).
        net.to(arg.train_pars.device)

    # defining the loss function; not explored in the publication
    if arg.train_pars.loss_fun == 'rms':
        criterion = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    elif arg.train_pars.loss_fun == 'L1':
        criterion = nn.L1Loss(reduction='mean').to(arg.train_pars.device)
    elif arg.train_pars.loss_fun == 'mixed':
        criterion = nn.L1Loss(reduction='mean').to(arg.train_pars.device) + 0.5 * nn.MSELoss(reduction='mean').to(arg.train_pars.device)


    # splitting data into learning and validation set; subsequently initialising the Dataloaders
    split = int(np.floor(len(X_train) * arg.train_pars.split))
    train_set, val_set = torch.utils.data.random_split(torch.from_numpy(X_train.astype(np.float32)),
                                                       [split, len(X_train) - split])
    # train loader loads the trianing data. We want to shuffle to make sure data order is modified each epoch and different data is selected each epoch.
    trainloader = utils.DataLoader(train_set,
                                   batch_size=arg.train_pars.batch_size,
                                   shuffle=True,
                                   drop_last=True)
    # validation data is loaded here. By not shuffling, we make sure the same data is loaded for validation every time. We can use substantially more data per batch as we are not training.
    inferloader = utils.DataLoader(val_set,
                                   batch_size=32 * arg.train_pars.batch_size,
                                   shuffle=False,
                                   drop_last=True)

    # defining the number of training and validation batches for normalisation later
    totalit = np.min([arg.train_pars.maxit, np.floor(split // arg.train_pars.batch_size)])
    batch_norm2 = np.floor(len(val_set) // (32 * arg.train_pars.batch_size))

    # defining optimiser
    if arg.train_pars.scheduler:
        optimizer, scheduler = load_optimizer(net, arg)
    else:
        optimizer = load_optimizer(net, arg)

    # Initialising parameters
    best = 1e16
    num_bad_epochs = 0
    loss_train = []
    loss_val = []
    prev_lr = 0
    # get_ipython().run_line_magic('matplotlib', 'inline')
    final_model = copy.deepcopy(net.state_dict())

    ## Train
    for epoch in range(1000):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        # initialising and resetting parameters
        net.train()
        running_loss_train = 0.
        running_loss_val = 0.
        for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
            if i > totalit:
                # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
                break
            # zero the parameter gradients
            optimizer.zero_grad()
            # put batch on GPU if pressent
            X_batch = X_batch.to(arg.train_pars.device)
            ## forward + backward + optimize
            if arg.net_pars.tri_exp:
                X_pred, _, _, Fp_pred, Dp_pred, _, _ = net(X_batch)
            else:
                X_pred, _, Fp_pred, Dp_pred, _ = net(X_batch)            # removing nans and too high/low predictions to prevent overshooting
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # determine loss for batch; note that the loss is determined by the difference between the predicted signal and the actual signal. The loss does not look at Dt, Dp or Fp.
            loss = criterion(X_pred, X_batch)
            # updating network
            loss.backward()
            optimizer.step()
            # total loss and determine max loss over all batches
            running_loss_train += loss.item()
        # show some figures if desired, to show whether there is a correlation between Dp and f
        if arg.fig:
            plt.figure(3)
            plt.clf()
            plt.plot(Dp_pred.tolist(), Fp_pred.tolist(), 'rx', markersize=5)
            plt.ion()
            plt.show()
        # after training, do validation in unseen data without updating gradients
        print('\n validation \n')
        net.eval()
        # validation is always done over all validation data
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            optimizer.zero_grad()
            X_batch = X_batch.to(arg.train_pars.device)
            # do prediction, only look at predicted IVIM signal
            if arg.net_pars.tri_exp:
                X_pred, _, _, _, _, _, _ = net(X_batch)
            else:
                X_pred, _, _, _, _ = net(X_batch)            # removing nans and too high/low predictions to prevent overshooting
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # validation loss
            loss = criterion(X_pred, X_batch)
            running_loss_val += loss.item()
        # scale losses
        running_loss_train = running_loss_train / totalit
        running_loss_val = running_loss_val / batch_norm2
        # save loss history for plot
        loss_train.append(running_loss_train)
        loss_val.append(running_loss_val)
        # as discussed in the article, LR is important. This approach allows to reduce the LR if we think it is too
        # high, and return to the network state before it went poorly
        if arg.train_pars.scheduler:
            scheduler.step(running_loss_val)
            if optimizer.param_groups[0]['lr'] < prev_lr:
                net.load_state_dict(final_model)
            prev_lr = optimizer.param_groups[0]['lr']
        # print stuff
        print("\nLoss: {loss}, validation_loss: {val_loss}, lr: {lr}".format(loss=running_loss_train,
                                                                             val_loss=running_loss_val,
                                                                             lr=optimizer.param_groups[0]['lr']))
        # early stopping criteria
        if running_loss_val < best:
            print("\n############### Saving good model ###############################")
            final_model = copy.deepcopy(net.state_dict())
            best = running_loss_val
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == arg.train_pars.patience:
                print("\nDone, best val loss: {}".format(best))
                break
        # plot loss and plot 4 fitted curves
        if epoch > 0:
            # plot progress and intermediate results (if enabled)
            plot_progress(X_batch.cpu(), X_pred.cpu(), bvalues.cpu(), loss_train, loss_val, arg)
    print("Done")
    # save final fits
    if arg.fig:
        if not os.path.isdir('plots'):
            os.makedirs('plots')
        plt.figure(1)
        plt.gcf()
        plt.savefig('plots/fig_fit.png')
        plt.figure(2)
        plt.gcf()
        plt.savefig('plots/fig_train.png')
        plt.close('all')
    # Restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)
    del trainloader
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    return net


def load_optimizer(net, arg):
    if arg.train_pars.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=arg.train_pars.lr, weight_decay=1e-4, amsgrad=True)
    elif arg.train_pars.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=arg.train_pars.lr, momentum=0.9, weight_decay=1e-4)
    elif arg.train_pars.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=arg.train_pars.lr, weight_decay=1e-4)
    if arg.train_pars.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2,
                                                         patience=round(arg.train_pars.plateau_size))
        return optimizer, scheduler
    else:
        return optimizer


def predict_IVIM(data, bvalues, net, arg):
    """
    This program takes a trained network and predicts the IVIM parameters from it.
    :param data: 2D array of IVIM data we want to predict the IVIM parameters from. First axis are the voxels and second axis are the b-values
    :param bvalues: a 1D array with the b-values
    :param net: the trained IVIM-NET network
    :param arg: an object with network design options, as explained in the publication check hyperparameters.py for
    options
    :return param: returns the predicted parameters
    """
    arg = checkarg(arg)

    ## normalise the signal to b=0 and remove data with nans
    data = normalise(data, bvalues, arg)
    mylist = isnan(np.mean(data, axis=1))
    sels = [not i for i in mylist]
    # remove data with non-IVIM-like behaviour. Estimating IVIM parameters in these data is meaningless anyways.
    sels = sels & (np.percentile(data[:, bvalues < 50], 0.95, axis=1) < 1.3) & (
                np.percentile(data[:, bvalues > 50], 0.95, axis=1) < 1.2) & (
                       np.percentile(data[:, bvalues > 150], 0.95, axis=1) < 1.0)
    # we need this for later
    lend = len(data)
    data = data[sels]

    # tell net it is used for evaluation
    net.eval()
    # initialise parameters and data
    Dp = np.array([])
    Dt = np.array([])
    Fp = np.array([])
    S0 = np.array([])
    if arg.net_pars.tri_exp:
        Fp2 = np.array([])
        Dp2 = np.array([])

    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = utils.DataLoader(torch.from_numpy(data.astype(np.float32)),
                                   batch_size=2056,
                                   shuffle=False,
                                   drop_last=False)
    # start predicting
    with torch.no_grad():
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            X_batch = X_batch.to(arg.train_pars.device)
            # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
            if arg.net_pars.tri_exp:
                _,  S0t, Dtt, Fpt, Dpt, Fp2t, Dp2t = net(X_batch)
                Dp2 = np.append(Dp2, (Dp2t.cpu()).numpy())
                Fp2 = np.append(Fp2, (Fp2t.cpu()).numpy())
            else:
                _, Dtt, Fpt, Dpt, S0t = net(X_batch)
            # Quick and dirty solution to deal with networks not predicting S0
            try:
                S0 = np.append(S0, (S0t.cpu()).numpy())
            except:
                S0 = np.append(S0, S0t)
            Dp = np.append(Dp, (Dpt.cpu()).numpy())
            Dt = np.append(Dt, (Dtt.cpu()).numpy())
            Fp = np.append(Fp, (Fpt.cpu()).numpy())
    # The 'abs' and 'none' constraint networks have no way of figuring out what is D and D* a-priori. However, they do
    # tend to pick one output parameter for D or D* consistently within the network. If the network has swapped D and
    # D*, we swap them back here.
    if arg.net_pars.tri_exp:
        if np.mean(Dp2) < np.mean(Dp):
            Dp22 = copy.deepcopy(Dp2)
            Dp2 = Dp
            Dp = Dp22
            Fp22 = copy.deepcopy(Fp2)
            Fp2 = Fp
            Fp = Fp22
    if np.mean(Dp) < np.mean(Dt):
        Dp22 = copy.deepcopy(Dt)
        Dt = Dp
        Dp = Dp22
        if arg.net_pars.tri_exp:
            Fp = 1 - Fp - Fp2
        else:
            Fp = 1 - Fp    # here we correct for the data that initially was removed as it did not have IVIM behaviour, by returning zero
    if arg.net_pars.tri_exp:
        if np.mean(Dp2) < np.mean(Dp):
            Dp22 = copy.deepcopy(Dp2)
            Dp2 = Dp
            Dp = Dp22
            Fp22 = copy.deepcopy(Fp2)
            Fp2 = Fp
            Fp = Fp22
    # estimates
    Dptrue = np.zeros(lend)
    Dttrue = np.zeros(lend)
    Fptrue = np.zeros(lend)
    S0true = np.zeros(lend)
    Dptrue[sels] = Dp
    Dttrue[sels] = Dt
    Fptrue[sels] = Fp
    S0true[sels] = S0
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    if arg.net_pars.tri_exp:
        Dp2true = np.zeros(lend)
        Fp2true = np.zeros(lend)
        Dp2true[sels] = Dp2
        Fp2true[sels] = Fp2
        return [S0true, Dttrue, Fptrue, Dptrue, Fp2true, Dp2true]
    else:
        return [Dttrue, Fptrue, Dptrue, S0true]


def normalise(X_train,bvalues,arg, bref=0):
    try:
        ## normalise the signal to b=0 and remove data with nans
        if arg.norm_data_full:
            S0 = np.mean(X_train, axis=1).astype('<f')
        else:
            S0 = np.mean(X_train[:, bvalues == bref], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        np.delete(X_train, isnan(np.mean(X_train, axis=1)), axis=0)
        # normalise neighbours
    except:
        S0 = torch.mean(X_train[:, bvalues == bref], axis=1)
        X_train = X_train / S0[:, None]
        np.delete(X_train, isnan(torch.mean(X_train, axis=1)), axis=0)
    return X_train


def isnan(x):
    # this program indicates what are NaNs 
    return x != x


def plot_progress(X_batch, X_pred, bvalues, loss_train, loss_val, arg):
    # this program plots the progress of the training. It will plot the loss and validatin loss, as well as 4 IVIM curve
    # fits to 4 data points from the input
    inds1 = np.argsort(bvalues)
    X_batch = X_batch[:, inds1]
    X_pred = X_pred[:, inds1]
    bvalues = bvalues[inds1]
    if arg.fig:
        matplotlib.use('TkAgg')
        plt.close('all')
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(bvalues, X_batch.data[0], 'o')
        axs[0, 0].plot(bvalues, X_pred.data[0])
        axs[0, 0].set_ylim(min(X_batch.data[0]) - 0.3, 1.2 * max(X_batch.data[0]))
        axs[1, 0].plot(bvalues, X_batch.data[1], 'o')
        axs[1, 0].plot(bvalues, X_pred.data[1])
        axs[1, 0].set_ylim(min(X_batch.data[1]) - 0.3, 1.2 * max(X_batch.data[1]))
        axs[0, 1].plot(bvalues, X_batch.data[2], 'o')
        axs[0, 1].plot(bvalues, X_pred.data[2])
        axs[0, 1].set_ylim(min(X_batch.data[2]) - 0.3, 1.2 * max(X_batch.data[2]))
        axs[1, 1].plot(bvalues, X_batch.data[3], 'o')
        axs[1, 1].plot(bvalues, X_pred.data[3])
        axs[1, 1].set_ylim(min(X_batch.data[3]) - 0.3, 1.2 * max(X_batch.data[3]))
        plt.legend(('data', 'estimate from network'))
        for ax in axs.flat:
            ax.set(xlabel='b-value (s/mm2)', ylabel='normalised signal')
        for ax in axs.flat:
            ax.label_outer()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        plt.figure(2)
        plt.clf()
        plt.plot(loss_train)
        plt.plot(loss_val)
        plt.yscale("log")
        plt.xlabel('epoch #')
        plt.ylabel('loss')
        plt.legend(('training loss', 'validation loss (after training epoch)'))
        plt.ion()
        plt.show()
        plt.pause(0.001)


def make_data_complete(dw_data,bvalues,fraction_threshold=0.2):
    """
    This function is specific to missing data. For example, due to motion, after image registration our dataset
    contained gaps of information in some patients. As the Neural Network might get confused by empty slots,
    this program was desigend to fill up these slots with more realistic data estimates.

    :param bvalues: Array with the b-values
    :param dw_data: 1D Array with diffusion-weighted signal at different b-values
    :param fraction_threshold: an optional parameter determining the maximum fraction of missing data allowed.
    if more data is missing, the algorithm will not correct to prrvent too unrealistic (noiseless) data.

    :return dw_data: corrected dataset
    """
    if len(np.shape(dw_data)) is 4:
        sx, sy, sz, n_b_values = dw_data.shape
        dw_data = np.reshape(dw_data, (sx * sy * sz, n_b_values))
        reshape = True
    dw_data[isnan(dw_data)] = 0
    zeros = (dw_data == 0)
    locs = np.mean(zeros,axis=1)
    sels = (locs > 0) & (locs < fraction_threshold)
    data_to_correct = dw_data[sels,:]
    print('correcting {} datapoints'.format(len(data_to_correct)))
    def parfun(i):
        datatemp = data_to_correct[i,:]
        nonzeros = datatemp > 0
        bvaltemp = bvalues[nonzeros]
        datatempf=datatemp[nonzeros]
        norm=np.nanmean(datatempf)
        datatemp = datatemp / norm
        datatempf = datatempf / norm
        [Dt,Fp,Dp,S0]=fit.fit_least_squares(bvaltemp, datatempf, S0_output=True, fitS0=True, bounds=([0, 0, 0, 0.8], [0.005, 0.7, 0.3, 3]))
        datatemp[~nonzeros] = fit.ivim(bvalues,Dt,Fp,Dp,S0)[~nonzeros]
        return datatemp * norm
    data_to_correct = Parallel(n_jobs=4,batch_size=64)(delayed(parfun)(i) for i in tqdm(range(len(data_to_correct)), position=0,
                                                                    leave=True))
    dw_data[sels, :] = data_to_correct
    if reshape:
        dw_data = np.reshape(dw_data, (sx, sy, sz, n_b_values))
    return dw_data


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
    if not hasattr(arg, 'plateau_size'):
        warnings.warn('arg.train.plateau_size not defined. Using default of 20')
        arg.plateau_size = 20
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
        arg.parallel = 'parallel'  # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
    if not hasattr(arg,'con'):
        warnings.warn('arg.net_pars.con not defined. Using default of ''sigmoid''')
        arg.con = 'sigmoid'  # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
    if not hasattr(arg,'fitS0'):
        warnings.warn('arg.net_pars.parallel not defined. Using default of False')
        arg.fitS0 = False  # indicates whether to fix S0 to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
    if not hasattr(arg,'depth'):
        warnings.warn('arg.net_pars.depth not defined. Using default value of 4')
        arg.depth = 4  # number of layers
    if not hasattr(arg, 'width'):
        warnings.warn('arg.net_pars.width not defined. Using default of number of b-values')
        arg.width = 0
    if not hasattr(arg, 'tri_exp'):
        warnings.warn('arg.net_pars.tri_exp not defined. Using default of False')
        arg.tri_exp = False
    if not hasattr(arg,'cons_min'):
        if arg.tri_exp:
            warnings.warn('arg.net_pars.cons_min not defined. Using default values [0., 0.0003, 0.0, 0.003, 0.0, 0.08]')
            arg.cons_min = [0., 0.0003, 0.0, 0.003, 0.0, 0.08]  #
        else:
            warnings.warn('arg.net_pars.cons_min not defined. Using default values [0, 0, 0.005, 0.7]')
            arg.cons_min = [0, 0, 0.005, 0.7]  #
    if not hasattr(arg,'cons_max'):
        if arg.tri_exp:
            warnings.warn('arg.net_pars.cons_max not defined. Using default values [2.5, 0.003, 1, 0.08, 1, 5]')
            arg.cons_max = [2.5, 0.003, 1, 0.08, 1, 5]  #
        else:
            warnings.warn('arg.net_pars.cons_max not defined. Using default values [0.005, 0.7, 0.2, 2.0]')
            arg.cons_max = [0.005, 0.7, 0.2, 2.0]  #
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
    if not hasattr(arg, 'norm_data_full'):
        warnings.warn('arg no norm_data_full. Using default of False')
        arg.norm_data_full = False
    arg.net_pars=checkarg_net_pars(arg.net_pars)
    arg.train_pars = checkarg_train_pars(arg.train_pars)
    arg.sim = checkarg_sim(arg.sim)
    arg.fit = fit.checkarg_lsq(arg.fit)
    return arg


class train_pars:
    def __init__(self):
        self.optim='adam' #these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
        self.lr = 0.0001 # this is the learning rate.
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
        self.plateau_size = 20


class net_pars:
    def __init__(self):
        # select a network setting
        # the optimized network settings
        self.dropout = 0.1 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
        self.batch_norm = True # False/True turns on batch normalistion
        self.parallel = 'parallel' # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
        self.tri_exp = False
        self.con = 'sigmoid' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
        #### only if sigmoid constraint is used!
        self.cons_min = [-0.0001, -0.05, -0.05, 0.7]  # Dt, Fp, Ds, S0
        self.cons_max = [0.005, 0.7, 0.3, 1.3]  # Dt, Fp, Ds, S0
        ####
        self.fitS0=True # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
        self.depth = 4 # number of layers
        self.width = 0 # new option that determines network width. Putting to 0 makes it as wide as the number of b-values


class lsqfit:
    def __init__(self):
        self.method = 'lsq' #seg, bayes or lsq
        self.do_fit = True # skip lsq fitting
        self.load_lsq = False # load the last results for lsq fit
        self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 in the least squares fit.
        self.jobs = 2 # number of parallel jobs. If set to 1, no parallel computing is used
        self.bounds = ([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3]) #Dt, Fp, Ds, S0


class sim:
    def __init__(self):
        self.bvalues = np.array([0, 2, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]) # array of b-values
        self.SNR = [15, 20, 30, 40, 50] # the SNRs to simulate at
        self.sims = 1000000 # number of simulations to run
        self.num_samples_eval = 100000 # number of simualtiosn te evaluate. This can be lower than the number run. Particularly to save time when fitting. More simulations help with generating sufficient data for the neural network
        self.repeats = 1 # this is the number of repeats for simulations
        self.rician = False # add rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0005, 0.05, 0.01],
                  [0.003, 0.55, 0.1])
