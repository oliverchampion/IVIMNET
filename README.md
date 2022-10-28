# IVIMNET
This repository contains the code regarding the publication: Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients. (example1.py, example2.py and example3.py)


it also contains the code for: Self-supervised neural network improves tri-exponential intravoxel incoherent motion model fitting compared to least-squares fitting in non-alcoholic fatty liver disease (tri_exp_examples.example_tri_exp.py)


Publication accompanying this code is available at:
https://doi.org/10.1002/mrm.28852 Improved unsupervised physics-informed deep learning for intravoxel incoherent motion modeling and evaluation in pancreatic cancer patients, Kaandorp et al. MRM 2021 86:4;2250-2265 


and


https://doi.org/10.3389/fphys.2022.942495 Self-supervised neural network improves tri-exponential intravoxel incoherent motion model fitting compared to least-squares fitting in non-alcoholic fatty liver disease, Troelstra et al. Frontiers in Physiology. 2022 


September 2020 by Oliver Gurney-Champion

oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl

https://www.github.com/ochampion


September 2020 by Misha Kaandorp

mishakaandorp@gmail.com / m.kaandorp1@amsterdamumc.nl

https://github.com/Mishakaandorp 


Built on code by Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim

##requirements:

numpy

torch

tqdm

matplotlib

scipy

joblib


To directly run the code, we added a '.yml' file which can be run in anaconda. 

To create a conda environment with the '.yml' file enter the command below in the terminal: conda env create -f environment.yml

This now creates an environment called 'ivim' that can be activated by: conda activate ivim


#For easy start, run the examples
##example1.py
This is a very simple visual example in which we simulate a square in a square in a square and show its performance. 

##example2.py
This is a example where we simulate a lot of varied data and calculate the accuracy and precission. Output will be tables with accuracy and precission.

##example3.py
Here we load 1 dataset, train the network, and do the IVIM fit in this patient. NOTE: this is a toy-example in which you will train and evaluate in the same single patient. For larger studies, we would advice you to train 1 single network that you apply to all patients, instead of training a network on a per patient basis. This requieres you to adapt the code.

##tri_exp_examples.example_tri_exp.py
This is similar to 2, but with tri-exponential model.
