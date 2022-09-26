# IVIMNET
This repository contains the code regarding our submitted publication: Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients

Publication accompanying this code is available at:
https://doi.org/10.1002/mrm.28852 Improved unsupervised physics-informed deep learning for intravoxel incoherent motion modeling and evaluation in pancreatic cancer patients, Kaandorp et al. MRM 2021 86:4;2250-2265 


September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

September 2020 by Misha Kaandorp
mishakaandorp@gmail.com / m.kaandorp1@amsterdamumc.nl
https://github.com/Mishakaandorp 

Built on code by Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim

requirements:
numpy
torch
tqdm
matplotlib
scipy
joblib

To directly run the code, we added a '.yml' file which can be run in anaconda. 
To create a conda environment with the '.yml' file enter the command below in the terminal: conda env create -f environment.yml
This now creates an environment called 'ivim' that can be activated by: conda activate ivim
