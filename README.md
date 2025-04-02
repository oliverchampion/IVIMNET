# IVIMNET
For more deep learning qMRI code and latest updates, see: https://mriresearch.amsterdam/software/ivim-net-and-dce-net/

This repository contains the code regarding the publication: Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients. (example1.py, example2.py and example3.py)
it also contains the code for: Self-supervised neural network improves tri-exponential intravoxel incoherent motion model fitting compared to least-squares fitting in non-alcoholic fatty liver disease (tri_exp_examples.example_tri_exp.py)


The publication accompanying this code is available at:

bi-exponential:
https://doi.org/10.1002/mrm.28852 Improved unsupervised physics-informed deep learning for intravoxel incoherent motion modeling and evaluation in pancreatic cancer patients, Kaandorp et al. MRM 2021 86:4;2250-2265 


and

tri-exponential
https://doi.org/10.3389/fphys.2022.942495 Self-supervised neural network improves tri-exponential intravoxel incoherent motion model fitting compared to least-squares fitting in non-alcoholic fatty liver disease, Troelstra et al. Frontiers in Physiology. 2022 


#### requirements:

To directly run the code, we added a '.yml' file which can be run in anaconda. 

To create a conda environment with the '.yml' file enter the command below in the terminal: conda env create -f environment.yml

This now creates an environment called 'ivim' that can be activated by: conda activate ivim


## For easy start, run the examples
### example1.py
This is a very simple visual example in which we simulate a square in a square in a square and show its performance. 

### example2.py
This is a example where we simulate a lot of varied data and calculate the accuracy and precission. Output will be tables with accuracy and precission.

### example3.py
Here we load 1 dataset, train the network, and do the IVIM fit in this patient. **NOTE:** this is a *toy-example* in which you will train and evaluate in **the same single patient**. For larger studies, we would advice you to train 1 single network that you apply to all patients, instead of training a network on a per patient basis. This requieres you to adapt the code.

### tri_exp_examples.example_tri_exp.py
This is similar to 2, but with tri-exponential model.

## Considerations when performing in your own data
We have received feedback from all of you working on this code, for which we are grateful (Please do reach out!). Generally, most of the feedback is related to how to adopt the code for your own usage. Therefore, we would like to provide you with some of the tips and lessons learnt over the years.
### Training in vivo
If you train in vivo, in our view there are two possible approaches: As a fitting tool (as we intend it) or as a traditional deep learning approach.
#### Fitting tool (intended use)
In this case, the combined physics-informed training and inference are seen as the "fit" (in a way, similar to neural implicit learning). In this case, all data from your study should be thrown into ONE 2D matrix (voxels * b-values), and the network should be trained and inferenced on ALL data together. This way, you ensure that any potential bias introduced by the network will be identical throughout your study. One common mistake we see happening is that people train 1 network per patient. this is easier to implement from example 3, but can cause biasses between patients.
#### Traditional deep learning
In this case, you split data into a train/validation/test set. You train the network on all training data, and then evaluate it in the test set. This would mimic the situation in which a trained version of IVIM-NET is used to evaluate a study. The reason why we have not attended IVIM-NET's use for this approach is that IVIM-NET needs retraining for any new b-value distribution. So having a trained network would only be useful for a specific acquisition and you would have to train a network for each acquisition (and acquire training data...).
### IVIM-NET in other organs
It has come to our attention that other organs, especially the brain, may requiere other fit boundaries. Moreover, it can be possible that some furhter tweaking of hyperparameters may be needed in these cases.
### IVIM-NET and preprocessing
One of the things we have noticed is that many IVIM data is strongly denoised to enable fitting. Come better fit-methods, less denoising may be required. In particular, we noticed that IVIM-NET performs well in non-denoised data, giving precise parameter maps while retaining the sharpness of the image. When strong denoising is applied, the least square fit may to do an equally good job, but at the cost of a loss in sharpness.
A second thing we have noticed is that IVIM-NET is happier with non-averaged images, such that it can better capture the noise behaviour. 
### IVIM-NET and small amount of b-values/averaged data
Advanced fit methods gain their advantage in how they deal with redundancy. If only a small amount of b-values have been acquiered and only 1 image is available per b-value (averaged TRACE image) then the added benefit of IVIM-NET may be limited.
### IVIM-NET as a reference fit method for when you did a better job!
We have seen some papers where the reported performance of IVIM-NET does not match our experience and reported performance. If you are reporting IVIM-NET as a reference fit method, please have a look at your local IVIM-NET performance as compared to our reported performance. If there is a big discrepancy, please do contact us and we can have a look at what may be going wrong. 


## authors 
September 2020 by 
Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Misha Kaandorp
mishakaandorp@gmail.com / m.kaandorp1@amsterdamumc.nl
https://github.com/Mishakaandorp 

Built on code by Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim
