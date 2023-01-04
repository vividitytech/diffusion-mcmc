# diffusion-mcmc
Code for reproducing some key results of our paper Speed up the inference of diffusion models via shortcut MCMC sampling (https://arxiv.org/abs/2301.01206).

## Requirement
jaxlib, tensorflow, etc

## run example:

python global_vdm_2d.py

## some important parameters:
### joinFlag = False # add the flag whether we need joined learning
### freeze_x_decoder = False
### shortcut = True
### K=10 default, which can change based on experiments


## reference
we demonstrate our approach based on a 2D swirl dataset and using MLPs, most code base are from VDM [Link to open in Colab](https://colab.research.google.com/github/google-research/vdm/blob/main/colab/2D_VDM_Example.ipynb).

if you think it is helpful please cite:
Gang Chen, Speed up the inference of diffusion models via shortcut MCMC sampling (https://arxiv.org/abs/2301.01206).
