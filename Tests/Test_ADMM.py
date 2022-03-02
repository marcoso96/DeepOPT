"""
Test ADMM and comparison
"""
#%% Import libraries
import os
import os,time, sys
os.chdir('.')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
sys.path.append('Reconstruction/')

import numpy as np
import random
import matplotlib.pyplot as plt
import DataLoading as DL
from Folders_cluster import *
import Reconstruction as RecTV
import ModelUtilities as modutils
import torch
from torch_radon import Radon, RadonFanbeam
from skimage.transform import radon, iradon
import torchvision
import model_torch as modl
import pickle
from tqdm import tqdm
from bayes_opt import BayesianOptimization


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%% ADMM test on data
proj_num = 72
augment_factor = 1
total_size = 5000
n_angles = 720
img_size = 100
det_count = int((img_size+0.5)*np.sqrt(2))
tv_iters = 3

tensor_path = datasets_folder + 'Proj_{}_augmentFactor_{}_totalSize_{}_FullY.pt'.format(proj_num, augment_factor, total_size)                                            

fullY = torch.load(tensor_path, map_location=torch.device('cpu'))
#%%

# Tensors in image space
# dataloaders = modutils.formDataloaders(datasets, proj_num, total_size, train_factor, val_factor, test_factor, batch_size, img_size, tensor_path, augment_factor, load_tensor = True, save_tensor = False)    

# Radon operator
angles = np.linspace(0, 2*np.pi, n_angles, endpoint = False)

Psi = lambda x,th:  RecTV.TVdenoise(x,2/th,tv_iters)
#  set the penalty function, to compute the objective
Phi = lambda x: RecTV.TVnorm(x)
hR = lambda x: radon(x, angles, circle = False)
hRT = lambda sino: iradon(sino, angles, circle = False)

# Test Image
print(callable(hR))
print(callable(hRT))
print(callable(Phi))
print(callable(Psi))

img_true = fullY[3, 0, ...].to(device).cpu().numpy().T

print(img_true.max())
print(img_true.min())

sino = hR(img_true)
img_rec_FBP = hRT(sino)
img_rec_ADMM,_,_,_ = RecTV.ADMM(y = sino, A =hR, AT = hRT, Den = Psi, alpha = 0.01, delta = 1, max_iter = 200, phi = Phi, tol = 10e-7, invert = 0, warm = 1, true_img = img_true)

# Have to send FiltX to Sinogram space in order to use ADMM
fig, ax = plt.subplots(1,4, figsize = (16,8))

ax[0].imshow(img_true)
ax[0].set_title('Image Ground Truth')
ax[1].imshow(sino)
ax[1].set_title('Sinogram')
ax[2].imshow(img_rec_ADMM)
ax[2].set_title('Reconstruction ADMM')
ax[3].imshow(np.abs(img_rec_ADMM-img_rec_FBP))
ax[3].set_title('Difference')

fig.savefig(results_folder+'TestADMM.pdf', bbox_inches = 'tight')







