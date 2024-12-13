# From https://github.com/brentyi/multimodalfilter/blob/master/scripts/push_task/train_push.py
import torch.optim as optim
import torch.nn as nn
import torch
import fannypack

import argparse
import sys
import os

sys.path.insert(0, os.getcwd())

from training_structures.Supervised_Learning import train, train_mixer
from fusions.common_fusions import TensorFusion # noqa

from unimodals.common_models import Sequential, Transpose, Reshape, MLP # noqa
from datasets.gentle_push.data_loader import PushTask # noqa
import unimodals.gentle_push.layers as layers # noqa
from datasets.gentle_push.data_loader import get_dataloader_nfold

import numpy as np
from sklearn.model_selection import KFold
from mwae.util import setup_seed, print_current_time

seed=123
print(f'seed = {seed}')

setup_seed(seed)
print_current_time()


n_samples = 1310

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

Task = PushTask
optimtype = optim.Adam
loss_state = nn.MSELoss()
# Parse args
parser = argparse.ArgumentParser()
Task.add_dataset_arguments(parser)
args = parser.parse_args()
dataset_args = Task.get_dataset_args(args)

fannypack.data.set_cache_path('datasets/gentle_push/cache')

optimtype = optim.Adam
loss_state = nn.MSELoss()

# torch.autograd.set_detect_anomaly(True)



beta = 0.1
print(f'beta = {beta}')


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

epoch_num = 20
print(f'epoch = {epoch_num}')


tolerance = 7
print(f'Tolerance of early stop = {tolerance}')


pts_dir = 'pts/gentle_push/' + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)

bests_m = []
bests_o = []
    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"-------------------Fold {fold}-------------------")
    train_dataloaer, valid_dataloader = get_dataloader_nfold(subsequence_length=16, train_index=train_index, val_index=valid_index, batch_size=32, drop_last=True, device=device)
    
     
    print(f"--------------Mixer {fold} starts:--------------")
    encoders_mixer = [
    Sequential(Transpose(0, 1), layers.observation_pos_layers(
        8), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.observation_sensors_layers(
        8), Transpose(0, 1)),
    Sequential(Transpose(0, 1), Reshape(
        [-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64]), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.control_layers(16), Transpose(0, 1)),
]
    fusion_mixer = TensorFusion()
    head_mixer = MLP((8 + 1) * (8 + 1) * (64 + 1) * (16 + 1), 256, 2)
    
    save_as_m = pts_dir + 'tf_mixer_fold' + str(fold) + '.pt'
    best_m = train_mixer(encoders_mixer, fusion_mixer, head_mixer,
      train_dataloaer, valid_dataloader,
      epoch_num,
      task='regression',
      optimtype=optimtype,
      objective=loss_state,
      lr=0.00001, device=device,save=save_as_m, beta=beta, tolerance=tolerance)
    
    bests_m.append(best_m.item())
    print(f"--------------Mixer {fold} ends!--------------")
      
      
    print(f'--------------ORIGINAL {fold} starts:-------------')
    encoders = [
    Sequential(Transpose(0, 1), layers.observation_pos_layers(
        8), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.observation_sensors_layers(
        8), Transpose(0, 1)),
    Sequential(Transpose(0, 1), Reshape(
        [-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64]), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.control_layers(16), Transpose(0, 1)),
]
    fusion = TensorFusion()
    head = MLP((8 + 1) * (8 + 1) * (64 + 1) * (16 + 1), 256, 2)

    save_as_o = pts_dir + 'tf_origin_fold' + str(fold) + '.pt'
    
    best_o = train(encoders, fusion, head,
      train_dataloaer, valid_dataloader,
      epoch_num,
      task='regression',
      optimtype=optimtype,
      objective=loss_state,
      lr=0.00001, device=device, save=save_as_o, tolerance=tolerance)
    
    bests_o.append(best_o.item())
    print(f'--------------ORIGINAL {fold} ends.-------------')
    
    
print(f'[Average] Mixer = {sum(bests_m)/5.0}, \n\t All = {bests_m}')  
print(f'[Average] Original = {sum(bests_o)/5.0}, \n\t All = {bests_o}')  
      

print_current_time()






