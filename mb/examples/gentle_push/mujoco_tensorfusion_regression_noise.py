# From https://github.com/brentyi/multimodalfilter/blob/master/scripts/push_task/train_push.py
import torch.optim as optim
import torch.nn as nn
import torch
import fannypack

import argparse
import sys
import os

sys.path.insert(0, os.getcwd())

from training_structures.Supervised_Learning import train, train_mixer, test
from fusions.common_fusions import TensorFusion # noqa

from unimodals.common_models import Sequential, Transpose, Reshape, MLP # noqa
from datasets.gentle_push.data_loader import PushTask # noqa
import unimodals.gentle_push.layers as layers # noqa
from datasets.gentle_push.data_loader import get_dataloader_noise

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

dataset_args['device'] = device


epoch_num = 200
print(f'epoch = {epoch_num}')


tolerance = 7
print(f'Tolerance of early stop = {tolerance}')


pts_dir = 'pts/gentle_push/' + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)

fold='noise'

task='regression'

noise_level = dataset_args['noise']

train_dataloaer, valid_dataloader, test_dataloader = get_dataloader_noise(subsequence_length=16,  batch_size=32, drop_last=True, dataset_args=dataset_args)
    
    
if True:

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
    
    save_as_m = pts_dir + 'tf_mixer_fold' + fold + str(noise_level) + '_' + str(epoch_num)+'.pt'
    best_m = train_mixer(encoders_mixer, fusion_mixer, head_mixer,
      train_dataloaer, valid_dataloader,
      epoch_num,
      task='regression',
      optimtype=optimtype,
      objective=loss_state,
      lr=0.00001, device=device,save=save_as_m, beta=beta, tolerance=tolerance)
    

      
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

    save_as_o = pts_dir + 'tf_origin_fold' + fold + str(noise_level) + '_' + str(epoch_num)+'.pt'
    
    best_o = train(encoders, fusion, head,
      train_dataloaer, valid_dataloader,
      epoch_num,
      task='regression',
      optimtype=optimtype,
      objective=loss_state,
      lr=0.00001, device=device, save=save_as_o, tolerance=tolerance)
    
print("Testing Mixer:")
model_mixer = torch.load(save_as_m).cuda()
test(model_mixer, test_dataloader, dataset='gentle push',
     task='regression', criterion=loss_state, mixer_type='mixer', no_robust=True)


print("Testing Origin:")
model_origin = torch.load(save_as_o).cuda()
test(model_origin, test_dataloader, dataset='gentle push',
     task='regression', criterion=loss_state, mixer_type='origin', no_robust=True)






