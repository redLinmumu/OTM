# From https://github.com/brentyi/multimodalfilter/blob/master/scripts/push_task/train_push.py
import torch.optim as optim
import torch.nn as nn
import torch
import fannypack

import argparse
import sys
import os




sys.path.insert(0, os.getcwd())

from training_structures.Supervised_Learning import test, MMDL
from fusions.common_fusions import TensorFusion # noqa

from unimodals.common_models import Sequential, Transpose, Reshape, MLP # noqa
from datasets.gentle_push.data_loader import PushTask # noqa
import unimodals.gentle_push.layers as layers # noqa
from datasets.gentle_push.data_loader import get_dataloader_nfold

import numpy as np
from sklearn.model_selection import KFold
from mwae.util import setup_seed, print_current_time

setup_seed()
print_current_time()

torch.backends.cudnn.enabled = False

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

is_packed = False

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

epoch_num = 20
print(f'epoch = {epoch_num}')

tolerance = 7
print(f'Tolerance of early stop = {tolerance}')


single_dir_base = 'best_pts/mujoco/tensor_fusion/single/'
# os.makedirs(pts_dir, exist_ok=True)

mixer_dir_base = 'best_pts/mujoco/tensor_fusion/' + str(beta) + '/'


view_num = 4
bests_list_mixer = [ [] for i in range(view_num) ]
bests_list_origin = [ [] for i in range(view_num) ]


encoders = [
    Sequential(Transpose(0, 1), layers.observation_pos_layers(
        8), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.observation_sensors_layers(
        8), Transpose(0, 1)),
    Sequential(Transpose(0, 1), Reshape(
        [-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64]), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.control_layers(16), Transpose(0, 1)),
]

head_dims = [8, 8, 64, 16]
task="regression"

train_loader, val_loader, test_loader = Task.get_dataloader(
    16, batch_size=32, drop_last=True)

    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"Begins -------------------Single Modality  [Fold] {fold}:")
    
    # train_dataloaer, valid_dataloader = get_dataloader_nfold(subsequence_length=16, train_index=train_index, val_index=valid_index, batch_size=32, drop_last=True, device=device)
    
    # print(f"--------------Mixer {fold} starts:--------------")
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
    
    mixer_file_path = mixer_dir_base + 'tf_mixer_fold' + str(fold) + '.pt'
    
    print(f"Load mixer model = {mixer_file_path}")
    
    model_mixer = MMDL(encoders_mixer, fusion_mixer, head_mixer, has_padding=is_packed).to(device)
    model_mixer = torch.load(mixer_file_path, map_location=device)
    model_mixer.to(device)

    test(model_mixer, test_loader, dataset='gentle push',
     task='regression', criterion=loss_state, mixer_type='mixer', fold_index=fold)

      
    # print(f'--------------ORIGINAL {fold} starts:-------------')
    encoders_origin = [
        Sequential(Transpose(0, 1), layers.observation_pos_layers(
            8), Transpose(0, 1)),
        Sequential(Transpose(0, 1), layers.observation_sensors_layers(
            8), Transpose(0, 1)),
        Sequential(Transpose(0, 1), Reshape(
            [-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64]), Transpose(0, 1)),
        Sequential(Transpose(0, 1), layers.control_layers(16), Transpose(0, 1)),
        ]
    fusion_origin = TensorFusion()
    
    head_origin = MLP((8 + 1) * (8 + 1) * (64 + 1) * (16 + 1), 256, 2)

    origin_file_path = mixer_dir_base + 'tf_origin_fold' + str(fold) + '.pt'
    
    print(f"Load orign model = {origin_file_path}")
    
    model_origin = MMDL(encoders_origin, fusion_origin, head_origin, has_padding=is_packed).to(device)
    model_origin = torch.load(origin_file_path, map_location=device)
    model_origin.to(device)
    
    test(model_origin, test_loader, dataset='gentle push',
     task='regression', criterion=loss_state, mixer_type='origin', fold_index=fold)

