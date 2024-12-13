# From https://github.com/brentyi/multimodalfilter/blob/master/scripts/push_task/train_push.py
import torch.optim as optim
import torch.nn as nn
import torch
import fannypack

import argparse
import sys
import os
import copy

sys.path.insert(0, os.getcwd())

from training_structures.Supervised_Learning import train_head_single_modality, MMDL, SingleMDL
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
    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"Begins -------------------Single Modality  [Fold] {fold}:")
    
    train_dataloaer, valid_dataloader = get_dataloader_nfold(subsequence_length=16, train_index=train_index, val_index=valid_index, batch_size=32, drop_last=True, device=device)
    
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
    
    for i in range(view_num):

        print(f"*** Modality {i} == > fold {fold}:")
        
        # encoder = copy.deepcopy(encoders[i])
        # head = MLP(head_dims[i], 256, 2)
        
        # single_model = SingleMDL(encoder, head=head, index=i, has_padding=is_packed).to(device)
        # single_model_dir = single_dir_base + "single_view_" + str(i) + "_fold_" + str(fold) + ".pt"
        # single_model = torch.load(single_model_dir)
        
        # ------mixer valid--------
        # mixer_single_model = copy.deepcopy(single_model)
        # mixer_single_model.encoder = model_mixer.encoders[i]
        
        encoder_mixer = copy.deepcopy(model_mixer.encoders[i])
        head_mixer = MLP(head_dims[i], 256, 2)
        
        best_mixer = train_head_single_modality(encoder_mixer, i, head_mixer, train_dataloaer, valid_dataloader, epoch_num, task="regression",
        optimtype=optimtype,  lr=0.00001, track_complexity=False, objective=loss_state,
        save=single_dir_base, device=device, tolerance=tolerance, fold_index=fold, type="mixer")
        
        bests_list_mixer[i].append(best_mixer.item())
        
        # ------origin valid---------
        # origin_single_model = copy.deepcopy(single_model)
        # origin_single_model.encoder = model_origin.encoders[i]
        encoder_origin = copy.deepcopy(model_origin.encoders[i])
        head_origin = MLP(head_dims[i], 256, 2)
        
        best_origin = train_head_single_modality(encoder_origin, i, head_origin, train_dataloaer, valid_dataloader, epoch_num, task="regression",
        optimtype=optimtype,  lr=0.00001, track_complexity=False, objective=loss_state,
        save=single_dir_base, device=device, tolerance=tolerance, fold_index=fold, type="origin")

        bests_list_origin[i].append(best_origin.item())
    
    # print(f"------------------Fold {fold} ends.-------------------")

print("---------Start Eval-----------")
for i in range(view_num):
  print(f"Results for modality {i}: ")
  print(f'[Average] Mixer = {sum(bests_list_mixer[i])/5.0}, total = {bests_list_mixer[i]}')    
  print(f'[Average] Origin = {sum(bests_list_origin[i])/5.0}, total = {bests_list_origin[i]}')   
print_current_time()