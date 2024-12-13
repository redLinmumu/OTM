import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader_nfold
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train_head_single_modality, MMDL, SingleMDL

import copy
import numpy as np
from sklearn.model_selection import KFold
from mwae.util import setup_seed, print_current_time

setup_seed()
print_current_time()

# device_default = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = 'datasets/avmnist/avmnist'
channels = 6


n_splits = 5  
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
n_samples = 70000


from utils.parse_args import parse_args

args = parse_args()

beta = 0.01
print(f'beta = {beta}')


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

epoch_num = 30
print(f'epoch = {epoch_num}')

tolerance = args.tolerance
print(f'Tolerance of early stop = {tolerance}')

mixer_dir = 'best_pts/avmnist/late_fusion/' + str(beta) + '/'

single_dir = 'best_pts/avmnist/late_fusion/single/'

# os.makedirs(pts_dir, exist_ok=True)

encoders_kernel_dim = [3, 5]
head_dim = [8, 32]

view_num = 2

bests_list_mixer = [ [] for i in range(view_num) ]
bests_list_origin = [ [] for i in range(view_num) ]


task = "classification"
is_packed = False


for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"Begins -------------------Single Modality  [Fold] {fold}:")
    
    traindata, valid_dataloader = get_dataloader_nfold(data_dir=data_dir, train_index=train_index, valid_index=valid_index)
    
    # mixer model
    encoders_mixer = [LeNet(1, channels, 3).to(device), LeNet(1, channels, 5).to(device)]
    head_mixer = MLP(channels*40, 100, 10).to(device)
    fusion_mixer = Concat().to(device)
    
    mixer_file_path = mixer_dir + 'slatefusion_mixer_' + str(fold)+ '.pt'
    print(f"Load mixer model = {mixer_file_path}")
    
    model_mixer = MMDL(encoders_mixer, fusion_mixer, head_mixer, has_padding=is_packed).to(device)
    model_mixer = torch.load(mixer_file_path, map_location=device)
    model_mixer.to(device)
    
    # origin model
    encoders_origin = [LeNet(1, channels, 3).to(device), LeNet(1, channels, 5).to(device)]
    head_origin = MLP(channels*40, 100, 10).to(device)
    fusion_origin = Concat().to(device)

    origin_file_path = mixer_dir + 'slatefusion_origin_' + str(fold)+ '.pt'
    print(f"Load orign model = {origin_file_path}")

    model_origin = MMDL(encoders_origin, fusion_origin, head_origin, has_padding=is_packed).to(device)
    model_origin = torch.load(origin_file_path, map_location=device)
    model_origin.to(device)
    
    
    for i in range(view_num):
      # encoder = LeNet(1, channels, encoders_kernel_dim[i]).to(device)
      
      # head = MLP((head_dim[i])*channels, 100, 10).to(device)
      
      # single_model = SingleMDL(encoder, head=head, index=i, has_padding=is_packed).to(device)
      # single_model_dir = single_dir + "single_view_" + str(i) + "_fold_" + str(fold) + ".pt"
      # single_model = torch.load(single_model_dir)
        
      print(f"*** Modality {i} == > fold {fold}:")
      
            # ------mixer valid--------
      # mixer_single_model = copy.deepcopy(single_model)
      # mixer_single_model.encoder = model_mixer.encoders[i]
      
      encoder_mixer = copy.deepcopy(model_mixer.encoders[i])
      head_mixer = MLP((head_dim[i])*channels, 100, 10).to(device)
      
      
      best_mixer = train_head_single_modality(encoder_mixer, i, head_mixer, traindata, valid_dataloader, epoch_num,
      optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001
      , track_complexity=False, save=single_dir, device=device, tolerance=tolerance, fold_index=fold, type="mixer")
            
      bests_list_mixer[i].append(best_mixer.item())
            
        # ------origin valid---------
      # origin_single_model = copy.deepcopy(single_model)
      # origin_single_model.encoder = model_origin.encoders[i]
      
      encoder_origin = copy.deepcopy(model_origin.encoders[i])
      head_origin = MLP((head_dim[i])*channels, 100, 10).to(device)
      
      
      best_origin = train_head_single_modality(encoder_origin, i, head_origin, traindata, valid_dataloader, epoch_num,
      optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001
      , track_complexity=False, save=single_dir, device=device, tolerance=tolerance, fold_index=fold, type="origin")

      bests_list_origin[i].append(best_origin.item())
    
    # print(f"------------------Fold {fold} ends.-------------------")


for i in range(view_num):
  print(f"Results for modality {i}: ")
  print(f'[Average] Mixer = {sum(bests_list_mixer[i])/5.0}, total = {bests_list_mixer[i]}')    
  print(f'[Average] Origin = {sum(bests_list_origin[i])/5.0}, total = {bests_list_origin[i]}')   
print_current_time()