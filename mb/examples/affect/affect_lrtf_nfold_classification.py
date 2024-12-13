import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


from private_test_scripts.all_in_one import all_in_one_train # noqa
from training_structures.Supervised_Learning import train, train_mixer
from unimodals.common_models import GRUWithLinear, MLP # noqa

from fusions.common_fusions import  LowRankTensorFusion # noqa

from datasets.affect.get_data import get_fulldata, get_fold_dataloader
from sklearn.model_selection import KFold

import numpy as np
from sklearn.model_selection import KFold
from mwae.util import setup_seed, print_current_time

setup_seed()
print_current_time()

full_data = get_fulldata(
    'data/affect/mosi/mosi_raw.pkl', robust_test=False, data_type='mosi')
labels = full_data['labels']
n_samples = labels.shape[0]


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

from utils.parse_args import parse_args
    
args = parse_args()

beta = args.beta
print(f'beta = {beta}')


pts_dir = 'pts/affect/mosi/lrtf/'  + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

epoch_num = args.epoch
print(f'epoch = {epoch_num}')

tolerance = args.tolerance
print(f'Tolerance of early stop = {tolerance}')
# run origin(default True)
origin = args.origin

# run mixer(default True)
mixer = args.mixer

if mixer:
    bests_m = []
    print('Run mixer...')
    
if origin:
    bests_o = []
    print('Run origin...')
    
    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"[Fold] {fold}")
    if origin or mixer:
     
      train_data = {key: value[train_index] for key, value in full_data.items()}
      valid_data = {key: value[valid_index] for key, value in full_data.items()}

      train_dataloaer, valid_dataloader = get_fold_dataloader(train_data=train_data, valid_data=valid_data, robust_test=False, task='classification')
    
    if mixer:
      encoders_mixer = [GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(74, 128, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(300, 512, 128, dropout=True, has_padding=True).to(device)]
      head_mixer = MLP(128, 512, 2).to(device)

      fusion_mixer = LowRankTensorFusion([32, 32, 128], 128, 32, device=device).to(device)
      print(f"--------------Mixer {fold} starts:--------------")
      
      save_as_m = pts_dir + 'lrtf_mixer_' + str(fold)+ '.pt'
      
      best_m = train_mixer(encoders_mixer, fusion_mixer, head_mixer, train_dataloaer, valid_dataloader, epoch_num, task="classification", optimtype=torch.optim.AdamW,
        early_stop=True, is_packed=True, lr=1e-3, save=save_as_m, weight_decay=0.01, device=device, beta=beta, tolerance=tolerance)
      
      bests_m.append(best_m.item())
      print(f"--------------Mixer {fold} ends.--------------")
    
    if origin:
      encoders = [GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(74, 128, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(300, 512, 128, dropout=True, has_padding=True).to(device)]
      head = MLP(128, 512, 2).to(device)

      fusion = LowRankTensorFusion([32, 32, 128], 128, 32, device=device).to(device)
      print(f"--------------Origin {fold} starts:--------------")
      save_as_o = pts_dir + 'lrtf_origin_' + str(fold)+ '.pt'
      best_o = train(encoders, fusion, head, train_dataloaer, valid_dataloader, epoch_num, task="classification", optimtype=torch.optim.AdamW,
        early_stop=True, is_packed=True, lr=1e-3, save=save_as_o, weight_decay=0.01, device=device)
      
      bests_o.append(best_o.item())
      print(f"--------------Origin {fold} starts:--------------")

if mixer:
  print(f'[Average] Mixer = {sum(bests_m)/5.0}, \n\t All = {bests_m}')  

if origin:
  print(f'[Average] Original = {sum(bests_o)/5.0}, \n\t All = {bests_o}')  

print_current_time()