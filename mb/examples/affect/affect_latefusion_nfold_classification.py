import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.Supervised_Learning import train, train_mixer
from unimodals.common_models import GRU, MLP

from fusions.common_fusions import Concat
import torch
from datasets.affect.get_data import get_fulldata, get_fold_dataloader
from sklearn.model_selection import KFold

import numpy as np
from sklearn.model_selection import KFold
from mwae.util import setup_seed, print_current_time

setup_seed()
print_current_time()

from utils.parse_args import parse_args

args = parse_args()

beta = 100.0
print(f'beta = {beta}')


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

epoch_num = 100
print(f'epoch = {epoch_num}')

tolerance = args.tolerance
print(f'Tolerance of early stop = {tolerance}')

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

timestep = 50
full_data = get_fulldata(
    'data/affect/mosi/mosi_raw.pkl', robust_test=False, data_type='mosi')
labels = full_data['labels']
n_samples = labels.shape[0]



n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

task="classification"

pts_dir = 'pts/affect/mosi/classification/latefusion/'  + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)

bests_m = []
bests_o = []

    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
      print(f"-------------------Fold {fold}-------------------")

      encoders = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(74, 200, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(300, 600, dropout=True, has_padding=True, batch_first=True).to(device)]
      head = MLP(870, 870, 2).to(device)

      fusion = Concat().to(device)
      
      
      encoders_mixer = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(74, 200, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(300, 600, dropout=True, has_padding=True, batch_first=True).to(device)]
      head_mixer = MLP(870, 870, 2).to(device)

      fusion_mixer = Concat().to(device)
      
      
      train_data = {key: value[train_index] for key, value in full_data.items()}
      valid_data = {key: value[valid_index] for key, value in full_data.items()}

      train_dataloaer, valid_dataloader = get_fold_dataloader(train_data=train_data, valid_data=valid_data, robust_test=False, data_type='mosi', task=task)
      
      print(f"--------------Mixer {fold} starts:--------------")
      
      save_as_m = pts_dir + 'mosi_latefusion_classify_mixer_' + str(fold) + '.pt'
      
      best_m = train_mixer(encoders_mixer, fusion_mixer, head_mixer, train_dataloaer, valid_dataloader, epoch_num, task=task, optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save=save_as_m, weight_decay=0.01, device=device, beta=beta)
      bests_m.append(best_m.item())
      
      print(f"--------------Mixer {fold} ends!--------------")
      
      
      print(f'--------------ORIGINAL {fold} starts:-------------')
      
      save_as_o = pts_dir + 'mosi_latefusion_classify_origin_' + str(fold) + '.pt'
      
      best_o = train(encoders, fusion, head, train_dataloaer, valid_dataloader, epoch_num, task=task, optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save=save_as_o, weight_decay=0.01, device=device)
      bests_o.append(best_o.item())
      
      print(f'--------------ORIGINAL {fold} ends.-------------')
      
print(f'[Average] Mixer = {sum(bests_m)/5.0}, \n\t All = {bests_m}')  
print(f'[Average] Original = {sum(bests_o)/5.0}, \n\t All = {bests_o}')  

      
print_current_time()