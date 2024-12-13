import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.Supervised_Learning import train, train_mixer, test
from unimodals.common_models import GRU, MLP

from fusions.common_fusions import Concat
import torch
from datasets.affect.get_data import get_fulldata, get_noise_dataloader
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


tolerance = args.tolerance
print(f'Tolerance of early stop = {tolerance}')


timestep = 50


full_data = get_fulldata(
    'data/affect/mosi/mosi_raw.pkl', task='classification', robust_test=False, max_pad=True, max_seq_len=timestep)

labels = full_data['labels']
n_samples = labels.shape[0]

from utils.parse_args import parse_args
    
args = parse_args()

epoch_num = 100
# epoch_num = args.epoch

print(f'epoch_num = {epoch_num}')

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

task="classification"

pts_dir = 'pts/affect/mosi/classification/latefusion/'  + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)

robust_test=True
noise_level = args.noise

train_dataloader, valid_dataloader, test_dataloader  = get_noise_dataloader(
    'data/affect/mosi/mosi_raw.pkl', robust_test=True, data_type='mosi'
    , raw_path="data/affect/mosi/mosi.hdf5", full_data=full_data, i=noise_level, device=device, task=task)

fold='noise'

    
if True:
    print(f"--------------Mixer {fold} starts:--------------")
      
    encoders_mixer = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(74, 200, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(300, 600, dropout=True, has_padding=True, batch_first=True).to(device)]
    head_mixer = MLP(870, 870, 2).to(device)

    fusion_mixer = Concat().to(device)
      
      
    save_as_m = pts_dir + 'mosi_latefusion_classify_mixer_' + fold + str(noise_level) +'_' + str(epoch_num) +'.pt'
      
    best_m = train_mixer(encoders_mixer, fusion_mixer, head_mixer, train_dataloader, valid_dataloader, epoch_num, task=task, optimtype=torch.optim.AdamW,
        early_stop=False, is_packed=True, lr=1e-3, save=save_as_m, weight_decay=0.01, device=device, beta=beta)



    print(f"--------------Origin {fold} starts:--------------") 
    encoders = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(74, 200, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(300, 600, dropout=True, has_padding=True, batch_first=True).to(device)]
    head = MLP(870, 870, 2).to(device)

    fusion = Concat().to(device)
    save_as_o = pts_dir + 'mosi_latefusion_classify_origin_' + fold + str(noise_level) +'_' + str(epoch_num) +'.pt'
      
    best_o = train(encoders, fusion, head, train_dataloader, valid_dataloader, epoch_num, task=task, optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save=save_as_o, weight_decay=0.01, device=device)


print("Testing Mixer:")    
model_mixer = torch.load(save_as_m).cuda()  
test(model=model_mixer, test_dataloaders_all=test_dataloader,
     dataset='mosi', is_packed=True, no_robust=True)


print("Testing Origin:")
model_origin = torch.load(save_as_o).cuda() 
test(model=model_origin, test_dataloaders_all=test_dataloader,
     dataset='mosi', is_packed=True, no_robust=True)

      
print_current_time()