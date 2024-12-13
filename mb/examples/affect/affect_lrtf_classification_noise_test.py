import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


from private_test_scripts.all_in_one import all_in_one_train # noqa
from training_structures.Supervised_Learning import train, train_mixer, test
from unimodals.common_models import GRUWithLinear, MLP # noqa

from fusions.common_fusions import  LowRankTensorFusion # noqa

from datasets.affect.get_data import get_fulldata, get_noise_dataloader
from sklearn.model_selection import KFold

import numpy as np
from sklearn.model_selection import KFold
from mwae.util import setup_seed, print_current_time

setup_seed()
print_current_time()

full_data = get_fulldata(
    'data/affect/mosi/mosi_raw.pkl', robust_test=False, data_type='mosi', task='classification')
labels = full_data['labels']
n_samples = labels.shape[0]


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

from utils.parse_args import parse_args
    
args = parse_args()

# beta = args.beta
beta=0.01
print(f'beta = {beta}')


pts_dir = 'pts/affect/mosi/lrtf/'  + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')


# epoch_num = args.epoch
epoch_num = 100
print(f'epoch = {epoch_num}')

tolerance = args.tolerance
print(f'Tolerance of early stop = {tolerance}')
# run origin(default True)

robust_test=True
noise_level = args.noise

task='classification'

train_dataloader, valid_dataloader, test_dataloader  = get_noise_dataloader(
    'data/affect/mosi/mosi_raw.pkl', robust_test=True, data_type='mosi'
    , raw_path="data/affect/mosi/mosi.hdf5",  full_data=full_data, i=noise_level, device=device, task=task)


noise_level = args.noise

fold = 'fullnoise'
    
if True:
  if True:
      encoders_mixer = [GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(74, 128, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(300, 512, 128, dropout=True, has_padding=True).to(device)]
      head_mixer = MLP(128, 512, 2).to(device)

      fusion_mixer = LowRankTensorFusion([32, 32, 128], 128, 32, device=device).to(device)
      
      print(f"--------------Mixer {fold} starts:--------------")
      
      save_as_m = pts_dir + 'lrtf_mixer_' + fold + str(noise_level) + '_' + str(epoch_num) +'.pt'
      
      best_m = train_mixer(encoders_mixer, fusion_mixer, head_mixer, train_dataloader, valid_dataloader, epoch_num, task="classification", optimtype=torch.optim.AdamW,
        early_stop=True, is_packed=True, lr=1e-3, save=save_as_m, weight_decay=0.01, device=device, beta=beta, tolerance=tolerance)
      


      encoders = [GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(74, 128, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(300, 512, 128, dropout=True, has_padding=True).to(device)]
      head = MLP(128, 512, 2).to(device)

      fusion = LowRankTensorFusion([32, 32, 128], 128, 32, device=device).to(device)
      
      print(f"--------------Origin {fold} starts:--------------")
      
      save_as_o = pts_dir + 'lrtf_origin_' + fold + str(noise_level) + '_' + str(epoch_num) +'.pt'
      
      best_o = train(encoders, fusion, head, train_dataloader, valid_dataloader, epoch_num, task="classification", optimtype=torch.optim.AdamW,
        early_stop=True, is_packed=True, lr=1e-3, save=save_as_o, weight_decay=0.01, device=device)



print("Testing Mixer:")    
model_mixer = torch.load(save_as_m).cuda()  
test(model=model_mixer, test_dataloaders_all=test_dataloader,
     dataset='mosi', is_packed=True, no_robust=True)


print("Testing Origin:")
model_origin = torch.load(save_as_o).cuda() 
test(model=model_origin, test_dataloaders_all=test_dataloader,
     dataset='mosi', is_packed=True, no_robust=True)


print_current_time()