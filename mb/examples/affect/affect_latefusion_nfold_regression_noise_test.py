import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.Supervised_Learning import train, train_mixer, test
from unimodals.common_models import GRU, MLP

from fusions.common_fusions import Concat

import torch
from datasets.affect.get_data import get_noise_dataloader,  get_fulldata


import numpy as np
from sklearn.model_selection import KFold

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from mwae.util import setup_seed, print_current_time


setup_seed()
print_current_time()


timestep = 50



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

# run origin(default True)
origin = args.origin

# run mixer(default True)
mixer = args.mixer

pts_dir = 'pts/affect/mosi/latefusion/'  + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

task = 'regression'
print(f'Task: {task} of total epoch_num = {epoch_num}')

n_samples = 2183
print(f'n_samples = {n_samples}')




full_data = get_fulldata(
    'data/affect/mosi/mosi_raw.pkl', robust_test=False, data_type='mosi',)
labels = full_data['labels']
n_samples = labels.shape[0]

robust_test=True
noise_level = args.noise

train_dataloader, valid_dataloader, test_dataloader  = get_noise_dataloader(
    'data/affect/mosi/mosi_raw.pkl', robust_test=True, data_type='mosi'
    , raw_path="data/affect/mosi/mosi.hdf5", full_data=full_data, i=noise_level, task=task, device=device)


'''


train len = 1283

valid len = 214

test len = 686

ratio:
1283/2183
0.5877233165368758
214/2183
0.09803023362345396
686/2183
0.3142464498396702

type(test_robust)
<class 'dict'>

test_robust.keys()
dict_keys(['robust_text', 'robust_vision', 'robust_audio', 'robust_timeseries'])

len(test_robust['robust_text'])
10

type(test_robust['robust_vision'][0])
<class 'torch.utils.data.dataloader.DataLoader'>

len(test_robust['robust_vision'][0].dataset)
686



'''

test_robust = True

# noise_dataset = test_robust[]
fold = 'fullnoise'

    
if True:
    if True:
        encoders_mixer = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).to(device),
                    GRU(74, 200, dropout=True, has_padding=True, batch_first=True).to(device),
                    GRU(300, 600, dropout=True, has_padding=True, batch_first=True).to(device)]
        head_mixer = MLP(870, 870, 1).to(device)

        fusion_mixer = Concat().to(device)
        
        print(f"--------------Mixer {fold} starts:--------------")
        save_as_m = pts_dir + 'mosi_latefusion_regression_mixer_' + fold + str(noise_level) +'.pt'
        best_m = train_mixer(encoders_mixer, fusion_mixer, head_mixer, train_dataloader, valid_dataloader, epoch_num, task=task, optimtype=torch.optim.AdamW,
        early_stop=False, is_packed=True, lr=1e-3, save=save_as_m, weight_decay=0.01, device=device, objective=torch.nn.L1Loss(), beta=beta, tolerance=tolerance)
        

      
    if True:
        print(f'--------------ORIGINAL {fold} starts:-------------')
        encoders = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).to(device),
                    GRU(74, 200, dropout=True, has_padding=True, batch_first=True).to(device),
                    GRU(300, 600, dropout=True, has_padding=True, batch_first=True).to(device)]
        head = MLP(870, 870, 1).to(device)

        fusion = Concat().to(device)
        
        save_as_o = pts_dir + 'mosi_latefusion_regression_origin_' + fold + str(noise_level) + '.pt'
        
        best_o = train(encoders, fusion, head, train_dataloader, valid_dataloader, epoch_num, task=task, optimtype=torch.optim.AdamW,
        early_stop=False, is_packed=True, lr=1e-3, save=save_as_o, weight_decay=0.01, device=device, objective=torch.nn.L1Loss(), tolerance=tolerance)
    

    
        print("Testing Mixer:")
        model_mixer = torch.load(save_as_m).cuda()
        test(model=model_mixer, test_dataloaders_all=test_dataloader, dataset='mosi', is_packed=True,
            criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)

        
        print("Testing Origin:")
        model_origin = torch.load(save_as_o).cuda()
        test(model=model_origin, test_dataloaders_all=test_dataloader, dataset='mosi', is_packed=True,
            criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)
        
