import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


from training_structures.Supervised_Learning import train, train_mixer, test
from unimodals.common_models import GRUWithLinear, MLP  # noqa

from fusions.common_fusions import  TensorFusion  # noqa

from datasets.affect.get_data import get_fulldata, get_noise_dataloader
from sklearn.model_selection import KFold

import numpy as np
from sklearn.model_selection import KFold
from mwae.util import setup_seed, print_current_time

setup_seed()
print_current_time()


full_data = get_fulldata(
    'data/affect/mosi/mosi_raw.pkl', robust_test=False, data_type='mosi', task='regression')
labels = full_data['labels']
n_samples = labels.shape[0]


from utils.parse_args import parse_args
    
args = parse_args()

beta = args.beta
print(f'beta = {beta}')


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

epoch_num = args.epoch
print(f'epoch = {epoch_num}')

tolerance = args.tolerance
print(f'Tolerance of early stop = {tolerance}')

pts_dir = 'pts/affect/mosi/' + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)

bests_m = []
bests_o = []

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

robust_test=True

fold = 'fullnoise'
noise_level = args.noise

train_dataloader, valid_dataloader, test_dataloader  = get_noise_dataloader(
    'data/affect/mosi/mosi_raw.pkl', robust_test=True, data_type='mosi'
    , raw_path="data/affect/mosi/mosi.hdf5", full_data=full_data, i=noise_level, device=device, task='regression')



if True:
   
    encoders_mixer = [GRUWithLinear(35, 64, 4, dropout=True, has_padding=True).to(device),
                GRUWithLinear(74, 128, 19, dropout=True, has_padding=True).to(device),
                GRUWithLinear(300, 512, 79, dropout=True, has_padding=True).to(device)]
    head_mixer = MLP(8000, 512, 1).to(device)

    fusion_mixer = TensorFusion().to(device)
    
    print(f"--------------Mixer {fold} starts:--------------")
    
    save_as_m = pts_dir + 'mosi_tf_mixer_regression' + fold + str(noise_level) + '_' + str(epoch_num) +'.pt'
    
    best_m = train_mixer(encoders_mixer, fusion_mixer, head_mixer, train_dataloader, valid_dataloader, epoch_num, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save=save_as_m, weight_decay=0.01, device=device, objective=torch.nn.L1Loss(), beta=beta, tolerance=tolerance)
    

 
    print(f"--------------Origin {fold} starts:--------------")
   
    encoders = [GRUWithLinear(35, 64, 4, dropout=True, has_padding=True).to(device),
                GRUWithLinear(74, 128, 19, dropout=True, has_padding=True).to(device),
                GRUWithLinear(300, 512, 79, dropout=True, has_padding=True).to(device)]
    head = MLP(8000, 512, 1).to(device)

    fusion = TensorFusion().to(device)
    
    save_as_o = pts_dir + 'mosi_tf_origin_regression' + fold + str(noise_level) + '_' + str(epoch_num) +'.pt'
        
    best_o = train(encoders, fusion, head, train_dataloader, valid_dataloader, epoch_num, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save=save_as_o, weight_decay=0.01, device=device, objective=torch.nn.L1Loss(), tolerance=tolerance)
    
    
    
    
    print("Testing Mixer:")
    model_mixer = torch.load(save_as_m).cuda()
    test(model=model_mixer, test_dataloaders_all=test_dataloader, dataset='mosi',
     is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)
 
    
    
    print("Testing Origin:")
    model_origin = torch.load(save_as_o).cuda()

    test(model=model_origin, test_dataloaders_all=test_dataloader, dataset='mosi',
      is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)


print_current_time()