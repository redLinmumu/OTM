import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader_nfold
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train_mixer, train
 
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
bests_m = []
bests_o = []


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

pts_dir = 'pts/avmnist/' + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)



for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"[Fold] {fold}:")
    
    traindata, validdata = get_dataloader_nfold(data_dir=data_dir, train_index=train_index, valid_index=valid_index)

    # mixer model
    encoders_mixer = [LeNet(1, channels, 3).to(device), LeNet(1, channels, 5).to(device)]
    head_mixer = MLP(channels*40, 100, 10).to(device)
    fusion_mixer = Concat().to(device)
    
    save_m = pts_dir + 'slatefusion_mixer_' + str(fold)+ '.pt'
    print(f"-------------------Mixer fold {fold} starts-------------------")
    best_m = train_mixer(encoders_mixer, fusion_mixer, head_mixer, traindata, validdata, epoch_num,
      optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001
      , track_complexity=False, save=save_m, device=device, beta=beta, tolerance=tolerance)
    
    bests_m.append(best_m.item())
    print(f"-------------------Mixer fold {fold} ends-------------------")
    # origin model
    encoders = [LeNet(1, channels, 3).to(device), LeNet(1, channels, 5).to(device)]
    head = MLP(channels*40, 100, 10).to(device)
    fusion = Concat().to(device)

    save_o = pts_dir + 'slatefusion_origin_' + str(fold)+ '.pt'
    print(f"-------------------Origin fold {fold} starts-------------------")
    best_o = train(encoders, fusion, head, traindata, validdata, epoch_num,
      optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001
      , track_complexity=False, save=save_o, device=device, tolerance=tolerance)
    
    bests_o.append(best_o.item())
    print(f"-------------------Origin fold {fold} ends-------------------")


print(f'[Average] Mixer = {sum(bests_m)/5.0}, \n\t All = {bests_m}')  
print(f'[Average] Original = {sum(bests_o)/5.0}, \n\t All = {bests_o}')  
   
print_current_time()