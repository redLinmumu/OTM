import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


from training_structures.Supervised_Learning import train, train_mixer
from unimodals.common_models import GRUWithLinear, MLP  # noqa

from fusions.common_fusions import  TensorFusion  # noqa

from datasets.affect.get_data import get_fulldata, get_fold_dataloader
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


    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"[Fold] {fold}")
       
    train_data = {key: value[train_index] for key, value in full_data.items()}
    valid_data = {key: value[valid_index] for key, value in full_data.items()}

    train_dataloaer, valid_dataloader = get_fold_dataloader(train_data=train_data, valid_data=valid_data, robust_test=False, task='regression')
   
   
    encoders_mixer = [GRUWithLinear(35, 64, 4, dropout=True, has_padding=True).to(device),
                GRUWithLinear(74, 128, 19, dropout=True, has_padding=True).to(device),
                GRUWithLinear(300, 512, 79, dropout=True, has_padding=True).to(device)]
    head_mixer = MLP(8000, 512, 1).to(device)

    fusion_mixer = TensorFusion().to(device)
    
    print(f"--------------Mixer {fold} starts:--------------")
    save_m = pts_dir + 'mosi_tf_mixer_regression' + str(fold) + '.pt'
    
    
    best_m = train_mixer(encoders_mixer, fusion_mixer, head_mixer, train_dataloaer, valid_dataloader, epoch_num, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save=save_m, weight_decay=0.01, device=device, objective=torch.nn.L1Loss(), beta=beta, tolerance=tolerance)
    
    bests_m.append(best_m.item())
    print(f"--------------Mixer {fold} starts:--------------")
   
   
    encoders = [GRUWithLinear(35, 64, 4, dropout=True, has_padding=True).to(device),
                GRUWithLinear(74, 128, 19, dropout=True, has_padding=True).to(device),
                GRUWithLinear(300, 512, 79, dropout=True, has_padding=True).to(device)]
    head = MLP(8000, 512, 1).to(device)

    fusion = TensorFusion().to(device)
    
    print(f"--------------Origin {fold} starts:--------------")
    
    save_o = pts_dir + 'mosi_tf_origin_regression' + str(fold) + '.pt'
    
    best_o = train(encoders, fusion, head, train_dataloaer, valid_dataloader, epoch_num, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save=save_o, weight_decay=0.01, device=device, objective=torch.nn.L1Loss(), tolerance=tolerance)
    
    bests_o.append(best_o.item())
    print(f"--------------Origin {fold} starts:--------------")


print(f'[Average] Mixer = {sum(bests_m)/5.0}, \n\t All = {bests_m}')  
print(f'[Average] Original = {sum(bests_o)/5.0}, \n\t All = {bests_o}')  


print_current_time()