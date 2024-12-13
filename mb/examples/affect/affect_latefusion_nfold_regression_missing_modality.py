import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.Supervised_Learning import train_head_single_modality, MMDL, SingleMDL
from unimodals.common_models import GRU, MLP
from fusions.common_fusions import Concat
import copy

import torch
from datasets.affect.get_data import get_fulldata, get_fold_dataloader


import numpy as np
from sklearn.model_selection import KFold

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from mwae.util import setup_seed, print_current_time

setup_seed()
print_current_time()


timestep = 50
full_data = get_fulldata(
    'data/affect/mosi/mosi_raw.pkl', robust_test=False, data_type='mosi')
labels = full_data['labels']
n_samples = labels.shape[0]


from utils.parse_args import parse_args
    
args = parse_args()


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


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

view_num = 3
bests_list_mixer = [ [] for i in range(view_num) ]
bests_list_origin = [ [] for i in range(view_num) ]


encoders_paras = [ [35, 70], [74, 200], [300, 600] ]

task = 'regression'
print(f'Task: {task} of total epoch_num = {epoch_num}')


beta_mixer = 100.0

# single is for the training single modality by itsefl without multimodal fusions.

beta_single = 1.0

print(f"mixer beta = {beta_mixer}, origin beta = {beta_single}")

is_packed=True
print(f"is_packed = {is_packed}")

mixer_dir_base = 'best_pts/mosi/regression/late_fusion/'  + str(beta_mixer) + '/'

mixer_base = mixer_dir_base + 'mosi_latefusion_regression_mixer_'
origin_base = mixer_dir_base + "mosi_latefusion_regression_origin_"


single_dir = 'best_pts/mosi/regression/late_fusion/single/'
is_packed=True


    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    
    print(f"Begins -------------------Single Modality  [Fold] {fold}:")
       
    train_data = {key: value[train_index] for key, value in full_data.items()}
    valid_data = {key: value[valid_index] for key, value in full_data.items()}

    train_dataloaer, valid_dataloader = get_fold_dataloader(train_data=train_data, valid_data=valid_data, robust_test=False, data_type='mosi', task=task, device=device)
    
    
    # ------------------------load mixer model--------------------------
    encoders_mixer = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).to(device),
                    GRU(74, 200, dropout=True, has_padding=True, batch_first=True).to(device),
                    GRU(300, 600, dropout=True, has_padding=True, batch_first=True).to(device)]
    head_mixer = MLP(870, 870, 1).to(device)
    fusion_mixer = Concat().to(device)
    
    mixer_file_path = mixer_base + str(fold) + '.pt'
    print(f"Load mixer model = {mixer_file_path}")
    
    model_mixer = MMDL(encoders_mixer, fusion_mixer, head_mixer, has_padding=is_packed).to(device)
    model_mixer = torch.load(mixer_file_path, map_location=device)
    model_mixer.to(device)
    # ------------------------load origin model-------------------------
    encoders_origin = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).to(device),
                    GRU(74, 200, dropout=True, has_padding=True, batch_first=True).to(device),
                    GRU(300, 600, dropout=True, has_padding=True, batch_first=True).to(device)]
    head_origin = MLP(870, 870, 1).to(device)
    fusion_origin = Concat().to(device)
    
    origin_file_path = origin_base + str(fold) + '.pt'
    print(f"Load orign model = {origin_file_path}")
    
    model_origin = MMDL(encoders_origin, fusion_origin, head_origin, has_padding=is_packed).to(device)
    model_origin = torch.load(origin_file_path, map_location=device)
    model_origin.to(device)
    #  ----------------------------------------
    
    for i in range(view_num):
        print(f"*** Eval Modality {i} == > fold {fold}:")
        
        # print("Load single modality:")
        # encoder = GRU(encoders_paras[i][0], encoders_paras[i][1], dropout=True, has_padding=True, batch_first=True).to(device)
        # head = MLP(encoders_paras[i][1], 870, 2).to(device)
        
        # single_model = SingleMDL(encoder, head=head, index=i, has_padding=is_packed).to(device)
        # single_model_dir = single_dir + "single_view_" + str(i) + "_fold_" + str(fold) + ".pt"
        # single_model = torch.load(single_model_dir)
        
        
        # ------mixer valid--------
        # mixer_single_model = copy.deepcopy(single_model)
        # mixer_single_model.encoder = model_mixer.encoders[i]
        encoder_mixer = copy.deepcopy(model_mixer.encoders[i])
        head_mixer = MLP(encoders_paras[i][1], 870, 1).to(device)
        
        # best_mixer = valid_single_modality(mixer_single_model, i,  valid_dataloader, is_packed=is_packed, 
        # task=task, objective=torch.nn.L1Loss(),  objective_args_dict=None, 
        # device=device, beta=beta, fold_index=fold)
        best_o = train_head_single_modality(encoder_mixer, i, head_mixer, train_dataloaer, valid_dataloader, epoch_num, task=task,
        optimtype=torch.optim.AdamW,  early_stop=False, is_packed=True, lr=1e-3, track_complexity=False, objective=torch.nn.L1Loss(),
        tolerance=tolerance, save=single_dir, weight_decay=0.01, device=device, fold_index=fold, type="mixer")
        
        print(f"Mixer valid result = {best_o.item()}")
        
        bests_list_mixer[i].append(best_o.item())
        
        # ------origin valid---------
        # origin_single_model = copy.deepcopy(single_model)
        # origin_single_model.encoder = model_origin.encoders[i]
        encoder_origin = copy.deepcopy(model_origin.encoders[i])
        head_origin = MLP(encoders_paras[i][1], 870, 1).to(device)
        
        best_origin = train_head_single_modality(encoder_origin, i, head_origin, train_dataloaer, valid_dataloader, epoch_num, task=task,
        optimtype=torch.optim.AdamW,  early_stop=False, is_packed=True, lr=1e-3, track_complexity=False, objective=torch.nn.L1Loss(),
        tolerance=tolerance, save=single_dir, weight_decay=0.01, device=device, fold_index=fold, type="origin")

        bests_list_origin[i].append(best_origin.item())
    
    # print(f"------------------Fold {fold} ends.-------------------")

print("---------Start Eval-----------")
for i in range(view_num):
  print(f"Results for modality {i}: ")
  print(f'[Average] Mixer = {sum(bests_list_mixer[i])/5.0}, total = {bests_list_mixer[i]}')    
  print(f'[Average] Origin = {sum(bests_list_origin[i])/5.0}, total = {bests_list_origin[i]}')   
print_current_time()
