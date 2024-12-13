import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.Supervised_Learning import valid_single_modality, MMDL, SingleMDL, train_head_single_modality
from unimodals.common_models import GRU, MLP

from fusions.common_fusions import Concat
import torch
from datasets.affect.get_data import get_fulldata, get_fold_dataloader
from sklearn.model_selection import KFold

import numpy as np
from sklearn.model_selection import KFold
from mwae.util import setup_seed, print_current_time

import copy

setup_seed()
print_current_time()


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

timestep = 50
full_data = get_fulldata(
    'data/affect/mosi/mosi_raw.pkl', robust_test=False, data_type='mosi')
labels = full_data['labels']
n_samples = labels.shape[0]

epoch_num = 100
print(f'epoch_num = {epoch_num}')

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

task="classification"

is_packed=True

beta = 100.0
print(f'beta = {beta}')

mixer_dir = 'best_pts/mosi/classification/late_fusion/' + str(beta) + '/'

single_dir = 'best_pts/mosi/classification/late_fusion/single/'

# os.makedirs(pts_dir, exist_ok=True)

view_num = 3
bests_train_head_mixer_list = [ [] for i in range(view_num) ]
bests_train_head_origin_list = [ [] for i in range(view_num) ]

bests_list_mixer = [ [] for i in range(view_num) ]
bests_list_origin = [ [] for i in range(view_num) ]

encoders_paras = [ [35, 70], [74, 200], [300, 600] ]

    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"Begins -------------------Single Modality  [Fold] {fold}:")
   
    train_data = {key: value[train_index] for key, value in full_data.items()}
    valid_data = {key: value[valid_index] for key, value in full_data.items()}

    train_dataloaer, valid_dataloader = get_fold_dataloader(train_data=train_data, valid_data=valid_data, robust_test=False, data_type='mosi', task=task)
    
      
    encoders_mixer = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(74, 200, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(300, 600, dropout=True, has_padding=True, batch_first=True).to(device)]
    head_mixer = MLP(870, 870, 2).to(device)

    fusion_mixer = Concat().to(device)
    
    mixer_file_path = mixer_dir + 'mosi_latefusion_classify_mixer_' + str(fold) + '.pt'
    print(f"Load mixer model = {mixer_file_path}")
    
    model_mixer = MMDL(encoders_mixer, fusion_mixer, head_mixer, has_padding=is_packed).to(device)
    model_mixer = torch.load(mixer_file_path, map_location=device)
    model_mixer.to(device)
    
    
    encoders_origin = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(74, 200, dropout=True, has_padding=True, batch_first=True).to(device),
                  GRU(300, 600, dropout=True, has_padding=True, batch_first=True).to(device)]
    head_origin = MLP(870, 870, 2).to(device)

    fusion_origin = Concat().to(device)
    
    origin_file_path = mixer_dir + 'mosi_latefusion_classify_origin_' + str(fold) + '.pt'
    print(f"Load orign model = {origin_file_path}")
    
    model_origin = MMDL(encoders_origin, fusion_origin, head_origin, has_padding=is_packed).to(device)
    model_origin = torch.load(origin_file_path, map_location=device)
    model_origin.to(device)
    
    
    for i in range(view_num):
        
        # encoder = GRU(encoders_paras[i][0], encoders_paras[i][1], dropout=True, has_padding=True, batch_first=True).to(device)
        # head = MLP(encoders_paras[i][1], 870, 2).to(device)
        
        # single_model = SingleMDL(encoder, head=head, index=i, has_padding=is_packed).to(device)
        # single_model_dir = single_dir + "single_view_" + str(i) + "_fold_" + str(fold) + ".pt"
        # single_model = torch.load(single_model_dir)
        
        print(f"*** Modality {i} == > fold {fold}:")

      # ------mixer valid--------
        # mixer_single_model = copy.deepcopy(single_model)
        # mixer_single_model.encoder = model_mixer.encoders[i]
        
        mixer_encoder_i =  copy.deepcopy(model_mixer.encoders[i])
        
        head_mixer_i = MLP(encoders_paras[i][1], 870, 2).to(device)
        
        best_o = train_head_single_modality(mixer_encoder_i, i, head_mixer_i, train_dataloaer, valid_dataloader, epoch_num, task=task,
        optimtype=torch.optim.AdamW, lr=1e-3, early_stop=False, is_packed=True, track_complexity=False, 
        save=single_dir, weight_decay=0.01, device=device, fold_index=fold, type='mixer')
        
        bests_list_mixer[i].append(best_o.item())
        # single_model_mixer = SingleMDL(mixer_encoder_i, head=head_mixer_i, index=i, has_padding=is_packed).to(device)
        
        print(f"Mixer Modality  result = {best_o}")
        # bests_train_head_mixer_list[i].append(best_o.item())
        
        # best_mixer = valid_single_modality(single_model_mixer, i,  valid_dataloader, is_packed=is_packed, 
        #     task=task, device=device, beta=beta, fold_index=fold)
        
        # print(f"Mixer Modality valid result = {best_mixer}")    
        
        # bests_list_mixer[i].append(best_mixer.item())
            
        # ------origin valid---------
        # origin_single_model = copy.deepcopy(single_model)
        # origin_single_model.encoder = model_origin.encoders[i]

        origin_encoder_i = copy.deepcopy(model_origin.encoders[i])
        
        head_origin_i = MLP(encoders_paras[i][1], 870, 2).to(device)
        
        best_o = train_head_single_modality(origin_encoder_i, i, head_origin_i, train_dataloaer, valid_dataloader, epoch_num, task=task,
        optimtype=torch.optim.AdamW, lr=1e-3, early_stop=False, is_packed=True, track_complexity=False, 
        save=single_dir, weight_decay=0.01, device=device, fold_index=fold, type='origin')
        
        # single_model_origin = SingleMDL(origin_encoder_i, head=head_origin_i, index=i, has_padding=is_packed).to(device)
        
        print(f"Origin Modality result = {best_o}")
        # bests_train_head_origin_list[i].append(best_o.item())
        
        # best_origin = valid_single_modality(single_model_origin, i,  valid_dataloader, is_packed=is_packed, 
        #     task=task,  device=device, beta=beta, fold_index=fold)
        
        # print(f"Origin Modality valid result = {best_origin}")    
        
        bests_list_origin[i].append(best_o.item())
    
    # print(f"------------------Fold {fold} ends.-------------------")


for i in range(view_num):
  print(f"Valid Results for modality {i}: ")
  print(f'[Average] Mixer = {sum(bests_list_mixer[i])/5.0}, total = {bests_list_mixer[i]}')    
  print(f'[Average] Origin = {sum(bests_list_origin[i])/5.0}, total = {bests_list_origin[i]}')   
print_current_time()
