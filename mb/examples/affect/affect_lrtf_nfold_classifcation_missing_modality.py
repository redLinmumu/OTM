import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


from private_test_scripts.all_in_one import all_in_one_train # noqa
from training_structures.Supervised_Learning import train_head_single_modality, MMDL, SingleMDL
from unimodals.common_models import GRUWithLinear, MLP # noqa

from fusions.common_fusions import  LowRankTensorFusion # noqa

from datasets.affect.get_data import get_fulldata, get_fold_dataloader
from sklearn.model_selection import KFold

import numpy as np
from sklearn.model_selection import KFold
from mwae.util import setup_seed, print_current_time
import copy

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

beta = 0.01
print(f'beta = {beta}')


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

view_num = 3
bests_list_mixer = [ [] for i in range(view_num) ]
bests_list_origin = [ [] for i in range(view_num) ]

mixer_dir = 'best_pts/mosi/classification/lrtf/' + str(beta) + '/'

single_dir = 'best_pts/mosi/classification/lrtf/single/'


encoder_dims = [ [35, 64, 32], [74, 128, 32], [300, 512, 128] ] 
head_dims = [ [32, 128], [32, 128], [128, 512] ]

early_stop=True

is_packed=True

task="classification"

    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"Begins -------------------Single Modality  [Fold] {fold}:")
    
     
    train_data = {key: value[train_index] for key, value in full_data.items()}
    valid_data = {key: value[valid_index] for key, value in full_data.items()}

    train_dataloaer, valid_dataloader = get_fold_dataloader(train_data=train_data, valid_data=valid_data, robust_test=False, task='classification')
    
    encoders_mixer = [GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(74, 128, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(300, 512, 128, dropout=True, has_padding=True).to(device)]
    
    head_mixer = MLP(128, 512, 2).to(device)

    fusion_mixer = LowRankTensorFusion([32, 32, 128], 128, 32, device=device).to(device)
    print(f"--------------Mixer {fold} starts:--------------")
      
    mixer_file_path = mixer_dir + 'lrtf_mixer_' + str(fold)+ '.pt'
    print(f"Load mixer model = {mixer_file_path}")
    
    model_mixer = MMDL(encoders_mixer, fusion_mixer, head_mixer, has_padding=is_packed).to(device)
    model_mixer = torch.load(mixer_file_path, map_location=device)
    model_mixer.to(device)
    
    
    # origin
    encoders_origin = [GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(74, 128, 32, dropout=True, has_padding=True).to(device),
              GRUWithLinear(300, 512, 128, dropout=True, has_padding=True).to(device)]
    head_origin = MLP(128, 512, 2).to(device)

    fusion_origin = LowRankTensorFusion([32, 32, 128], 128, 32, device=device).to(device)
    
    origin_file_path = mixer_dir + 'lrtf_origin_' + str(fold)+ '.pt'
    
    print(f"Load orign model = {origin_file_path}")
    
    model_origin = MMDL(encoders_origin, fusion_origin, head_origin, has_padding=is_packed).to(device)
    model_origin = torch.load(origin_file_path, map_location=device)
    model_origin.to(device)
    
    
    for i in range(view_num):
      
      # encoder = GRUWithLinear(encoder_dims[i][0], encoder_dims[i][1], encoder_dims[i][2], dropout=True, has_padding=True).to(device)
      
      # head = MLP(head_dims[i][0], head_dims[i][1], 2).to(device)
      
      # single_model = SingleMDL(encoder, head=head, index=i, has_padding=is_packed).to(device)
      # single_model_dir = single_dir + "single_view_" + str(i) + "_fold_" + str(fold) + ".pt"
      # single_model = torch.load(single_model_dir)
        
      print(f"*** Modality {i} == > fold {fold}:")

      # ------mixer valid--------
      # mixer_single_model = copy.deepcopy(single_model)
      # mixer_single_model.encoder = model_mixer.encoders[i]
      
      encoder_mixer = copy.deepcopy(model_mixer.encoders[i])
      head_mixer = MLP(head_dims[i][0], head_dims[i][1], 2).to(device)
      
            
      best_mixer = train_head_single_modality(encoder_mixer, i, head_mixer, train_dataloaer, valid_dataloader, epoch_num, task="classification",
        optimtype=torch.optim.AdamW,  early_stop=True, is_packed=True, lr=1e-3, track_complexity=False, 
        save=single_dir, weight_decay=0.01, device=device, fold_index=fold, type="mixer")
      
      print(f"Mixer valid result = {best_mixer.item()}")
        
      bests_list_mixer[i].append(best_mixer.item())
            
        # ------origin valid---------
      # origin_single_model = copy.deepcopy(single_model)
      # origin_single_model.encoder = model_origin.encoders[i]
      encoder_origin = copy.deepcopy(model_origin.encoders[i])
      head_origin = MLP(head_dims[i][0], head_dims[i][1], 2).to(device)
      
      best_origin = train_head_single_modality(encoder_origin, i, head_origin, train_dataloaer, valid_dataloader, epoch_num, task="classification",
        optimtype=torch.optim.AdamW,  early_stop=True, is_packed=True, lr=1e-3, track_complexity=False, 
        save=single_dir, weight_decay=0.01, device=device, fold_index=fold, type="origin")

      print(f"Origin valid result = {best_origin.item()}")
      
      bests_list_origin[i].append(best_origin.item())
    
    # print(f"------------------Fold {fold} ends.-------------------")


for i in range(view_num):
  print(f"Results for modality {i}: ")
  print(f'[Average] Mixer = {sum(bests_list_mixer[i])/5.0}, total = {bests_list_mixer[i]}')    
  print(f'[Average] Origin = {sum(bests_list_origin[i])/5.0}, total = {bests_list_origin[i]}')   
print_current_time()
