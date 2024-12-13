import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import copy
from training_structures.Supervised_Learning import train_head_single_modality, MMDL, SingleMDL
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

# beta = args.beta

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

epoch_num = args.epoch
print(f'epoch = {epoch_num}')

tolerance = args.tolerance
print(f'Tolerance of early stop = {tolerance}')

single_dir = 'best_pts/mosi/regression/tensor_fusion/single/'
# os.makedirs(pts_dir, exist_ok=True)

is_packed=True

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

view_num = 3
bests_list_mixer = [ [] for i in range(view_num) ]
bests_list_origin = [ [] for i in range(view_num) ]

# beta_mixer = 10
beta_mixer = 10.0
print(f'beta_mixer = {beta_mixer}')

beta_single = 1.0

print(f"beta mixer = {beta_mixer}, beta single = {beta_single}")

encoder_dims = [ [35, 64, 4], [74, 128, 19], [300, 512, 79] ]

pts_base = 'best_pts/mosi/regression/tensor_fusion/' + str(beta_mixer) + '/'

task="regression"

    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"Begins -------------------Single Modality  [Fold] {fold}:")
       
    train_data = {key: value[train_index] for key, value in full_data.items()}
    valid_data = {key: value[valid_index] for key, value in full_data.items()}

    train_dataloaer, valid_dataloader = get_fold_dataloader(train_data=train_data, valid_data=valid_data, robust_test=False, task='regression')

    #   loadmixer  
    encoders_mixer = [GRUWithLinear(35, 64, 4, dropout=True, has_padding=True).to(device),
                GRUWithLinear(74, 128, 19, dropout=True, has_padding=True).to(device),
                GRUWithLinear(300, 512, 79, dropout=True, has_padding=True).to(device)]
    head_mixer = MLP(8000, 512, 1).to(device)

    fusion_mixer = TensorFusion().to(device)
    
    mixer_file_path = pts_base + 'mosi_tf_mixer_regression' + str(fold) + '.pt'
    
    print(f"Load mixer model = {mixer_file_path}")
    
    model_mixer = MMDL(encoders_mixer, fusion_mixer, head_mixer, has_padding=is_packed).to(device)
    model_mixer = torch.load(mixer_file_path, map_location=device)
    model_mixer.to(device)

  #    loadorigin  
    encoders_origin = [GRUWithLinear(35, 64, 4, dropout=True, has_padding=True).to(device),
                GRUWithLinear(74, 128, 19, dropout=True, has_padding=True).to(device),
                GRUWithLinear(300, 512, 79, dropout=True, has_padding=True).to(device)]
    head_origin = MLP(8000, 512, 1).to(device)

    fusion_origin = TensorFusion().to(device)
    
    
    origin_file_path = pts_base + 'mosi_tf_origin_regression' + str(fold) + '.pt'
    print(f"Load orign model = {origin_file_path}")
    
    model_origin = MMDL(encoders_origin, fusion_origin, head_origin, has_padding=is_packed).to(device)
    model_origin = torch.load(origin_file_path, map_location=device)
    model_origin.to(device)
       
    for i in range(view_num):

      # encoder = GRUWithLinear(encoder_dims[i][0], encoder_dims[i][1], encoder_dims[i][2], dropout=True, has_padding=True).to(device)
      # head = MLP(encoder_dims[i][2], encoder_dims[i][2], 1).to(device)
      
      # single_model = SingleMDL(encoder, head=head, index=i, has_padding=is_packed).to(device)
      # single_model_dir = single_dir + "single_view_" + str(i) + "_fold_" + str(fold) + ".pt"
      # single_model = torch.load(single_model_dir)
      
      print(f"*** Modality {i} == > fold {fold}:")
      
      # ------mixer valid--------
      # mixer_single_model = copy.deepcopy(single_model)
      # mixer_single_model.encoder = model_mixer.encoders[i]
      encoder_mixer = copy.deepcopy(model_mixer.encoders[i])
      head_mixer = MLP(encoder_dims[i][2], encoder_dims[i][2], 1).to(device)
      
      best_mixer = train_head_single_modality(encoder_mixer, i, head_mixer, train_dataloaer, valid_dataloader, epoch_num, task="regression",
        optimtype=torch.optim.AdamW,  early_stop=False, is_packed=True, lr=1e-3, track_complexity=False, 
        save=single_dir, weight_decay=0.01, device=device, objective=torch.nn.L1Loss(), tolerance=tolerance, fold_index=fold, type="mixer")
        
      bests_list_mixer[i].append(best_mixer.item())
        
        # ------origin valid---------
      # origin_single_model = copy.deepcopy(single_model)
      # origin_single_model.encoder = model_origin.encoders[i]
      
      encoder_origin = copy.deepcopy(model_origin.encoders[i])
      head_origin = MLP(encoder_dims[i][2], encoder_dims[i][2], 1).to(device)
      
      best_origin = train_head_single_modality(encoder_origin, i, head_origin, train_dataloaer, valid_dataloader, epoch_num, task="regression",
        optimtype=torch.optim.AdamW,  early_stop=False, is_packed=True, lr=1e-3, track_complexity=False, 
        save=single_dir, weight_decay=0.01, device=device, objective=torch.nn.L1Loss(), tolerance=tolerance, fold_index=fold, type="origin")

      bests_list_origin[i].append(best_origin.item())
    
    # print(f"------------------Fold {fold} ends.-------------------")


for i in range(view_num):
  print(f"Results for modality {i}: ")
  print(f'[Average] Mixer = {sum(bests_list_mixer[i])/5.0}, total = {bests_list_mixer[i]}')    
  print(f'[Average] Origin = {sum(bests_list_origin[i])/5.0}, total = {bests_list_origin[i]}')   
print_current_time()