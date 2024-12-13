
import sys
import os
from torch import nn
sys.path.append(os.getcwd())

from unimodals.common_models import  Linear,  VGG11Slim
import torch # noqa

from datasets.enrico.get_data import get_dataloader_nfold, EnricoDatasetMixer # noqa
from fusions.common_fusions import MultiplicativeInteractions2Modal

from training_structures.Supervised_Learning import train_head_single_modality, MMDL, SingleMDL  # noqa
from sklearn.model_selection import KFold
import csv
import numpy as np
import copy

from mwae.util import setup_seed, print_current_time

setup_seed()
print_current_time()

    
n_splits = 5       
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)


data_dir = 'datasets/enrico/dataset'

csv_file = os.path.join(data_dir, "design_topics.csv")
with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    example_list = list(reader)

# the wireframe files are corrupted for these files
IGNORES = set(["50105", "50109"])
example_list = [
    e for e in example_list if e['screen_id'] not in IGNORES]
n_samples = len(example_list)

from utils.parse_args import parse_args
    
args = parse_args()

beta = 0.1 
print(f'beta = {beta}')

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

epoch_num = 50
print(f'epoch = {epoch_num}')

tolerance = args.tolerance
print(f'Tolerance of early stop = {tolerance}')
# pts_dir = 'pts/enrico/' + str(beta) + '/'
# os.makedirs(pts_dir, exist_ok=True)

view_num = 2
bests_list_mixer = [ [] for i in range(view_num) ]
bests_list_origin = [ [] for i in range(view_num) ]


mixer_dir = 'best_pts/enrico/mi_matrix/' + str(beta) + '/'

single_dir = 'best_pts/enrico/mi_matrix/single/'

is_packed = False

task = 'classification'

    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"Begins -------------------Single Modality  [Fold] {fold}:")
      
    ds_train = EnricoDatasetMixer(data_dir, input_keys=train_index)
    ds_val = EnricoDatasetMixer(data_dir, input_keys=valid_index)
    
    dls, weights = get_dataloader_nfold(ds_train=ds_train, ds_val=ds_val)
    
    train_dataloader, valid_dataloader = dls
    
     
    encoders_mixer = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device)]
    head_mixer = Linear(32, 20).to(device)
    fusion_mixer = MultiplicativeInteractions2Modal([16, 16], 32, "matrix").to(device)
    
    mixer_file_path = mixer_dir + 'mi_mixer_fold' + str(fold)+ '.pt'
    print(f"Load mixer model = {mixer_file_path}")
    
    model_mixer = MMDL(encoders_mixer, fusion_mixer, head_mixer, has_padding=is_packed).to(device)
    model_mixer = torch.load(mixer_file_path, map_location=device)
    model_mixer.to(device)

    
    encoders_origin = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device)]
    head_origin = Linear(32, 20).to(device)
    fusion_origin = MultiplicativeInteractions2Modal([16, 16], 32, "matrix").to(device)
    
    origin_file_path = mixer_dir + 'mi_origin_fold' + str(fold)+ '.pt'
    print(f"Load orign model = {origin_file_path}")

    model_origin = MMDL(encoders_origin, fusion_origin, head_origin, has_padding=is_packed).to(device)
    model_origin = torch.load(origin_file_path, map_location=device)
    model_origin.to(device)
    
    
    for i in range(view_num):
        print(f"*** Eval Modality {i} == > fold {fold}:")
        
        # print("Load single modality:")
        
        # encoder = VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device)
        # head = Linear(16, 20).to(device)

        # single_model = SingleMDL(encoder, head=head, index=i, has_padding=is_packed).to(device)
        # single_model_dir = single_dir + "single_view_" + str(i) + "_fold_" + str(fold) + ".pt"
        # single_model = torch.load(single_model_dir)
        
        # ------mixer valid--------
        # mixer_single_model = copy.deepcopy(single_model)
        # mixer_single_model.encoder = model_mixer.encoders[i]
        
        encoder_mixer = copy.deepcopy(model_mixer.encoders[i])
        
        head_mixer = Linear(16, 20).to(device)
        
        best_mixer = train_head_single_modality(encoder_mixer, i, head_mixer, train_dataloader, valid_dataloader, epoch_num,
        optimtype=torch.optim.Adam, lr=0.0001, weight_decay=0.0
        , track_complexity=False, save=single_dir, device=device, tolerance=tolerance, fold_index=fold, type="mixer")
        
        bests_list_mixer[i].append(best_mixer.item())
        
        # ------origin valid---------
        # origin_single_model = copy.deepcopy(single_model)
        # origin_single_model.encoder = model_origin.encoders[i]
        
        encoder_origin = copy.deepcopy(model_origin.encoders[i])
        
        head_origin = Linear(16, 20).to(device)
        
        best_origin = train_head_single_modality(encoder_origin, i, head_origin, train_dataloader, valid_dataloader, epoch_num,
        optimtype=torch.optim.Adam, lr=0.0001, weight_decay=0.0
        , track_complexity=False, save=single_dir, device=device, tolerance=tolerance, fold_index=fold, type="origin")

        bests_list_origin[i].append(best_origin.item())
    
    # print(f"------------------Fold {fold} ends.-------------------")

print("---------Start Eval-----------")
for i in range(view_num):
  print(f"Results for modality {i}: ")
  print(f'[Average] Mixer = {sum(bests_list_mixer[i])/5.0}, total = {bests_list_mixer[i]}')    
  print(f'[Average] Origin = {sum(bests_list_origin[i])/5.0}, total = {bests_list_origin[i]}')   
print_current_time()