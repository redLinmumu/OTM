
import sys
import os
from torch import nn
sys.path.append(os.getcwd())

from unimodals.common_models import  Linear,  VGG11Slim
import torch # noqa

from datasets.enrico.get_data import get_dataloader_nfold, EnricoDatasetMixer # noqa
from fusions.common_fusions import MultiplicativeInteractions2Modal

from training_structures.Supervised_Learning import train_mixer, train # noqa
from sklearn.model_selection import KFold
import csv
import numpy as np

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

beta = args.beta
print(f'beta = {beta}')


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

epoch_num = args.epoch
print(f'epoch = {epoch_num}')

tolerance = args.tolerance
print(f'Tolerance of early stop = {tolerance}')

pts_dir = 'pts/enrico/' + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)

bests_m = []
bests_o = []

    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"[Fold] {fold}:")
      
    ds_train = EnricoDatasetMixer(data_dir, input_keys=train_index)
    ds_val = EnricoDatasetMixer(data_dir, input_keys=valid_index)
    
    dls, weights = get_dataloader_nfold(ds_train=ds_train, ds_val=ds_val)
    traindata, validdata = dls
    
    save_as_mixer = pts_dir + 'tensormatrix_mixer_fold' + str(fold)+ '.pt'
    save_as_origin = pts_dir + 'tensormatrix_fold' + str(fold)+ '.pt'
    
     
    encoders_mix = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device)]
    head_mix = Linear(32, 20).to(device)
    fusion_mix = MultiplicativeInteractions2Modal([16, 16], 32, "matrix", True).to(device)


    print(f'------------Fold:{fold} Mixer begins--------------------')
    best_m = train_mixer(encoders_mix, fusion_mix, head_mix, traindata, validdata, epoch_num, optimtype=torch.optim.Adam,
            lr=0.0001, weight_decay=0, track_complexity=False, save=save_as_mixer, device=device, beta=beta, tolerance=tolerance)
    
    bests_m.append(best_m.item())
    print(f'------------Fold:{fold} Mixer ends--------------------')
    
    
    encoders = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device)]
    head = Linear(32, 20).to(device)
    fusion = MultiplicativeInteractions2Modal([16, 16], 32, "matrix", True).to(device)

    print(f'------------Fold:{fold} Origin begins--------------------')
    best_o = train(encoders, fusion, head, traindata, validdata, epoch_num, optimtype=torch.optim.Adam,
            lr=0.0001, weight_decay=0, track_complexity=False, save=save_as_origin,device=device, tolerance=tolerance)
    
    bests_o.append(best_o.item())
    print(f'------------Fold:{fold} Origin ends--------------------')
    
    print(f"Fold {fold} ends.")
    
    
print(f'[Average] Mixer = {sum(bests_m)/5.0}, \n\t All = {bests_m}')  
print(f'[Average] Original = {sum(bests_o)/5.0}, \n\t All = {bests_o}')  
        
print_current_time()