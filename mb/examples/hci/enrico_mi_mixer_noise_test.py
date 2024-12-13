
import sys
import os
from torch import nn
sys.path.append(os.getcwd())

from unimodals.common_models import  Linear,  VGG11Slim
import torch # noqa

from datasets.enrico.get_data import get_dataloader
from fusions.common_fusions import MultiplicativeInteractions2Modal

from training_structures.Supervised_Learning import test, MMDL, SingleMDL  # noqa
from sklearn.model_selection import KFold
import csv
import numpy as np
import copy

from mwae.util import setup_seed, print_current_time

setup_seed()
print_current_time()
torch.backends.cudnn.enabled = False



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


dls, weights = get_dataloader('datasets/enrico/dataset', train_shuffle=False)

traindata, validdata, testdata = dls

    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    
    print(f"Begins -------------------Single Modality  [Fold] {fold}:")
    
     
    encoders_mixer = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device)]
    head_mixer = Linear(32, 20).to(device)
    fusion_mixer = MultiplicativeInteractions2Modal([16, 16], 32, "matrix").to(device)
    
    mixer_file_path = mixer_dir + 'mi_mixer_fold' + str(fold)+ '.pt'
    print(f"Load mixer model = {mixer_file_path}")
    
    model_mixer = MMDL(encoders_mixer, fusion_mixer, head_mixer, has_padding=is_packed).to(device)
    model_mixer = torch.load(mixer_file_path, map_location=device)
    model_mixer.to(device)

    print("Testing mixer:")

    test(model_mixer, testdata, dataset='enrico', mixer_type='mixer', fold_index=fold, method_name='MI')
    
    
    encoders_origin = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(device)]
    head_origin = Linear(32, 20).to(device)
    fusion_origin = MultiplicativeInteractions2Modal([16, 16], 32, "matrix").to(device)
    
    origin_file_path = mixer_dir + 'mi_origin_fold' + str(fold)+ '.pt'
    print(f"Load orign model = {origin_file_path}")

    model_origin = MMDL(encoders_origin, fusion_origin, head_origin, has_padding=is_packed).to(device)
    model_origin = torch.load(origin_file_path, map_location=device)
    model_origin.to(device)
    
    
    print("Testing origin:")
    test(model_origin, testdata, dataset='enrico', mixer_type='origin', fold_index=fold, method_name='MI')
