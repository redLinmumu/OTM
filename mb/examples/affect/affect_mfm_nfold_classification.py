import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from unimodals.MVAE import TSEncoder, TSDecoder # noqa
from utils.helper_modules import Sequential2 # noqa
from objective_functions.objectives_for_supervised_learning import MFM_objective # noqa

from unimodals.common_models import MLP # noqa
from training_structures.Supervised_Learning import train, train_mixer
from datasets.affect.get_data import get_fulldata, get_fold_dataloader
from fusions.common_fusions import Concat # noqa
from sklearn.model_selection import KFold
import numpy as np

from mwae.util import setup_seed, print_current_time

setup_seed()
print_current_time()


classes = 2
n_latent = 256
dim_0 = 35
dim_1 = 74
dim_2 = 300
timestep = 50

# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
full_data = get_fulldata(
    'data/affect/mosi/mosi_raw.pkl', task='classification', robust_test=False, max_pad=True, max_seq_len=timestep)


# save_as = 'mosi_mfm_best_nfold.pt'
track_complexity = False

    
n_splits = 5       
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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

pts_dir = 'pts/affect/mosi/mfm/'  + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)

# common objective
objective = MFM_objective(2.0, [torch.nn.MSELoss(
), torch.nn.MSELoss(), torch.nn.MSELoss()], [1.0, 1.0, 1.0], device=device)


bests_m = []
bests_o = []

    
for fold, (train_index, valid_index) in enumerate(kfold.split(X=np.zeros(n_samples))):
    print(f"[Fold] {fold}")
    
    # Origin   
    encoders = [TSEncoder(dim_0, 30, n_latent, timestep, returnvar=False).to(device), TSEncoder(
        dim_1, 30, n_latent, timestep, returnvar=False).to(device), TSEncoder(dim_2, 30, n_latent, timestep, returnvar=False).to(device)]

    decoders = [TSDecoder(dim_0, 30, n_latent, timestep, device=device).to(device), TSDecoder(
        dim_1, 30, n_latent, timestep, device=device).to(device), TSDecoder(dim_2, 30, n_latent, timestep, device=device).to(device)]

    fuse = Sequential2(Concat(), MLP(3*n_latent, n_latent, n_latent//2)).to(device)

    intermediates = [MLP(n_latent, n_latent//2, n_latent//2).to(device), MLP(n_latent,
                                                                        n_latent//2, n_latent//2).to(device), MLP(n_latent, n_latent//2, n_latent//2).to(device)]

    head = MLP(n_latent//2, 20, classes).to(device)

    argsdict = {'decoders': decoders, 'intermediates': intermediates}

    additional_modules = decoders+intermediates

    # Mixer  
    encoders_mixer = [TSEncoder(dim_0, 30, n_latent, timestep, returnvar=False).to(device), TSEncoder(
        dim_1, 30, n_latent, timestep, returnvar=False).to(device), TSEncoder(dim_2, 30, n_latent, timestep, returnvar=False).to(device)]

    decoders_mixer = [TSDecoder(dim_0, 30, n_latent, timestep, device=device).to(device), TSDecoder(
        dim_1, 30, n_latent, timestep, device=device).to(device), TSDecoder(dim_2, 30, n_latent, timestep, device=device).to(device)]

    fuse_mixer = Sequential2(Concat(), MLP(3*n_latent, n_latent, n_latent//2)).to(device)

    intermediates_mixer = [MLP(n_latent, n_latent//2, n_latent//2).to(device), MLP(n_latent,
                                                                        n_latent//2, n_latent//2).to(device), MLP(n_latent, n_latent//2, n_latent//2).to(device)]

    head_mixer = MLP(n_latent//2, 20, classes).to(device)

    argsdict_mixer = {'decoders': decoders_mixer, 'intermediates': intermediates_mixer}

    additional_modules_mixer = decoders_mixer + intermediates_mixer
    
     
    train_data = {key: value[train_index] for key, value in full_data.items()}
    valid_data = {key: value[valid_index] for key, value in full_data.items()}

    train_dataloaer, valid_dataloader = get_fold_dataloader(train_data=train_data, valid_data=valid_data, task='classification', robust_test=False, max_pad=True, max_seq_len=timestep)
    
    print(f"-------------------Mixer fold{fold} starts-------------------")
    save_as_m = pts_dir + 'mfm_mixer_' + str(fold)+ '.pt'
    best_m = train_mixer(encoders_mixer, fuse_mixer, head_mixer, train_dataloaer, valid_dataloader, epoch_num, additional_modules_mixer,
        objective=objective, objective_args_dict=argsdict_mixer, save=save_as_m
        , track_complexity=track_complexity, device=device, beta=beta, tolerance=tolerance)
    
    bests_m.append(best_m.item())
    print(f"-------------------Mixer fold{fold} ends!-------------------\n")
    
    print(f"-------------------Origin fold{fold} starts-------------------")
    save_as_o = pts_dir + 'mfm_origin_' + str(fold)+ '.pt'
    best_o = train(encoders, fuse, head, train_dataloaer, valid_dataloader, epoch_num, additional_modules,
        objective=objective, objective_args_dict=argsdict, save=save_as_o
        , track_complexity=track_complexity, device=device, tolerance=tolerance)
    
    bests_o.append(best_o.item())
    print(f"-------------------Origin fold{fold} ends!-------------------")
   
    print(f"-------------------Fold {fold} Ends!-------------------")

print(f'[Average] Mixer = {sum(bests_m)/5.0}, \n\t All = {bests_m}')  
print(f'[Average] Original = {sum(bests_o)/5.0}, \n\t All = {bests_o}')  

print_current_time()