import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from unimodals.MVAE import TSEncoder, TSDecoder # noqa
from utils.helper_modules import Sequential2 # noqa
from objective_functions.objectives_for_supervised_learning import MFM_objective # noqa

from unimodals.common_models import MLP # noqa
from training_structures.Supervised_Learning import train, train_mixer, test
from datasets.affect.get_data import get_fulldata, get_noise_dataloader
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

beta=0.1
# beta = args.beta
print(f'beta = {beta}')


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'device={device}')

# epoch_num = args.epoch

epoch_num=100
print(f'epoch = {epoch_num}')

tolerance = args.tolerance
print(f'Tolerance of early stop = {tolerance}')

pts_dir = 'pts/affect/mosi/mfm/'  + str(beta) + '/'
os.makedirs(pts_dir, exist_ok=True)

# common objective
objective = MFM_objective(2.0, [torch.nn.MSELoss(
), torch.nn.MSELoss(), torch.nn.MSELoss()], [1.0, 1.0, 1.0], device=device)


robust_test=True
noise_level = args.noise

task='classification'

train_dataloader, valid_dataloader, test_dataloader  = get_noise_dataloader(
    'data/affect/mosi/mosi_raw.pkl', robust_test=True, data_type='mosi'
    , raw_path="data/affect/mosi/mosi.hdf5",  full_data=full_data, i=noise_level, device=device, task=task, max_pad=True, max_seq_len=timestep)


noise_level = args.noise

fold = 'fullnoise'


if True:
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
    

    print(f"-------------------Mixer fold{fold} starts-------------------")
    save_as_m = pts_dir + 'mfm_mixer_' + fold + str(noise_level) + '_' + str(epoch_num) +'.pt'
    best_m = train_mixer(encoders_mixer, fuse_mixer, head_mixer, train_dataloader, valid_dataloader, epoch_num, additional_modules_mixer,
        objective=objective, objective_args_dict=argsdict_mixer, save=save_as_m
        , track_complexity=track_complexity, device=device, beta=beta, tolerance=tolerance)
    

    
    print(f"-------------------Origin fold{fold} starts-------------------")
    save_as_o = pts_dir + 'mfm_origin_' + fold + str(noise_level) + '_' + str(epoch_num) +'.pt'
    best_o = train(encoders, fuse, head, train_dataloader, valid_dataloader, epoch_num, additional_modules,
        objective=objective, objective_args_dict=argsdict, save=save_as_o
        , track_complexity=track_complexity, device=device, tolerance=tolerance)
 
    
print("Testing Mixer:")    
model_mixer = torch.load(save_as_m).cuda()  
test(model=model_mixer, test_dataloaders_all=test_dataloader,
     dataset='mosi', is_packed=False, no_robust=True)


print("Testing Origin:")
model_origin = torch.load(save_as_o).cuda() 
test(model=model_origin, test_dataloaders_all=test_dataloader,
     dataset='mosi', is_packed=False, no_robust=True)


print_current_time()

