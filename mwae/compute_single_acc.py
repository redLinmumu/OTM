import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
from utils.config import global_cfg
from utils.cmd_args import parse_args
from utils.logger import update_logger
from utils.util import setup_seed, get_data

from utils.mwae_data import check_dir
from models.KeyModule import MultiEncoders, MultiDecoders
from utils.kmeans_valid import kmeans_for_single_view
import logging

logger = logging.getLogger("mwae")


if __name__ == '__main__':

    init_cfg = global_cfg

    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)

    update_logger(init_cfg, clear_before_add=True)

    if init_cfg.verbose > 0:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARN
        logger.warning("Skip DEBUG/INFO messages")

    logger.setLevel(logging_level)

    setup_seed(init_cfg.seed)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f'device = {device}')


    data_loaders, cfg = get_data(init_cfg, device)
    logger.info(f'cfg = {cfg}')

    data_type = cfg.data.type 
    feature_dims = cfg.data.modality_feature_dims 
    num_views = cfg.data.num_views  
    epochs = cfg.train.local_update_steps

    z_dim = cfg.model.z_dim
    h_dim = cfg.model.h_dim

    mvencoders = MultiEncoders(x_dims=feature_dims,
                               z_dim=z_dim,
                               h_dim=h_dim,
                               device=device)

    mvdecoders = MultiDecoders(x_dims=feature_dims,
                               z_dim=z_dim,
                               h_dim=h_dim,
                               device=device)


    dir_pth = 'models/'+ data_type
    check_dir(dir_pth)


    load_filtered = False
    
    if load_filtered:
        print('Load filtered trained model...')
        encoders_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_' + str(cfg.mixer.fuse) + '_encoders_bm_epoch' + str(epochs) + '_filter.pth'
        decoders_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_' + str(cfg.mixer.fuse) + '_decoders_bm_epoch' + str(epochs) + '_filter.pth'
    else: 
        print('Load origin(not filtered) trained model...')
        encoders_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_' + str(cfg.mixer.fuse) + '_encoders_bm_epoch' + str(epochs) + '.pth'
        decoders_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_' + str(cfg.mixer.fuse) + '_decoders_bm_epoch' + str(epochs) + '.pth'


    if os.path.exists(encoders_pth):
        if torch.cuda.is_available():
            mvencoders.load_state_dict(torch.load(encoders_pth))
        else:
            mvencoders.load_state_dict(torch.load(encoders_pth, map_location=torch.device('cpu')))
        logger.info(f'Load {encoders_pth} --- ')
    else:
        logger.info(f'>>> No file: {encoders_pth}! No saved model.')

        
    # load decoders
    if os.path.exists(decoders_pth):
        if torch.cuda.is_available():
            mvdecoders.load_state_dict(torch.load(decoders_pth))
        else:
            mvdecoders.load_state_dict(torch.load(decoders_pth, map_location=torch.device('cpu')))
        logger.info(f'Load {decoders_pth} --- ')
    else:
        logger.info(f'>>> No file: {decoders_pth}! No saved model.')


    mvencoders.eval()
    mvdecoders.eval()
    
    batch_num_valid = len(data_loaders['valid'])
    

    for i in range(num_views):
        
        this_epoch_acc = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
        this_epoch_nmi = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
        this_epoch_ami = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
        this_epoch_ri = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
        
        for batch_idx, (inputs, labels, index) in enumerate(data_loaders['valid']):
            with torch.no_grad():

                zi = mvencoders.encoders[i](inputs[i])
                xhat_i = mvdecoders.decoders[i](zi)
                    
                acc, nmi, ami, ari = kmeans_for_single_view(cfg, zi, labels)
                num_samples = inputs[0].shape[0]
                this_epoch_acc[batch_idx], this_epoch_nmi[batch_idx], this_epoch_ami[batch_idx], this_epoch_ri[
                    batch_idx] = acc * num_samples, nmi * num_samples, ami * num_samples, ari * num_samples
                
        epoch_acc_avg = torch.sum(this_epoch_acc) / cfg.data.valid_samples_num
        epoch_nmi_avg = torch.sum(this_epoch_nmi) / cfg.data.valid_samples_num
        epoch_ami_avg = torch.sum(this_epoch_ami) / cfg.data.valid_samples_num
        epoch_ari_avg = torch.sum(this_epoch_ri) / cfg.data.valid_samples_num
        
        print(f'【View {i}】acc = {epoch_acc_avg.item()}, nmi = {epoch_nmi_avg.item()}, ari = {epoch_ari_avg.item()}, ami = {epoch_ami_avg.item()}')

    
