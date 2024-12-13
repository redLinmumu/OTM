import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
from utils.config import global_cfg
from utils.cmd_args import parse_args
from utils.logger import update_logger
from utils.util import setup_seed, get_data
from utils.kmeans_valid import kmeans_for_multiview
from utils.mwae_data import check_dir
from models.KeyModule import MultiEncoders, MultiDecoders

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
    
    epoch_acc = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
    epoch_nmi = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
    epoch_ami = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
    epoch_ri = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
    
    best_result_batch = {'epoch': 0, 'batch': 0, 'test_acc_avg': 0.0, 'test_nmi_avg': 0.0, 'test_ami_avg': 0.0,
                         'test_ari_avg': 0.0}
    

    for batch_idx, (inputs, labels, index) in enumerate(data_loaders['valid']):
        
        with torch.no_grad():
            zs = mvencoders(xs=inputs)
                              
        test_acc_avg, test_nmi_avg, test_ami_avg, test_ri_avg, best_result_batch = kmeans_for_multiview(cfg, inputs, zs,
                                                                                                  labels,
                                                                                                  device,
                                                                                                  best_result_batch)
        num_samples = inputs[0].shape[0]
        
        epoch_acc[batch_idx], epoch_nmi[batch_idx], epoch_ami[batch_idx], epoch_ri[
                    batch_idx] = test_acc_avg * num_samples, test_nmi_avg * num_samples, test_ami_avg * num_samples, test_ri_avg * num_samples
        
    epoch_acc_avg = torch.sum(epoch_acc) / cfg.data.valid_samples_num
    epoch_nmi_avg = torch.sum(epoch_nmi) / cfg.data.valid_samples_num
    epoch_ami_avg = torch.sum(epoch_ami) / cfg.data.valid_samples_num
    epoch_ri_avg = torch.sum(epoch_ri) / cfg.data.valid_samples_num
    
    
    print(f'acc = {epoch_acc_avg.item()}, nmi = {epoch_nmi_avg.item()}, ri = {epoch_ri_avg.item()}, ami = {epoch_ami_avg.item()}')

    
