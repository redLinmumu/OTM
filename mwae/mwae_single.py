import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
from utils.get_training import get_optimizer, get_scheduler
from utils.get_distance import reconstruction_loss, cal_spectral_loss

from utils.config import global_cfg
from utils.cmd_args import parse_args
from utils.logger import update_logger
from utils.util import setup_seed, get_data
from utils.kmeans_valid import update_epoch_best_result
from utils.mwae_data import check_dir
from models.KeyModule import MultiEncoders, MultiDecoders


import logging
from utils.kmeans_valid import kmeans_for_single_view

logger = logging.getLogger("mwae")


# single modality training 


def run_mwae_m(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f'device = {device}')

    data_loaders, cfg = get_data(cfg, device)
    logger.info(f'cfg = {cfg}')

    data_type = cfg.data.type 
    feature_dims = cfg.data.modality_feature_dims  
    num_views = len(feature_dims) 
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
   
    dir_pth = 'models/'+ cfg.data.type+'/single'
    check_dir(dir_pth)
    
    if cfg.data.is_filter:
        encoders_root_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_encode_Viewf_'
        decoders_root_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_decode_Viewf_'
    else:
        encoders_root_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_encode_View_'
        decoders_root_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_decode_View_'


    train_cfg = cfg['train']  
    train_cfg.optimizer.clear_aux_info()
    train_cfg.scheduler.clear_aux_info()

    
    optimizer_list = []
    for i in range(num_views):
        paras =  list(mvencoders.encoders[i].parameters()) + list(mvdecoders.decoders[i].parameters())
        optimizer_i = get_optimizer(paras, **train_cfg.optimizer)
        optimizer_list.append(optimizer_i)
    
    scheduler_list = []  
    for i in range(num_views):  
        scheduler_i = get_scheduler(optimizer_list[i], **train_cfg.scheduler)    
        scheduler_list.append(scheduler_i)

    is_early_stop = cfg.train.early_stop
    patience = cfg.train.patience
    metric = cfg.metric
    metric_value = 0.0
    metric_key = 'test_acc_avg'
    if metric == 1:
        metric_key = 'test_nmi_avg'
    elif metric == -1:
        metric_key = 'test_ami_avg'
    
    # ss_dict = construct_ss_dict(cfg=cfg)
    best_result_epoch = {'epoch': 0, 'test_acc_avg': 0.0, 'test_nmi_avg': 0.0, 'test_ami_avg': 0.0, 'test_ari_avg': 0.0}
    # best_result_epoch.update(ss_dict)
    mvencoders_state_dict = [dict() for i in range(num_views)]
    mvdecoders_state_dict = [dict() for i in range(num_views)]
    has_stopped = [False for i in range(num_views)]
    patience_count = [0 for i in range(num_views)]
    previous_valid_loss = [0.0 for i in range(num_views)]
    best_result_epoch_records = [best_result_epoch.copy() for i in range(num_views)]
    
    for n in range(epochs): 
        # train -- begin
        for i in range(num_views):
            if has_stopped[i]:
                print(f'Stopped modality {i}: No train().')
                continue
            mvencoders.encoders[i].train()
            mvdecoders.decoders[i].train()

        for batch_idx, (inputs, labels, index) in enumerate(data_loaders['train']):
            # print(batch_idx)
            for  i in range(num_views):
                if has_stopped[i]:
                    print(f'Stopped modality {i}: No  batch training.')
                    continue
                
                optimizer_list[i].zero_grad()
                    
                zi = mvencoders.encoders[i](inputs[i])
                    
                xhat_i = mvdecoders.decoders[i](zi)

                loss_rec_i = reconstruction_loss(xhat_i, inputs[i])

                loss_cluster_i = cal_spectral_loss(cfg, zi)

                loss_all_i = loss_cluster_i + cfg.split.tau * loss_rec_i

                loss_all_i.backward()
                optimizer_list[i].step()

                loss_log = {
                            'train_loss_cluster': loss_cluster_i.item(), 
                            'train_loss_rec': loss_rec_i.item(),
                            'train_loss_all': loss_all_i.item()
                                }

                result = {'Modality:': i, 'epoch': n, 'batch': batch_idx,
                            data_type: loss_log}

                logger.info(str(result))
        # valid
        batch_num_valid = len(data_loaders['valid'])
        
        # eval -- begin
        for i in range(num_views):
            if has_stopped[i]:
                print(f'Stopped modality {i}: No eval().')
                continue
            mvencoders.encoders[i].eval()
            mvdecoders.decoders[i].eval()

            this_epoch_acc = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
            this_epoch_nmi = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
            this_epoch_ami = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
            this_epoch_ri = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
            
            present_valid_loss = 0.0
            
            for batch_idx, (inputs, labels, index) in enumerate(data_loaders['valid']):
                with torch.no_grad():
                    zi = mvencoders.encoders[i](inputs[i])
                    xhat_i = mvdecoders.decoders[i](zi)
                    
                    acc, nmi, ami, ari = kmeans_for_single_view(cfg, zi, labels)
                    num_samples = inputs[0].shape[0]
                    this_epoch_acc[batch_idx], this_epoch_nmi[batch_idx], this_epoch_ami[batch_idx], this_epoch_ri[
                        batch_idx] = acc * num_samples, nmi * num_samples, ami * num_samples, ari * num_samples
                    present_valid_loss += reconstruction_loss(xhat_i, inputs[i]) + cfg.split.tau * cal_spectral_loss(cfg, zi)

            if is_early_stop:
                if previous_valid_loss[i] == present_valid_loss.item():
                    patience_count[i] += 1
                else:
                    previous_valid_loss[i] = present_valid_loss.item()
                
                
                if patience_count[i] >= patience:
                    has_stopped[i] = True
                    print(f"Modality {i} has early stopped.")
                
                
            epoch_acc_avg = torch.sum(this_epoch_acc) / cfg.data.valid_samples_num
            epoch_nmi_avg = torch.sum(this_epoch_nmi) / cfg.data.valid_samples_num
            epoch_ami_avg = torch.sum(this_epoch_ami) / cfg.data.valid_samples_num
            epoch_ari_avg = torch.sum(this_epoch_ri) / cfg.data.valid_samples_num

            if metric == 0:
                metric_value = epoch_acc_avg.item()
            elif metric == 1:
                metric_value = epoch_nmi_avg.item()
            elif metric == -1:
                metric_value = epoch_ami_avg.item()

            scheduler_list[i].step(metric_value) 
            
            best_result_epoch_records[i] = update_epoch_best_result(cfg, epoch_acc_avg=epoch_acc_avg, epoch_nmi_avg=epoch_nmi_avg, epoch_ami_avg=epoch_ami_avg, epoch_ari_avg=epoch_ari_avg, n=n,
                                                        best_result_epoch=best_result_epoch_records[i], modality=i)
            
            if cfg.model.is_save and metric_value == best_result_epoch_records[i][metric_key]:
                mvencoders_state_dict[i] = mvencoders.encoders[i].state_dict()
                mvdecoders_state_dict[i] = mvdecoders.decoders[i].state_dict()
 
    if True:
        
        print(f'Models saved to {encoders_root_pth}')
        print('Best result:')
        for i in range(num_views):
            encoder_path_i = encoders_root_pth + str(i) + '.pth'
            decoder_path_i = decoders_root_pth + str(i) + '.pth'
            torch.save(mvencoders_state_dict[i], encoder_path_i)
            torch.save(mvdecoders_state_dict[i], decoder_path_i)
            print(f"[View {i}] = {best_result_epoch_records[i]}")
 


if __name__ == '__main__':

    init_cfg = global_cfg #.clone()
    
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

    logger.info('Begin mwae_single:')
    run_mwae_m(init_cfg)
