import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
from utils.get_training import get_optimizer, get_scheduler
from utils.get_distance import mixer_ot, reconstruction_loss, cal_spectral_loss
from utils.gw_transport import dist, emd2
from utils.config import global_cfg
from utils.cmd_args import parse_args
from utils.logger import update_logger
from utils.util import setup_seed, get_data, update_cfg, construct_ss_dict, plot_tSNE
from utils.kmeans_valid import kmeans_for_multiview, update_epoch_best_result
from utils.mwae_data import check_dir
from models.KeyModule import MultiEncoders, MultiDecoders
from plot import plot_tsne2
import logging

logger = logging.getLogger("mwae")

import nni


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
 
    dir_pth = 'models/'+ cfg.data.type
    check_dir(dir_pth)
    
    if cfg.data.is_filter:
        encoders_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_' + str(cfg.mixer.fuse) + '_encoders_bm_epoch' + str(epochs) + '_filter.pth'
        decoders_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_' + str(cfg.mixer.fuse) + '_decoders_bm_epoch' + str(epochs) + '_filter.pth'
    else:
        encoders_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_' + str(cfg.mixer.fuse) + '_encoders_bm_epoch' + str(epochs) + '.pth'
        decoders_pth = dir_pth + '/'  + str(int(cfg.data.unaligned_rate)) + '_' + str(cfg.mixer.fuse) + '_decoders_bm_epoch' + str(epochs) + '.pth'

    if cfg.model.is_load:
        # load encoders
        if os.path.exists(encoders_pth):
            mvencoders.load_state_dict(torch.load(encoders_pth))
            logger.info(f'Load {encoders_pth} --- ')
        else:
            logger.info(f'>>> No file: {encoders_pth}! Start a new model.')
        
        # load decoders
        if os.path.exists(decoders_pth):
            mvdecoders.load_state_dict(torch.load(decoders_pth))
            logger.info(f'Load {decoders_pth} --- ')
        else:
            logger.info(f'>>> No file: {decoders_pth}! Start a new model.')

    train_cfg = cfg['train'] 
    train_cfg.optimizer.clear_aux_info()
    train_cfg.scheduler.clear_aux_info()

    parameters = list(mvencoders.parameters()) + list(mvdecoders.parameters())
    optimizer = get_optimizer(parameters, **train_cfg.optimizer)
    gradient_scheduler = get_scheduler(optimizer, **train_cfg.scheduler)    

    best_result_batch = {'epoch': 0, 'batch': 0, 'test_acc_avg': 0.0, 'test_nmi_avg': 0.0, 'test_ami_avg': 0.0,
                         'test_ari_avg': 0.0}

    ss_dict = construct_ss_dict(cfg=cfg)
    best_result_epoch = {'epoch': 0, 'test_acc_avg': 0.0, 'test_nmi_avg': 0.0, 'test_ami_avg': 0.0, 'test_ari_avg': 0.0}
    best_result_epoch.update(ss_dict)

    
    metric = cfg.metric
    metric_value = 0.0
    metric_key = 'test_acc_avg'
    if metric == 1:
        metric_key = 'test_nmi_avg'
    elif metric == -1:
        metric_key = 'test_ami_avg'
    
 

    
    for n in range(cfg.train.local_update_steps): 
        # train -- begin
        mvencoders.train()
        mvdecoders.train()

        for batch_idx, (inputs, labels, index) in enumerate(data_loaders['train']):

            optimizer.zero_grad()
            zs = mvencoders(xs=inputs)
            
            zmix, trans = mixer_ot(zs, outer_iter=cfg.mixer.inner_iter, method=cfg.mixer.gw_method,
                                   reg=cfg.mixer.gamma, f_alpha=cfg.mixer.f_alpha)
            
            zmix_inputs = [torch.matmul(torch.transpose(trans[i], 0, 1), zmix) for i in range(num_views)]

            xhats = mvdecoders(zs)
            xhats_mixer = mvdecoders(zmix_inputs)

            # loss 1: reconstruction loss
            loss_rec = 0
            loss_mix = 0
            loss_reg = 0

            for i in range(len(inputs)):
                loss_rec += reconstruction_loss(xhats[i], inputs[i])
                loss_mix += reconstruction_loss(xhats_mixer[i], inputs[i])

     
                if i + 1 <= len(inputs) - 1:
                    s = torch.ones_like(zs[i][:, 0]) / inputs[i].shape[0]

                    t = torch.ones_like(zs[i + 1][:, 0]) / xhats[i].shape[0]

                    C = dist(zs[i], zs[i + 1])

                    loss_reg += emd2(s, t, C)



            if int(cfg.data.unaligned_rate) == 0:
                if cfg.mixer.fuse in ['add', 'bary']:
                    zs_stack = torch.stack(zs)
                    out = torch.sum(zs_stack, dim=0)
                elif cfg.mixer.fuse == 'con':
                    out = torch.cat(zs, dim=1)
            elif int(cfg.data.unaligned_rate) == 1:
                if cfg.mixer.fuse in ['add', 'bary']:
                    out = 0
                    for i in range(len(zs)):
                        out += trans[i] @ zs[i]
                elif cfg.mixer.fuse == 'con':
                    zc = zs.copy()
                    for i in range(len(zs)):
                        zc[i] = trans[i] @ zs[i]
                    out = torch.cat(zc, dim=1)

            loss_cluster = cal_spectral_loss(cfg, out)

            loss_all = loss_cluster + cfg.split.tau * loss_rec + cfg.split.alpha * loss_mix + cfg.split.eta * loss_reg

            loss_all.backward()
            optimizer.step()

            loss_log = {'train_loss_regularize': loss_reg.item(), 'train_loss_mix': loss_mix.item(),
                        'train_loss_cluster': loss_cluster.item(), 'train_loss_rec': loss_rec.item(),
                        'train_loss_all': loss_all.item()}

            result = {'Role': 'Loss #', 'epoch': n, 'batch': batch_idx,
                      data_type: loss_log}

            logger.info(str(result))


        mvencoders.eval()
        mvdecoders.eval()


        batch_num_valid = len(data_loaders['valid'])
        
        epoch_acc = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
        epoch_nmi = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
        epoch_ami = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)
        epoch_ri = torch.zeros(batch_num_valid, dtype=torch.float32, device=device)

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

        if metric == 0:
            metric_value = epoch_acc_avg.item()
        elif metric == 1:
            metric_value = epoch_nmi_avg.item()
        elif metric == -1:
            metric_value = epoch_ami_avg.item()

        gradient_scheduler.step(metric_value) 

        best_result_epoch = update_epoch_best_result(cfg, epoch_acc_avg, epoch_nmi_avg, epoch_ami_avg, epoch_ri_avg, n,
                                                     best_result_epoch)
        if metric_value == best_result_epoch[metric_key]:
            if cfg.nni:
                nni.report_intermediate_result(metric_value)
            if cfg.model.is_save:
                torch.save(mvencoders.state_dict(), encoders_pth)
                torch.save(mvdecoders.state_dict(), decoders_pth)
    
    if cfg.nni:
        nni.report_final_result(best_result_epoch[metric_key])

    if cfg.data.valid_tsne:
        plot_tsne2(cfg)


        


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

    logger.info(f'init_cfg.nni = {init_cfg.nni}')
    if init_cfg.nni:
        next_paras = nni.get_next_parameter()
        logger.info(f'next_paras = {next_paras}')
        init_cfg = update_cfg(init_cfg=init_cfg, next_paras=next_paras)
    logger.info('Begin mwae_bm:')
    run_mwae_m(init_cfg)
