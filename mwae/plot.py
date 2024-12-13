import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
from utils.config import global_cfg
from utils.cmd_args import parse_args
from utils.logger import update_logger
from utils.util import setup_seed, get_data, update_cfg, plot_tSNE
from utils.get_distance import mixer_ot
from utils.mwae_data import check_dir
from models.KeyModule import MultiEncoders, MultiDecoders

import logging

logger = logging.getLogger("mwae")


def plot_tsne2(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f'device = {device}')

    data_loaders, cfg = get_data(cfg, device)
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
    

    cfg.model.is_load = True

    dir_pth = 'models/'+ data_type
    check_dir(dir_pth)

    # save data for plot 
    result_dict_z = {key: [] for key in range(num_views)}
    result_dict_x = {key: [] for key in range(num_views)}

    valid_labels=data_loaders['valid'].dataset.labels.detach().cpu().numpy()

    if cfg.data.valid_tsne:
        # 1.save data
        for batch_idx, (inputs, labels, index) in enumerate(data_loaders['valid']):
            zs = mvencoders(xs=inputs)
            for i in range(num_views):
                # save raw data
                xi = inputs[i]
                result_dict_x[i].extend(xi.detach().cpu().numpy())
                # save zi for each view
                zi = zs[i]
                result_dict_z[i].extend(zi.detach().cpu().numpy())


        # delete the first two most classes in cal7
        if data_type in ['cal7', 'caltech7']:
            unique_labels, label_counts = np.unique(valid_labels, return_counts=True)
            # print label count
            sorted_indices = np.argsort(-label_counts)  # sort count

         # print as count 
            for idx in sorted_indices:
                print(f"Label {unique_labels[idx]} appears count: {label_counts[idx]}")
            # find the first two
            most_common_indices = np.argsort(-label_counts)[:2]

        #   find the label 
            labels_to_remove = unique_labels[most_common_indices]

            #find the index
            indices_to_remove = np.isin(valid_labels, labels_to_remove)
            indices_to_remove = np.array(indices_to_remove)
            # delete data and label
            for i in range(num_views):
                fetures_x_raw = np.array(result_dict_x[i])
                result_dict_x[i] = fetures_x_raw[~indices_to_remove]
                fetures_z_view = np.array(result_dict_z[i])
                result_dict_z[i] = fetures_z_view[~indices_to_remove]
            valid_labels = np.array(valid_labels)[~indices_to_remove]
            
            print(f"Filtered data: {valid_labels}")
            
        # 2.plot tsne
        dir = 'plots/'+ cfg.data.type
        for i in range(num_views):
            # plot raw data -> x
            file_xi = str(int(cfg.data.unaligned_rate))+'_view'+str(i)+'_raw'
            logger.info(f'tsne raw data begins! => {file_xi}')
            xi = result_dict_x[i]
            plot_tSNE(xi, valid_labels, random_state=cfg.seed, save_dir=dir, save_filename=file_xi)
            
            
        logger.info('All tsne--end!')
    else:
        logger.info(f'cfg.data.valid_tsne = {cfg.data.valid_tsne}')


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

    
    logger.info('Begin plot---')

    

    plot_tsne2(init_cfg)
