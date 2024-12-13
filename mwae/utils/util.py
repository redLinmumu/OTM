import numpy as np
import torch
import random
from utils.mwae_data import prepare_dataloaders
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_data(config, device):
    # Fix the seed for data generation
    setup_seed(12345)
    dataset, modified_config = prepare_dataloaders(config, device)
    modified_config.clear_aux_info()
    setup_seed(config.seed)
    return dataset, modified_config


def update_cfg(init_cfg, next_paras):
    for key, value in next_paras.items():
        if key == 'inner_iter':
            init_cfg.mixer.inner_iter = int(value)
        elif key == 'h_dim':
            init_cfg.model.h_dim = int(value)
        elif key == 'z_dim':
            init_cfg.model.z_dim = int(value)
        elif key == 'lr':
            init_cfg.train.optimizer.lr = value
        elif key == 'tau':
            init_cfg.split.tau = value
        elif key == 'eta':
            init_cfg.split.eta = value
        elif key == 'alpha':
            init_cfg.split.alpha = value
        elif key == 'batch_size':
            init_cfg.train.batch_size = int(value)
        elif key == 'epoch':
            init_cfg.train.local_update_steps = int(value)
        elif key == 'local_update_steps':
            init_cfg.train.local_update_steps = int(value)
        elif key == 'f_alpha':
            init_cfg.mixer.f_alpha = value
        elif key == 'beta':
            init_cfg.mixer.beta = value
        elif key == 'gw_method':
            init_cfg.mixer.gw_method = value
        elif key == 'valid_tsne':
            init_cfg.data.valid_tsne = value
        elif key == 'is_save':
            init_cfg.model.is_save = value
        elif key == 'is_load':
            init_cfg.model.is_load = value
        elif key == 'metric':
            init_cfg.metric = value
        elif key == 'fuse':
            init_cfg.mixer.fuse = value
        elif key == 'splits':
            init_cfg.data.splits = value
        elif key == "is_filter":
            init_cfg.data.is_filter = value
        elif key == "filter_num":
            init_cfg.data.filter_num = int(value)
    return init_cfg


def construct_ss_dict(cfg):
    ss_dict = dict()
    ss_dict['dataset'] = cfg.data.type
    ss_dict['unaligned_rate'] = cfg.data.unaligned_rate

    ss_dict['z_dim'] = cfg.model.z_dim
    ss_dict['h_dim'] = cfg.model.h_dim

    ss_dict['batch_size'] = cfg.train.batch_size
    ss_dict['lr'] = cfg.train.optimizer.lr

    ss_dict['tau'] = cfg.split.tau
    ss_dict['eta'] = cfg.split.eta
    ss_dict['alpha'] = cfg.split.alpha

    ss_dict['inner_iter'] = cfg.mixer.inner_iter
    ss_dict['gw_method'] = cfg.mixer.gw_method
    ss_dict['f_alpha'] = cfg.mixer.f_alpha

    ss_dict['fuse'] = cfg.mixer.fuse

    ss_dict['is_filter'] = cfg.data.is_filter
    
    return ss_dict


def plot_tSNE(outputs, labels, n_components=2, random_state=42, save_dir='.', save_filename='tSNE', title=None):

    tsne = TSNE(n_components=n_components, random_state=random_state)

    embedded_data = tsne.fit_transform(outputs)

    plt.figure(figsize=(8, 6))

    colors = [
    'green', 'red',
    'orange', 'purple', 'brown', 'thistle', 'indigo',
    'olive', 'teal', 'lime', 'navy', 'magenta',
    'deepskyblue', 'gold', 'black', 'blue', 'dimgrey'
    ]

    l = len(np.unique(labels))

    if l <= len(colors):
        colors_n = colors[:l]
        cmap_custom = ListedColormap(colors_n)
        plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels, cmap=cmap_custom)
    else:
        cmap = plt.cm.get_cmap('tab20c', l)
        plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels, cmap=cmap)
    

    save_to_file = os.path.join(save_dir, save_filename + '.pdf')
    plt.savefig(save_to_file)
    print(f'Saved to {save_to_file}')
    
    plt.colorbar()
    save_to_file = os.path.join(save_dir, save_filename + '2.pdf')
    plt.savefig(save_to_file)
    # plt.show()
    print(f'Saved Copy to {save_to_file}')
