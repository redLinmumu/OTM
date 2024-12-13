import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.io as io
import scipy.sparse as sp
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from mvlearn.datasets import load_UCImultifeature
# from util import plot_tSNE

class MultiViewSampler(Dataset):
    """
    Sample each single-view data in batch independently.
    """

    def __init__(self, features: dict, labels: list, rate: float = 0.9, seed: int = 42, correspondence: bool = True,
                 device='cpu'):
        """
        This function initialize the class
        Args:
            features: {view_name: feature ndarray with size (num_samples, view_dimension)}
                Note that differing from FullViewSampler, here the samples in each view are unaligned
            labels: {view_name: a numpy array of labels}
        """
        # self.device = device
        # print(f'feature[0].shape = {features[0].shape}')
        self.num_samples = features[0].shape[0]

        self.num_unaligned_samples = int(rate * self.num_samples)

        self.num_views = len(features)
        # print(self.num_samples, self.num_views)

        idx = np.random.RandomState(seed=seed).permutation(self.num_samples)
        # print(idx.shape)
        labels = np.asarray(labels)

        labels = labels[idx]

        self.labels = []
        self.features = []

        for n in range(self.num_views):

            feature = features[n][idx, :]

            if correspondence is False:
                if self.num_unaligned_samples > 0:
                    idx_n = np.random.RandomState(seed=int(seed + 1 + (n + 1) ** 2)).permutation(
                        self.num_unaligned_samples)
                    # print(f'idx_n = {idx_n}')
                    # print(labels[:self.num_unaligned_samples])
                    n_label = labels.copy()
                    n_label[:self.num_unaligned_samples] = labels[idx_n]
                    # print(n_label[:self.num_unaligned_samples])
                    # print('--')
                    feature[:self.num_unaligned_samples, :] = feature[idx_n, :]
                    label = torch.from_numpy(n_label).type(torch.LongTensor).to(device)
                    self.labels.append(label)
            feature = torch.as_tensor(feature, dtype=torch.float32, device=device)
            self.features.append(feature)

        if len(self.labels) == 0:
            self.labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)


    def __len__(self):
        """
        This function returns the size of the data set
        """
        if isinstance(self.labels, list):
            return self.labels[0].size(0)
        return self.labels.size(0)  # the number of samples

    def __getitem__(self, idx):
        """
        Given an index of sample, return the data and the label
        Args:
            idx: the index of sample

        Returns:
            features_multiview: a list [view1_feature, view2_feature, ...]
            labels: the labels corresponding to the features
        """
        if isinstance(self.labels, list):
            features = []
            labels = []
            for i in range(self.num_views):
                features.append(self.features[i][idx, :])
                labels.append(self.labels[i][idx])
            return features, labels, idx
        return [self.features[i][idx, :] for i in range(self.num_views)], self.labels[idx], idx


def load_handwritten_data(cfg):
    fname = os.path.join(cfg.data.root, 'handwritten.mat')
    data = io.loadmat(fname)
    labels = data['Y'][:, 0]
    features = {'Pix': data['X'][0, 0],
                'Fou': data['X'][0, 1],
                'Fac': data['X'][0, 2],
                'ZER': data['X'][0, 3],
                'KAR': data['X'][0, 4],
                'MOR': data['X'][0, 5]}
    view_names = ['Pix', 'Fou', 'Fac', 'ZER', 'KAR', 'MOR']
    view_dimensions = [features['Pix'].shape[1],
                       features['Fou'].shape[1],
                       features['Fac'].shape[1],
                       features['ZER'].shape[1],
                       features['KAR'].shape[1],
                       features['MOR'].shape[1]]
    cfg.data.modality_feature_names = view_names
    cfg.data.modality_feature_dims = view_dimensions

    return split_by_rate(cfg, features, labels)


def load_caltech7_data(cfg):
    fname = os.path.join(cfg.data.root, 'Caltech101_7.mat')
    data = io.loadmat(fname)
    labels = data['Y'][:, 0] - 1
    features = {'Gabor': data['X'][0, 0],
                'WM': data['X'][0, 1],
                'CENTRIST': data['X'][0, 2],
                'HOG': data['X'][0, 3],
                'GIST': data['X'][0, 4],
                'LBP': data['X'][0, 5]}
    view_names = ['Gabor', 'WM', 'CENTRIST', 'HOG', 'GIST', 'LBP']
    view_dimensions = [features['Gabor'].shape[1],
                       features['WM'].shape[1],
                       features['CENTRIST'].shape[1],
                       features['HOG'].shape[1],
                       features['GIST'].shape[1],
                       features['LBP'].shape[1]]

    cfg.data.modality_feature_names = view_names
    cfg.data.modality_feature_dims = view_dimensions

    return split_by_rate(cfg, features, labels)


def load_caltech20_data(cfg):
    fname = os.path.join(cfg.data.root, 'Caltech101_20.mat')
    data = io.loadmat(fname)
    labels = data['Y'][:, 0] - 1
    features = {'Gabor': data['X'][0, 0],
                'WM': data['X'][0, 1],
                'CENTRIST': data['X'][0, 2],
                'HOG': data['X'][0, 3],
                'GIST': data['X'][0, 4],
                'LBP': data['X'][0, 5]}
    view_names = ['Gabor', 'WM', 'CENTRIST', 'HOG', 'GIST', 'LBP']
    view_dimensions = [features['Gabor'].shape[1],
                       features['WM'].shape[1],
                       features['CENTRIST'].shape[1],
                       features['HOG'].shape[1],
                       features['GIST'].shape[1],
                       features['LBP'].shape[1]]

    # round number of samples for simplifying the following process
    cfg.data.modality_feature_names = view_names
    cfg.data.modality_feature_dims = view_dimensions

    return split_by_rate(cfg, features, labels)


def load_cathgen_data(cfg):
    """
    Load data and do feature normalization
    :param args:
    :return:
    """
    fname = os.path.join(cfg.data.root, 'cathgen_seed1.pkl')
    with open(fname, 'rb') as f:
        [features_train, features_valid, features_test,
         e_train, e_valid, e_test,
         view_names, view_dimensions] = pickle.load(f)

    num_views = len(features_train)

    features = {}
    for n in range(num_views):
        view = view_names[n]
        features[view] = np.concatenate((features_train[n], features_valid[n], features_test[n]), axis=0)
    labels = e_train + e_valid + e_test

    cfg.data.modality_feature_names = view_names
    cfg.data.modality_feature_dims = view_dimensions

    return split_by_rate(cfg, features, labels)


def load_previous_data(cfg, data_name):
    data_only, labels, features = load_full_data(data_name)
    view_names = list(features.keys())
    view_dimensions = [item.shape[1] for item in data_only]

    # round number of samples for simplifying the following process
    cfg.data.modality_feature_names = view_names
    cfg.data.modality_feature_dims = view_dimensions

    return split_by_rate(cfg, features, labels)


def split_by_rate(cfg, features, labels):
    split_rates = cfg.data.splits
    # round number of samples for simplifying the following process

    num_views = len(features)
    cfg.data.num_views = num_views
    cfg.data.cluster_num = int(np.max(labels) - np.min(labels) + 1)
    view_names = cfg.data.modality_feature_names
    view_dimensions = cfg.data.modality_feature_dims
    train_labels = []
    valid_labels = []
    test_labels = []
    train_samples = {}
    valid_samples = {}
    test_samples = {}
    epsilon = 1e-8


    dir = 'plots/'+ cfg.data.type
    check_dir(folder_name=dir)



    if cfg.data.is_filter:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-label_counts) 
        
        for idx in sorted_indices:
            print(f"Total labels:  {unique_labels[idx]} appear count: {label_counts[idx]}")
        
        most_common_indices = np.argsort(-label_counts)[:cfg.data.filter_num]
        
        labels_to_remove = unique_labels[most_common_indices]
        print(f'labels_to_remove = {labels_to_remove}')
        
 
        indices_to_remove = np.isin(labels, labels_to_remove)
        indices_to_remove = np.array(indices_to_remove)
        keys = list(features.keys())
      
        for i in range(cfg.data.num_views):
            fetures_x_raw = np.array(features[keys[i]])
            features[keys[i]] = fetures_x_raw[~indices_to_remove]
   
        labels = np.array(labels)[~indices_to_remove]
        print(f"Filtered labels = {labels}, length = {len(labels)}")

    if split_rates[0] == 1.0:
        train_labels = labels.tolist()
        for n in range(num_views):
            view = view_names[n]
            train_data = features[view]
            tmp_m = np.mean(train_data, axis=0)
            tmp_v = np.std(train_data, axis=0)
            tmp_v = np.where(tmp_v == 0, epsilon, tmp_v)
            samples1 = (train_data - tmp_m) / tmp_v
            train_samples[n] = samples1

    elif split_rates[0] + split_rates[1] == 1.0: 
        train_labels, valid_labels = train_test_split(labels, train_size=split_rates[0], random_state=cfg.seed) 
        train_labels = train_labels.tolist()
        valid_labels = valid_labels.tolist()
        for n in range(num_views):
            view = view_names[n]
            train_data, valid_data = train_test_split(features[view], train_size=split_rates[0], random_state=cfg.seed)
            tmp_m = np.mean(train_data, axis=0)
            tmp_v = np.std(train_data, axis=0)
            tmp_v = np.where(tmp_v == 0, epsilon, tmp_v)
            samples1 = (train_data - tmp_m) / tmp_v
            samples2 = (valid_data - tmp_m) / tmp_v
            train_samples[n] = samples1
            valid_samples[n] = samples2

    elif split_rates[0] + split_rates[2] == 1.0:  
        train_labels, test_labels = train_test_split(labels, train_size=split_rates[0], random_state=cfg.seed)  # 1200
        train_labels = train_labels.tolist()
        test_labels = test_labels.tolist()
        for n in range(num_views):
            view = view_names[n]
            train_data, test_data = train_test_split(features[view], train_size=split_rates[0], random_state=cfg.seed)
            tmp_m = np.mean(train_data, axis=0)
            tmp_v = np.std(train_data, axis=0)
            tmp_v = np.where(tmp_v == 0, epsilon, tmp_v)
            samples1 = (train_data - tmp_m) / tmp_v
            samples3 = (test_data - tmp_m) / tmp_v
            train_samples[n] = samples1
            test_samples[n] = samples3

    elif split_rates[0] <= 0.0:  
        pass

    else:
        valid_rate = split_rates[1] / (split_rates[1] + split_rates[2])
        train_labels, test_labels = train_test_split(labels, train_size=split_rates[0], random_state=cfg.seed)
        valid_labels, test_labels = train_test_split(test_labels, train_size=valid_rate, random_state=cfg.seed)
        train_labels = train_labels.tolist()
        valid_labels = valid_labels.tolist()
        test_labels = test_labels.tolist()

        for n in range(num_views):
            view = view_names[n]
            train_data, test_data = train_test_split(features[view], train_size=split_rates[0], random_state=cfg.seed)
            valid_data, test_data = train_test_split(test_data, train_size=valid_rate, random_state=cfg.seed)
            tmp_m = np.mean(train_data, axis=0)
            tmp_v = np.std(train_data, axis=0)
            tmp_v = np.where(tmp_v == 0, epsilon, tmp_v)
            samples1 = (train_data - tmp_m) / tmp_v
            samples2 = (valid_data - tmp_m) / tmp_v
            samples3 = (test_data - tmp_m) / tmp_v
            train_samples[n] = samples1
            valid_samples[n] = samples2
            test_samples[n] = samples3

  
    if cfg.data.raw_tsne:
        if len(valid_labels) > 0:
            for i in range(num_views):
                filename = 'view'+str(i)+'_valid'
                # plot_tSNE(outputs=valid_samples[i], labels=valid_labels, save_dir=dir, save_filename=filename)

    return train_samples, valid_samples, test_samples, train_labels, \
        valid_labels, test_labels, view_names, view_dimensions


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=15)  # 'small')
    ax.set_yticklabels(row_labels, fontsize=15)  # 'small')

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-75, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=15)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            if data[i, j] > 0.005:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
    return texts


def check_dir(folder_name):
    if os.path.exists(folder_name):
        print(f"The '{folder_name}' folder already exists.")
    else:
        os.makedirs(folder_name)
        print(f"The '{folder_name}' folder has been created.")


def prepare_dataloaders(cfg, device):
    data_name = cfg.data.type
    if data_name == 'cathgen':
        train_samples, valid_samples, test_samples, train_labels, \
            valid_labels, test_labels, view_names, view_dimensions = load_cathgen_data(cfg)
    elif data_name == 'caltech7':
        train_samples, valid_samples, test_samples, train_labels, \
            valid_labels, test_labels, view_names, view_dimensions = load_caltech7_data(cfg)
    elif data_name == 'caltech20':
        train_samples, valid_samples, test_samples, train_labels, \
            valid_labels, test_labels, view_names, view_dimensions = load_caltech20_data(cfg)
    elif data_name == 'handwritten':
        train_samples, valid_samples, test_samples, train_labels, \
            valid_labels, test_labels, view_names, view_dimensions = load_handwritten_data(cfg)
    else:  # 3source rt orl pro
        train_samples, valid_samples, test_samples, train_labels, \
            valid_labels, test_labels, view_names, view_dimensions = load_previous_data(cfg, data_name)

    data_loaders = {}

    if len(train_labels) > 0:

        train_data_sampler = MultiViewSampler(train_samples, labels=train_labels,
                                                rate=cfg.data.unaligned_rate, seed=cfg.seed,
                                                correspondence=cfg.data.correspondence, device=device)
        cfg.data.train_samples_num = len(train_data_sampler)  
        train_dataloader = DataLoader(train_data_sampler, batch_size=cfg.train.batch_size, shuffle=True)
        data_loaders['train'] = train_dataloader
    else:
    
        cfg.data.train_samples_num = 0
        train_dataloader = None

    # valid: dataset - dataloader
    if len(valid_labels) > 0:
            
        valid_data_sampler = MultiViewSampler(valid_samples, labels=valid_labels,
                                              rate=0.0, seed=cfg.seed, correspondence=True, device=device)
        cfg.data.valid_samples_num = len(valid_data_sampler)
        valid_dataloader = DataLoader(valid_data_sampler, batch_size=cfg.train.batch_size, shuffle=False)
        data_loaders['valid'] = valid_dataloader
    else: 
        cfg.data.valid_samples_num = cfg.data.train_samples_num
        valid_samples = train_samples
        valid_labels = train_labels
        valid_data_sampler = MultiViewSampler(valid_samples, labels=valid_labels,
                                              rate=0.0, seed=cfg.seed, correspondence=True, device=device)
        valid_dataloader = DataLoader(valid_data_sampler, batch_size=cfg.train.batch_size, shuffle=False)
        data_loaders['valid'] = valid_dataloader

    # test: dataset - dataloader
    if len(test_labels) > 0:

        test_data_sampler = MultiViewSampler(test_samples, labels=test_labels,
                                             rate=0.0, seed=cfg.seed, correspondence=True, device=device)
        cfg.data.test_samples_num = len(test_data_sampler)
        test_dataloader = DataLoader(test_data_sampler, batch_size=cfg.train.batch_size, shuffle=False)
        data_loaders['test'] = test_dataloader
    else:
        cfg.data.test_samples_num = 0
        test_dataloader = None

    print('load_{}_data_mv-> '.format(data_name), view_names)
    print('load_{}_data_mv-> '.format(data_name), view_dimensions)

    print('#training samples = {}'.format(cfg.data.train_samples_num))
    print('#unaligned training samples = {}'.format(train_data_sampler.num_unaligned_samples))

    print('#validation samples = {}'.format(cfg.data.valid_samples_num))
    print('#testing samples = {}'.format(cfg.data.test_samples_num))

    return data_loaders, cfg


# not changed -- for future
def prepare_dataloaders_remove_view(args, view_id):
    if args.dataname == 'cathgen':
        train_samples, valid_samples, test_samples, train_labels, \
            valid_labels, test_labels, view_names, view_dimensions = load_cathgen_data(args)
    elif args.dataname == 'caltech7':
        train_samples, valid_samples, test_samples, train_labels, \
            valid_labels, test_labels, view_names, view_dimensions = load_caltech7_data(args)
    elif args.dataname == 'caltech20':
        train_samples, valid_samples, test_samples, train_labels, \
            valid_labels, test_labels, view_names, view_dimensions = load_caltech20_data(args)
    elif args.dataname == 'handwritten':
        train_samples, valid_samples, test_samples, train_labels, \
            valid_labels, test_labels, view_names, view_dimensions = load_handwritten_data(args)

    # view_names[view_id] = []
    num = len(view_names)
    view_names = [view_names[i] for i in range(num) if i != view_id]
    # view_dimensions[view_id] = []
    view_dimensions = [view_dimensions[i] for i in range(num) if i != view_id]
    train_samples[view_id] = None
    valid_samples[view_id] = None
    test_samples[view_id] = None
    train_samples_new = {}
    valid_samples_new = {}
    test_samples_new = {}
    n = 0
    for key in train_samples.keys():
        if train_samples[key] is not None:
            train_samples_new[n] = train_samples[key]
            valid_samples_new[n] = valid_samples[key]
            test_samples_new[n] = test_samples[key]
            n += 1

    train_data_sampler = MultiViewSampler(train_samples_new, labels=train_labels,
                                          rate=args.rate, seed=args.seed, correspondence=args.correspondence)
    valid_data_sampler = MultiViewSampler(valid_samples_new, labels=valid_labels,
                                          rate=0.0, seed=args.seed, correspondence=True)
    test_data_sampler = MultiViewSampler(test_samples_new, labels=test_labels,
                                         rate=0.0, seed=args.seed, correspondence=True)


    data_loaders = {'train': DataLoader(train_data_sampler,
                                        batch_size=args.batch_size,  # define your batch size here
                                        shuffle=True),
                    'valid': DataLoader(valid_data_sampler,
                                        batch_size=args.batch_size,  # define your batch size here
                                        shuffle=False),
                    'test': DataLoader(test_data_sampler,
                                       batch_size=args.batch_size,  # define your batch size here
                                       shuffle=False)
                    }
    return data_loaders, view_dimensions, view_names, len(train_data_sampler)


def get_mat_data(name, classes=0):
    if classes != 0:  # cal
        dataset = io.loadmat("./data/Caltech101_" + str(classes) + ".mat")  
    else:
        if name == "orl":
            dataset = io.loadmat("./data/" + name + ".mat")
            dataset_y = io.loadmat("./data/" + name + "_y.mat")
            dataset["Y"] = dataset_y["y"][0, 0]
        else:
            dataset = io.loadmat("./data/" + name + ".mat")
    full_data = np.squeeze(dataset["X"])
    full_labels = np.squeeze(dataset["Y"]) 
    full_labels = (full_labels - np.min(full_labels)).astype("int64")
    data = []
    index = np.random.permutation(len(full_labels))
    for i in range(len(full_data)):
        data.append(full_data[i][index].astype("float64"))
    return data, full_labels[index]


def load_full_data(name):
    feature_data = {}  # hw/cal7/cal20 

    name = name.lower() 

    if name in ['hw', 'handwritten']:
        full_data, full_labels = load_UCImultifeature(views=[0, 1, 2, 3, 4, 5])
        full_labels = full_labels.astype("int64") 
    elif name in ['cal7', 'caltech7']:
        full_data, full_labels = get_mat_data("cal", 7)
    elif name in ['cal20', 'caltech20']:
        full_data, full_labels = get_mat_data("cal", 20)
    if name in ['rt', 'reuters']:
        full_data, full_labels = get_mat_data("Reuters")
        feature_name = ['English', 'French', 'German', 'Spanish', 'Italian']

        feature_data = {key: sp.csr_matrix(value).toarray() for key, value in zip(feature_name, full_data)}

    elif name == 'orl':
        full_data, full_labels = get_mat_data("orl")
        feature_name = [str(i + 1) for i in range(len(full_data))]

        feature_data = {key: value for key, value in zip(feature_name, full_data)}

    elif name in ['mv', 'movie', 'movies']:
        actors = np.loadtxt("./data/my_movies/M2.txt")
        keywords = np.loadtxt("./data/my_movies/M1.txt")
        full_data = [actors, keywords]
        full_labels = np.loadtxt("./data/my_movies/Mact.txt").astype("int64")

        feature_data = {'actors': actors, 'keywords': keywords}

    elif name in ['pro', 'prokaryotic']:
        dataset = io.loadmat("./data/prokaryotic.mat")
        full_data = [dataset["text"].astype("float64"), dataset["proteome_comp"].astype("float64"),
                     dataset["gene_repert"].astype("float64")]
        full_labels = dataset["truth"].squeeze().astype("int64")

        feature_data = {"text": dataset["text"].astype("float64"),
                        "proteome_comp": dataset["proteome_comp"].astype("float64"),
                        "gene_repert": dataset["gene_repert"].astype("float64")}

    elif name in ['3s', '3-sources', '3sources', '3_sources']:
        dataset = io.loadmat("./data/3-sources.mat")

        full_data = [dataset["bbc"].A.astype("float64"), dataset["guardian"].A.astype("float64"),
                     dataset["reuters"].A.astype("float64")]
        full_labels = dataset["truth"].squeeze().astype("int64")

        feature_data = {"bbc": dataset["bbc"].A.astype("float64"), "guardian": dataset["guardian"].A.astype("float64"),
                        "reuters": dataset["reuters"].A.astype("float64")}

    return full_data, full_labels, feature_data
