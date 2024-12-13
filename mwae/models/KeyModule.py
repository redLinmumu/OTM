import numpy as np
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, x_dim: int, h_dim: int):
        super(MLP, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.layer1 = nn.Linear(self.x_dim, self.h_dim)
        self.layer2 = nn.Linear(self.h_dim, self.h_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        """
        Define the feed-forward process of the model, i.e., output = model(inputs)
        :param x: concatenated latent representations of all views
        :return: the representation used to classify the data
        """
        z = self.layer1(x)
        return self.layer2(self.tanh(z))


class MultiMLP(nn.Module):
    def __init__(self, num_views: int, x_dim: int, h_dim: int):
        super(MultiMLP, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.mlps = nn.ModuleList()
        for i in range(num_views):
            self.mlps.append(MLP(x_dim=x_dim, h_dim=h_dim))

    def forward(self, zs: torch.Tensor):
        """
        Define the feed-forward process of the model, i.e., output = model(inputs)
        :param x: concatenated latent representations of all views
        :return: the representation used to classify the data

        Args:
            zs: the output of wae model
        """
        ts = []
        for i in range(len(zs)):
            t = self.mlps[i](zs[i])
            ts.append(t)
        return ts


class SingleEncoder(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, h_dim: int = 50):
        super(SingleEncoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z = self.fc2(h)
        z_normalized = F.normalize(z)
        return z_normalized


class MultiEncoders(nn.Module):
    def __init__(self, x_dims: list, z_dim: int, h_dim: int = 50, device='cpu'):
        super(MultiEncoders, self).__init__()
        self.num_views = len(x_dims)
        self.x_dims = np.asarray(x_dims)
        self.z_dim = z_dim
        self.encoders = nn.ModuleList()
        for n in range(self.num_views):
            encoder = SingleEncoder(x_dim=self.x_dims[n], z_dim=z_dim, h_dim=h_dim)
            self.encoders.append(encoder)
        self.encoders.to(device)
        # print(f'encoders to device = {device}')

    def forward(self, xs: list):
        zs = []
        for n in range(self.num_views):
            z = self.encoders[n](xs[n])
            zs.append(z)
        return zs


class SingleDecoder(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, h_dim: int = 50):
        super(SingleDecoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        h = F.relu(self.fc3(z))
        xhat = self.fc4(h)
        xhat_normalized = F.normalize(xhat)
        return xhat_normalized


class MultiDecoders(nn.Module):
    def __init__(self, x_dims: list, z_dim: int, h_dim: int = 50, device='cpu'):
        super(MultiDecoders, self).__init__()
        self.num_views = len(x_dims)
        self.x_dims = np.asarray(x_dims)
        self.z_dim = z_dim
        self.decoders = nn.ModuleList()
        for n in range(self.num_views):
            decoder = SingleDecoder(x_dim=self.x_dims[n], z_dim=z_dim, h_dim=h_dim)
            self.decoders.append(decoder)
        self.decoders.to(device)
        # print(f'encoders to device = {device}')

    def forward(self, zs: list):
        xhats = []
        for n in range(self.num_views):
            xhat = self.decoders[n](zs[n])
            xhats.append(xhat)
        return xhats


class AutoEncoder(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, h_dim: int = 50):
        super(AutoEncoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        z = self.fc2(h)
        return z

    def decode(self, z):
        h = F.relu(self.fc3(z))
        xhat = self.fc4(h)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)


class MultiAutoEncoders(nn.Module):
    def __init__(self, x_dims: list, z_dim: int, h_dim: int = 50):
        super(MultiAutoEncoders, self).__init__()
        self.num_views = len(x_dims)
        self.x_dims = np.asarray(x_dims)
        self.z_dim = z_dim
        self.autoencoders = nn.ModuleList()
        for n in range(self.num_views):
            self.autoencoders.append(AutoEncoder(x_dim=self.x_dims[n], z_dim=z_dim, h_dim=h_dim))

    def forward(self, xs: list):
        zs = []
        xhats = []
        for n in range(self.num_views):
            z, xhat = self.autoencoders[n](xs[n])
            zs.append(z)
            xhats.append(xhat)
        return zs, xhats
