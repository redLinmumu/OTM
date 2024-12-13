import torch
import ot
from utils.gw_transport import gromov_wasserstein, fused_gromov_wasserstein
from torch.nn import functional as F
# from ot.gromov import fused_gromov_wasserstein
from torch.distributions.dirichlet import Dirichlet
import logging
logger = logging.getLogger("get_distance")

def reconstruction_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='mean')


def classification_loss(yhat, y):
    return F.cross_entropy(yhat, y, reduction='sum')


def distance_tensor(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.abs(x_col - y_row) ** p
    return distance.sum(dim=2)


def mixer_ot(zs: list, outer_iter: int = 5, b_size: int = None, method: str = 'w', reg: float = 1e-2, f_alpha: float = '0.5'):
    """
    Optimal transport mixer: Wasserstein mixer or GW mixer
    :param zs: a list of tensor
    :param outer_iter: the number of outer iterations
    :param b_size: the size of barycenter
    :param method: wasserstein barycenter or gw barycenter
    :param reg: the weight of entropic regularizer
    :return:
    """
    # initialization
    if b_size is None:
        z = zs[0].detach()
    else:
        z = zs[0][:b_size, :].detach()
    a = torch.ones_like(z[:, 0]) / z.shape[0]
    bs = [torch.ones_like(zs[i][:, 0]) / zs[i].shape[0] for i in range(len(zs))]

    logger.info(f' >>> method = {method}')

    # compute ot plans
    for m in range(outer_iter):
        trans = []
        tmp = 0
        for i in range(len(zs)):
            if method == 'w':
                cost = distance_tensor(pts_src=z,
                                       pts_dst=zs[i].detach())

                tran = ot.lp.emd(a=a, b=bs[i], M=cost)
            elif method == 'gw':
                cost1 = distance_tensor(pts_src=z,
                                        pts_dst=z)
                cost2 = distance_tensor(pts_src=zs[i].detach(),
                                        pts_dst=zs[i].detach())

                tran = gromov_wasserstein(C1=cost1, C2=cost2, p=a, q=bs[i], loss_fun='square_loss')
            elif method == 'fgw':
                M = distance_tensor(pts_src=z, pts_dst=zs[i].detach())
                cost1 = distance_tensor(pts_src=z,
                                        pts_dst=z)
                cost2 = distance_tensor(pts_src=zs[i].detach(),
                                        pts_dst=zs[i].detach())
                tran = fused_gromov_wasserstein(M=M, C1=cost1, C2=cost2, p=a, q=bs[i], loss_fun='square_loss', alpha=f_alpha)
            trans.append(tran)

            tmp += tran @ zs[i].detach() 
        z = tmp

    # mixing
    beta = 0.5 * torch.ones(len(zs))
    dirichlet_dist = Dirichlet(beta)
    masks = dirichlet_dist.sample((z.shape[0], z.shape[1])).to(zs[0].device)
    mix = 0
    for i in range(len(zs)):

        mix += masks[:, :, i] * (trans[i] @ zs[i])
    return mix, trans


def cal_spectral_loss(cfg, out):

    cost1 = distance_tensor(out, out)
    tmp = torch.ones(5, 5)
    for c in range(cfg.data.cluster_num - 1):
        tmp = torch.block_diag(tmp, torch.ones(5, 5))
    cost2 = 1 - tmp.to(out.device)

    a = torch.ones_like(out[:, 0]) / out.shape[0]
    b = torch.ones_like(cost2[:, 0]) / cost2.shape[0]

    tran = gromov_wasserstein(C1=cost1.detach(), C2=cost2, p=a, q=b, loss_fun='square_loss')

    cost = (cost1 ** 2) @ torch.ones_like(tran) / out.shape[0] + (
            torch.ones_like(tran) / cost2.shape[0]) @ (cost2 ** 2) - 2 * cost1 @ tran @ cost2.T
    loss_cluster = (cost * tran).sum()

    return loss_cluster

