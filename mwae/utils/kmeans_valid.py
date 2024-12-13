import torch
from kmeans_gpu import KMeans
from utils.metrics import contingency_matrix, adjusted_mutual_info_score, adjusted_rand_score2, \
    cluster_accuracy2, normalized_mutual_info_score
import logging
logger = logging.getLogger("mwae")


def kmeans_for_single_view(cfg, out, labels):
    if out.shape[0] >= cfg.data.cluster_num:
        with torch.no_grad():
            kmeans = KMeans(
                n_clusters=cfg.data.cluster_num,
                max_iter=100,
                tolerance=1e-4,
                distance='euclidean',
                sub_sampling=None,
                max_neighbors=15,
            )

            predict_labels2, _ = kmeans.fit_predict(out)
            del _
            contingency = contingency_matrix(labels, predict_labels2, sparse=True)

            acc = cluster_accuracy2(labels, predict_labels2)
            nmi = normalized_mutual_info_score(labels, predict_labels2, contingency)
            ami = adjusted_mutual_info_score(labels, predict_labels2, contingency)
            ari = adjusted_rand_score2(labels, predict_labels2, contingency)

    return acc, nmi, ami, ari


def kmeans_for_multiview(cfg, inputs, zs, labels, device, best_result_batch):
    num_views = len(inputs)
    acc_all = torch.zeros(num_views, device=device)
    nmi_all = torch.zeros(num_views, device=device)
    ami_all = torch.zeros(num_views, device=device)
    ri_all = torch.zeros(num_views, device=device)
    # data_type = cfg.data.type
    for i in range(num_views):
        out = zs[i]
        if isinstance(labels, list):
            expected_label = labels[i]
        else:
            expected_label = labels
        acc, nmi, ami, ari = kmeans_for_single_view(cfg, out, expected_label)
        acc_all[i], nmi_all[i], ami_all[i], ri_all[i] = acc, nmi, ami, ari

    with torch.no_grad():
        test_acc_avg = torch.mean(acc_all)
        test_nmi_avg = torch.mean(nmi_all)
        test_ami_avg = torch.mean(ami_all)
        test_ri_avg = torch.mean(ri_all)


    return test_acc_avg, test_nmi_avg, test_ami_avg, test_ri_avg, best_result_batch


def update_epoch_best_result(cfg, epoch_acc_avg, epoch_nmi_avg, epoch_ami_avg, epoch_ari_avg, n, best_result_epoch, modality=-1):
    data_type = cfg.data.type
    metric = cfg.metric
    # log epoch metric
    loss_log = {'test_acc_avg': epoch_acc_avg.item(),
                'test_nmi_avg': epoch_nmi_avg.item(),
                'test_ami_avg': epoch_ami_avg.item(),
                'test_ari_avg': epoch_ari_avg.item()}
    if modality == -1:
        result = {'Role': 'Epoch Avg#', 'epoch': n, data_type: loss_log}
    else:
        result = {'Modality': modality, 'epoch': n, data_type: loss_log}
    logger.info(str(result))
    #  0: acc, 1:nmi, -1:ami
    is_update = False
    if metric == 1:
        if epoch_nmi_avg.item() > best_result_epoch['test_nmi_avg']:
            is_update = True
    elif metric == 0:  # default acc
        if epoch_acc_avg.item() > best_result_epoch['test_acc_avg']:
            is_update = True
    elif metric == -1:
        if epoch_ami_avg.item() > best_result_epoch['test_ami_avg']:
            is_update = True

    if is_update:
        best_result_epoch['epoch'] = n
        best_result_epoch['test_acc_avg'] = epoch_acc_avg.item()
        best_result_epoch['test_nmi_avg'] = epoch_nmi_avg.item()
        best_result_epoch['test_ami_avg'] = epoch_ami_avg.item()
        best_result_epoch['test_ari_avg'] = epoch_ari_avg.item()
        logger.info(f'Modality {modality}: Update best result epoch = {best_result_epoch}')

    return best_result_epoch






