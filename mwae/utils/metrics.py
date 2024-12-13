import torch

# cpython
from sklearn.metrics.cluster._expected_mutual_info_fast import expected_mutual_information
from scipy.optimize import _lsap_module


def entropy(labels):
    if labels.numel() == 0:
        return torch.tensor(1.0, dtype=torch.float64)

    label_idx = torch.unique(labels, return_inverse=True)[1]
    minlength = int(labels.max().item() + 1)
    pi = torch.bincount(label_idx, minlength=minlength).to(torch.float64)
    pi = pi[pi > 0]
    pi_sum = pi.sum()
    return -torch.sum((pi / pi_sum) * (torch.log(pi) - torch.log(pi_sum)))


def _generalized_average(U, V, average_method):
    if average_method == "min":
        return torch.min(U, V)
    elif average_method == "geometric":
        return torch.sqrt(U * V)
    elif average_method == "arithmetic":
        return (U + V) / 2.0
    elif average_method == "max":
        return torch.max(U, V)
    else:
        raise ValueError(
            "'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'"
        )


def contingency_matrix(labels_true, labels_pred, sparse=True, eps=None):
    """
    return: contingency_matrix
    """
   
    classes, class_idx = torch.unique(labels_true, return_inverse=True)

   
    clusters, cluster_idx = torch.unique(labels_pred, return_inverse=True)

  
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]

    v1 = torch.stack((class_idx, cluster_idx), dim=0)

    device = v1.device

    v2 = torch.ones(class_idx.shape[0], dtype=torch.int64).to(device)

    contingency = torch.sparse_coo_tensor(
        v1,
        v2,
        (n_classes, n_clusters),
        dtype=torch.int64,
    )
    if sparse:
        pass

    else:
       
        contingency = contingency.to_dense()
        if eps is not None:
            contingency = contingency + eps
    return contingency


def mutual_info_score(U, V, contingency):
    if contingency is None:
        contingency = contingency_matrix(U, V, sparse=True)


    contingency_coo = contingency.coalesce()
    nonzero_indices = contingency_coo.indices()
    nzx = nonzero_indices[0]
    nzy = nonzero_indices[1]

    contingency_dense = contingency.to_dense()

    nz_val = contingency_dense[nzx, nzy]

    contingency_sum = contingency_dense.sum()

    pi = torch.ravel(contingency_dense.sum(dim=1))
    pj = torch.ravel(contingency_dense.sum(dim=0))

    log_contingency_nm = torch.log(nz_val)

    contingency_nm = nz_val / contingency_sum

    outer = pi.take(nzx) * pj.take(nzy)

    log_outer = -torch.log(outer) + torch.log(pi.sum()) + torch.log(pj.sum())

    mi = (
            contingency_nm * (log_contingency_nm - torch.log(contingency_sum))
            + contingency_nm * log_outer
    )

    eps = torch.finfo(mi.dtype).eps
    mi = torch.where(torch.abs(mi) < eps, 0.0, mi)

    mi_sum = mi.sum()
    mi_sum = torch.clamp(mi_sum, 0.0, float('inf'))

    return mi_sum


def normalized_mutual_info_score(
        labels_true, labels_pred, contingency, *, average_method="arithmetic"
):

    classes = torch.unique(labels_true)
    clusters = torch.unique(labels_pred)

    if (
            classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    if contingency is None:
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True) 
    contingency = contingency.to(torch.float64, non_blocking=True)
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    if mi == 0:
        return 0.0

    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)

    normalizer = _generalized_average(h_true, h_pred, average_method)
    return mi / normalizer


def adjusted_mutual_info_score(labels_true, labels_pred, contingency, average_method="arithmetic"):
    n_samples = labels_true.shape[0]
    classes = torch.unique(labels_true)
    clusters = torch.unique(labels_pred)

    if (
            classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    if contingency is None:
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    contingency = contingency.to(torch.float64, non_blocking=True)
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    device = mi.device
    contingency_dense = contingency.to_dense()

    v = contingency_dense.detach().cpu().numpy()
    emi = expected_mutual_information(v, n_samples)
    emi = torch.tensor(emi, dtype=torch.float64).to(device)

    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    normalizer = _generalized_average(h_true, h_pred, average_method)
    denominator = normalizer - emi

    eps = torch.finfo(torch.float64).eps
    eps = torch.tensor(eps, dtype=torch.float64, device=device)
    if denominator < 0:
        denominator = torch.min(denominator, eps)
    else:
        denominator = torch.max(denominator, eps)

    ami = (mi - emi) / denominator

    return ami


def pair_confusion_matrix(labels_true, labels_pred, contingency):
    n_samples = torch.tensor(labels_true.shape[0], dtype=torch.int64)

    if contingency is None:
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True) 
    contingency = contingency.to(torch.float64, non_blocking=True)
  
    contingency_dense = contingency.to_dense()

    n_c = contingency_dense.sum(dim=1).squeeze()
    n_k = contingency_dense.sum(dim=0).squeeze()
    sum_squares = (contingency_dense ** 2).sum()

    C = torch.empty((2, 2), dtype=torch.int64, device=labels_true.device)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = (contingency_dense.matmul(n_k).sum() - sum_squares)
    C[1, 0] = (contingency_dense.t().matmul(n_c).sum() - sum_squares)
    C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares

    return C


def adjusted_rand_score2(labels_true, labels_pred, contingency):
    """Rand index adjusted for chance.

    The raw RI score is then "adjusted for chance" into the ARI score
    using the following scheme::

        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

    The adjusted Rand index is thus ensured to have a value close to
    0.0 for random labeling independently of the number of clusters and
    samples and exactly 1.0 when the clusterings are identical (up to
    a permutation).

    ARI is a symmetric measure.

    """
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred, contingency)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    # Calculate the Adjusted Rand Index
    numerator = 2.0 * (tp * tn - fn * fp)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return numerator / denominator


def cluster_accuracy2(true_labels, predict_labels):

    device = predict_labels.device

    predict_max = torch.max(predict_labels)
    true_max = torch.max(true_labels)

    dim = max(predict_max, true_max) + 1

    cost_matrix = torch.zeros((dim, dim), dtype=torch.int64, device=device)

    class_num = predict_labels.shape[0]

    predict_labels = predict_labels.to(torch.int64)
    true_labels = true_labels.to(torch.int64)

    for i in range(class_num):
        cost_matrix[predict_labels[i].item(), true_labels[i].item()] += 1
    res = torch.max(cost_matrix) - cost_matrix
    cos = res.detach().cpu().numpy()
    ind = _lsap_module.calculate_assignment(cos) 
    ind_torch = (torch.from_numpy(ind[0]).to(device), torch.from_numpy(ind[1]).to(device))
    ind_transposed = torch.stack(ind_torch, dim=1)

    total_cost = torch.sum(torch.stack([cost_matrix[i, j] for i, j in ind_transposed]))

    result = total_cost * 1.0 / class_num
    return result


def adjusted_mutual_info_score2(labels_true, labels_pred):
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    ami = adjusted_mutual_info_score(labels_true, labels_pred, contingency)
    return ami
