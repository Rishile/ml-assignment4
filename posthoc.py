import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

ALPHA = {"majority": 0.0, "sqrt_prior": 0.5, "prior": 1.0, "purity": 0.0}


def metrics_from_cm(cm):
    """Accuracy, balanced accuracy, macro/weighted F1, multiclass MCC and Cohen's kappa."""
    cm = cm.astype(np.float64)
    N = cm.sum()
    tp = np.diag(cm)
    t = cm.sum(1)             # true counts per class
    p = cm.sum(0)             # predicted counts per class
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(t + p > 0, 2 * tp / (t + p), 0.0)
        rec = np.where(t > 0, tp / t, 0.0)
    present = t > 0
    c = tp.sum()
    mcc_den = np.sqrt((N ** 2 - (p ** 2).sum()) * (N ** 2 - (t ** 2).sum()))
    pe = (t * p).sum() / N ** 2
    return dict(accuracy=c / N, balanced_accuracy=rec[present].mean(),
                macro_f1=f1[present].mean(), weighted_f1=(f1 * t).sum() / N,
                mcc=(c * N - (p * t).sum()) / mcc_den if mcc_den > 0 else 0.0,
                kappa=(c / N - pe) / (1 - pe) if pe < 1 else 0.0)


def confusion(y, pred, n_classes):
    return np.bincount(y * n_classes + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


class CodebookClusterer:
    """Caches linkage / k-means fits so that many (k, rule) combinations share work."""
    def __init__(self, W, hits):
        self.W, self.hits = W, hits
        self._Z = None
        self._cache = {}

    def assign(self, method, k, weighting):
        M = self.W.shape[0]
        if method == "none" or k is None or k >= M:
            return np.arange(M)
        key = (method, k, weighting)
        if key in self._cache:
            return self._cache[key]
        if method == "ward":
            if self._Z is None:
                self._Z = linkage(self.W.astype(np.float64), "ward")
            lab = fcluster(self._Z, t=k, criterion="maxclust") - 1
        elif method == "kmeans":
            sw = (self.hits + 1e-3) if weighting == "hits" else None
            lab = KMeans(n_clusters=k, n_init=3, random_state=None).fit(self.W, sample_weight=sw).labels_
        else:
            raise ValueError(method)
        _, lab = np.unique(lab, return_inverse=True)
        self._cache[key] = lab
        return lab


def build_label_map(W, xy, clusters, bmu_lab, y_lab, n_classes, rule="majority", tau=0.7,
                    fallback="weight"):
    """Return (neuron_labels [M], info) given neuron->cluster assignment and labelled data."""
    M = W.shape[0]
    ncl = int(clusters.max()) + 1
    Cn = np.bincount(bmu_lab * n_classes + y_lab, minlength=M * n_classes).reshape(M, n_classes)
    Cc = np.bincount(clusters[bmu_lab] * n_classes + y_lab, minlength=ncl * n_classes).reshape(ncl, n_classes)
    totals = Cn.sum(0).astype(np.float64)
    w = 1.0 / np.maximum(totals, 1.0) ** ALPHA[rule]
    cl_label = (Cc * w).argmax(1)
    cl_known = Cc.sum(1) > 0
    lab = cl_label[clusters]
    known = cl_known[clusters]
    if rule == "purity":
        purity = Cc.max(1) / np.maximum(Cc.sum(1), 1)
        use_n = (purity[clusters] < tau) & (Cn.sum(1) > 0)
        lab = np.where(use_n, (Cn * w).argmax(1), lab)
        known = np.where(use_n, True, known)
    lab = lab.copy()
    n_unknown = int((~known).sum())
    if n_unknown and known.any():
        ui, ki = np.where(~known)[0], np.where(known)[0]
        P = xy if fallback == "grid" else W
        sq_u, sq_k = (P[ui] ** 2).sum(1), (P[ki] ** 2).sum(1)
        d = sq_u[:, None] + sq_k[None, :] - 2 * P[ui] @ P[ki].T
        lab[ui] = lab[ki[d.argmin(1)]]
    elif n_unknown:
        lab[:] = 0
    purity_w = float(Cn.max(1).sum() / max(Cn.sum(), 1))
    return lab.astype(np.int64), dict(n_clusters=ncl, neuron_purity=purity_w, n_unlabelled=n_unknown)


def sweep_posthoc(W, xy, bmu_tr, y_tr, bmu_eval, y_eval, posthoc_list, n_classes, label_idx=None):
    """Evaluate many post-hoc configurations on ONE trained map (no retraining)."""
    M = W.shape[0]
    hits = np.bincount(bmu_tr, minlength=M)
    clus = CodebookClusterer(W, hits)
    if label_idx is None:
        b_l, y_l = bmu_tr, y_tr
    else:
        b_l, y_l = bmu_tr[label_idx], y_tr[label_idx]
    rows = []
    for ph in posthoc_list:
        c = clus.assign(ph["cluster_method"], ph["k"], ph["cluster_weighting"])
        lab, info = build_label_map(W, xy, c, b_l, y_l, n_classes, ph["label_rule"],
                                    ph["purity_tau"], ph["fallback"])
        m = metrics_from_cm(confusion(y_eval, lab[bmu_eval], n_classes))
        rows.append({**ph, **m, **info, "empty_frac": float((hits == 0).mean())})
    return rows
