import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import config as C
from som_lib import SOM, map_shape
from posthoc import CodebookClusterer, build_label_map


def make_som(cfg, X):
    rows, cols = map_shape(cfg["map_side"], cfg["aspect"], X)
    return SOM(rows, cols, topology=cfg["topology"], neighbourhood=cfg["neighbourhood"],
               sigma0_frac=cfg["sigma0_frac"], sigma_final=cfg["sigma_final"], lr0=cfg["lr0"],
               lr_final=cfg["lr_final"], decay=cfg["decay"], epochs=cfg["epochs"], init=cfg["init"],
               mode=cfg["mode"], metric=cfg["metric"])


def stratified_label_subset(y, fraction):
    if fraction >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng()
    idx = []
    for c in np.unique(y):
        rows = np.where(y == c)[0]
        n = max(1, int(np.ceil(fraction * len(rows))))
        idx.append(rng.choice(rows, n, replace=False))
    return np.sort(np.concatenate(idx))


class SOMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cfg=None, label_fraction=1.0):
        self.cfg = cfg
        self.label_fraction = label_fraction

    def fit(self, X, y):
        cfg = {**C.DEFAULT, **(self.cfg or {})}
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        self.som_ = make_som(cfg, X).fit(X)
        self.bmu_train_ = self.som_.bmus(X)[0][:, 0]
        self.y_train_ = y_idx
        self.hits_ = np.bincount(self.bmu_train_, minlength=self.som_.n_neurons)
        self.clusterer_ = CodebookClusterer(self.som_.W_, self.hits_)
        self.cfg_ = cfg
        return self.relabel(self.label_fraction)

    def relabel(self, label_fraction=1.0, **posthoc):
        """(Re)build the neuron->class map from stored BMUs; no retraining. Returns self."""
        cfg = {**self.cfg_, **posthoc}
        self.cfg_ = cfg
        li = stratified_label_subset(self.y_train_, label_fraction)
        clusters = self.clusterer_.assign(cfg["cluster_method"], cfg["k"], cfg["cluster_weighting"])
        self.cluster_of_neuron_ = clusters
        self.neuron_labels_, self.label_info_ = build_label_map(
            self.som_.W_, self.som_.xy, clusters, self.bmu_train_[li], self.y_train_[li],
            len(self.classes_), cfg["label_rule"], cfg["purity_tau"], cfg["fallback"])
        return self

    def predict(self, X):
        b = self.som_.bmus(X)[0][:, 0]
        return self.classes_[self.neuron_labels_[b]]

    def predict_topm(self, X, m=3):
        """Soft prediction: inverse-distance weighted vote over the m best-matching neurons."""
        idx, dist = self.som_.bmus(X, m)
        w = 1.0 / (dist + 1e-6)
        votes = np.zeros((len(idx), len(self.classes_)))
        for j in range(idx.shape[1]):
            np.add.at(votes, (np.arange(len(idx)), self.neuron_labels_[idx[:, j]]), w[:, j])
        return self.classes_[votes.argmax(1)]
