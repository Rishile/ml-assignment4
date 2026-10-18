import time
import numpy as np
from numba import njit

MEXICAN_NEG_DAMP = 0.1     # scale of the negative lobe of the Mexican-hat neighbourhood
KIND_CODE = {"gaussian": 0, "bubble": 1, "triangle": 2, "mexican_hat": 3}


def new_rng():
    ss = np.random.SeedSequence()          # fresh OS entropy: deliberately not seeded
    return np.random.default_rng(ss), str(ss.entropy)


# ----------------------------------------------------------------------------
# Grid geometry
# ----------------------------------------------------------------------------
def grid_coords(rows, cols, topology):
    r, c = np.divmod(np.arange(rows * cols), cols)
    if topology == "rect":
        xy = np.stack([c, r], 1)
    elif topology == "hex":                       # offset rows, unit centre spacing
        xy = np.stack([c + 0.5 * (r % 2), r * np.sqrt(3) / 2], 1)
    else:
        raise ValueError(topology)
    return xy.astype(np.float32)


def grid_sqdist(xy):
    sq = (xy ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * xy @ xy.T
    return np.maximum(d2, 0).astype(np.float32)


def map_shape(side, aspect, X):
    """(rows, cols) with rows*cols ~ side^2. 'eigen' uses sqrt(lambda1/lambda2) of the data."""
    m = side * side
    if aspect == "square":
        return side, side
    sub = X[:50000] - X[:50000].mean(0)
    ev = np.linalg.eigvalsh(sub.T @ sub / len(sub))
    ratio = float(np.clip(np.sqrt(max(ev[-1], 1e-12) / max(ev[-2], 1e-12)), 1.0, 4.0))
    rows = max(2, int(round(np.sqrt(m / ratio))))
    cols = max(2, int(round(m / rows)))
    return rows, cols


def neighbourhood(d2, sigma, kind):
    s2 = sigma * sigma
    if kind == "gaussian":
        return np.exp(-d2 / (2 * s2))
    if kind == "bubble":
        return (d2 <= s2).astype(np.float32)
    if kind == "triangle":
        return np.maximum(0.0, 1.0 - np.sqrt(d2) / sigma)
    if kind == "mexican_hat":                       # negative lobe damped for stability
        h = (1.0 - d2 / s2) * np.exp(-d2 / (2 * s2))
        return np.where(h < 0, MEXICAN_NEG_DAMP * h, h)
    raise ValueError(kind)


def schedule(v0, v1, t, T, decay):
    frac = t / max(T - 1, 1)
    if decay == "linear":
        return v0 + (v1 - v0) * frac
    return v0 * (v1 / v0) ** frac


# ----------------------------------------------------------------------------
# Best-matching-unit search
# ----------------------------------------------------------------------------
def bmu_search(X, W, metric="euclidean", k=1, chunk=None):
    """Indices and distances of the k nearest codebook vectors for each row of X.

    Returns (idx [N,k] int32, dist [N,k] float32). Euclidean uses the expansion
    |x-w|^2 = |x|^2 - 2 x.w + |w|^2 (BLAS); cosine uses unit vectors; manhattan uses cdist.
    """
    N, M = X.shape[0], W.shape[0]
    k = min(k, M)
    chunk = chunk or max(500, int(2.5e7 // M))
    if metric == "manhattan":
        chunk = max(200, chunk // 8)
    idx_out = np.empty((N, k), dtype=np.int32)
    dist_out = np.empty((N, k), dtype=np.float32)
    if metric == "euclidean":
        W2 = (W ** 2).sum(1)
    elif metric == "cosine":
        Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    for s in range(0, N, chunk):
        Xc = X[s:s + chunk]
        if metric == "euclidean":
            part = W2[None, :] - 2.0 * (Xc @ W.T)
            xx = (Xc ** 2).sum(1)
        elif metric == "cosine":
            Xn = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
            part = -(Xn @ Wn.T)
            xx = np.ones(len(Xc), dtype=np.float32)
        elif metric == "manhattan":
            from scipy.spatial.distance import cdist
            part = cdist(Xc, W, "cityblock").astype(np.float32)
            xx = np.zeros(len(Xc), dtype=np.float32)
        else:
            raise ValueError(metric)
        if k == 1:
            j = part.argmin(1)[:, None]
        else:
            j = np.argpartition(part, k - 1, axis=1)[:, :k]
            o = np.argsort(np.take_along_axis(part, j, 1), axis=1)
            j = np.take_along_axis(j, o, 1)
        d = np.take_along_axis(part, j, 1) + xx[:, None]
        if metric == "euclidean":
            d = np.sqrt(np.maximum(d, 0))
        idx_out[s:s + chunk] = j
        dist_out[s:s + chunk] = d
    return idx_out, dist_out


# ----------------------------------------------------------------------------
# Online (sequential) training kernel
# ----------------------------------------------------------------------------
@njit(cache=True, nogil=True, fastmath=True)
def _online_epoch(X, W, D2, order, t0, total, sigma0, sigma_f, lr0, lr_f, decay, kind):
    N = order.shape[0]
    M, D = W.shape
    for n in range(N):
        t = t0 + n
        frac = t / (total - 1) if total > 1 else 0.0
        if decay == 0:
            sigma = sigma0 + (sigma_f - sigma0) * frac
            lr = lr0 + (lr_f - lr0) * frac
        else:
            sigma = sigma0 * (sigma_f / sigma0) ** frac
            lr = lr0 * (lr_f / lr0) ** frac
        s2 = sigma * sigma
        cutoff = 16.0 * s2
        xi = order[n]
        best = 0
        bd = 1e30
        for j in range(M):
            d = 0.0
            for q in range(D):
                diff = X[xi, q] - W[j, q]
                d += diff * diff
            if d < bd:
                bd = d
                best = j
        for j in range(M):
            d2 = D2[best, j]
            if d2 > cutoff:
                continue
            if kind == 0:
                h = np.exp(-d2 / (2.0 * s2))
            elif kind == 1:
                h = 1.0 if d2 <= s2 else 0.0
            elif kind == 2:
                h = max(0.0, 1.0 - np.sqrt(d2) / sigma)
            else:
                h = (1.0 - d2 / s2) * np.exp(-d2 / (2.0 * s2))
                if h < 0:
                    h = h * 0.1
            if abs(h) < 1e-3:
                continue
            a = lr * h
            for q in range(D):
                W[j, q] += a * (X[xi, q] - W[j, q])


# ----------------------------------------------------------------------------
# The SOM
# ----------------------------------------------------------------------------
class SOM:
    def __init__(self, rows, cols, topology="rect", neighbourhood="gaussian", sigma0_frac=0.25,
                 sigma_final=1.0, lr0=0.3, lr_final=0.01, decay="exponential", epochs=20,
                 init="pca", mode="batch", metric="euclidean", eval_size=5000, verbose=False):
        if neighbourhood == "mexican_hat" and mode == "batch":
            raise ValueError("mexican_hat is only supported in online mode")
        if mode == "online" and metric != "euclidean":
            raise ValueError("online mode supports the Euclidean metric only")
        self.rows, self.cols = rows, cols
        self.topology, self.neighbourhood = topology, neighbourhood
        self.sigma0_frac, self.sigma_final = sigma0_frac, sigma_final
        self.lr0, self.lr_final = lr0, lr_final
        self.decay, self.epochs = decay, epochs
        self.init, self.mode, self.metric = init, mode, metric
        self.eval_size, self.verbose = eval_size, verbose

    # -- cached grid matrices (not pickled) ------------------------------------
    @property
    def n_neurons(self):
        return self.rows * self.cols

    def _grid(self):
        if getattr(self, "_xy", None) is None:
            self._xy = grid_coords(self.rows, self.cols, self.topology)
            self._d2 = grid_sqdist(self._xy)
            thr = 2.0001 if self.topology == "rect" else 1.0001      # 8- / 6-neighbourhood
            self._adj = (self._d2 > 0) & (self._d2 <= thr)
        return self._xy, self._d2, self._adj

    @property
    def xy(self):
        return self._grid()[0]

    def __getstate__(self):
        s = dict(self.__dict__)
        for k in ("_xy", "_d2", "_adj"):
            s.pop(k, None)
        return s

    # -- initialisation -----------------------------------------------------------
    def _init_weights(self, X, rng):
        M = self.n_neurons
        if self.init == "random":
            return X[rng.choice(len(X), M, replace=len(X) < M)].astype(np.float32).copy()
        sub = X if len(X) <= 50000 else X[rng.choice(len(X), 50000, replace=False)]
        mu = sub.mean(0)
        Xc = sub - mu
        vals, vecs = np.linalg.eigh(Xc.T @ Xc / len(Xc))
        pc1, pc2 = vecs[:, -1], vecs[:, -2] if vecs.shape[1] > 1 else np.zeros(vecs.shape[0])
        s1, s2 = np.sqrt(max(vals[-1], 0)), np.sqrt(max(vals[-2], 0)) if len(vals) > 1 else 0.0
        r, c = np.divmod(np.arange(M), self.cols)
        u = np.linspace(-1, 1, self.cols)[c]
        v = np.linspace(-1, 1, self.rows)[r]
        a, b = (u, v) if self.cols >= self.rows else (v, u)
        W = mu + 2 * s1 * a[:, None] * pc1[None, :] + 2 * s2 * b[:, None] * pc2[None, :]
        return W.astype(np.float32)

    # -- training -----------------------------------------------------------------
    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        t_start = time.time()
        rng, self.entropy_ = new_rng()
        xy, d2, _ = self._grid()
        W = self._init_weights(X, rng)
        M, N = W.shape[0], X.shape[0]
        side = max(self.rows, self.cols)
        sigma0 = max(self.sigma0_frac * side, self.sigma_final)
        ev = X[rng.choice(N, min(self.eval_size, N), replace=False)]
        lo, hi = X.min(0), X.max(0)
        pad = 0.5 * (hi - lo)
        lo, hi = lo - pad, hi + pad          # safeguard: codebook may not leave the data range
        self.history_ = []
        decay_code = 0 if self.decay == "linear" else 1
        kind = KIND_CODE[self.neighbourhood]
        for ep in range(self.epochs):
            if self.mode == "batch":
                sigma = schedule(sigma0, self.sigma_final, ep, self.epochs, self.decay)
                b = bmu_search(X, W, self.metric, 1)[0][:, 0]
                n = np.bincount(b, minlength=M).astype(np.float32)
                import scipy.sparse as sp
                S = sp.csr_matrix((np.ones(N, np.float32), (b, np.arange(N))), shape=(M, N)) @ X
                H = neighbourhood(d2, sigma, self.neighbourhood).astype(np.float32)
                num, den = H @ S, H @ n
                ok = den > 1e-8
                W[ok] = num[ok] / den[ok, None]
                lr = np.nan
            else:
                order = rng.permutation(N).astype(np.int64)
                _online_epoch(X, W, d2, order, ep * N, self.epochs * N, sigma0, self.sigma_final,
                              self.lr0, self.lr_final, decay_code, kind)
                frac = (ep + 1) / self.epochs
                sigma = schedule(sigma0, self.sigma_final, ep, self.epochs, self.decay)
                lr = schedule(self.lr0, self.lr_final, ep, self.epochs, self.decay)
            np.clip(W, lo, hi, out=W)
            self.W_ = W
            qe, te = self._qe_te(ev)
            self.history_.append(dict(epoch=ep + 1, sigma=float(sigma), lr=float(lr), qe=qe, te=te))
            if self.verbose:
                print(f"  epoch {ep+1:3d}/{self.epochs}  sigma={sigma:6.2f}  qe={qe:.4f}  te={te:.4f}")
        self.W_ = W
        self.fit_time_ = time.time() - t_start
        return self

    # -- inference and diagnostics --------------------------------------------------
    def bmus(self, X, k=1):
        return bmu_search(np.ascontiguousarray(X, dtype=np.float32), self.W_, self.metric, k)

    def _qe_te(self, X):
        idx, dist = bmu_search(X, self.W_, self.metric, 2)
        adj = self._grid()[2]
        return float(dist[:, 0].mean()), float((~adj[idx[:, 0], idx[:, 1]]).mean())

    def quantization_error(self, X):
        return float(self.bmus(X, 1)[1][:, 0].mean())

    def topographic_error(self, X):
        idx = self.bmus(X, 2)[0]
        return float((~self._grid()[2][idx[:, 0], idx[:, 1]]).mean())

    def hit_counts(self, X):
        return np.bincount(self.bmus(X, 1)[0][:, 0], minlength=self.n_neurons)

    def umatrix(self):
        """Mean Euclidean weight-space distance from each neuron to its grid neighbours."""
        adj = self._grid()[2]
        sq = (self.W_ ** 2).sum(1)
        dw = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * self.W_ @ self.W_.T, 0))
        return (dw * adj).sum(1) / np.maximum(adj.sum(1), 1)
