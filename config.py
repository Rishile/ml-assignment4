import itertools
from pathlib import Path


# Data and splits
DATA_PATH = "data/networkTraffic.csv"
TARGET = "attack_cat"
CAT_COLS = ["proto", "state", "service"]
DROP_COLS = ["id"]
LABEL_MAP = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Backdoor",
    3: "DoS",
    4: "Exploits",
    5: "Analysis",
    6: "Fuzzers",
    7: "Worms",
    8: "Shellcode",
    9: "Generic"
}
TEST_SIZE = 0.20 
SPLIT_SEED = 27
N_FOLDS = 5
CV_SEED = 27
SCREEN_SEED = 2024

# Rare-category grouping
RARE_MIN_COUNT = {"proto": 500, "state": 100}

RESULTS_DIR = Path("results")
FIGS_DIR = Path("figs")
MODELS_DIR = Path("models")

A2_TEST_METRICS = {
    "KNN": dict(accuracy=0.7972, balanced_accuracy=0.5030, macro_f1=0.5214, weighted_f1=0.7951, mcc=0.7363, kappa=0.7361),
    "PCA KNN": dict(accuracy=0.8002, balanced_accuracy=0.5018, macro_f1=0.5233, weighted_f1=0.7972, mcc=0.7399, kappa=0.7396),
    "Decision tree": dict(accuracy=0.7978, balanced_accuracy=0.6024, macro_f1=0.5941, weighted_f1=0.8074, mcc=0.7480, kappa=0.7433),
}
A2_PER_CLASS_F1 = {
    "KNN": {"Analysis": 0.17, "Backdoor": 0.07, "DoS": 0.38, "Exploits": 0.69, "Fuzzers": 0.57, "Generic": 0.99,
            "Normal": 0.90, "Reconnaissance": 0.75, "Shellcode": 0.42, "Worms": 0.26},
    "PCA KNN": {"Analysis": 0.18, "Backdoor": 0.08, "DoS": 0.37, "Exploits": 0.69, "Fuzzers": 0.58, "Generic": 0.99,
                "Normal": 0.91, "Reconnaissance": 0.76, "Shellcode": 0.49, "Worms": 0.18},
    "Decision tree": {"Analysis": 0.18, "Backdoor": 0.14, "DoS": 0.47, "Exploits": 0.68, "Fuzzers": 0.64, "Generic": 0.99,
                      "Normal": 0.89, "Reconnaissance": 0.83, "Shellcode": 0.60, "Worms": 0.52},
}

# ----------------------------------------------------------------------------
PREP_KEYS = ["log1p", "scaler", "redundancy", "redundancy_param", "onehot_weight"]
SOM_KEYS = ["map_side", "aspect", "topology", "neighbourhood", "sigma0_frac",
            "sigma_final", "lr0", "lr_final", "decay", "epochs", "init", "mode", "metric"]
POST_KEYS = ["cluster_method", "k", "cluster_weighting", "label_rule", "purity_tau", "fallback"]

DEFAULT = dict(
    log1p=True,
    scaler="standard",            # standard | minmax
    redundancy="corr",            # corr | pca | none
    redundancy_param=0.90,        # |r| threshold for corr, explained-variance ratio for pca
    onehot_weight=1.0,            # multiplier on the one-hot block

    map_side=40,                  # neurons = side * side
    aspect="square",              # square | eigen (side ratio from the two leading PCA eigenvalues)
    topology="rect",              # rect | hex
    neighbourhood="gaussian",     # gaussian | bubble | triangle | mexican_hat (online only)
    sigma0_frac=0.25,             # initial radius as a fraction of the longer map side
    sigma_final=1.0,              # final radius (grid units)
    lr0=0.3,                      # online mode only
    lr_final=0.01,                # online mode only
    decay="exponential",          # linear | exponential
    epochs=20,
    init="pca",                   # random (random training rows) | pca
    mode="batch",                 # batch | online
    metric="euclidean",           # euclidean | cosine | manhattan (batch only)

    cluster_method="none",        # none (neuron level) | kmeans | ward
    k=None,                       # number of codebook clusters
    cluster_weighting="none",     # none | hits (k-means only: weight neurons by training hits)
    label_rule="majority",        # majority | sqrt_prior | prior | purity
    purity_tau=0.7,               # purity rule: clusters below tau fall back to neuron-level labels
    fallback="weight",            # weight | grid : nearest labelled neuron for unlabelled neurons
)

def _dim(key, values):
    return [{key: v} for v in values]

REDUNDANCY_OPTIONS = [
    {"redundancy": "corr", "redundancy_param": 0.85},
    {"redundancy": "corr", "redundancy_param": 0.90},
    {"redundancy": "corr", "redundancy_param": 0.95},
    {"redundancy": "pca", "redundancy_param": 0.90},
    {"redundancy": "pca", "redundancy_param": 0.95},
    {"redundancy": "pca", "redundancy_param": 0.99},
    {"redundancy": "none", "redundancy_param": None},
]

# Stage 1 - preprocessing (full factorial: 2 x 2 x 7 x 3 = 84 configurations)
STAGE1_DIMS = [
    _dim("log1p", [True, False]),
    _dim("scaler", ["standard", "minmax"]),
    REDUNDANCY_OPTIONS,
    _dim("onehot_weight", [1.0, 0.5, 0.25]),
]

# Stage 2 - map structure (full factorial: 5 x 2 x 2 = 20 configurations)
STAGE2_DIMS = [
    _dim("map_side", [20, 30, 40, 50, 60]),
    _dim("topology", ["rect", "hex"]),
    _dim("aspect", ["square", "eigen"]),
]

# Stage 3 - training dynamics: coordinate search (one parameter at a time, in this
# order, each time fixing the best value found so far).
STAGE3_ORDER = ["mode", "init", "neighbourhood", "decay", "sigma0_frac",
                "sigma_final", "epochs", "lr0"]
STAGE3_SPACE = {
    "mode": ["batch", "online"],
    "init": ["random", "pca"],
    "neighbourhood": ["gaussian", "bubble", "triangle", "mexican_hat"],
    "decay": ["linear", "exponential"],
    "sigma0_frac": [0.5, 0.25, 0.1],
    "sigma_final": [1.0, 0.5],
    "epochs": [3, 5, 10, 20, 40],
    "lr0": [0.1, 0.3, 0.5],
}

# Stage 4 - post-hoc clustering and labelling (swept on every trained map; cheap)
K_LIST = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

def posthoc_grid(k_list=K_LIST, methods=("kmeans", "ward"), weightings=("none", "hits"),
                 rules=(("majority", 0.7), ("sqrt_prior", 0.7), ("prior", 0.7),
                        ("purity", 0.5), ("purity", 0.7)),
                 fallbacks=("weight", "grid"), include_neuron=True):
    grid = []
    base = dict(DEFAULT)
    if include_neuron:
        for rule, tau in rules:
            if rule == "purity":      # purity rule needs clusters
                continue
            for fb in fallbacks:
                grid.append(dict(base, cluster_method="none", k=None, cluster_weighting="none",
                                 label_rule=rule, purity_tau=tau, fallback=fb))
    for method in methods:
        for k in k_list:
            for w in (weightings if method == "kmeans" else ("none",)):
                for rule, tau in rules:
                    for fb in fallbacks:
                        grid.append(dict(base, cluster_method=method, k=k, cluster_weighting=w,
                                         label_rule=rule, purity_tau=tau, fallback=fb))
    return [{kk: g[kk] for kk in POST_KEYS} for g in grid]

DEFAULT_POSTHOC = [{k: DEFAULT[k] for k in POST_KEYS}]     # neuron-level majority vote

# Stage 5 - joint local refinement around the best settings found so far
STAGE5_PARAMS = ["map_side", "sigma0_frac", "epochs"]      # plus the neighbouring k values
STAGE5_SPACE = {"map_side": STAGE2_DIMS[0], "sigma0_frac": _dim("sigma0_frac", STAGE3_SPACE["sigma0_frac"]),
                "epochs": _dim("epochs", STAGE3_SPACE["epochs"])}

# Stage 6/7 - final evaluation and ablations
N_FINAL_REPEATS = 5
LABEL_BUDGETS = [0.01, 0.05, 0.10, 0.25, 1.00]
LABEL_BUDGET_DRAWS = 5
TOPM_VOTES = [1, 3, 5, 9]
METRIC_ABLATION = ["euclidean", "cosine", "manhattan"]
N_BOOTSTRAP = 1000


SCREEN_FRAC = 0.25        # stratified fraction of the training set used when screening
SCREEN_FOLDS = 2          # number of the 5 folds evaluated when screening
CONFIRM_TOP = 5           # configurations re-evaluated with full 5-fold CV on the full training set


def product_dims(base, dims):
    """Full factorial over dimensions (each a list of dict updates)."""
    out = []
    for combo in itertools.product(*dims):
        cfg = dict(base)
        for upd in combo:
            cfg.update(upd)
        out.append(cfg)
    return out


def validate_cfg(cfg):
    """Return None if the configuration is valid, else a string describing why not."""
    if cfg["neighbourhood"] == "mexican_hat" and cfg["mode"] == "batch":
        return "mexican_hat has negative weights; only supported in online mode"
    if cfg["mode"] == "online" and cfg["metric"] != "euclidean":
        return "online mode supports the Euclidean metric only"
    return None


def apply_quick():
    """Shrink every grid so the whole pipeline can be smoke-tested in minutes (--quick)."""
    global STAGE1_DIMS, STAGE2_DIMS, STAGE3_SPACE, K_LIST, STAGE5_SPACE, SCREEN_FRAC, SCREEN_FOLDS
    global CONFIRM_TOP, N_FINAL_REPEATS, LABEL_BUDGET_DRAWS, N_BOOTSTRAP, DEFAULT
    DEFAULT = dict(DEFAULT, map_side=14, epochs=6)
    STAGE1_DIMS = [_dim("log1p", [True, False]), _dim("scaler", ["standard"]),
                   REDUNDANCY_OPTIONS[1:2] + REDUNDANCY_OPTIONS[3:4], _dim("onehot_weight", [1.0, 0.5])]
    STAGE2_DIMS = [_dim("map_side", [10, 14]), _dim("topology", ["rect", "hex"]), _dim("aspect", ["square"])]
    STAGE3_SPACE = {"mode": ["batch", "online"], "init": ["random", "pca"],
                    "neighbourhood": ["gaussian", "mexican_hat"], "decay": ["linear", "exponential"],
                    "sigma0_frac": [0.5, 0.25], "sigma_final": [1.0, 0.5], "epochs": [3, 6], "lr0": [0.1, 0.3]}
    STAGE5_SPACE = {"map_side": STAGE2_DIMS[0], "sigma0_frac": _dim("sigma0_frac", STAGE3_SPACE["sigma0_frac"]),
                    "epochs": _dim("epochs", STAGE3_SPACE["epochs"])}
    K_LIST = [5, 10, 30, 60]
    SCREEN_FRAC, SCREEN_FOLDS, CONFIRM_TOP = 0.5, 2, 2
    N_FINAL_REPEATS, LABEL_BUDGET_DRAWS, N_BOOTSTRAP = 2, 2, 100
