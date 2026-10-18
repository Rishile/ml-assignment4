import csv
import json
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import config as C
from classifier import make_som
from posthoc import sweep_posthoc
from preprocessing import build_preprocessor

CFG_KEYS = C.PREP_KEYS + C.SOM_KEYS


def run_fold(data, cfg, posthoc_list, tr, va, fold, label_fraction=1.0):
    err = C.validate_cfg(cfg)
    if err:
        return []
    from classifier import stratified_label_subset
    t0 = time.time()
    prep = build_preprocessor(cfg, data.num_cols, data.cat_cols, data.ref_n)
    Xtr = prep.fit_transform(data.Xtr.iloc[tr]).astype(np.float32)
    Xva = prep.transform(data.Xtr.iloc[va]).astype(np.float32)
    ytr, yva = data.ytr[tr], data.ytr[va]
    som = make_som(cfg, Xtr).fit(Xtr)
    btr, bva = som.bmus(Xtr)[0][:, 0], som.bmus(Xva)[0][:, 0]
    li = None if label_fraction >= 1.0 else stratified_label_subset(ytr, label_fraction)
    rows = sweep_posthoc(som.W_, som.xy, btr, ytr, bva, yva, posthoc_list, len(data.classes), li)
    cfg_json = json.dumps({k: cfg[k] for k in CFG_KEYS}, sort_keys=True)
    extra = dict(cfg_json=cfg_json, fold=fold, n_features=Xtr.shape[1], n_neurons=som.n_neurons,
                 map_shape=f"{som.rows}x{som.cols}", qe_val=som.quantization_error(Xva),
                 te_val=som.topographic_error(Xva), qe_train=som.history_[-1]["qe"],
                 te_train=som.history_[-1]["te"], train_seconds=som.fit_time_,
                 total_seconds=time.time() - t0, run_entropy=som.entropy_,
                 label_fraction=label_fraction)
    out = []
    for r in rows:
        r = {**{k: cfg[k] for k in CFG_KEYS}, **r, **extra}
        r["ph_json"] = json.dumps({k: r[k] for k in C.POST_KEYS}, sort_keys=True, default=str)
        out.append(r)
    return out


def evaluate_configs(data, configs, posthoc, folds, csv_path, tag, n_jobs=1, label_fraction=1.0):
    """Run every (config, fold); stream rows to csv_path; return all rows as a DataFrame.

    `posthoc` is a list of post-hoc dicts or a callable cfg -> list.
    """
    configs = [c for c in configs if C.validate_cfg(c) is None]
    tasks = [(cfg, fi, tr, va) for cfg in configs for fi, (tr, va) in enumerate(folds)]
    print(f"[{tag}] {len(configs)} configurations x {len(folds)} folds = {len(tasks)} map trainings")
    lock, done, t_start = threading.Lock(), [0], time.time()
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(exist_ok=True)
    state = {"writer": None, "file": open(csv_path, "w", newline="")}
    all_rows = []

    def work(cfg, fi, tr, va):
        ph = posthoc(cfg) if callable(posthoc) else posthoc
        rows = run_fold(data, cfg, ph, tr, va, fi, label_fraction)
        with lock:
            done[0] += 1
            for r in rows:
                r["tag"] = tag
            all_rows.extend(rows)
            if rows:
                if state["writer"] is None:
                    state["writer"] = csv.DictWriter(state["file"], fieldnames=list(rows[0].keys()))
                    state["writer"].writeheader()
                state["writer"].writerows(rows)
                state["file"].flush()
            el = time.time() - t_start
            eta = el / done[0] * (len(tasks) - done[0])
            print(f"[{tag}] {done[0]}/{len(tasks)} done  elapsed {el/60:.1f} min  eta {eta/60:.1f} min", flush=True)
        return None

    import os
    n_cpu = os.cpu_count() or 1
    with threadpool_limits(limits=max(1, n_cpu // max(1, n_jobs))):
        Parallel(n_jobs=n_jobs, prefer="threads")(delayed(work)(*t) for t in tasks)
    state["file"].close()
    return pd.DataFrame(all_rows)


def summarise(df):
    """Mean +- SD across folds for every (configuration, post-hoc) pair, best first."""
    g = df.groupby(["cfg_json", "ph_json"], sort=False)
    agg = g.agg(macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                balanced_accuracy=("balanced_accuracy", "mean"), mcc=("mcc", "mean"),
                accuracy=("accuracy", "mean"), weighted_f1=("weighted_f1", "mean"),
                kappa=("kappa", "mean"), qe_val=("qe_val", "mean"), te_val=("te_val", "mean"),
                neuron_purity=("neuron_purity", "mean"), empty_frac=("empty_frac", "mean"),
                n_clusters=("n_clusters", "mean"), train_seconds=("train_seconds", "mean"),
                n_folds=("fold", "nunique")).reset_index()
    agg["macro_f1_sd"] = agg["macro_f1_sd"].fillna(0.0)
    return agg.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def row_to_config(row):
    cfg = {**C.DEFAULT, **json.loads(row["cfg_json"]), **json.loads(row["ph_json"])}
    return cfg


def pick_best(summary):
    """Highest mean macro-F1; among results within one SD of it, prefer the simplest
    (smaller map, then fewer clusters, then fewer epochs)."""
    best = summary.iloc[0]
    cand = summary[summary["macro_f1"] >= best["macro_f1"] - best["macro_f1_sd"]].copy()
    cfgs = cand.apply(lambda r: json.loads(r["cfg_json"]), axis=1)
    cand["_side"] = [c["map_side"] for c in cfgs]
    cand["_ep"] = [c["epochs"] for c in cfgs]
    cand = cand.sort_values(["_side", "n_clusters", "_ep", "macro_f1"], ascending=[True, True, True, False])
    return cand.iloc[0], best
