import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd
import config as C
import engine
import evaluate as ev
import visualise as viz
from classifier import SOMClassifier, make_som
from data import load_data, make_folds
from posthoc import confusion
from preprocessing import build_preprocessor

BEST = C.RESULTS_DIR / "best_config.json"


# ----------------------------------------------------------------------------- helpers
def load_best():
    if BEST.exists():
        return {**C.DEFAULT, **json.loads(BEST.read_text())}
    return dict(C.DEFAULT)


def save_best(cfg, note):
    C.RESULTS_DIR.mkdir(exist_ok=True)
    BEST.write_text(json.dumps(cfg, indent=2, default=str))
    print(f"[best] saved after {note}:\n" + json.dumps({k: cfg[k] for k in C.PREP_KEYS + C.SOM_KEYS + C.POST_KEYS}, default=str))


def top_configs(summary, n):
    seen, out = set(), []
    for _, r in summary.iterrows():
        if r["cfg_json"] not in seen:
            seen.add(r["cfg_json"])
            out.append(engine.row_to_config(r))
        if len(out) == n:
            break
    return out


def tune(ctx, stage, configs, posthoc, screen=True):
    """Screen all configs on a stratified subsample, confirm the best few with full 5-fold CV."""
    data = ctx["data"]
    name = f"stage{stage}"
    if screen and not ctx["no_screen"]:
        folds = make_folds(data.ytr, frac=C.SCREEN_FRAC, n_run=C.SCREEN_FOLDS)
        df = engine.evaluate_configs(data, configs, posthoc, folds, C.RESULTS_DIR / f"{name}_screen.csv",
                                     f"{name}-screen", ctx["n_jobs"])
        summ = engine.summarise(df)
        summ.to_csv(C.RESULTS_DIR / f"{name}_screen_summary.csv", index=False)
        configs = top_configs(summ, C.CONFIRM_TOP)
        print(f"[{name}] confirming the top {len(configs)} configurations with full 5-fold CV")
    df = engine.evaluate_configs(data, configs, posthoc, make_folds(data.ytr), C.RESULTS_DIR / f"{name}_full.csv",
                                 f"{name}-full", ctx["n_jobs"])
    summ = engine.summarise(df)
    summ.to_csv(C.RESULTS_DIR / f"{name}_full_summary.csv", index=False)
    chosen, top = engine.pick_best(summ)
    print(f"[{name}] best CV macro-F1 {top['macro_f1']:.4f} +- {top['macro_f1_sd']:.4f}; "
          f"selected (simplest within 1 SD): {chosen['macro_f1']:.4f}")
    return engine.row_to_config(chosen), summ


def _screen_or_full(stage, full_summary):
    """All screened configurations if a screening run exists (more points), else the full-CV summary."""
    p = C.RESULTS_DIR / f"stage{stage}_screen_summary.csv"
    return pd.read_csv(p) if p.exists() else full_summary


def neighbours(space, current):
    """Values adjacent to `current` in an ordered list of dimension updates."""
    key = list(space[0].keys())[0]
    vals = [list(s.values())[0] for s in space]
    i = vals.index(current[key]) if current[key] in vals else 0
    return [space[j] for j in range(max(0, i - 1), min(len(space), i + 2))]


# ----------------------------------------------------------------------------- stages
def stage0(ctx):
    """Timing pilot: measures the cost of one map and prints a cost estimate per stage."""
    data, cfg = ctx["data"], load_best()
    prep = build_preprocessor(cfg, data.num_cols, data.cat_cols, data.ref_n)
    Xt = prep.fit_transform(data.Xtr).astype(np.float32)
    N = len(Xt)
    print(f"[pilot] training rows {N}, features after preprocessing {Xt.shape[1]}")
    n_fit = min(N, 20000)
    make_som(dict(cfg, map_side=5, epochs=1, mode="online"), Xt[:200]).fit(Xt[:200])   # warm up numba (JIT)
    rows = []
    for side in (20, 40, 60):
        t = time.time()
        make_som(dict(cfg, map_side=side, epochs=2, mode="batch"), Xt[:n_fit]).fit(Xt[:n_fit])
        tb = (time.time() - t) / 2 / n_fit          # seconds per sample per epoch (batch)
        t = time.time()
        n_on = min(N, 4000)
        make_som(dict(cfg, map_side=side, epochs=1, mode="online"), Xt[:n_on]).fit(Xt[:n_on])
        to = (time.time() - t) / n_on
        rows.append((side, tb, to))
        print(f"[pilot] {side}x{side}: batch {tb*1e6:.1f} us/sample/epoch, online {to*1e6:.1f} us/sample/epoch")

    def est(n_cfg, folds, frac, side, epochs, mode="batch"):
        r = min(rows, key=lambda x: abs(x[0] - side))
        per = r[1] if mode == "batch" else r[2]
        scale = (side / r[0]) ** 2
        return n_cfg * folds * (N * frac * 0.8) * per * scale * epochs / 3600
    s1 = len(C.product_dims(cfg, C.STAGE1_DIMS))
    s2 = len(C.product_dims(cfg, C.STAGE2_DIMS))
    s3 = sum(len(v) for v in C.STAGE3_SPACE.values())
    d = cfg["map_side"]
    print("\n[pilot] rough compute estimates (single process, hours):")
    print(f"  stage 1 screen  {s1} configs : {est(s1, C.SCREEN_FOLDS, C.SCREEN_FRAC, d, cfg['epochs']):6.2f}   "
          f"+ confirm {C.CONFIRM_TOP} : {est(C.CONFIRM_TOP, 5, 1.0, d, cfg['epochs']):6.2f}")
    print(f"  stage 2 screen  {s2} configs : {est(s2, C.SCREEN_FOLDS, C.SCREEN_FRAC, 40, cfg['epochs']):6.2f}   "
          f"+ confirm {C.CONFIRM_TOP} : {est(C.CONFIRM_TOP, 5, 1.0, 50, cfg['epochs']):6.2f}")
    print(f"  stage 3 screen  {s3} steps   : {est(s3, C.SCREEN_FOLDS, C.SCREEN_FRAC, d, 20):6.2f} (online steps cost more)")
    print(f"  stage 6 final   {C.N_FINAL_REPEATS} full runs   : {est(C.N_FINAL_REPEATS, 1, 1.25, d, cfg['epochs']):6.2f}")
    print("  If these are too large, lower SCREEN_FRAC / SCREEN_FOLDS / CONFIRM_TOP in config.py or use --n-jobs.")


def stage1(ctx):
    cfgs = C.product_dims(load_best(), C.STAGE1_DIMS)
    best, summ = tune(ctx, 1, cfgs, C.DEFAULT_POSTHOC)
    viz.plot_param_effects(_screen_or_full(1, summ), ["log1p", "scaler", "redundancy_setting", "onehot_weight"],
                           "stage1_param_effects.png", "Stage 1 - preprocessing (every screened configuration)")
    save_best(best, "stage 1")


def stage2(ctx):
    cfgs = C.product_dims(load_best(), C.STAGE2_DIMS)
    best, summ = tune(ctx, 2, cfgs, C.DEFAULT_POSTHOC)
    viz.plot_param_effects(_screen_or_full(2, summ), ["map_side", "topology", "aspect"],
                           "stage2_param_effects.png", "Stage 2 - map structure (every screened configuration)")
    save_best(best, "stage 2")


def stage3(ctx):
    """Coordinate search: vary one parameter at a time, keep the best, move on."""
    data = ctx["data"]
    cur = load_best()
    start = dict(cur)
    folds = make_folds(data.ytr, frac=C.SCREEN_FRAC, n_run=C.SCREEN_FOLDS)
    trace = []
    for p in C.STAGE3_ORDER:
        if p == "lr0" and cur["mode"] != "online":
            print("[stage3] lr0 skipped (batch mode has no learning rate)")
            continue
        cands = [dict(cur, **{p: v}) for v in C.STAGE3_SPACE[p]]
        cands = [c for c in cands if C.validate_cfg(c) is None]
        df = engine.evaluate_configs(data, cands, C.DEFAULT_POSTHOC, folds, C.RESULTS_DIR / f"stage3_{p}.csv",
                                     f"stage3-{p}", ctx["n_jobs"])
        summ = engine.summarise(df)
        chosen, top = engine.pick_best(summ)
        cur = engine.row_to_config(chosen)
        for _, r in summ.iterrows():
            trace.append(dict(parameter=p, value=json.loads(r["cfg_json"])[p], macro_f1=r["macro_f1"],
                              macro_f1_sd=r["macro_f1_sd"], qe_val=r["qe_val"], te_val=r["te_val"],
                              train_seconds=r["train_seconds"], selected=(r["cfg_json"] == chosen["cfg_json"])))
        print(f"[stage3] {p} -> {cur[p]}")
    pd.DataFrame(trace).to_csv(C.RESULTS_DIR / "stage3_trace.csv", index=False)
    # confirm: tuned vs. starting configuration with full 5-fold CV
    df = engine.evaluate_configs(data, [start, cur], C.DEFAULT_POSTHOC, make_folds(data.ytr),
                                 C.RESULTS_DIR / "stage3_full.csv", "stage3-confirm", ctx["n_jobs"])
    summ = engine.summarise(df)
    summ.to_csv(C.RESULTS_DIR / "stage3_full_summary.csv", index=False)
    chosen, _ = engine.pick_best(summ)
    save_best(engine.row_to_config(chosen), "stage 3")


def stage4(ctx):
    """Train the best map once per fold and sweep ALL post-hoc settings on it."""
    data, best = ctx["data"], load_best()
    grid = C.posthoc_grid(k_list=C.K_LIST)
    print(f"[stage4] {len(grid)} post-hoc configurations per trained map")
    df = engine.evaluate_configs(data, [best], grid, make_folds(data.ytr), C.RESULTS_DIR / "stage4_full.csv",
                                 "stage4", ctx["n_jobs"])
    summ = engine.summarise(df)
    summ.to_csv(C.RESULTS_DIR / "stage4_full_summary.csv", index=False)
    viz.plot_k_curve(summ)
    chosen, top = engine.pick_best(summ)
    save_best(engine.row_to_config(chosen), "stage 4")


def stage5(ctx):
    """Small full-factorial grid around the best settings; post-hoc k varied for free."""
    data, best = ctx["data"], load_best()
    dims = [neighbours(C.STAGE5_SPACE[p], best) for p in C.STAGE5_PARAMS]
    cfgs = C.product_dims(best, dims)
    ks = [k for k in C.K_LIST]
    if best["cluster_method"] != "none" and best["k"] in ks:
        i = ks.index(best["k"])
        ks = ks[max(0, i - 1): i + 2]
    elif best["cluster_method"] != "none":
        ks = [best["k"]]
    ph = [dict({k: best[k] for k in C.POST_KEYS}, k=(k if best["cluster_method"] != "none" else None)) for k in ks]
    new, summ = tune(ctx, 5, cfgs, ph)
    save_best(new, "stage 5")


def _final_models():
    return pickle.load(open(C.MODELS_DIR / "final_runs.pkl", "rb"))


def stage6(ctx):
    """Final evaluation: retrain the chosen configuration on the FULL training set, test set used once."""
    data, best = ctx["data"], load_best()
    K = len(data.classes)
    C.MODELS_DIR.mkdir(exist_ok=True)
    rows, models, preds = [], [], []
    for r in range(C.N_FINAL_REPEATS):
        t = time.time()
        prep = build_preprocessor(best, data.num_cols, data.cat_cols, data.ref_n)
        Xtr = prep.fit_transform(data.Xtr).astype(np.float32)
        clf = SOMClassifier(best).fit(Xtr, data.ytr)
        p = clf.predict(prep.transform(data.Xte).astype(np.float32))
        m = ev.all_metrics(data.yte, p, K)
        rows.append(dict(run=r, run_entropy=clf.som_.entropy_, train_seconds=clf.som_.fit_time_,
                         qe_train=clf.som_.history_[-1]["qe"], te_train=clf.som_.history_[-1]["te"],
                         neuron_purity=clf.label_info_["neuron_purity"], **m))
        models.append(dict(prep=prep, clf=clf))
        preds.append(p)
        print(f"[stage6] run {r+1}/{C.N_FINAL_REPEATS}: macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  ({time.time()-t:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "stage6_final_runs.csv", index=False)
    print(df[["macro_f1", "balanced_accuracy", "mcc", "accuracy", "weighted_f1", "kappa"]].agg(["mean", "std"]).round(4))
    pickle.dump(models, open(C.MODELS_DIR / "final_runs.pkl", "wb"))
    np.save(C.RESULTS_DIR / "final_preds.npy", np.array(preds))
    viz.plot_training_curves([m["clf"].som_.history_ for m in models])


def stage7(ctx):
    """Ablations and extensions. Not used for model selection."""
    from sklearn.metrics import roc_auc_score
    data, best = ctx["data"], load_best()
    K = len(data.classes)
    models = _final_models()
    # (a) label budget and (c) top-m soft voting use the maps trained in stage 6 (no retraining)
    rows_a, rows_c = [], []
    for ri, mdl in enumerate(models):
        prep, clf = mdl["prep"], mdl["clf"]
        Xte = prep.transform(data.Xte).astype(np.float32)
        b_te = clf.som_.bmus(Xte)[0][:, 0]
        for f in C.LABEL_BUDGETS:
            for d in range(C.LABEL_BUDGET_DRAWS if f < 1 else 1):
                clf.relabel(label_fraction=f)
                m = ev.all_metrics(data.yte, clf.neuron_labels_[b_te], K)
                rows_a.append(dict(run=ri, draw=d, label_fraction=f, **m))
        clf.relabel(label_fraction=1.0)
        for m_votes in C.TOPM_VOTES:
            p = clf.predict_topm(Xte, m_votes)
            rows_c.append(dict(run=ri, m=m_votes, **ev.all_metrics(data.yte, p, K)))
    da = pd.DataFrame(rows_a)
    da.to_csv(C.RESULTS_DIR / "stage7_label_budget.csv", index=False)
    pd.DataFrame(rows_c).to_csv(C.RESULTS_DIR / "stage7_topm.csv", index=False)
    viz.plot_label_budget(da)
    print(da.groupby("label_fraction")["macro_f1"].agg(["mean", "std"]).round(4))
    # (b) distance metric (batch mode only), screened on the subsample
    base = dict(best, mode="batch", neighbourhood=("gaussian" if best["neighbourhood"] == "mexican_hat" else best["neighbourhood"]))
    cfgs = [dict(base, metric=mt) for mt in C.METRIC_ABLATION]
    ph = [{k: best[k] for k in C.POST_KEYS}]
    df = engine.evaluate_configs(data, cfgs, ph, make_folds(data.ytr, frac=C.SCREEN_FRAC, n_run=C.SCREEN_FOLDS),
                                 C.RESULTS_DIR / "stage7_metric.csv", "stage7-metric", ctx["n_jobs"])
    print(engine.summarise(df)[["cfg_json", "macro_f1", "macro_f1_sd"]].assign(
        cfg_json=lambda d: d["cfg_json"].map(lambda s: json.loads(s)["metric"])))
    # (d) anomaly detector: SOM trained on Normal traffic only, quantization error as the score
    nid = [i for i, c in enumerate(data.classes) if c.lower() == "normal"]
    if nid:
        nid = nid[0]
        prep = build_preprocessor(best, data.num_cols, data.cat_cols, data.ref_n)
        Xtr = prep.fit_transform(data.Xtr).astype(np.float32)
        Xn = Xtr[data.ytr == nid]
        perm = np.random.default_rng().permutation(len(Xn))
        n_cal = len(Xn) // 5
        som = make_som(best, Xn[perm[n_cal:]]).fit(Xn[perm[n_cal:]])
        cal = som.bmus(Xn[perm[:n_cal]])[1][:, 0]
        Xte = prep.transform(data.Xte).astype(np.float32)
        score = som.bmus(Xte)[1][:, 0]
        yb = data.yte != nid
        out = [dict(threshold="AUC", value=roc_auc_score(yb, score))]
        for q in (0.90, 0.95, 0.99):
            pr = score > np.quantile(cal, q)
            tp, fp, fn = (pr & yb).sum(), (pr & ~yb).sum(), (~pr & yb).sum()
            out.append(dict(threshold=f"q{int(q*100)}", precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                            f1=2 * tp / max(2 * tp + fp + fn, 1), fpr=fp / max((~yb).sum(), 1)))
        pd.DataFrame(out).to_csv(C.RESULTS_DIR / "stage7_anomaly.csv", index=False)
        print(pd.DataFrame(out).round(4))
    else:
        print("[stage7] no class named 'Normal' - anomaly experiment skipped")


def _norm(s):
    return str(s).lower().replace(" ", "").rstrip("s")      # Backdoor/Backdoors, Fuzzers/Fuzzer ...


def stage8(ctx):
    """Compare the SOM with the assignment 2 results (taken from the report), plus figures."""
    data = ctx["data"]
    K, classes = len(data.classes), data.classes
    models = _final_models()
    final = pd.read_csv(C.RESULTS_DIR / "stage6_final_runs.csv")
    mets = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc", "kappa"]
    med = int((final["macro_f1"] - final["macro_f1"].median()).abs().idxmin())
    som_pred = np.load(C.RESULTS_DIR / "final_preds.npy")[med]
    m = ev.all_metrics(data.yte, som_pred, K)
    ci, ci_pc = ev.bootstrap_ci(data.yte, som_pred, K, C.N_BOOTSTRAP)
    rows = [dict(model="SOM (mean of runs)", **final[mets].mean().to_dict()),
            dict(model="SOM (SD over runs)", **final[mets].std().to_dict()),
            dict(model="SOM (median run)", **m, macro_f1_ci_low=ci[0], macro_f1_ci_high=ci[1])]
    for name, mm in C.A2_TEST_METRICS.items():
        rows.append(dict(model=f"{name} (assignment 2)", **mm))
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(C.RESULTS_DIR / "stage8_comparison.csv")
    print(table.round(4).to_string())
    print(f"\nSOM median-run macro-F1 95% bootstrap CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    for name, mm in C.A2_TEST_METRICS.items():
        inside = ci[0] <= mm["macro_f1"] <= ci[1]
        print(f"  {name}: macro-F1 {mm['macro_f1']:.4f}  (SOM mean - {name}) = {final['macro_f1'].mean() - mm['macro_f1']:+.4f}"
              f"  inside SOM CI: {inside}")
    # per-class F1
    rep = ev.class_report(data.yte, som_pred, classes)
    rep.to_csv(C.RESULTS_DIR / "stage8_class_report_SOM.csv")
    pc = pd.DataFrame({"SOM": rep["f1"]})
    for name, d in C.A2_PER_CLASS_F1.items():
        look = {_norm(k): v for k, v in d.items()}
        pc[name] = [look.get(_norm(c), np.nan) for c in classes]
    pc.to_csv(C.RESULTS_DIR / "stage8_per_class_f1.csv")
    print(pc.round(2).to_string())
    pd.DataFrame(ci_pc, index=classes, columns=["som_f1_ci_low", "som_f1_ci_high"]).to_csv(
        C.RESULTS_DIR / "stage8_som_per_class_ci.csv")
    # figures
    clf, prep = models[med]["clf"], models[med]["prep"]
    viz.plot_overview(clf)
    viz.plot_component_planes(clf, prep.get_feature_names_out())
    viz.plot_class_maps(clf, classes)
    viz.plot_confusion(confusion(data.yte, som_pred, K), classes, "confusion_som.png",
                       "SOM - row-normalised confusion (%)")
    viz.plot_per_class_f1(pc[["SOM", "Decision tree", "PCA KNN", "KNN"]])
    print(f"[stage8] figures written to {C.FIGS_DIR}/ ; SOM predictions from run {med} (median macro-F1)")


STAGES = {0: stage0, 1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6, 7: stage7, 8: stage8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="0", help="0-8 or 'all' (comma separated allowed)")
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel map trainings (threads)")
    ap.add_argument("--no-screen", action="store_true", help="skip subsample screening; full CV for every config")
    ap.add_argument("--data", default=None, help="override config.DATA_PATH")
    ap.add_argument("--synthetic", type=int, default=0, help="use N rows of synthetic data (smoke test)")
    ap.add_argument("--quick", action="store_true", help="tiny grids (smoke test)")
    a = ap.parse_args()
    if a.quick:
        C.apply_quick()
    if a.data:
        C.DATA_PATH = a.data
    C.RESULTS_DIR.mkdir(exist_ok=True)
    C.FIGS_DIR.mkdir(exist_ok=True)
    if a.synthetic:
        from synthetic import make_synthetic
        data = load_data(make_synthetic(a.synthetic))
    else:
        data = load_data()
    print(f"[data] train {len(data.Xtr)}  test {len(data.Xte)}  classes {len(data.classes)}  "
          f"numeric {len(data.num_cols)}  categorical {len(data.cat_cols)}")
    ctx = dict(data=data, n_jobs=a.n_jobs, no_screen=a.no_screen)
    todo = list(range(9)) if a.stage == "all" else [int(s) for s in a.stage.split(",")]
    for s in todo:
        print(f"\n=== stage {s} ===")
        STAGES[s](ctx)


if __name__ == "__main__":
    main()
