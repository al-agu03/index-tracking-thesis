"""
Ablation of Phase A: stock-selection rules and mappings are varied separately.

Selection rules: autoencoder and PCA reconstruction error, on levels or on daily
returns, each with HPW's both-ends rule or the n most communal stocks; market
capitalisation; correlation with the index; and random selections. The
autoencoder's sparsity penalty is optional, as in Phase A. Random selections are
matched by capitalisation decile to the seed-0 selection of --null-rule, use one
network seed per draw, and serve as a descriptive reference, not a formal test.

Mappings from the selected stocks to the index: an equally weighted basket of the
first ten, OLS, ridge, and the DFP policy, with penalties cross-validated as in
Phase A. ae_hpw + dfp reproduces Phase A and is checked against its saved results.

    python ablation_phaseA.py --n 60 --ae-beta 0 --phaseA-dir phaseA_outputs_v2_beta0_n60_max60
"""

# Single-threaded linear algebra, set before NumPy loads: the summation order
# inside matrix products, and hence the results, depends on the thread count.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phaseA_deep_frontier as phase_a

HERE = Path(__file__).resolve().parent
OUT = HERE / "ablation_phaseA_outputs"
OUT.mkdir(exist_ok=True)

CALIB = phase_a.CALIB
VERIF = phase_a.VERIF
LAMBDAS = phase_a.LAMBDAS
# Ridge lives on another scale than the policy penalty, so its grid spans more
# decades, in the same half-decade steps as LAMBDAS.
RIDGE_LAMBDAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1,
                 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4]
N_COMMUNAL = phase_a.N_COMMUNAL
CAP_FILE = "Daily cap.csv"


# ===========================================================================
# Rankings
# ===========================================================================
def to_returns(frame):
    """Daily simple returns of a level panel, as the input to a returns ranker.

    Invariant to the rebasing applied downstream, so it does not matter whether
    the frame handed in has been divided by its first row. One row is lost.
    """
    return frame.pct_change().iloc[1:]


def rank_pca(frame, k=4):
    """Per-column reconstruction error from the leading k principal components.

    A linear low-rank counterpart of the autoencoder ranking: the same operation
    with the subspace of the leading principal components. A difference between
    the two reflects the autoencoder's nonlinearity, sparsity penalty, ridge term
    and optimisation jointly.
    """
    X = frame.values
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    Zhat = (U[:, :k] * S[:k]) @ Vt[:k]
    err = ((Zhat - Z) ** 2).mean(0)
    return pd.Series(err, index=frame.columns).sort_values()


def rank_correlation(frame, index):
    """Correlation of daily returns with the index, descending, as an error-like
    score so that 'smaller is more communal' holds for every rule."""
    r = frame.pct_change().iloc[1:]
    ri = index.pct_change().iloc[1:]
    corr = r.corrwith(ri)
    return (1.0 - corr).sort_values()


def rank_marketcap(caps):
    """Average market equity over whatever window is passed, largest first."""
    return (-caps.dropna()).sort_values()


def market_equity_panel(cap, columns, span):
    """Daily market equity as a wide frame whose columns match the price panel.

    Keyed by PERMNO, since some securities change ticker inside the window, and
    mapped to the symbol the panel uses (the first one observed in the study
    period, as in crsp_panel.py). Duplicated security-days are dropped only if
    the copies agree; otherwise the run stops.
    """
    cap = cap.copy()
    cap["DlyCap"] = pd.to_numeric(cap.DlyCap, errors="coerce")
    key = ["PERMNO", "DlyCalDt"]
    copies = cap[cap.duplicated(key, keep=False)]
    # Identical means the same number or both missing.
    distinct = copies.assign(v=copies.DlyCap.fillna(-1.0)).groupby(key).v.nunique()
    if (distinct > 1).any():
        bad = distinct[distinct > 1].index[:5].tolist()
        raise SystemExit(f"{int((distinct > 1).sum())} security-days carry conflicting "
                         f"market equity in {CAP_FILE}, e.g. {bad}; fix the extract.")
    cap = cap.drop_duplicates(key)
    inside = cap[cap.DlyCalDt.between(*span)].dropna(subset=["Ticker"])
    name = inside.sort_values("DlyCalDt").groupby("PERMNO").Ticker.first()
    wide = cap.pivot_table(index="DlyCalDt", columns="PERMNO", values="DlyCap")
    wide.columns = [name.get(p) for p in wide.columns]
    wide = wide.loc[:, [bool(c) and c in set(columns) for c in wide.columns]]
    dup = pd.Index(wide.columns)[pd.Index(wide.columns).duplicated()].unique()
    if len(dup):
        raise SystemExit(
            f"two PERMNOs map to the same panel symbol: {', '.join(map(str, dup))}."
            f"\nDropping one would pick a security by column order and defeat the "
            f"point of\nkeying on PERMNO. Resolve the mapping instead.")
    return wide


def portfolio(pa, rank, n, rule):
    """The both-ends rule of HPW, taken from Phase A so the de-duplication of a
    name appearing at both ends is handled identically, or simply the top n.

    Returns None unless exactly n distinct names with a finite score are
    available; a short ranking is treated as a failed ranking, not a smaller
    portfolio, so it can never be compared with full-size ones by accident.
    """
    rank = rank[np.isfinite(rank.values)]
    if len(rank) < n:
        return None
    names = pa.portfolio_from(rank, n) if rule == "hpw" else rank.index[:n].tolist()
    return names if len(set(names)) == n else None


# ===========================================================================
# Maps
# ===========================================================================
def fit_ridge(X, y, lam):
    """Ridge with an unpenalised intercept, solved by least squares.

    Uses the augmented least-squares form rather than the normal equations, which
    square the condition number of X.
    """
    mx, my = X.mean(0), y.mean()
    Xc, yc = X - mx, y - my
    if lam > 0:
        Xc = np.vstack([Xc, np.sqrt(lam) * np.eye(X.shape[1])])
        yc = np.concatenate([yc, np.zeros(X.shape[1])])
    beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    return beta, my - mx @ beta


def predict_ridge(model, X):
    beta, b0 = model
    return X @ beta + b0


SEEDS = phase_a.SEEDS      # seeds shared with Phase A
MAPS = ("basket", "ols", "ridge", "dfp")


def fit_predict(pa, kind, Xtr, ytr, Xva, lam, seed):
    """Fit a ridge or the policy on training paths; return the training and
    validation predictions. A failed policy fit predicts NaN, as in Phase A."""
    if kind == "ridge":
        m = fit_ridge(Xtr, ytr, lam)
        return predict_ridge(m, Xtr), predict_ridge(m, Xva)
    d = pa.DFP(n_in=Xtr.shape[1], seed=seed, **dict(pa.DFP_PARAMS, lam=lam)).fit(Xtr, ytr)
    return d.predict(Xtr), d.predict(Xva)


def cv_select(pa, frame, index, rule, n, kind, seeds, fixed_names=None, verbose=False):
    """
    Choose the map's penalty on the anchored folds, selection inside the loop.

    The score of a penalty is the mean validation TE over folds and seeds. A
    penalty that fails in any fold or seed -- no ranking, a failed fit, or a
    non-positive path in training or in validation -- is excluded, the same
    eligibility rule as Phase A. The validation TEs are averaged in the
    same order as Phase A, seed by seed and fold by fold, so the score of ae_hpw +
    dfp reproduces Phase A's te_val. `fixed_names` is used for random draws, whose
    stocks do not depend on a fold.
    """
    folds = pa.anchored_folds(len(frame), pa.N_FOLDS, pa.MIN_TRAIN_DAYS)
    grid = RIDGE_LAMBDAS if kind == "ridge" else LAMBDAS
    te = {lam: [] for lam in grid}       # validation TE per (seed, fold), NaN if invalid
    for s in seeds:
        for (t0a, t0b), (t1a, t1b) in folds:
            tr, va = frame.iloc[t0a:t0b], frame.iloc[t1a:t1b]
            itr, iva = index.iloc[t0a:t0b], index.iloc[t1a:t1b]
            cols = (list(fixed_names) if fixed_names is not None
                    else refit_names(pa, tr, itr, rule, n, s))
            if cols is None:                       # the ranking failed
                for lam in grid:
                    te[lam].append(np.nan)
                continue
            # Preserve Phase A's portfolio_from order: network initial weights
            # are attached to input positions.
            Xtr = (tr[cols] / tr[cols].iloc[0]).values
            ytr = (itr / itr.iloc[0]).values
            Xva = (va[cols] / va[cols].iloc[0]).values
            yva = (iva / iva.iloc[0]).values
            for lam in grid:
                p_tr, p_va = fit_predict(pa, kind, Xtr, ytr, Xva, lam, s)
                te_tr = pa.level_stats(p_tr, ytr)[0]
                te_va = pa.level_stats(p_va, yva)[0]
                ok = np.isfinite(te_tr) and np.isfinite(te_va)
                te[lam].append(te_va if ok else np.nan)
            if verbose:
                print(f"      seed {s}, fold ending {frame.index[t1b - 1].date()} done, "
                      f"{time.strftime('%H:%M:%S')}", flush=True)
    # A penalty with any invalid fit is ineligible; the rest are scored by the mean.
    scores = np.array([np.mean(v) if np.all(np.isfinite(v)) else np.inf
                       for v in te.values()])
    if not np.isfinite(scores).any():
        return np.nan, np.inf
    return grid[int(np.argmin(scores))], float(scores.min())


RANKERS = {}    # name -> f(frame, index, seed) -> ranked Series, filled in main
RULES = {}      # rule name -> (ranker name, "hpw" or "top"), filled in main
SEEDED = {"ae", "ae_ret"}   # rankers that depend on the seed
_RANK_CACHE = {}


def ranking(frame, index, ranker, seed):
    """
    The ranking a window produces, computed once and kept; None if it failed.

    The key carries the seed, the dates and the column identities. Rankers that
    do not depend on the seed share one entry across seeds.
    """
    s = seed if ranker in SEEDED else None
    key = (ranker, s, frame.index[0], frame.index[-1], tuple(frame.columns))
    if key not in _RANK_CACHE:
        try:
            _RANK_CACHE[key] = RANKERS[ranker](frame, index, seed)
        except phase_a.TrainingFailed as err:
            print(f"      {ranker} ranking failed: {err}")
            _RANK_CACHE[key] = None
    return _RANK_CACHE[key]


def refit_names(pa, frame, index, rule, n, seed):
    ranker, kind = RULES[rule]
    rk = ranking(frame, index, ranker, seed)
    return None if rk is None else portfolio(pa, rk, n, kind)


def evaluate(pa, kind, names, basket_names, lam, seed, c_cal, i_cal, c_ver, i_ver):
    """Calibration and verification figures for one map on one selection."""
    yc = (i_cal / i_cal.iloc[0]).values
    yv = (i_ver / i_ver.iloc[0]).values
    if kind == "basket":
        pc = (c_cal[basket_names] / c_cal[basket_names].iloc[0]).mean(axis=1).values
        pv = (c_ver[basket_names] / c_ver[basket_names].iloc[0]).mean(axis=1).values
    else:
        Xc = (c_cal[names] / c_cal[names].iloc[0]).values
        Xv = (c_ver[names] / c_ver[names].iloc[0]).values
        if kind in ("ols", "ridge"):
            m = fit_ridge(Xc, yc, lam)
            pc, pv = predict_ridge(m, Xc), predict_ridge(m, Xv)
        else:
            d = pa.DFP(n_in=len(names), seed=seed,
                       **dict(pa.DFP_PARAMS, lam=lam)).fit(Xc, yc)
            pc, pv = d.predict(Xc), d.predict(Xv)
    te_c, rm_c = pa.level_stats(pc, yc)
    te_v, rm_v = pa.level_stats(pv, yv)
    return dict(te_calib=te_c, rmse_calib=rm_c, mar_calib=pa.active_return(pc, yc),
                te_verif=te_v, rmse_verif=rm_v, mar_verif=pa.active_return(pv, yv),
                valid=bool(np.isfinite(te_c) and np.isfinite(te_v)))


def nan_metrics():
    return dict(te_calib=np.nan, rmse_calib=np.nan, mar_calib=np.nan,
                te_verif=np.nan, rmse_verif=np.nan, mar_verif=np.nan, valid=False)


def resume(partial, cfg, fingerprint, key_cols, expected_of):
    """
    Completed units from an interrupted run of the same configuration.

    A unit (a rule, or a random draw) counts as complete only if it holds exactly
    the expected rows, each once; rows of incomplete units are discarded by
    rewriting the file atomically, so they cannot accumulate. The fingerprint
    carries hashes of the data and of both scripts, so results from different
    data, code or settings are never merged.

    Each stage (grid, random draws) has its own config file next to its own
    partial file; a partial file is accepted only if its own config file exists
    and matches.
    """
    import json
    text = json.dumps(fingerprint, sort_keys=True, default=str)
    if partial.exists() and not (cfg.exists() and cfg.read_text() == text):
        raise SystemExit(f"{partial.name} belongs to a run with a different (or "
                         f"unrecorded) configuration, data or code; remove it to "
                         f"start afresh")
    cfg.write_text(text)
    done = {}
    if partial.exists():
        prev = pd.read_csv(partial, float_precision="round_trip")
        keep = []
        unit = key_cols[0]
        for u, g in prev.groupby(unit, sort=False):
            pairs = list(zip(*(g[c] for c in key_cols[1:])))
            want = expected_of(u)
            if len(pairs) == len(want) and set(pairs) == set(want):
                done[u] = g
                keep.append(g)
        dropped = len(prev) - sum(len(g) for g in keep)
        tmp = partial.with_name(partial.name + ".tmp")
        if keep:
            pd.concat(keep).to_csv(tmp, index=False)
            os.replace(tmp, partial)
        else:
            partial.unlink()
        print(f"  resuming {partial.name}: {len(done)} complete, "
              f"{dropped} rows of incomplete units discarded")
    return done


def phase_a_results(pa, fingerprint_of_run, folder=None):
    """
    Phase A's grid and per-seed verification for the same configuration.

    `folder` defaults to Phase A's default output folder for this beta; pass the
    folder of an override run (e.g. --n 60) to compare with that run instead.
    Stops, naming the reason, unless the Phase A folder holds one finished run with
    the same code, data, universe, seeds, encoder, policy and penalty grid: no
    partial file left, and the config (written when the run starts), the grid and
    the verification file written in that order.
    """
    import json
    d = Path(folder) if folder else pa.output_dir(fingerprint_of_run["ae_params"]["beta"])
    files = [d / "phaseA_run_config.json", d / "phaseA_grid_total.csv",
             d / "phaseA_verification_seeds_total.csv"]
    hint = "\nRun Phase A at this configuration first, or pass --skip-phaseA-check."
    if not all(f.exists() for f in files):
        raise SystemExit(f"Phase A check: {d.name} lacks its results.{hint}")
    if (d / "phaseA_grid_long_total.partial.csv").exists():
        raise SystemExit(f"Phase A check: {d.name} holds an unfinished run.{hint}")
    t = [f.stat().st_mtime for f in files]
    if not t[0] <= t[1] <= t[2]:
        raise SystemExit(f"Phase A check: the files in {d.name} are not from one "
                         f"finished run.{hint}")
    cfg = json.loads(files[0].read_text())
    norm = lambda x: json.loads(json.dumps(x, default=str))          # noqa: E731
    for key, ours in (("script_sha256", fingerprint_of_run["sha256_phaseA_deep_frontier.py"]),
                      ("panel_sha256", fingerprint_of_run["sha256_panel_total.csv"]),
                      ("universe", fingerprint_of_run["universe"]),
                      ("seeds", fingerprint_of_run["seeds"]),
                      ("ae_params", fingerprint_of_run["ae_params"]),
                      ("dfp_params", pa.DFP_PARAMS),
                      ("lambdas", LAMBDAS)):
        if norm(cfg.get(key)) != norm(ours):
            raise SystemExit(f"Phase A check: {key} differs between Phase A's run "
                             f"and this one.{hint}")
    grid_a = pd.read_csv(files[1], float_precision="round_trip")
    n = fingerprint_of_run["n"]
    if not (grid_a[grid_a.n_stocks == n].eligible.any()):
        raise SystemExit(f"Phase A check: Phase A has no eligible penalty at n={n}.{hint}")
    return grid_a, pd.read_csv(files[2], float_precision="round_trip")


def check_against_phase_a(phase_a_ref, n, ae_rows, tol=1e-12):
    """
    ae_hpw + dfp here is Phase A's procedure at a fixed n, so it must give Phase
    A's numbers: the same chosen penalty with the same cross-validation score
    and, if Phase A selected that same (n, penalty), the same verification TE in
    every seed. Any difference stops the run; the tolerance only absorbs
    floating-point summation order.
    """
    grid_a, ver_a = phase_a_ref
    at_n = grid_a[(grid_a.n_stocks == n) & grid_a.eligible]
    best = at_n.loc[at_n.te_val.idxmin()]
    ours = ae_rows[ae_rows.mapping == "dfp"].sort_values("seed")
    lam, score = float(ours.lam.iloc[0]), float(ours.cv_te.iloc[0])
    rel = lambda a, b: abs(a - b) / abs(b)                            # noqa: E731
    problems = []
    if lam != float(best.lam):
        problems.append(f"penalty {lam:.0e} here, {best.lam:.0e} in Phase A")
    elif rel(score, best.te_val) > tol:
        problems.append(f"CV score {score:.12f} here, {best.te_val:.12f} in Phase A")
    print(f"  Phase A check at n={n}: penalty {lam:.0e} vs {best.lam:.0e}, "
          f"CV TE {score:.10f} vs {best.te_val:.10f}")
    sel = ver_a[ver_a.model == ver_a.model.iloc[0]]     # the selected model comes first
    if (not problems and len(sel) and int(sel.n_stocks.iloc[0]) == n
            and float(sel.lam.iloc[0]) == lam):
        a = sel.set_index("seed").te_verif
        b = ours.set_index("seed").te_verif
        if set(a.index) != set(b.index):
            problems.append("different seeds in the verification")
        else:
            worst = float(np.max([rel(b[s], a[s]) for s in a.index]))  # NaN propagates
            print(f"  Phase A selected this cell: verification TE per seed agrees "
                  f"to a relative {worst:.1e}")
            if not worst <= tol:              # also catches a NaN on either side
                problems.append(f"verification TE differs (relative {worst:.1e})")
    elif not problems and len(sel):
        print(f"  Phase A selected a different cell (n={int(sel.n_stocks.iloc[0])}), "
              f"so only the cross-validation is compared")
    if problems:
        raise SystemExit("Phase A check FAILED: " + "; ".join(problems))
    print("  Phase A check passed")


def summarise(per_seed):
    """
    Rule x map table: mean and spread over valid seeds, with failures counted.

    A cell with a failed seed is marked incomplete rather than averaged silently.
    """
    metrics = ["te_calib", "rmse_calib", "mar_calib", "te_verif", "rmse_verif", "mar_verif"]
    keys = ["selection", "mapping"]
    v = per_seed[per_seed.valid].groupby(keys)
    out = (per_seed.groupby(keys)[["lam"]].first()
           .join(v[metrics].mean())
           .join(v[metrics].std(ddof=1).add_suffix("_seed_sd")))
    c = per_seed.groupby(keys).valid.agg(["sum", "size"])
    out["valid_seeds"] = c["sum"].astype(int)
    out["failed_seeds"] = (c["size"] - c["sum"]).astype(int)
    out["complete"] = out["failed_seeds"] == 0
    return out.reset_index()


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50,
                    help="portfolio size; defaults to the 50-stock budget")
    ap.add_argument("--ae-beta", type=float, default=None,
                    help="encoder sparsity weight; defaults to Phase A's 3")
    ap.add_argument("--ae-lambda", type=float, default=None,
                    help="ridge of the encoder; defaults to Phase A's 1e-4")
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--cv-replicates", type=int, default=20,
                    help="random draws on which the penalty is also cross-validated, "
                         "to check whether holding it fixed matters")
    ap.add_argument("--verbose", action="store_true",
                    help="report each cross-validation fold as it finishes")
    ap.add_argument("--random-seed", type=int, default=0,
                    help="controls the random stock draws only")
    ap.add_argument("--null-rule", default="ae_hpw",
                    help="the selection rule the matched random reference and the "
                         "primary comparisons are built around")
    ap.add_argument("--skip-phaseA-check", action="store_true",
                    help="do not compare ae_hpw + dfp with Phase A's results; only "
                         "for configurations Phase A was not run at")
    ap.add_argument("--phaseA-dir", default=None,
                    help="Phase A output folder to check against; defaults to "
                         "phaseA_outputs_v2 (or its _beta variant)")
    args = ap.parse_args()
    for name, v in (("--ae-beta", args.ae_beta), ("--ae-lambda", args.ae_lambda)):
        if v is not None and not (np.isfinite(v) and v >= 0):
            ap.error(f"{name} must be finite and non-negative, got {v}")

    pa = phase_a
    seeds = list(SEEDS)
    ae_params = dict(pa.AE_PARAMS)
    if args.ae_beta is not None:
        ae_params["beta"] = args.ae_beta
    if args.ae_lambda is not None:
        ae_params["lam"] = args.ae_lambda

    panel = pd.read_csv(HERE / "panel_total.csv", index_col=0, parse_dates=True)
    idx, stocks = panel["SP500"], panel.drop(columns=["SP500"])
    keep_full = pa.full_coverage(stocks, stocks.columns, *pa.FULL_SPAN)
    keep_cal = pa.full_coverage(stocks, stocks.columns, *pa.CALIB)
    lost = sorted(set(keep_cal) - set(keep_full))
    print(f"universe: {len(keep_cal)} cover calibration, {len(keep_full)} cover "
          f"calibration and verification; {len(lost)} excluded by the through-2015 "
          f"requirement\n  ({len(lost) / max(1, len(keep_cal)):.1%} of the "
          f"calibration universe, decided with dates after it)")
    stocks = stocks[keep_full]
    if not N_COMMUNAL <= args.n <= stocks.shape[1]:
        raise SystemExit(f"--n must lie between {N_COMMUNAL} and {stocks.shape[1]}")
    if args.replicates <= 0:
        raise SystemExit("--replicates must be positive")
    if not 0 <= args.cv_replicates <= args.replicates:
        raise SystemExit("--cv-replicates must lie between 0 and --replicates")
    c_cal, i_cal, c_ver, i_ver = pa.checked_windows(stocks, idx)
    print(f"{stocks.shape[1]} eligible securities, portfolio size n = {args.n}, "
          f"seeds {seeds}")

    cap_raw = pd.read_csv(HERE / CAP_FILE, parse_dates=["DlyCalDt"], low_memory=False)
    cap_wide = market_equity_panel(cap_raw, stocks.columns, pa.FULL_SPAN)
    cal_cap = cap_wide.loc[CALIB[0]:CALIB[1]].mean()
    print(f"market equity available for {cal_cap.notna().sum()} of them")
    if cal_cap.notna().sum() < args.n:
        raise SystemExit(f"only {cal_cap.notna().sum()} securities carry market "
                         f"equity, fewer than the requested n = {args.n}")
    print(f"encoder: {ae_params['n_hidden']} units, rho={ae_params['rho']}, "
          f"beta={ae_params['beta']:g}, ridge={ae_params['lam']:.0e}")

    tag = (f"b{ae_params['beta']:g}_n{args.n}_r{args.replicates}"
           f"_cv{args.cv_replicates}_rs{args.random_seed}_s{len(seeds)}_{args.null_rule}")
    tag_csv = lambda part: f"ablation_{part}_{tag}.csv"          # noqa: E731

    # Rankings, each re-estimable inside a fold, and the rules built from them.
    _RANK_CACHE.clear()
    RANKERS.update({
        "ae":  lambda f, i, s: pa.rank_universe(f, seed=s, params=ae_params)[0],
        "pca": lambda f, i, s: rank_pca(f),
        "cor": lambda f, i, s: rank_correlation(f, i),
        "ae_ret":  lambda f, i, s: pa.rank_universe(to_returns(f), seed=s,
                                                   params=ae_params)[0],
        "pca_ret": lambda f, i, s: rank_pca(to_returns(f)),
        # Averaged over the window actually passed, so inside a fold the ranking
        # uses that fold's training dates only.
        "cap": lambda f, i, s: rank_marketcap(cap_wide.reindex(f.index)
                                              .reindex(columns=f.columns).mean()),
    })
    RULES.update({"ae_hpw": ("ae", "hpw"), "ae_communal": ("ae", "top"),
                  "pca_hpw": ("pca", "hpw"), "pca_communal": ("pca", "top"),
                  "ae_ret_hpw": ("ae_ret", "hpw"),
                  "ae_ret_communal": ("ae_ret", "top"),
                  "pca_ret_hpw": ("pca_ret", "hpw"),
                  "pca_ret_communal": ("pca_ret", "top"),
                  "correlation": ("cor", "top"), "marketcap": ("cap", "top")})
    if args.null_rule not in RULES:
        raise SystemExit(f"--null-rule must be one of: {', '.join(RULES)}")

    import hashlib
    fingerprint = dict(
        version="ablation v2", tag=tag, n=args.n, seeds=seeds, ae_params=ae_params,
        dfp_params=pa.DFP_PARAMS, lambdas=LAMBDAS, ridge_lambdas=RIDGE_LAMBDAS,
        replicates=args.replicates, cv_replicates=args.cv_replicates,
        random_seed=args.random_seed, null_rule=args.null_rule,
        universe=list(stocks.columns),
        **{f"sha256_{f}": hashlib.sha256((HERE / f).read_bytes()).hexdigest()
           for f in ("panel_total.csv", CAP_FILE, "ablation_phaseA.py",
                     "phaseA_deep_frontier.py")})

    # ---------------- saved progress of both stages --------------------------
    # Both partial files are checked before any fitting, so a stale random-stage
    # file stops the run now rather than after the grid has been recomputed.
    grid_part = OUT / f"ablation_grid_long_{tag}.partial.csv"
    want_grid = {(m, s) for m in MAPS for s in seeds}
    done_grid = resume(grid_part, OUT / f"ablation_run_config_grid_{tag}.json",
                       fingerprint, ["selection", "mapping", "seed"],
                       lambda u: want_grid)
    null_part = OUT / f"ablation_null_{tag}.partial.csv"

    def want_null(r):
        fixed = {(k, "fixed") for k in MAPS}
        return fixed | ({("ridge", "cv"), ("dfp", "cv")} if r < args.cv_replicates
                        else set())

    done_null = resume(null_part, OUT / f"ablation_run_config_null_{tag}.json",
                       fingerprint, ["replicate", "mapping", "regime"], want_null)

    # ---------------- the grid: rule x map x seed --------------------------

    # ae_hpw runs first (RULES starts with it) and is checked against Phase A
    # before anything else is computed, so a drift costs minutes, not hours.
    # Phase A's files are validated now, before any fitting.
    phase_a_ref = (None if args.skip_phaseA_check
                   else phase_a_results(pa, fingerprint, args.phaseA_dir))
    rows = []
    for rule in RULES:
        if rule in done_grid:
            print(f"  {rule:16s} (resumed)")
            rows += done_grid[rule].to_dict("records")
            if rule == "ae_hpw" and not args.skip_phaseA_check:
                check_against_phase_a(phase_a_ref, args.n, done_grid[rule])
            continue
        t_rule = time.time()
        names = {s: refit_names(pa, c_cal, i_cal, rule, args.n, s) for s in seeds}
        base = names[seeds[0]]
        lam = {"basket": np.nan, "ols": 0.0}
        cv = {"basket": np.nan, "ols": np.nan}
        for kind in ("ridge", "dfp"):
            lam[kind], cv[kind] = cv_select(pa, c_cal, i_cal, rule, args.n, kind, seeds,
                                            verbose=args.verbose)
        rule_rows = []
        for kind in MAPS:
            for s in seeds:
                sel = names[s]
                ok_lam = kind in ("basket", "ols") or np.isfinite(lam[kind])
                r = (evaluate(pa, kind, sel, sel[:N_COMMUNAL], lam[kind], s,
                              c_cal, i_cal, c_ver, i_ver)
                     if sel is not None and ok_lam else nan_metrics())
                rule_rows.append(dict(
                    selection=rule, mapping=kind, seed=s, lam=lam[kind],
                    cv_te=cv[kind], **r,
                    names_shared_with_seed0=(np.nan if sel is None or base is None
                                             else len(set(sel) & set(base)))))
        g = pd.DataFrame(rule_rows)
        if rule == "ae_hpw" and not args.skip_phaseA_check:
            check_against_phase_a(phase_a_ref, args.n, g)   # before saving it
        g.to_csv(grid_part, mode="a", index=False, header=not grid_part.exists())
        rows += rule_rows
        line = "  ".join(
            f"{k} {g[(g.mapping == k) & g.valid].te_verif.mean():.4f}"
            + ("*" if (~g[g.mapping == k].valid).any() else "") for k in MAPS)
        print(f"  {rule:16s} verif TE (mean over seeds)  {line}   "
              f"lam ridge={lam['ridge']:.0e} dfp={lam['dfp']:.0e}  "
              f"[{time.time() - t_rule:5.0f}s]", flush=True)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / tag_csv("grid_seeds"), index=False)
    grid = summarise(per_seed)
    grid.to_csv(OUT / tag_csv("grid"), index=False)
    grid_part.unlink(missing_ok=True)
    print("  (* = at least one seed failed; that cell is marked incomplete)")
    chosen = {(r.selection, r.mapping): r.lam for r in grid.itertuples()}

    # ---------------- the matched random reference -------------------------
    # Stratifying by market-cap decile keeps the random sets comparable in size
    # composition to the null rule's selection (seed 0), so the contrast is about
    # communality and not about which names happen to be large. The basket is
    # matched separately to the ten names it actually holds.
    print(f"\n{args.replicates} stratified random selections ...")
    deciles = pd.qcut(cal_cap.dropna(), 10, labels=False, duplicates="drop")

    def composition_of(names):
        d = deciles.reindex(names).dropna().astype(int)
        return d.value_counts().reindex(sorted(deciles.unique()), fill_value=0)

    null_sel = refit_names(pa, c_cal, i_cal, args.null_rule, args.n, seeds[0])
    if null_sel is None:
        raise SystemExit(f"the {args.null_rule} ranking failed at seed {seeds[0]}; "
                         f"the random reference cannot be matched to it")
    comp_full = composition_of(null_sel)
    comp_basket = composition_of(null_sel[:N_COMMUNAL])
    print(f"decile composition matched to {args.null_rule} (seed {seeds[0]}), "
          f"deciles 0 (smallest) to 9")
    print(f"  {args.n} names : " + ", ".join(f"{d}:{c}" for d, c in comp_full.items()))
    print(f"  basket of {N_COMMUNAL}: "
          + ", ".join(f"{d}:{c}" for d, c in comp_basket.items()))

    def draw(composition, size, rng):
        """A matched random set in a fixed order (initial weights attach to
        column position, so a shuffled order would add initialisation noise)."""
        picks = []
        for d, want in composition.items():
            pool = deciles.index[deciles == d].tolist()
            picks += list(rng.choice(pool, min(int(want), len(pool)), replace=False))
        extra = [t for t in rng.permutation(deciles.index) if t not in picks]
        return sorted((picks + extra)[:size])

    rng = np.random.default_rng(args.random_seed)
    null_rows = []
    for r in range(args.replicates):
        # Always draw, so the random sequence is the same whether or not earlier
        # draws were resumed.
        names = draw(comp_full, args.n, rng)
        basket_names = draw(comp_basket, N_COMMUNAL, rng)
        if r in done_null:
            null_rows += done_null[r].to_dict("records")
            continue
        s = seeds[r % len(seeds)]        # one network seed per draw, cycling
        draw_rows = []
        for kind in MAPS:
            lam = {"basket": np.nan, "ols": 0.0}.get(kind, chosen[(args.null_rule, kind)])
            m = (evaluate(pa, kind, names, basket_names, lam, s, c_cal, i_cal, c_ver, i_ver)
                 if (kind in ("basket", "ols") or np.isfinite(lam)) else nan_metrics())
            draw_rows.append(dict(replicate=r, mapping=kind, regime="fixed", seed=s,
                                  lam=lam, te_verif=m["te_verif"],
                                  rmse_verif=m["rmse_verif"], valid=m["valid"]))
            # On a subset of draws the penalty is re-tuned for that draw, on its
            # own seed, instead of inherited; the difference is paired.
            if r < args.cv_replicates and kind in ("ridge", "dfp"):
                lam_cv, _ = cv_select(pa, c_cal, i_cal, None, args.n, kind, [s],
                                      fixed_names=names)
                m = (evaluate(pa, kind, names, basket_names, lam_cv, s,
                              c_cal, i_cal, c_ver, i_ver)
                     if np.isfinite(lam_cv) else nan_metrics())
                draw_rows.append(dict(replicate=r, mapping=kind, regime="cv", seed=s,
                                      lam=lam_cv, te_verif=m["te_verif"],
                                      rmse_verif=m["rmse_verif"], valid=m["valid"]))
        pd.DataFrame(draw_rows).to_csv(null_part, mode="a", index=False,
                                       header=not null_part.exists())
        null_rows += draw_rows
        if (r + 1) % 10 == 0:
            print(f"  {r + 1}/{args.replicates}  {time.strftime('%H:%M:%S')}",
                  flush=True)
    null = pd.DataFrame(null_rows)
    null.to_csv(OUT / tag_csv("null"), index=False)
    null_part.unlink(missing_ok=True)

    fail = null[(null.regime == "fixed") & ~null.valid]
    if len(fail):
        print("\nrandom draws producing an invalid path, excluded:")
        print(fail.groupby("mapping").size().to_string())
    ok = null[(null.regime == "fixed") & null.valid]

    # ---------------- position of the null rule ----------------------------
    nr = args.null_rule
    print(f"\n{nr} (mean over seeds) against its matched random reference "
          f"(verification TE):")
    pos = []
    for kind in MAPS:
        d = ok[ok.mapping == kind].te_verif.values
        row = grid[(grid.selection == nr) & (grid.mapping == kind)].iloc[0]
        v = row.te_verif
        pos.append(dict(mapping=kind, selection=nr, te=v,
                        te_seed_sd=row.te_verif_seed_sd,
                        complete=bool(row.complete), n_valid=len(d),
                        pct_of_random_beaten=float((d > v).mean()) if len(d) else np.nan,
                        null_median=float(np.median(d)) if len(d) else np.nan,
                        null_q05=float(np.quantile(d, .05)) if len(d) else np.nan))
    pos = pd.DataFrame(pos)
    pos.to_csv(OUT / tag_csv("position"), index=False)
    print(pos[["mapping", "te", "te_seed_sd", "null_median", "null_q05",
               "pct_of_random_beaten", "n_valid", "complete"]].round(4).to_string(index=False))

    print("\nOf the four mappings only the basket is a portfolio: equal amounts "
          "bought once\nand held. The other three return a fitted level series "
          "with unconstrained\ncoefficients, so their tracking error measures "
          "reconstruction by a function and\nnot the experience of holding "
          "anything. The columns are comparable as procedures,\nnot as trackers.")
    print("\nall selection rules, verification TE (mean over seeds):")
    print(grid.pivot_table(index="selection", columns="mapping",
                           values="te_verif")[list(MAPS)].round(4).to_string())
    print("\nspread across seeds (sd of verification TE):")
    print(grid.pivot_table(index="selection", columns="mapping",
                           values="te_verif_seed_sd")[list(MAPS)].round(4).to_string())
    inc = grid[~grid.complete]
    if len(inc):
        print("\nINCOMPLETE cells (a seed failed; mean over the valid seeds only):")
        print(inc[["selection", "mapping", "valid_seeds", "failed_seeds"]]
              .to_string(index=False))
    print("\nand their annualised mean active return, which the selection "
          "criterion does not\nsee: a respectable tracking error can sit on top "
          "of a large persistent shortfall")
    print(grid.pivot_table(index="selection", columns="mapping",
                           values="mar_verif")[list(MAPS)].round(4).to_string())

    paired = null[null.regime == "cv"]
    if len(paired):
        print("\nPENALTY REGIME ON THE RANDOM DRAWS (paired, same selections)")
        for kind in ("ridge", "dfp"):
            a = null[(null.regime == "fixed") & (null.mapping == kind) & null.valid
                     & (null.replicate < args.cv_replicates)].set_index("replicate").te_verif
            b = paired[(paired.mapping == kind) & paired.valid].set_index("replicate").te_verif
            j = a.index.intersection(b.index)
            if not len(j):
                continue
            diff = b[j] - a[j]
            print(f"  {kind:6s} fixed median {a[j].median():.4f}, cross-validated "
                  f"{b[j].median():.4f}, mean paired difference {diff.mean():+.4f}; "
                  f"cross-validating helps on {(diff < 0).mean():.0%} of {len(j)} draws")

    print("\nPRIMARY COMPARISONS (means over seeds)")
    g = lambda r, m: grid[(grid.selection == r) & (grid.mapping == m)].iloc[0]  # noqa: E731
    gd, gr = g(nr, "dfp"), g(nr, "ridge")
    print(f"  non-linear policy vs linear ridge, same selection: {nr}+dfp "
          f"{gd.te_verif:.4f} (sd {gd.te_verif_seed_sd:.4f}) vs {nr}+ridge "
          f"{gr.te_verif:.4f} (sd {gr.te_verif_seed_sd:.4f})")
    p = pos[pos.mapping == "ridge"].iloc[0]
    n_bad = int(null[(null.regime == "fixed") & ~null.valid].replicate.nunique())
    print(f"  value of the selection  : {nr}+ridge beats "
          f"{p.pct_of_random_beaten:.1%} of the {int(p.n_valid)} valid random draws\n"
          f"                            (random median {p.null_median:.4f}; {n_bad} "
          f"draws failed under at least one map and are excluded)")
    for a_, b_ in (("ae_hpw", "ae_ret_hpw"), ("pca_hpw", "pca_ret_hpw")):
        print(f"  levels vs returns ({a_.split('_')[0]}): "
              + ", ".join(f"{m} {g(a_, m).te_verif:.4f} -> {g(b_, m).te_verif:.4f}"
                          for m in ("ols", "ridge", "dfp")))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, kind in zip(axes, MAPS):
        d = ok[ok.mapping == kind].te_verif.values
        ax.hist(d, bins=30, color="#cfe3f7", edgecolor="#7fb3e0")
        ax.axvline(g(nr, kind).te_verif, color="crimson", lw=1.6, label=nr)
        for rule, c in (("ae_hpw", "navy"), ("ae_ret_hpw", "teal"),
                        ("pca_hpw", "darkorange"), ("marketcap", "seagreen"),
                        ("correlation", "purple")):
            if rule != nr:
                ax.axvline(g(rule, kind).te_verif, color=c, lw=1.0, ls="--", label=rule)
        ax.set_title(kind); ax.set_xlabel("verification tracking error")
    axes[0].set_ylabel("random selections"); axes[0].legend(fontsize=7)
    fig.suptitle(f"Rules: mean over {len(seeds)} seeds. Random draws: one seed each. "
                 f"Each panel has its own x-axis.", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / tag_csv("null").replace(".csv", ".png"), dpi=140)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
