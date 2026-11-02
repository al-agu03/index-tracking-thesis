"""
Phase A: the HPW smart-indexing example with model selection by cross-validation.

A sparse autoencoder (sparsity penalty optional; the main specification uses
beta = 0) ranks stocks by reconstruction error; a deep feature policy (DFP) maps
the 10 most communal plus the most individualistic stocks to the index level.
Portfolio size n and policy penalty lambda' are explored by anchored walk-forward
cross-validation on 2010-2013, with the ranking refitted inside every fold, and
the model is verified on 2014-2015. The verification window is not used for
fitting or selection, but it was examined during development.

The thesis uses n = 60, chosen by inspecting the validation frontier rather than
as its minimum, with lambda' chosen by cross-validation at that size. The
universe is the stocks with a complete observed history over 2010-2015, a
retrospective availability restriction.

    python phaseA_deep_frontier.py --ae-beta 0 --n 60 --max-size 60
"""

# Single-threaded linear algebra, set before NumPy loads: the summation order
# inside matrix products, and hence the results, depends on the thread count.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODE = "total"                                   # which cached panel to use

FULL_SPAN = ("2010-01-01", "2015-12-31")         # complete observed history required
CALIB     = ("2010-01-01", "2013-12-31")         # fit weights + cross-validate here
VERIF     = ("2014-01-01", "2015-12-31")         # not used to fit or select, but examined

N_COMMUNAL      = 10                             # "10 most communal + x individualistic"
PORTFOLIO_SIZES = [20, 30, 40, 50, 60, 80, 100]  # total stocks n (IBB used 25/45/65)
# Penalty grid in half-decade steps, applied at every size.
LAMBDAS         = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
N_FOLDS         = 4                              # walk-forward CV folds

# Length of the first fitting window, in trading days; the rest of the calibration
# period is divided into N_FOLDS equal validation blocks. Fixing it keeps the
# earliest fold from having fewer observations than the policy has weights
# (4n+17, i.e. 417 at n=100).
MIN_TRAIN_DAYS  = 500

BASE_SEED  = 0
# Each seed seeds both the autoencoder (so the selected stocks can change) and the
# policy; the configuration is chosen on the mean over seeds.
SEEDS      = [0, 1, 2, 3, 4]
# Reporting threshold: a fit "ends above its best" when its last training loss
# exceeds the best by this fraction. The best weights are kept in any case.
DETERIORATION_TOL = 0.01
AE_PARAMS  = dict(n_hidden=4, rho=0.01, beta=3.0, lam=1e-4, lr=1e-2, epochs=4000,
                  adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8)
DFP_PARAMS = dict(hidden=(4, 2), lr=5e-3, epochs=12000,
                  adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8)  # lam is swept

# Portfolio budget: the frontier runs to 100 stocks, but the configuration carried
# forward has at most this many. --max-size changes it.
MAX_SIZE = 50

NAIVE_N, NAIVE_LAM = 20, 1e-6                    # the working paper's configuration
TRADING_DAYS = 252

# Guard against reading the wrong panel (a full panel leaves about 350 stocks);
# set to 0 to disable.
MIN_EXPECTED_UNIVERSE = 100

# Paths are anchored to this file, so the script can be run from any directory.
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_FILE = SCRIPT_DIR / f"panel_{MODE}.csv"
# Set in main, so importing this module (as ablation_phaseA.py does) creates no folder.
OUTPUT_DIR = None


def output_dir(beta, n_seeds=None, n=None, lam=None, max_size=None):
    """
    Output folder named after everything that departs from the default run, so a
    quick check or an override can never overwrite the full specification.
    """
    name = "phaseA_outputs_v2"
    if beta != 3.0:
        name += f"_beta{beta:g}"
    if n_seeds is not None and n_seeds != len(SEEDS):
        name += f"_seeds{n_seeds}"
    if n is not None:
        name += f"_n{n}"
    if lam is not None:
        name += f"_lam{lam:.0e}"
    if max_size is not None and max_size != MAX_SIZE:
        name += f"_max{max_size}"
    return SCRIPT_DIR / name


class TrainingFailed(RuntimeError):
    """A network's loss or gradient became non-finite; `model` is the failed fit."""

    def __init__(self, msg, model):
        super().__init__(msg)
        self.model = model


def _all_finite(*arrays):
    return all(np.all(np.isfinite(a)) for a in arrays)


# ===========================================================================
# Models (same architecture as the replication script)
# ===========================================================================
class SparseAutoencoder:
    """N -> n_hidden -> N autoencoder with an optional KL sparsity penalty (beta)."""

    def __init__(self, n_features, n_hidden=4, rho=0.01, beta=3.0,
                 lam=1e-4, lr=1e-2, epochs=4000, seed=0,
                 adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
        self.n_features, self.n_hidden = n_features, n_hidden
        self.rho, self.beta, self.lam = rho, beta, lam
        self.lr, self.epochs, self.seed = lr, epochs, seed
        self.adam_beta1, self.adam_beta2 = adam_beta1, adam_beta2
        self.adam_epsilon = adam_epsilon
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, np.sqrt(1.0 / n_features), (n_hidden, n_features))
        self.b1 = np.zeros(n_hidden)
        self.W2 = r.normal(0, np.sqrt(1.0 / n_hidden), (n_features, n_hidden))
        self.b2 = np.zeros(n_features)

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    _KEYS = ("W1", "b1", "W2", "b2")

    def _loss_from(self, X, H, Xhat):
        """The objective the gradient below differentiates."""
        rho_hat = np.clip(H.mean(0), 1e-6, 1 - 1e-6)
        kl = np.sum(self.rho * np.log(self.rho / rho_hat)
                    + (1 - self.rho) * np.log((1 - self.rho) / (1 - rho_hat)))
        ridge = 0.5 * self.lam * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return float(np.mean((Xhat - X) ** 2) + self.beta * kl + ridge)

    def loss(self, X):
        H = self.activations(X)
        return self._loss_from(X, H, H @ self.W2.T + self.b2)

    def _loss_and_grad(self, X):
        """The objective at the current weights and its gradient."""
        T, N = X.shape
        H = self._sig(X @ self.W1.T + self.b1)
        Xhat = H @ self.W2.T + self.b2
        cur = self._loss_from(X, H, Xhat)
        dOut = (2.0 / (T * N)) * (Xhat - X)
        g = {"W2": dOut.T @ H + self.lam * self.W2, "b2": dOut.sum(0)}
        dH = dOut @ self.W2
        rho_hat = np.clip(H.mean(0), 1e-6, 1 - 1e-6)
        dKL = self.beta * (-self.rho / rho_hat + (1 - self.rho) / (1 - rho_hat))
        dZ1 = (dH + dKL / T) * H * (1 - H)
        g["W1"] = dZ1.T @ X + self.lam * self.W1
        g["b1"] = dZ1.sum(0)
        return cur, g

    def fit(self, X):
        """
        Full-batch Adam, keeping the weights at the lowest training loss.

        The checkpoint uses the training objective only, so no validation data is
        involved. loss_min, loss_last and epoch_min record what happened, so a fit
        whose last iterate ended above its best can be counted.

        A non-finite loss or gradient stops the fit and marks it failed
        (fail_epoch records when). A failed fit is invalid, whatever its earlier
        checkpoint looked like.
        """
        self.failed, self.fail_epoch = False, None
        m = {k: np.zeros_like(getattr(self, k)) for k in self._KEYS}
        v = {k: np.zeros_like(getattr(self, k)) for k in self._KEYS}
        b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_epsilon
        best, best_epoch, keep = np.inf, 0, None
        for e in range(1, self.epochs + 1):
            cur, g = self._loss_and_grad(X)              # loss before this update
            if not _all_finite(cur, *g.values()):
                self.failed, self.fail_epoch = True, e - 1
                break
            if cur < best:
                best, best_epoch = cur, e - 1
                keep = {k: getattr(self, k).copy() for k in self._KEYS}
            for k in g:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * (g[k] ** 2)
                mhat = m[k] / (1 - b1 ** e)
                vhat = v[k] / (1 - b2 ** e)
                setattr(self, k, getattr(self, k) - self.lr * mhat / (np.sqrt(vhat) + eps))
        if not self.failed:
            self.loss_last = self.loss(X)                # after the final update
            if not np.isfinite(self.loss_last):
                self.failed, self.fail_epoch = True, self.epochs
        if self.failed:
            self.loss_last = np.nan
            self.loss_min = best if keep is not None else np.nan
            self.epoch_min = best_epoch if keep is not None else np.nan
            return self
        if self.loss_last <= best:
            best, best_epoch = self.loss_last, self.epochs
        else:
            for k in self._KEYS:
                setattr(self, k, keep[k])
        self.loss_min, self.epoch_min = best, best_epoch
        return self

    def activations(self, X):
        return self._sig(X @ self.W1.T + self.b1)

    def per_feature_mse(self, X):
        Xhat = self.activations(X) @ self.W2.T + self.b2
        return ((Xhat - X) ** 2).mean(axis=0)


class DFP:
    """Deep feature policy: n_in -> 4 -> 2 -> 1, tanh hidden, linear output."""

    def __init__(self, n_in, hidden=(4, 2), lam=1e-6, lr=5e-3, epochs=12000, seed=0,
                 adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
        self.n_in, self.hidden = n_in, tuple(hidden)
        self.lam, self.lr, self.epochs, self.seed = lam, lr, epochs, seed
        self.adam_beta1, self.adam_beta2 = adam_beta1, adam_beta2
        self.adam_epsilon = adam_epsilon
        r = np.random.default_rng(seed)
        sizes = [n_in] + list(hidden) + [1]
        self.W = [r.normal(0, np.sqrt(1.0 / sizes[i]), (sizes[i + 1], sizes[i]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def _forward(self, X):
        a, acts, pre = X, [X], []
        for i in range(len(self.W) - 1):
            z = a @ self.W[i].T + self.b[i]; pre.append(z)
            a = np.tanh(z); acts.append(a)
        z = a @ self.W[-1].T + self.b[-1]; pre.append(z); acts.append(z)
        return acts, pre

    def _loss_from(self, pred, y):
        """The objective the gradient below differentiates."""
        ridge = 0.5 * self.lam * sum(np.sum(w ** 2) for w in self.W)
        return float(np.mean((pred - y) ** 2) + ridge)

    def loss(self, X, y):
        return self._loss_from(self._forward(X)[0][-1], y.reshape(-1, 1))

    def _loss_and_grad(self, X, y):
        """The objective at the current weights and its gradient; y is a column."""
        T = X.shape[0]
        acts, pre = self._forward(X)
        cur = self._loss_from(acts[-1], y)
        dz = (2.0 / T) * (acts[-1] - y)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = dz.T @ acts[i] + self.lam * self.W[i]
            gb[i] = dz.sum(0)
            if i > 0:
                dz = (dz @ self.W[i]) * (1 - np.tanh(pre[i - 1]) ** 2)
        return cur, gW, gb

    def fit(self, X, y):
        """
        Full-batch Adam, keeping the weights at the lowest training loss.

        As for the auto-encoder: training objective only, loss_min, loss_last and
        epoch_min are kept for the convergence report, and a non-finite loss or
        gradient marks the fit failed. A failed policy predicts NaN, so its paths
        are invalid wherever it is used.
        """
        self.failed, self.fail_epoch = False, None
        y = y.reshape(-1, 1)
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps, t = self.adam_beta1, self.adam_beta2, self.adam_epsilon, 0
        best, best_epoch, keep = np.inf, 0, None
        for ep in range(self.epochs):
            cur, gW, gb = self._loss_and_grad(X, y)      # loss before this update
            if not _all_finite(cur, *gW, *gb):
                self.failed, self.fail_epoch = True, ep
                break
            if cur < best:
                best, best_epoch = cur, ep
                keep = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            t += 1
            for i in range(len(self.W)):
                for cm, cv, param, grad in ((mW, vW, self.W, gW), (mb, vb, self.b, gb)):
                    cm[i] = b1 * cm[i] + (1 - b1) * grad[i]
                    cv[i] = b2 * cv[i] + (1 - b2) * (grad[i] ** 2)
                    mhat = cm[i] / (1 - b1 ** t); vhat = cv[i] / (1 - b2 ** t)
                    param[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)
        if not self.failed:
            self.loss_last = self.loss(X, y)             # after the final update
            if not np.isfinite(self.loss_last):
                self.failed, self.fail_epoch = True, self.epochs
        if self.failed:
            self.loss_last = np.nan
            self.loss_min = best if keep is not None else np.nan
            self.epoch_min = best_epoch if keep is not None else np.nan
            return self
        if self.loss_last <= best:
            best, best_epoch = self.loss_last, self.epochs
        else:
            self.W, self.b = keep
        self.loss_min, self.epoch_min = best, best_epoch
        return self

    def predict(self, X):
        if getattr(self, "failed", False):
            return np.full(X.shape[0], np.nan)
        return self._forward(X)[0][-1].ravel()


# ===========================================================================
# Helpers
# ===========================================================================
def zscore_cols(A):
    mu = A.mean(0); sd = A.std(0); sd[sd == 0] = 1.0
    return (A - mu) / sd


def rebase(df):
    """Rebase every column to 1.0 at the first observation of the window."""
    return df / df.iloc[0]


def level_stats(lvl_p, lvl_i):
    """Annualised tracking error of daily active returns, and level RMSE."""
    lvl_p = np.asarray(lvl_p, dtype=float)
    lvl_i = np.asarray(lvl_i, dtype=float)
    rmse = float(np.sqrt(np.mean((lvl_p - lvl_i) ** 2)))
    if np.any(lvl_p <= 0) or np.any(lvl_i <= 0):
        return float("nan"), rmse
    rp = np.diff(lvl_p) / lvl_p[:-1]
    ri = np.diff(lvl_i) / lvl_i[:-1]
    te = float(np.std(rp - ri, ddof=1) * np.sqrt(TRADING_DAYS))
    return te, rmse


def active_return(lvl_p, lvl_i):
    """Annualised mean daily active return, the first moment of the tracking gap.

    Reported next to the tracking error (the second moment), never used to
    select. The level RMSE mixes the two and, over long windows, is dominated by
    this mean term.
    """
    lvl_p = np.asarray(lvl_p, dtype=float)
    lvl_i = np.asarray(lvl_i, dtype=float)
    if np.any(lvl_p <= 0) or np.any(lvl_i <= 0):
        return float("nan")
    rp = np.diff(lvl_p) / lvl_p[:-1]
    ri = np.diff(lvl_i) / lvl_i[:-1]
    return float(np.mean(rp - ri) * TRADING_DAYS)


def full_coverage(df, cols, a, b):
    span = df.loc[a:b]
    return [c for c in cols if c in span.columns and span[c].notna().all()]


def anchored_folds(n_obs, n_folds, min_train):
    """
    Anchored walk-forward cross-validation.

    The first `min_train` observations are reserved for fitting and are never
    scored; what remains is divided into `n_folds` equal contiguous validation
    blocks. Fold k is fitted on everything up to the start of its own block, so
    training data always precede validation data.
    """
    if n_obs - min_train < n_folds * 20:
        raise SystemExit(
            f"MIN_TRAIN_DAYS={min_train} leaves only {n_obs - min_train} days for "
            f"{n_folds} validation blocks; lower it or use fewer folds."
        )
    edges = np.linspace(min_train, n_obs, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        tr, va = (0, edges[k]), (edges[k], edges[k + 1])
        if va[1] - va[0] >= 20:
            folds.append((tr, va))
    return folds


def dfp_param_count(n_in, hidden):
    """Weights plus biases of the policy, used only for the sizing diagnostic."""
    sizes = [n_in] + list(hidden) + [1]
    return sum(sizes[i + 1] * sizes[i] + sizes[i + 1] for i in range(len(sizes) - 1))


def rank_universe(frame, seed=BASE_SEED, params=None):
    """Fit the auto-encoder on `frame` and rank its columns by reconstruction error.

    The standardisation statistics come from `frame` alone, so when this is called
    on a fold's training blocks nothing about the validation block enters either
    the transform or the ranking. `params` overrides AE_PARAMS.
    """
    X = zscore_cols(frame.values)
    ae = SparseAutoencoder(n_features=X.shape[1], seed=seed,
                           **(params or AE_PARAMS)).fit(X)
    if ae.failed:
        raise TrainingFailed(f"auto-encoder became non-finite at epoch "
                             f"{ae.fail_epoch} (seed {seed})", ae)
    rank = pd.Series(ae.per_feature_mse(X), index=frame.columns,
                     name="reconstruction_mse").sort_values()
    return rank, ae, X


def portfolio_from(rank, n):
    """10 most communal + (n-10) most individualistic, as in the IBB example."""
    if not N_COMMUNAL <= n <= len(rank):
        raise ValueError(f"portfolio size {n} must lie between {N_COMMUNAL} and "
                         f"the {len(rank)} ranked stocks")
    communal = rank.index[:N_COMMUNAL].tolist()
    individualistic = [t for t in rank.index[::-1] if t not in communal]
    return communal + individualistic[: n - N_COMMUNAL]


# ===========================================================================
# Main
# ===========================================================================
def parse_overrides():
    """Command-line options; with none given the default configuration runs.

    --n and --lam fix the portfolio size and penalty instead of selecting them.
    --ae-beta sets the autoencoder's sparsity weight and redirects output.
    --seeds runs on the first k seeds, for quick checks only.
    """
    import argparse
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--n", type=int, default=None,
                    help="portfolio size to carry to verification, overriding "
                         "the minimum of the validation grid")
    ap.add_argument("--lam", type=float, default=None,
                    help="policy penalty to carry to verification")
    ap.add_argument("--ae-beta", type=float, default=AE_PARAMS["beta"],
                    help="auto-encoder sparsity weight; 3 is primary, 0 the "
                         "robustness run")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="use the first k prespecified seeds (default all five)")
    ap.add_argument("--max-size", type=int, default=MAX_SIZE,
                    help=f"largest portfolio size that may be selected "
                         f"(default {MAX_SIZE}); the grid still runs to "
                         f"{max(PORTFOLIO_SIZES)}")
    a = ap.parse_args()
    if a.max_size < N_COMMUNAL:
        ap.error(f"--max-size must be at least {N_COMMUNAL}")
    if a.n is not None and a.n > a.max_size:
        ap.error(f"--n {a.n} exceeds --max-size {a.max_size}; raise --max-size")
    if not 1 <= a.seeds <= len(SEEDS):
        ap.error(f"--seeds must lie between 1 and {len(SEEDS)}")
    for name, v in (("--lam", a.lam), ("--ae-beta", a.ae_beta)):
        if v is not None and not (np.isfinite(v) and v >= 0):
            ap.error(f"{name} must be finite and non-negative, got {v}")
    return a


def conv_entry(kind, loss_min, loss_last, epoch_min, epochs, failed, fail_epoch, **ctx):
    """One line of the convergence log."""
    ok = (not failed) and np.isfinite(loss_min) and np.isfinite(loss_last)
    excess = (loss_last - loss_min) / max(abs(loss_min), 1e-12) if ok else np.nan
    return dict(network=kind, **ctx, failed=bool(failed), fail_epoch=fail_epoch,
                loss_min=loss_min, loss_last=loss_last, epoch_min=epoch_min,
                epochs=epochs, last_above_min=excess,
                ended_above_best=bool(ok and excess > DETERIORATION_TOL))


def convergence_row(model, kind, **ctx):
    """Convergence log line for a fitted network."""
    return conv_entry(kind, model.loss_min, model.loss_last, model.epoch_min,
                      model.epochs, model.failed, model.fail_epoch, **ctx)


def select_cell(grid, max_size):
    """
    The configuration carried forward, and the unconstrained minimum for the record.

    Both are the minimum mean validation TE among eligible cells; the first is
    restricted to portfolios of at most max_size stocks.
    """
    ok = grid[grid.eligible]
    if ok.empty:
        raise SystemExit("no grid cell gave a valid path in every fold and seed; "
                         "nothing can be selected")
    within = ok[ok.n_stocks <= max_size]
    if within.empty:
        raise SystemExit(f"no eligible cell with at most {max_size} stocks")
    return within.loc[within.te_val.idxmin()], ok.loc[ok.te_val.idxmin()]


def summarise_cell(n, lam, cell):
    """
    One grid cell from its fold-by-seed fits.

    Eligible only if every fit gave a valid (positive) path. An ineligible cell
    gets no validation score at all, so it cannot be selected; averaging over the
    surviving fits instead would reward configurations that sometimes break.
    The seed spread is the standard deviation of the per-seed fold means.
    """
    c = pd.DataFrame(cell)
    n_valid, n_fits = int(c.valid.sum()), len(c)
    eligible = n_valid == n_fits
    per_seed = c.groupby("seed").te_val.mean()
    return dict(n_stocks=n, lam=lam,
                te_in=c.te_in.mean(), rmse_in=c.rmse_in.mean(),
                te_val=c.te_val.mean() if eligible else np.nan,
                rmse_val=c.rmse_val.mean() if eligible else np.nan,
                te_val_seed_sd=(per_seed.std(ddof=1)
                                if eligible and len(per_seed) > 1 else np.nan),
                valid_fits=n_valid, fits=n_fits, eligible=eligible)


def checked_windows(stocks, idx):
    """
    The calibration and verification windows, refused if they are not usable.

    Dates must be unique and chronologically sorted. Windows must be ordered and
    non-overlapping, non-empty, with a complete
    benchmark and finite, positive prices. Days are never dropped silently: the
    full-coverage filter should leave no gaps, so a gap is an error.
    """
    for name, ix in (("price panel", stocks.index), ("benchmark", idx.index)):
        if not ix.is_unique:
            raise SystemExit(f"{name} has {int(ix.duplicated().sum())} duplicated dates")
        if not ix.is_monotonic_increasing:
            raise SystemExit(f"{name} dates are not in chronological order")
    c0, c1, v0, v1 = (pd.Timestamp(d) for d in (*CALIB, *VERIF))
    if not c0 < c1 < v0 < v1:
        raise SystemExit(f"windows must be ordered and non-overlapping: "
                         f"calibration {CALIB}, verification {VERIF}")
    out = []
    for name, (a, b) in (("calibration", CALIB), ("verification", VERIF)):
        s = stocks.loc[a:b]
        i = idx.loc[a:b].reindex(s.index)
        problems = []
        if s.empty:
            problems.append("no days")
        if s.isna().any().any():
            problems.append(f"{int(s.isna().any(axis=1).sum())} days with a missing price")
        if i.isna().any():
            problems.append(f"benchmark missing on {int(i.isna().sum())} days")
        vals = np.concatenate([s.values.ravel(), i.values])
        if not np.all(np.isfinite(vals[~np.isnan(vals)])):
            problems.append("non-finite values")
        if np.any(vals[~np.isnan(vals)] <= 0):
            problems.append("non-positive prices")
        if problems:
            raise SystemExit(f"{name} window unusable: " + "; ".join(problems))
        out += [s, i]
    return out


def resume_state(out_dir, partial, fingerprint, expected):
    """
    Completed grid cells from an interrupted run of the same configuration.

    `expected` is the set of (seed, fold) pairs a complete cell must contain. A
    cell counts as complete only if it has exactly those pairs, each once. Rows of
    incomplete cells (a run stopped while writing) are removed by rewriting the
    file atomically before anything is appended, so they cannot accumulate.

    The configuration, including hashes of the price panel and of this script, is
    stored beside the partial results, and a partial file is refused if the stored
    configuration differs.
    """
    import json
    import os
    cfg = out_dir / "phaseA_run_config.json"
    text = json.dumps(fingerprint, sort_keys=True, default=str)
    if cfg.exists() and cfg.read_text() != text and partial.exists():
        raise SystemExit(f"{out_dir} holds a partial run with a different "
                         f"configuration; remove {partial.name} or use another folder")
    cfg.write_text(text)
    done = {}
    if partial.exists():
        # round_trip parsing so resumed cells are bit-identical to recomputed ones;
        # the default parser can change the last bit and so, in a near-tie, the choice
        prev = pd.read_csv(partial, float_precision="round_trip")
        keep_rows = []
        for (n, lam), g in prev.groupby(["n_stocks", "lam"], sort=False):
            pairs = list(zip(g.seed.astype(int), g.fold.astype(int)))
            if len(pairs) == len(expected) and set(pairs) == set(expected):
                done[(int(n), float(lam))] = g.to_dict("records")
                keep_rows.append(g)
        dropped = len(prev) - sum(len(g) for g in keep_rows)
        tmp = partial.with_name(partial.name + ".tmp")
        if keep_rows:
            pd.concat(keep_rows).to_csv(tmp, index=False)
            os.replace(tmp, partial)
        else:
            partial.unlink()
        print(f"\n    resuming: {len(done)} grid cells already complete in "
              f"{partial.name}; {dropped} rows of incomplete cells discarded")
    return done


def main():
    global OUTPUT_DIR
    # Parsed first, so a mistyped argument fails in a second rather than after
    # the grid has run.
    args = parse_overrides()
    seeds = SEEDS[: args.seeds]
    ae_params = dict(AE_PARAMS, beta=args.ae_beta)
    OUTPUT_DIR = output_dir(args.ae_beta, args.seeds, args.n, args.lam, args.max_size)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PHASE A (v2) -- efficient deep frontier on the S&P 500 smart-indexing example")
    print("=" * 78)
    print(f"Script folder : {SCRIPT_DIR}")
    print(f"Output folder : {OUTPUT_DIR}")
    print(f"Price panel   : {CACHE_FILE}")
    print(f"Seeds         : {seeds}   sparsity weight beta = {args.ae_beta:g}")
    if not CACHE_FILE.exists():
        raise SystemExit(
            f"\nERROR: price panel not found.\n"
            f"This script never downloads; it expects panel_{MODE}.csv to sit next\n"
            f"to this file. Copy the cached panel produced by the replication script\n"
            f"into {SCRIPT_DIR} and run again."
        )
    print("                (found)\n")

    # ---------------- data ------------------------------------------------
    px = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    idx = px["SP500"]
    stocks = px.drop(columns=["SP500"])
    keep = full_coverage(stocks, stocks.columns, *FULL_SPAN)
    stocks = stocks[keep]
    print(f"Universe with a complete {FULL_SPAN[0][:4]}-{FULL_SPAN[1][:4]} series: "
          f"{len(keep)} stocks")
    print(f"    first names: {', '.join(keep[:8])}"
          f"{' ...' if len(keep) > 8 else ''}")
    if len(keep) < MIN_EXPECTED_UNIVERSE:
        raise SystemExit(
            f"\nERROR: only {len(keep)} stocks survive the full-span filter, but at least\n"
            f"{MIN_EXPECTED_UNIVERSE} were expected for an S&P 500 panel. The file\n"
            f"{CACHE_FILE}\nis almost certainly the wrong one (a truncated or synthetic test\n"
            f"panel). Check the ticker names printed above: they should be real symbols such\n"
            f"as AAPL or MSFT. Replace it with the panel produced by the replication script,\n"
            f"or set MIN_EXPECTED_UNIVERSE = 0 to run on a small universe on purpose."
        )

    s_cal, i_cal, s_ver, i_ver = checked_windows(stocks, idx)
    print(f"Calibration {CALIB[0]}..{CALIB[1]}: {len(s_cal)} days")
    print(f"Verification {VERIF[0]}..{VERIF[1]}: {len(s_ver)} days  "
          f"(not used to fit or select; examined during development)")

    sizes = [n for n in PORTFOLIO_SIZES if n <= s_cal.shape[1]]
    if not sizes:
        raise SystemExit(f"Universe too small ({s_cal.shape[1]} stocks) for any size")
    if len(sizes) < len(PORTFOLIO_SIZES):
        print(f"    note: universe has {s_cal.shape[1]} stocks, dropping sizes "
              f"{[n for n in PORTFOLIO_SIZES if n not in sizes]}")

    # Checked before the grid runs, so a mistyped size fails in seconds, not hours.
    if args.n is not None and not N_COMMUNAL <= args.n <= s_cal.shape[1]:
        raise SystemExit(f"--n must lie between {N_COMMUNAL} and {s_cal.shape[1]}, "
                         f"got {args.n}")

    folds = anchored_folds(len(s_cal), N_FOLDS, MIN_TRAIN_DAYS)

    # How much data does the earliest fit have, relative to what it must estimate?
    first_train = folds[0][0][1] - folds[0][0][0]
    print(f"\nFold sizing: first fit on {first_train} days, "
          f"{len(folds)} validation blocks of about "
          f"{folds[0][1][1] - folds[0][1][0]} days")
    for n in sizes:
        p = dfp_param_count(n, DFP_PARAMS["hidden"])
        flag = "  <- thin" if first_train < 2 * p else ""
        print(f"    n={n:4d}  parameters={p:5d}  observations per parameter="
              f"{first_train / p:5.1f}{flag}")

    conv = []          # convergence log, one row per network fit

    # ---------------- step I, inside each fold, for each seed -------------
    # The auto-encoder is refitted on every fold's training blocks alone, so the
    # ranking a fold is scored on was fitted without its validation block (within
    # the fixed universe, whose eligibility uses the full 2010-2015 span).
    # The seed is part of the procedure, so each seed gets its own ranking.
    print(f"\n[I] Auto-encoding within each of the {len(folds)} folds, "
          f"for {len(seeds)} seeds (training blocks only) ...")
    fold_ranks = {}
    for s in seeds:
        for k, ((t0, t1), (v0, v1)) in enumerate(folds):
            try:
                rk, ae_k, _ = rank_universe(s_cal.iloc[t0:t1], seed=s,
                                            params=ae_params)
            except TrainingFailed as err:
                # No ranking for this fold and seed: every policy fit that needed
                # it is recorded as invalid below.
                rk, ae_k = None, err.model
                print(f"    seed {s}, fold {k + 1}: {err}")
            fold_ranks[(s, k)] = rk
            conv.append(convergence_row(ae_k, "autoencoder", stage="fold",
                                        fold=k + 1, seed=s, n_stocks=np.nan,
                                        lam=ae_params["lam"]))
        print(f"    seed {s}: fold communal sets "
              + " | ".join("FAILED" if fold_ranks[(s, k)] is None
                           else ", ".join(fold_ranks[(s, k)].index[:3]) + " ..."
                           for k in range(len(folds))), flush=True)

    # ---------------- steps II-III: calibrate + validate ------------------
    print(f"\n[II-III] Calibrating and cross-validating "
          f"({len(sizes)}x{len(LAMBDAS)} grid, {len(folds)} walk-forward folds, "
          f"{len(seeds)} seeds) ...")

    # Each completed cell is appended to a partial file, with a fingerprint of the
    # configuration beside it, so an interrupted run picks up where it stopped and
    # a run with different settings cannot mix its results into it.
    partial = OUTPUT_DIR / f"phaseA_grid_long_{MODE}.partial.csv"
    import hashlib
    done = resume_state(OUTPUT_DIR, partial, dict(
        version="v2", seeds=seeds, ae_params=ae_params, dfp_params=DFP_PARAMS,
        sizes=sizes, lambdas=LAMBDAS, folds=folds, calib=CALIB, verif=VERIF,
        universe=keep, calib_days=len(s_cal), verif_days=len(s_ver),
        panel_sha256=hashlib.sha256(CACHE_FILE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        expected={(s, k + 1) for s in seeds for k in range(len(folds))})

    long_rows, rows = [], []
    for n in sizes:
        for lam in LAMBDAS:
            if (int(n), float(lam)) in done:
                cell = done[(int(n), float(lam))]
                print(f"    n={n:3d}  lambda'={lam:.0e}   (resumed from the partial file)")
            else:
                cell = []
                for s in seeds:
                    for k, ((t0, t1), (v0, v1)) in enumerate(folds):
                        base = dict(n_stocks=n, lam=lam, seed=s, fold=k + 1)
                        if fold_ranks[(s, k)] is None:
                            cell.append(dict(base, te_in=np.nan, rmse_in=np.nan,
                                             te_val=np.nan, rmse_val=np.nan,
                                             valid=False, ae_failed=True, failed=False,
                                             fail_epoch=np.nan, loss_min=np.nan,
                                             loss_last=np.nan, epoch_min=np.nan))
                            continue
                        feats = portfolio_from(fold_ranks[(s, k)], n)
                        S_tr = rebase(s_cal.iloc[t0:t1][feats]).values
                        I_tr = rebase(i_cal.iloc[t0:t1]).values
                        S_va = rebase(s_cal.iloc[v0:v1][feats]).values
                        I_va = rebase(i_cal.iloc[v0:v1]).values
                        d = DFP(n_in=len(feats), lam=lam, seed=s,
                                **DFP_PARAMS).fit(S_tr, I_tr)
                        te_in, rmse_in = level_stats(d.predict(S_tr), I_tr)
                        te_va, rmse_va = level_stats(d.predict(S_va), I_va)
                        valid = bool(not d.failed and np.isfinite(te_in)
                                     and np.isfinite(te_va))
                        cell.append(dict(base, te_in=te_in, rmse_in=rmse_in,
                                         te_val=te_va, rmse_val=rmse_va, valid=valid,
                                         ae_failed=False, failed=d.failed,
                                         fail_epoch=d.fail_epoch, loss_min=d.loss_min,
                                         loss_last=d.loss_last, epoch_min=d.epoch_min))
                pd.DataFrame(cell).to_csv(partial, mode="a", index=False,
                                          header=not partial.exists())
            long_rows += cell
            for c in cell:
                if not c["ae_failed"]:
                    conv.append(conv_entry("dfp", c["loss_min"], c["loss_last"],
                                           c["epoch_min"], DFP_PARAMS["epochs"],
                                           c["failed"], c["fail_epoch"], stage="fold",
                                           fold=c["fold"], seed=c["seed"],
                                           n_stocks=n, lam=lam))
            rows.append(summarise_cell(n, lam, cell))
            r = rows[-1]
            n_valid, n_fits, eligible = r["valid_fits"], r["fits"], r["eligible"]
            shown = (f"TE val={r['te_val']:.4f} (seed sd {r['te_val_seed_sd']:.4f})"
                     if eligible else f"INELIGIBLE: {n_fits - n_valid} of {n_fits} fits "
                                      f"failed or gave a non-positive path")
            print(f"    n={n:3d}  lambda'={lam:.0e}   {shown}", flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(OUTPUT_DIR / f"phaseA_grid_{MODE}.csv", index=False)
    pd.DataFrame(long_rows).to_csv(OUTPUT_DIR / f"phaseA_grid_long_{MODE}.csv",
                                   index=False)
    partial.unlink(missing_ok=True)       # complete; the full file supersedes it

    ok = grid[grid.eligible]
    n_out = int((~grid.eligible).sum())
    best, unc = select_cell(grid, args.max_size)
    best_n, best_lam = int(best["n_stocks"]), float(best["lam"])
    unc_line = (f"n={int(unc['n_stocks'])}, lambda'={float(unc['lam']):.0e} "
                f"(TE_val={unc['te_val']:.4f}, sd {unc['te_val_seed_sd']:.4f})")
    print(f"\n    {n_out} of {len(grid)} cells ineligible (a fit failed or gave a "
          f"non-positive path)")
    print(f"    Unconstrained minimum over all sizes: {unc_line}")
    print(f"    Selected with at most {args.max_size} stocks: "
          f"n={best_n}, lambda'={best_lam:.0e} (TE_val={best['te_val']:.4f})")

    sel_score, sel_sd = float(best["te_val"]), float(best["te_val_seed_sd"])
    sel_note = (f"minimum of the mean validation TE among eligible cells with at "
                f"most {args.max_size} stocks")
    if args.n is not None or args.lam is not None:
        n_req = int(args.n) if args.n is not None else best_n
        if not N_COMMUNAL <= n_req <= s_cal.shape[1]:
            raise SystemExit(f"--n must lie between {N_COMMUNAL} and "
                             f"{s_cal.shape[1]}, got {n_req}")
        if args.lam is not None:
            lam_req = float(args.lam)
        else:
            # The penalty chosen for another size says nothing about this one.
            at_n = ok[ok.n_stocks == n_req]
            if at_n.empty:
                raise SystemExit(f"--n {n_req} has no eligible penalty in the grid; "
                                 f"pass --lam as well")
            lam_req = float(at_n.loc[at_n.te_val.idxmin(), "lam"])
        row = grid[(grid.n_stocks == n_req) & np.isclose(grid.lam, lam_req)]
        if len(row) and not bool(row.eligible.iloc[0]):
            raise SystemExit(f"n={n_req}, lambda'={lam_req:.0e} is ineligible: "
                             f"{int(row.fits.iloc[0] - row.valid_fits.iloc[0])} of "
                             f"{int(row.fits.iloc[0])} fits failed or gave a "
                             f"non-positive path")
        best_n, best_lam = n_req, lam_req
        if len(row):
            sel_score, sel_sd = float(row.te_val.iloc[0]), float(row.te_val_seed_sd.iloc[0])
            sel_note = "set on the command line; validation score of that cell"
        else:
            sel_score = sel_sd = np.nan
            sel_note = "set on the command line; not evaluated in the validation grid"
        print(f"    OVERRIDDEN from the command line: n={best_n}, lambda'={best_lam:.0e} "
              f"({sel_note}" + (f", TE_val={sel_score:.4f})" if len(row) else ")"))

    # ---------------- final selection on the full calibration window ------
    # One fresh instance of the validated procedure per seed, fitted on all of
    # calibration. Seed 0 supplies the diagnostics and the figure; every seed is
    # verified and reported.
    print("\n[I-II] Refitting the selected procedure on the full calibration window, "
          "per seed ...")
    final = {}
    for s in seeds:
        try:
            rank_s, ae_s, X_cal = rank_universe(s_cal, seed=s, params=ae_params)
            final[s] = (rank_s, ae_s, X_cal)
        except TrainingFailed as err:
            ae_s, final[s] = err.model, None
            print(f"    seed {s}: {err}; this seed fails verification")
        conv.append(convergence_row(ae_s, "autoencoder", stage="final", fold=np.nan,
                                    seed=s, n_stocks=np.nan, lam=ae_params["lam"]))
    ok_seeds = [s for s in seeds if final[s] is not None]
    if not ok_seeds:
        raise SystemExit("the auto-encoder failed on the full calibration window for "
                         "every seed")
    s_diag = ok_seeds[0]          # supplies the diagnostics: seed 0 unless it failed
    rank, ae, X_cal = final[s_diag]
    rho_hat = ae.activations(X_cal).mean(0)
    print(f"    seed {s_diag} achieved activations: "
          f"{', '.join(f'{v:.4f}' for v in rho_hat)}  "
          f"(mean {rho_hat.mean():.4f}, target rho = {ae.rho})")
    communal = rank.index[:N_COMMUNAL].tolist()
    print(f"    seed {s_diag} 10 most communal:", communal)
    shared = ["failed" if final[s] is None
              else len(set(final[s][0].index[:N_COMMUNAL]) & set(communal)) for s in seeds]
    print(f"    communal names shared with seed {s_diag}, by seed: "
          + ", ".join(f"{s}:{c}" for s, c in zip(seeds, shared)))

    fold_rows = []
    for k, ((t0, t1), (v0, v1)) in enumerate(folds, start=1):
        fr = fold_ranks[(s_diag, k - 1)]
        top = [] if fr is None else fr.index[:N_COMMUNAL].tolist()
        fold_rows.append({"fold": k, "train_days": t1 - t0, "valid_days": v1 - v0,
                          "overlap_with_final": (np.nan if fr is None
                                                 else len(set(top) & set(communal))),
                          "communal_stocks": "auto-encoder failed" if fr is None
                          else ", ".join(top)})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUTPUT_DIR / f"phaseA_fold_selection_{MODE}.csv", index=False)
    mean_overlap = float(fold_df["overlap_with_final"].mean())
    print(f"    mean overlap of seed-{s_diag} fold selections with the final one: "
          f"{mean_overlap:.2f} / {N_COMMUNAL}")

    rank_df = rank.rename_axis("ticker").reset_index()
    rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))
    largest = set(portfolio_from(rank, max(sizes)))
    rank_df["selection_group"] = "unused"
    rank_df.loc[rank_df["ticker"].isin(largest), "selection_group"] = "in_largest_portfolio"
    rank_df.loc[rank_df["ticker"].isin(communal), "selection_group"] = "communal_top10"
    rank_df.to_csv(OUTPUT_DIR / f"phaseA_ranking_{MODE}.csv", index=False)

    # ---------------- step IV: frontiers ----------------------------------
    print("\n[IV] Drawing the efficient deep frontiers ...")
    per_n = ok.loc[ok.groupby("n_stocks")["te_val"].idxmin()].sort_values("n_stocks")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(per_n["n_stocks"], per_n["te_val"], yerr=per_n["te_val_seed_sd"],
                fmt="o-", color="C0", capsize=3)
    pad = 0.05 * (max(sizes) - min(sizes)) if len(sizes) > 1 else 1
    ax.set_xlim(min(sizes) - pad, max(sizes) + pad)
    # When n is set on the command line the figure marks it with a dotted line.
    by_hand = args.n is not None or args.lam is not None
    ax.axvline(best_n, color="0.5", lw=1, ls=":")
    if not by_hand:
        if args.max_size < max(sizes):
            ax.axvspan(args.max_size, max(sizes) + pad, color="0.9", zorder=0,
                       label=f"above the {args.max_size}-stock budget")
            ax.legend(fontsize=8, loc="best")
        _right = best_n > (min(sizes) + max(sizes)) / 2
        ax.annotate(f"selected: n={best_n}", xy=(best_n, per_n["te_val"].max()),
                    xytext=(-5 if _right else 5, -2), textcoords="offset points",
                    fontsize=8, color="0.4", va="top",
                    ha="right" if _right else "left")
    ax.set_xlabel("stocks in the deep portfolio")
    ax.set_ylabel("cross-validated tracking error (mean over seeds)")
    ax.set_title(f"Efficient deep frontier over portfolio size; "
                 f"bars: sd across {len(seeds)} seeds", fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"frontier_stocks_{MODE}.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in sizes:
        sub = grid[grid["n_stocks"] == n].sort_values("lam")
        ax.plot(sub["lam"], sub["te_val"], "o-", label=f"n={n}")
    ax.set_xscale("log")
    ax.set_xlabel(r"DFP ridge penalty $\lambda'$")
    ax.set_ylabel("validation tracking error (mean over seeds)")
    ax.set_title(f"Validation error against regularisation; "
                 f"gaps are ineligible cells", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"frontier_lambda_{MODE}.png", dpi=140)
    plt.close(fig)

    # ---------------- verification, every seed ----------------------------
    print("\n[IV] Verifying on 2014-2015, every seed ...")

    def failed_row(label, s, n, lam):
        return (dict(model=label, seed=s, n_stocks=n, lam=lam,
                     **{m_: np.nan for m_ in ("te_calib", "rmse_calib", "mar_calib",
                                              "te_verif", "rmse_verif", "mar_verif")}),
                None)

    # Labels. A configuration set on the command line is not called "selected".
    overridden = args.n is not None or args.lam is not None
    sel_label = (f"DFP, n={best_n}, lambda'={best_lam:.0e}" if overridden
                 else "DFP selected by frontier")
    hpw_label = f"HPW configuration (n={NAIVE_N}, lambda'={NAIVE_LAM:.0e}, beta=3)"
    basket_label = "Equal-weight basket (10)"

    # The HPW configuration uses HPW's encoder (beta=3). When this run uses another
    # beta, that ranking is fitted here, once per seed, on the full calibration.
    if ae_params == AE_PARAMS:
        hpw_rank = {s: None if final[s] is None else final[s][0] for s in seeds}
    else:
        hpw_rank = {}
        for s in seeds:
            try:
                hpw_rank[s], ae_h, _ = rank_universe(s_cal, seed=s, params=AE_PARAMS)
            except TrainingFailed as err:
                hpw_rank[s], ae_h = None, err.model
                print(f"    seed {s}: HPW encoder {err}")
            conv.append(convergence_row(ae_h, "autoencoder", stage="final (HPW beta=3)",
                                        fold=np.nan, seed=s, n_stocks=np.nan,
                                        lam=AE_PARAMS["lam"]))

    def refit_and_verify(n, lam, label, s, rank_s):
        if rank_s is None:
            return failed_row(label, s, n, lam)
        feats = portfolio_from(rank_s, n)
        S_cal, I_cal = rebase(s_cal[feats]).values, rebase(i_cal).values
        S_ver, I_ver = rebase(s_ver[feats]).values, rebase(i_ver).values
        dfp = DFP(n_in=len(feats), lam=lam, seed=s, **DFP_PARAMS).fit(S_cal, I_cal)
        conv.append(convergence_row(dfp, "dfp", stage=f"final: {label}", fold=np.nan,
                                    seed=s, n_stocks=n, lam=lam))
        pc, pv = dfp.predict(S_cal), dfp.predict(S_ver)
        te_c, rmse_c = level_stats(pc, I_cal)
        te_v, rmse_v = level_stats(pv, I_ver)
        return (dict(model=label, seed=s, n_stocks=n, lam=lam,
                     te_calib=te_c, rmse_calib=rmse_c, mar_calib=active_return(pc, I_cal),
                     te_verif=te_v, rmse_verif=rmse_v, mar_verif=active_return(pv, I_ver)),
                (pc, pv))

    per_seed_rows, paths, paths_cal = [], {}, {}
    for s in seeds:
        for label, n, lam, rk in (
                (sel_label, best_n, best_lam, None if final[s] is None else final[s][0]),
                (hpw_label, NAIVE_N, NAIVE_LAM, hpw_rank[s])):
            r, pp = refit_and_verify(n, lam, label, s, rk)
            per_seed_rows.append(r)
            paths_cal[(label, s)], paths[(label, s)] = pp if pp else (None, None)
        if final[s] is None:
            r, _ = failed_row(basket_label, s, N_COMMUNAL, np.nan)
            per_seed_rows.append(r)
            print(f"    seed {s}: FAILED (no ranking)", flush=True)
            continue
        com_s = final[s][0].index[:N_COMMUNAL].tolist()
        bc = rebase(s_cal[com_s]).mean(axis=1).values
        bv = rebase(s_ver[com_s]).mean(axis=1).values
        te_bc, rmse_bc = level_stats(bc, rebase(i_cal).values)
        te_bv, rmse_bv = level_stats(bv, rebase(i_ver).values)
        per_seed_rows.append(dict(model=basket_label, seed=s,
                                  n_stocks=N_COMMUNAL, lam=np.nan,
                                  te_calib=te_bc, rmse_calib=rmse_bc,
                                  mar_calib=active_return(bc, rebase(i_cal).values),
                                  te_verif=te_bv, rmse_verif=rmse_bv,
                                  mar_verif=active_return(bv, rebase(i_ver).values)))
        paths[(basket_label, s)] = bv
        paths_cal[(basket_label, s)] = bc
        print(f"    seed {s} done", flush=True)

    per_seed = pd.DataFrame(per_seed_rows)
    # A seed is valid only if both its calibration and verification paths are.
    per_seed["valid"] = np.isfinite(per_seed.te_calib) & np.isfinite(per_seed.te_verif)
    per_seed.to_csv(OUTPUT_DIR / f"phaseA_verification_seeds_{MODE}.csv", index=False)
    metrics = ["te_calib", "rmse_calib", "mar_calib", "te_verif", "rmse_verif", "mar_verif"]
    order = [sel_label, hpw_label, basket_label]
    # Means and spreads over the valid seeds only, with the failures counted and
    # the result marked incomplete, so a failed seed is visible wherever the
    # number is quoted rather than silently dropped from it.
    valid = per_seed[per_seed.valid]
    va = valid.groupby("model", sort=False)
    summary = (per_seed.groupby("model", sort=False)[["n_stocks", "lam"]].first()
               .join(va[metrics].mean())
               .join(va[metrics].std(ddof=1).add_suffix("_seed_sd")))
    counts = per_seed.groupby("model", sort=False).valid.agg(["sum", "size"])
    summary["valid_seeds"] = counts["sum"].astype(int)
    summary["failed_seeds"] = (counts["size"] - counts["sum"]).astype(int)
    summary["complete"] = summary["failed_seeds"] == 0
    summary = summary.reindex(order).reset_index()
    summary.to_csv(OUTPUT_DIR / f"phaseA_summary_{MODE}.csv", index=False)

    def seeds_note(r):
        return ("" if r["complete"] else
                f" ({int(r['valid_seeds'])} of {len(seeds)} seeds; INCOMPLETE)")

    for _, r in summary.iterrows():
        print(f"    {r['model']:28s} verif TE={r['te_verif']:.4f} "
              f"(sd {r['te_verif_seed_sd']:.4f})  RMSE={r['rmse_verif']:.4f}  "
              f"active={r['mar_verif']:+.4f}{seeds_note(r)}")

    # ---------------- path figures: in-sample and verification -----------
    # One function draws both, so the two figures cannot drift apart. Each line
    # is one seed's path (the first valid seed), and the legend gives that seed's
    # own TE and mean active return, so it describes the curve drawn; the means
    # over seeds are in the summary. The policy is not constrained to start at 1.
    def draw_paths(dates, index_series, model_paths, te_col, title, filename):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(dates, rebase(index_series).values, "k", lw=2, label="S&P 500")
        for label, col in zip(order, ("C0", "C3", "C2")):
            shown = [s for s in seeds if model_paths.get((label, s)) is not None
                     and np.all(np.isfinite(model_paths[(label, s)]))]
            if not shown:
                continue
            r = per_seed[(per_seed.model == label) & (per_seed.seed == shown[0])].iloc[0]
            ax.plot(dates, model_paths[(label, shown[0])], col, lw=1.4,
                    label=f"{label}: TE {r[te_col]:.3f}, mean active return "
                          f"{r[te_col.replace('te_', 'mar_')]:+.1%}")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylabel("rebased level")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=140)
        plt.close(fig)

    draw_paths(s_cal.index, i_cal, paths_cal, "te_calib",
               f"In-sample fit, calibration window {CALIB[0][:4]}-{CALIB[1][:4]}; "
               f"seed {s_diag} (paths and legend values)", f"insample_{MODE}.png")
    draw_paths(s_ver.index, i_ver, paths, "te_verif",
               f"Verification window {VERIF[0][:4]}-{VERIF[1][:4]}; "
               f"seed {s_diag} (paths and legend values)", f"verification_{MODE}.png")

    # ---------------- convergence log -------------------------------------
    cv = pd.DataFrame(conv)
    cv.to_csv(OUTPUT_DIR / f"phaseA_convergence_{MODE}.csv", index=False)
    conv_lines = []
    for kind, g in cv.groupby("network"):
        fine = g[~g.failed]
        conv_lines.append(
            f"{kind}: {len(g)} fits; {int(g.failed.sum())} failed (non-finite loss or "
            f"gradient); {int(fine.ended_above_best.sum())} ended more than "
            f"{DETERIORATION_TOL:.0%} above their best training loss (final-iterate "
            f"deterioration; best weights kept); best reached at a median "
            f"{(fine.epoch_min / fine.epochs).median():.0%} of the epochs; largest "
            f"excess {fine.last_above_min.max():.2%}")
    print("\nConvergence:")
    for l in conv_lines:
        print("    " + l)

    # ---------------- reproducibility report ------------------------------
    lines = []

    def section(title, d):
        lines.append(title)
        lines.append("-" * len(title))
        for k, v in d.items():
            lines.append(f"{k}: {v}")
        lines.append("")

    section("DATA / DESIGN", {
        "version": "v2 (24 October 2026); v1 in archive/phaseA_deep_frontier_2026-10-11.py",
        "data_mode": MODE,
        "price_panel": CACHE_FILE.name,
        "full_coverage_span": f"{FULL_SPAN[0]} to {FULL_SPAN[1]}",
        "eligible_stocks": len(keep),
        "calibration_window": f"{CALIB[0]} to {CALIB[1]}",
        "calibration_days": len(s_cal),
        "verification_window": f"{VERIF[0]} to {VERIF[1]}",
        "verification_days": len(s_ver),
        "verification_status": ("never used to fit or select; examined repeatedly "
                                "during development, so a historical comparison "
                                "rather than an untouched test"),
        "design": "forward split: calibration precedes verification",
        "cross_validation": (f"{len(folds)} anchored walk-forward folds inside calibration; "
                             f"first fit on {first_train} days, validation blocks of about "
                             f"{folds[0][1][1] - folds[0][1][0]} days"),
        "min_train_days": MIN_TRAIN_DAYS,
        "selection_inside_cv": ("the auto-encoder, the standardisation statistics and "
                                "the portfolio are re-derived from each fold's training "
                                "blocks, conditional on a fixed universe; that universe "
                                "(stocks with a complete series over the full span) is "
                                "chosen using availability after the calibration window"),
        "seeds": f"{seeds}; each seeds both the auto-encoder and the policy",
        "selection_criterion": ("mean validation tracking error over folds and seeds, "
                                "among cells valid in every fold and seed"),
        "ineligible_cells": f"{n_out} of {len(grid)}",
        "fold_selection_overlap": (f"{mean_overlap:.2f} / {N_COMMUNAL} names shared on "
                                   f"average between a seed-{s_diag} fold's communal "
                                   f"set and the final one"),
        "communal_shared_across_seeds": ", ".join(
            f"seed {s}: " + (c if isinstance(c, str) else f"{c}/{N_COMMUNAL}")
            for s, c in zip(seeds, shared)),
        "final_model": "refitted on the whole calibration window, once per seed",
        "portfolio_rule": "10 most communal + (n-10) most individualistic",
        "portfolio_sizes": sizes,
        "lambda_grid": (f"{LAMBDAS} (half-decade steps, added after the n=50 tuning "
                        f"found a minimum between decade points; disclosed)"),
        "basket_construction": "equal weights at window start, buy-and-hold",
        "trading_days_per_year": TRADING_DAYS,
        "tracking_error": ("sample std of daily active returns (ddof=1) * sqrt(252); "
                           "the selection criterion"),
        "mean_active_return": "mean daily active return * 252; reported, not used to select",
        "rmse": ("RMSE between rebased level paths; approximately the cumulative "
                 "active return, hence dominated by the mean rather than the "
                 "dispersion over long windows"),
        "threads": ("single-threaded linear algebra: a seed reproduces exactly on the "
                    "same machine and library versions, not necessarily across them"),
        "failed_fits": ("a non-finite loss or gradient marks a fit failed and invalid; "
                        "see the convergence section"),
    })

    section("AUTOENCODER HYPERPARAMETERS", {
        "architecture": f"{ae.n_features} -> {ae.n_hidden} -> {ae.n_features}",
        "encoder_activation": "logistic", "decoder_activation": "linear",
        "sparsity_target_rho": ae.rho, "sparsity_weight_beta": ae.beta,
        "achieved_activation_per_unit": ", ".join(f"{v:.4f}" for v in rho_hat),
        "achieved_activation_mean": f"{rho_hat.mean():.4f}",
        "l2_lambda": ae.lam, "learning_rate": ae.lr, "epochs": ae.epochs,
        "optimizer": "Adam (full batch); weights at the lowest training loss kept",
        "input_preprocessing": "column-wise z-score, statistics from the fitting window only",
    })

    section("DFP HYPERPARAMETERS", {
        "hidden_layers": DFP_PARAMS["hidden"],
        "hidden_activation": "tanh", "output_activation": "linear",
        "output_constraint": ("none; implied weights are not restricted to sum to one "
                              "or to be non-negative"),
        "learning_rate": DFP_PARAMS["lr"], "epochs": DFP_PARAMS["epochs"],
        "optimizer": "Adam (full batch); weights at the lowest training loss kept",
        "l2_lambda": "swept over the grid; selected by cross-validation",
        "input_preprocessing": "rebased selected stock levels",
    })

    lines.append("CONVERGENCE")
    lines.append("-----------")
    lines += conv_lines
    lines.append("")

    lines.append("SELECTED CONFIGURATION")
    lines.append("----------------------")
    lines.append(f"portfolio size n : {best_n}")
    lines.append(f"ridge penalty    : {best_lam:.0e}")
    lines.append(f"chosen by        : {sel_note}")
    lines.append(f"size budget      : at most {args.max_size} stocks")
    lines.append(f"unconstrained    : {unc_line} (minimum over all sizes, for the record)")
    lines.append(f"validation TE    : " + (f"{sel_score:.6f} (sd across seeds {sel_sd:.6f})"
                                           if np.isfinite(sel_score)
                                           else "not evaluated in the validation grid"))
    lines.append("")
    lines.append(f"communal stocks (final, seed {s_diag}): " + ", ".join(communal))
    lines.append("")

    lines.append("VERIFICATION 2014-2015 (mean over seeds, sd in brackets)")
    lines.append("---------------------------------------------------------")
    for _, r in summary.iterrows():
        lines.append(f"{r['model']}{seeds_note(r)}")
        lines.append(f"  calibration : TE={r['te_calib']:.6f} ({r['te_calib_seed_sd']:.6f}), "
                     f"RMSE={r['rmse_calib']:.6f}, "
                     f"mean active return={r['mar_calib']:+.6f}")
        lines.append(f"  verification: TE={r['te_verif']:.6f} ({r['te_verif_seed_sd']:.6f}), "
                     f"RMSE={r['rmse_verif']:.6f}, "
                     f"mean active return={r['mar_verif']:+.6f}")
    lines.append("")

    report = "\n".join(lines)
    with open(OUTPUT_DIR / f"run_summary_phaseA_{MODE}.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n" + "=" * 78)
    print(report)
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
