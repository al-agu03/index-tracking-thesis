"""
Replication of the smart-indexing example in Heaton, Polson and Witte (2016),
"Deep Learning in Finance" (arXiv:1602.06561), Section 4: an equally weighted
basket of the ten most communal stocks, ranked by a sparse autoencoder, and a deep
feature policy (DFP) on the ten most communal and ten most individualistic stocks.

As in the paper, the models are fitted on 2014-2015 and evaluated on the earlier
period. The paper's two evaluation windows are merged into 2010-2013 and rebased
once, so the results are not directly comparable with the paper's per-window
figures; with a nonlinear mapping, rebasing the inputs can also change its output.
The paper reports only the autoencoder's hidden width and sparsity target and the
DFP's hidden layers; all other settings in Config are choices of this replication.

    python hpw_replication_total.py
"""

# Single-threaded linear algebra, set before NumPy loads: the summation order
# inside matrix products, and hence the results, depends on the thread count.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ---------------------------------------------------------------------------
# Config: every reconstructed choice lives here
# ---------------------------------------------------------------------------
TRAIN = ("2014-01-01", "2015-12-31")
# One out-of-sample period, instead of the paper's two separate windows.
OOS = ("2010-01-01", "2013-12-31")
N_TRACK, N_IND = 10, 10
DATA_MODE = "total"                                  # total-return convention only

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "hpw_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = SCRIPT_DIR / "panel_total.csv"
PANEL_SPAN = "not loaded"   # filled in by load_panel, recorded in the run report

REQUIRE_FULL_SPAN = True                             # keep DFP evaluable in every window
FULL_SPAN = ("2010-01-01", "2015-12-31")             # union of all windows

TRADING_DAYS = 252

# Guard against reading the wrong panel (a full panel leaves about 350 stocks);
# set to 0 to disable.
MIN_EXPECTED_UNIVERSE = 100

# Reported by the paper: autoencoder hidden width 4 and sparsity target 0.01, DFP
# hidden layers (4, 2). Everything else is a choice of this replication.
BASE_SEED = 0
AE_PARAMS = dict(n_hidden=4, rho=0.01, beta=3.0, lam=1e-4, lr=1e-2, epochs=4000,
                 adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8)
DFP_PARAMS = dict(hidden=(4, 2), lam=1e-6, lr=5e-3, epochs=12000,
                  adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8)

# ===========================================================================
# Models (pure NumPy)
# ===========================================================================
class SparseAutoencoder:
    def __init__(self, n_features, n_hidden=4, rho=0.01, beta=3.0,
                 lam=1e-4, lr=1e-2, epochs=4000, seed=0,
                 adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.rho, self.beta, self.lam = rho, beta, lam
        self.lr, self.epochs, self.seed = lr, epochs, seed
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, np.sqrt(1.0 / n_features), (n_hidden, n_features))
        self.b1 = np.zeros(n_hidden)
        self.W2 = r.normal(0, np.sqrt(1.0 / n_hidden), (n_features, n_hidden))
        self.b2 = np.zeros(n_features)

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _adam_init(self):
        self._m = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self._t = 0

    def _adam_step(self, grads):
        self._t += 1
        b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_epsilon
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            setattr(self, k, getattr(self, k) - self.lr * mhat / (np.sqrt(vhat) + eps))

    def fit(self, X):
        T, N = X.shape
        self._adam_init()
        for _ in range(self.epochs):
            H = self._sig(X @ self.W1.T + self.b1)
            Xhat = H @ self.W2.T + self.b2
            dOut = (2.0 / (T * N)) * (Xhat - X)
            dW2 = dOut.T @ H + self.lam * self.W2
            db2 = dOut.sum(0)
            dH = dOut @ self.W2
            rho_hat = np.clip(H.mean(0), 1e-6, 1 - 1e-6)
            dKL = self.beta * (-self.rho / rho_hat + (1 - self.rho) / (1 - rho_hat))
            dH = dH + dKL / T
            dZ1 = dH * H * (1 - H)
            dW1 = dZ1.T @ X + self.lam * self.W1
            db1 = dZ1.sum(0)
            self._adam_step({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2})
        return self

    def per_feature_mse(self, X):
        Xhat = self._sig(X @ self.W1.T + self.b1) @ self.W2.T + self.b2
        return ((Xhat - X) ** 2).mean(axis=0)


class DFP:
    def __init__(self, n_in, hidden=(4, 2), lam=1e-6, lr=5e-3, epochs=12000, seed=0,
                 adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
        self.n_in = n_in
        self.hidden = tuple(hidden)
        self.lam, self.lr, self.epochs, self.seed = lam, lr, epochs, seed
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
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

    def fit(self, X, y):
        y = y.reshape(-1, 1); T = X.shape[0]
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps, t = self.adam_beta1, self.adam_beta2, self.adam_epsilon, 0
        for _ in range(self.epochs):
            acts, pre = self._forward(X)
            dz = (2.0 / T) * (acts[-1] - y)
            gW, gb = [None] * len(self.W), [None] * len(self.W)
            for i in reversed(range(len(self.W))):
                gW[i] = dz.T @ acts[i] + self.lam * self.W[i]
                gb[i] = dz.sum(0)
                if i > 0:
                    dz = (dz @ self.W[i]) * (1 - np.tanh(pre[i - 1]) ** 2)
            t += 1
            for i in range(len(self.W)):
                for cm, cv, param, grad in ((mW, vW, self.W, gW), (mb, vb, self.b, gb)):
                    cm[i] = b1 * cm[i] + (1 - b1) * grad[i]
                    cv[i] = b2 * cv[i] + (1 - b2) * (grad[i] ** 2)
                    mhat = cm[i] / (1 - b1 ** t); vhat = cv[i] / (1 - b2 ** t)
                    param[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)
        return self

    def predict(self, X):
        return self._forward(X)[0][-1].ravel()


# ===========================================================================
# Helpers
# ===========================================================================
def zscore_cols(A):
    mu = A.mean(0); sd = A.std(0); sd[sd == 0] = 1.0
    return (A - mu) / sd


def rebase(df):
    """Rebase to 1.0 at the first observation of the window."""
    return df / df.iloc[0]


def level_stats(lvl_p, lvl_i):
    """Annualised tracking error of daily active returns, and level RMSE."""
    lvl_p = np.asarray(lvl_p, dtype=float)
    lvl_i = np.asarray(lvl_i, dtype=float)
    rmse = float(np.sqrt(np.mean((lvl_p - lvl_i) ** 2)))
    rp = np.diff(lvl_p) / lvl_p[:-1]
    ri = np.diff(lvl_i) / lvl_i[:-1]
    te = float(np.std(rp - ri, ddof=1) * np.sqrt(TRADING_DAYS))
    return te, rmse


def active_return(lvl_p, lvl_i):
    """Annualised mean daily active return, the first moment of the tracking gap.

    Reported next to the tracking error (the second moment). The level RMSE mixes
    the two and, over long windows, is dominated by this mean term.
    """
    lvl_p = np.asarray(lvl_p, dtype=float)
    lvl_i = np.asarray(lvl_i, dtype=float)
    if np.any(lvl_p <= 0) or np.any(lvl_i <= 0):
        return float("nan")
    rp = np.diff(lvl_p) / lvl_p[:-1]
    ri = np.diff(lvl_i) / lvl_i[:-1]
    return float(np.mean(rp - ri) * TRADING_DAYS)


def load_panel():
    """Return the price panel: stock levels in columns, the benchmark in SP500.

    The panel is built by crsp_panel.py from the CRSP daily extract and defines
    the universe; a missing panel is an error.
    """
    global PANEL_SPAN
    if not CACHE_FILE.exists():
        raise SystemExit(
            f"\nERROR: the price panel is missing.\n  {CACHE_FILE}\n\n"
            f"Build it from the CRSP extract before running this script:\n"
            f"    python crsp_panel.py --csv <crsp_daily_extract>.csv\n"
        )
    px = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    if "SP500" not in px.columns:
        raise SystemExit(f"{CACHE_FILE.name} has no SP500 benchmark column")
    PANEL_SPAN = f"{px.index.min().date()} to {px.index.max().date()}"
    print(f"[total] panel: {CACHE_FILE.name}, {px.shape[1] - 1} securities, "
          f"{len(px)} days, {PANEL_SPAN}")
    return px


def full_coverage(df, cols, a, b):
    span = df.loc[a:b]
    return [c for c in cols if c in span.columns and span[c].notna().all()]


# ===========================================================================
# Replication
# ===========================================================================
def run_replication():
    """Run the replication once under the total-return convention."""
    mode = DATA_MODE
    px = load_panel()
    idx = px["SP500"]
    stocks = px.drop(columns=["SP500"])

    # Restrict the universe to names with a complete series over the whole span,
    # so that every selected stock is tradable in both windows and the DFP is
    # evaluable everywhere.
    if REQUIRE_FULL_SPAN:
        keep = full_coverage(stocks, stocks.columns, *FULL_SPAN)
        stocks = stocks[keep]
        print(f"[{mode}] universe restricted to full-span coverage: {len(keep)} stocks")

    # Guard against silently running on the wrong cached panel.
    eligible = list(stocks.columns)
    print(f"[{mode}] first names: {', '.join(eligible[:8])}"
          f"{' ...' if len(eligible) > 8 else ''}")
    if len(eligible) < MIN_EXPECTED_UNIVERSE:
        raise SystemExit(
            f"\nERROR: only {len(eligible)} stocks are eligible, but at least "
            f"{MIN_EXPECTED_UNIVERSE} were\nexpected for an S&P 500 panel. The file\n"
            f"{CACHE_FILE}\nis almost certainly the wrong one (a truncated or synthetic "
            f"test panel). Check the\nticker names printed above: they should be real "
            f"symbols such as AAPL or MSFT.\nDelete or replace that file and run again, or "
            f"set MIN_EXPECTED_UNIVERSE = 0 to run\non a small universe on purpose."
        )

    # --- selection: auto-encode the 2014/15 cross-section ---------------------
    tr = stocks.loc[TRAIN[0]:TRAIN[1]].dropna()
    X = zscore_cols(tr.values)
    ae = SparseAutoencoder(n_features=X.shape[1], seed=BASE_SEED, **AE_PARAMS).fit(X)

    # Achieved sparsity: the paper states the target rho but not beta, so the
    # activations actually obtained are reported.
    rho_hat = ae._sig(X @ ae.W1.T + ae.b1).mean(0)
    print(f"[{mode}] achieved activations: "
          f"{', '.join(f'{v:.4f}' for v in rho_hat)}  "
          f"(mean {rho_hat.mean():.4f}, target rho = {ae.rho})")

    rank = pd.Series(ae.per_feature_mse(X), index=tr.columns,
                     name="reconstruction_mse").sort_values()
    communal = rank.index[:N_TRACK].tolist()
    individ = rank.index[-N_IND:].tolist()

    # Save the complete AE ranking.
    rank_df = rank.rename_axis("ticker").reset_index()
    rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))
    rank_df["selection_group"] = "middle"
    rank_df.loc[rank_df["ticker"].isin(communal), "selection_group"] = "communal_top10"
    rank_df.loc[rank_df["ticker"].isin(individ), "selection_group"] = "individualistic_bottom10"
    rank_df.to_csv(OUTPUT_DIR / "ae_full_ranking.csv", index=False)
    print(f"[{mode}] communal: {communal}")
    print(f"[{mode}] individualistic: {individ}")

    def window_levels(a, b, cols):
        s = stocks.loc[a:b, cols].dropna()
        common = s.index.intersection(idx.loc[a:b].index)
        return rebase(s.loc[common]), rebase(idx.loc[common])

    # Train the DFP once, in-sample, on the 20 selected stocks.
    feats = communal + individ
    Sin, Iin = window_levels(*TRAIN, feats)
    dfp = DFP(n_in=len(feats), seed=BASE_SEED, **DFP_PARAMS).fit(Sin[feats].values, Iin.values)

    res = {}

    def evaluate(label, a, b):
        c_ok = full_coverage(stocks, communal, a, b)
        dropped = sorted(set(communal) - set(c_ok))
        if dropped:
            warnings.warn(f"[{mode}/{label}] AE basket dropping: {dropped}")
        # Equal-weighted at the start of the window and never rebalanced, i.e.
        # buy-and-hold. The paper does not state whether it rebalances.
        Snorm, Inorm = window_levels(a, b, c_ok)
        lvl_i = Inorm.values
        lvl_ae = Snorm[c_ok].mean(axis=1).values
        te_ae, rmse_ae = level_stats(lvl_ae, lvl_i)
        mar_ae = active_return(lvl_ae, lvl_i)

        f_ok = full_coverage(stocks, feats, a, b)
        if len(f_ok) == len(feats):
            Sd, Id = window_levels(a, b, feats)
            lvl_dfp = dfp.predict(Sd[feats].values)
            te_dfp, rmse_dfp = level_stats(lvl_dfp, Id.values)
            mar_dfp = active_return(lvl_dfp, Id.values)
            dfp_dates, dfp_ok = Sd.index, True
            # Under REQUIRE_FULL_SPAN the two paths span identical dates; warn if
            # that assumption ever breaks, because n_days below is taken from the
            # basket path and the two series are plotted on a shared axis.
            if len(Sd.index) != len(Snorm.index):
                warnings.warn(f"[{mode}/{label}] basket and DFP date ranges differ "
                              f"({len(Snorm.index)} vs {len(Sd.index)} days)")
        else:
            warnings.warn(f"[{mode}/{label}] DFP not evaluable: "
                          f"missing {sorted(set(feats) - set(f_ok))}")
            lvl_dfp = te_dfp = rmse_dfp = mar_dfp = np.nan
            dfp_dates, dfp_ok = None, False

        res[label] = dict(dates=Snorm.index, n_days=len(Snorm.index),
                          lvl_i=lvl_i, lvl_ae=lvl_ae, te_ae=te_ae, rmse_ae=rmse_ae,
                          mar_ae=mar_ae, lvl_dfp=lvl_dfp, dfp_dates=dfp_dates,
                          te_dfp=te_dfp, rmse_dfp=rmse_dfp, mar_dfp=mar_dfp,
                          dfp_ok=dfp_ok)
        print(f"[{mode}/{label}] AE TE={te_ae:6.4f} RMSE={rmse_ae:6.4f} "
              f"active={mar_ae:+7.4f}   "
              f"DFP TE={te_dfp:6.4f} RMSE={rmse_dfp:6.4f} active={mar_dfp:+7.4f}")

    # Evaluate the training sample and the whole 2010-2013 OOS period.
    evaluate("in_sample", *TRAIN)
    evaluate("oos", *OOS)

    # --- one in-sample figure and one OOS figure ---------------------------
    plot_specs = [
        ("In-sample 2014/15", "in_sample", "insample", TRAIN),
        ("OOS 2010-2013", "oos", "oos", OOS),
    ]
    for title, key, tag, win in plot_specs:
        R = res[key]
        fig, a = plt.subplots(figsize=(9, 3.5))
        a.plot(R["dates"], R["lvl_i"], "k", lw=2, label="S&P 500")
        a.plot(R["dates"], R["lvl_ae"], "C0", lw=1.4,
               label=f"AE basket (TE {R['te_ae']:.3f}, RMSE {R['rmse_ae']:.3f})")
        if R["dfp_ok"]:
            a.plot(R["dfp_dates"], R["lvl_dfp"], "C3", lw=1.4,
                   label=f"DFP (TE {R['te_dfp']:.3f}, RMSE {R['rmse_dfp']:.3f})")
        a.set_title(f"{title} — total return", fontsize=10)
        a.legend(fontsize=8)
        a.grid(alpha=.3)
        a.set_xlim(pd.Timestamp(win[0]), pd.Timestamp(win[1]))
        a.xaxis.set_major_locator(mdates.YearLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"figure_total_{tag}.png", dpi=140)
        plt.close(fig)

    run_info = {
        "eligible_stocks": X.shape[1],
        "communal_stocks": communal,
        "individualistic_stocks": individ,
        "dfp_inputs": len(feats),
        "ae_rho_hat": rho_hat,
    }
    return res, run_info, ae, dfp


# ===========================================================================
# Reporting
# ===========================================================================
def print_and_save_summary(res, run_info, ae, dfp):
    """Print and save a report of the settings and results of both models,
    reading every hyperparameter off the fitted objects."""
    ae_hyper = {
        "architecture": f"{ae.n_features} -> {ae.n_hidden} -> {ae.n_features}",
        "input_units": ae.n_features,
        "hidden_units": ae.n_hidden,
        "encoder_activation": "sigmoid",
        "decoder_activation": "linear",
        "sparsity_target_rho": ae.rho,
        "sparsity_weight_beta": ae.beta,
        "achieved_activation_per_unit": ", ".join(f"{v:.4f}" for v in run_info["ae_rho_hat"]),
        "achieved_activation_mean": f"{np.mean(run_info['ae_rho_hat']):.4f}",
        "l2_lambda": ae.lam,
        "learning_rate": ae.lr,
        "epochs": ae.epochs,
        "objective": "reconstruction MSE + KL sparsity penalty + L2 weight penalty",
        "optimizer": "Adam (full batch)",
        "adam_beta1": ae.adam_beta1,
        "adam_beta2": ae.adam_beta2,
        "adam_epsilon": ae.adam_epsilon,
        "seed": ae.seed,
        "weight_initialization": "Normal(0, sqrt(1/fan_in))",
        "bias_initialization": "zeros",
        "input_preprocessing": "column-wise z-score on 2014-2015 training levels",
        "ranking_metric": "per-stock reconstruction MSE",
        "output_use": "rank stocks by communal information",
    }

    dfp_hyper = {
        "architecture": f"{dfp.n_in} -> " + " -> ".join(map(str, dfp.hidden)) + " -> 1",
        "input_units": dfp.n_in,
        "hidden_layers": dfp.hidden,
        "hidden_activation": "tanh",
        "output_activation": "linear",
        "output_constraint": ("none; the implied weights are not restricted to sum "
                              "to one or to be non-negative, so the fitted path is "
                              "not directly investible as a portfolio"),
        "target": "rebased S&P 500 Total Return index level",
        "training_window": f"{TRAIN[0]} to {TRAIN[1]}",
        "loss": "MSE on rebased index levels",
        "l2_lambda": dfp.lam,
        "learning_rate": dfp.lr,
        "epochs": dfp.epochs,
        "optimizer": "Adam (full batch)",
        "adam_beta1": dfp.adam_beta1,
        "adam_beta2": dfp.adam_beta2,
        "adam_epsilon": dfp.adam_epsilon,
        "seed": dfp.seed,
        "weight_initialization": "Normal(0, sqrt(1/fan_in))",
        "bias_initialization": "zeros",
        "input_preprocessing": "rebased selected stock levels",
        "input_stock_rule": "10 most communal + 10 most individualistic stocks",
        "input_stocks": ", ".join(run_info["communal_stocks"] + run_info["individualistic_stocks"]),
    }

    data_hyper = {
        "data_mode": "total return only",
        "price_panel": CACHE_FILE.name,
        "panel_span": PANEL_SPAN,
        "training_window": f"{TRAIN[0]} to {TRAIN[1]}",
        "training_days_effective": res["in_sample"]["n_days"],
        "oos_window": f"{OOS[0]} to {OOS[1]}",
        "oos_days_effective": res["oos"]["n_days"],
        "oos_design": "back-test: the evaluation period precedes the training period",
        "full_coverage_required": REQUIRE_FULL_SPAN,
        "full_coverage_span": f"{FULL_SPAN[0]} to {FULL_SPAN[1]}",
        "n_communal": N_TRACK,
        "n_individualistic": N_IND,
        "eligible_stocks_after_filters": run_info["eligible_stocks"],
        "benchmark": "S&P 500 Total Return (^SP500TR)",
        "stock_adjustment": ("CRSP daily total return (DlyRet) compounded to a "
                             "level; see crsp_panel.py"),
        "basket_construction": ("equal weights at the start of each window, "
                                "buy-and-hold (no rebalancing)"),
        "rebasing": "each evaluation window starts at 1; OOS rebased once at start of 2010",
        "trading_days_per_year": TRADING_DAYS,
        "tracking_error": "sample std of daily active returns (ddof=1) * sqrt(252)",
        "mean_active_return": "mean daily active return * 252",
        "rmse": ("RMSE between rebased level paths; approximately the cumulative "
                 "active return, hence dominated by the mean rather than the "
                 "dispersion over long windows"),
    }

    lines = []

    def section(title, d):
        lines.append(title)
        lines.append("-" * len(title))
        for k, v in d.items():
            lines.append(f"{k}: {v}")
        lines.append("")

    section("DATA / EVALUATION SETTINGS", data_hyper)
    section("MODEL 1 - AUTOENCODER (AE) HYPERPARAMETERS", ae_hyper)

    lines.append("AE SELECTED STOCKS")
    lines.append("------------------")
    lines.append("communal: " + ", ".join(run_info["communal_stocks"]))
    lines.append("individualistic: " + ", ".join(run_info["individualistic_stocks"]))
    lines.append("")

    section("MODEL 2 - DEEP FEATURE POLICY (DFP) HYPERPARAMETERS", dfp_hyper)

    lines.append("MODEL PERFORMANCE")
    lines.append("-----------------")
    rows = []
    for label in ["in_sample", "oos"]:
        R = res[label]
        rows.append({"period": label, "n_days": R["n_days"],
                     "AE_TE": R["te_ae"], "AE_RMSE": R["rmse_ae"],
                     "AE_active_return": R["mar_ae"],
                     "DFP_TE": R["te_dfp"], "DFP_RMSE": R["rmse_dfp"],
                     "DFP_active_return": R["mar_dfp"]})
        lines.append(f"{label.upper()}")
        lines.append(f"  AE : TE={R['te_ae']:.6f}, RMSE={R['rmse_ae']:.6f}, "
                     f"mean active return={R['mar_ae']:+.6f}")
        lines.append(f"  DFP: TE={R['te_dfp']:.6f}, RMSE={R['rmse_dfp']:.6f}, "
                     f"mean active return={R['mar_dfp']:+.6f}")
    lines.append("")

    # A compact DFP-only block makes it impossible to confuse the two models.
    lines.append("DFP RESULTS ONLY")
    lines.append("----------------")
    lines.append(f"in-sample DFP TE   : {res['in_sample']['te_dfp']:.6f}")
    lines.append(f"in-sample DFP RMSE : {res['in_sample']['rmse_dfp']:.6f}")
    lines.append(f"OOS DFP TE         : {res['oos']['te_dfp']:.6f}")
    lines.append(f"OOS DFP RMSE       : {res['oos']['rmse_dfp']:.6f}")
    lines.append("")

    summary_text = "\n".join(lines)
    print("\n" + "=" * 78)
    print("TOTAL-RETURN REPLICATION SUMMARY")
    print("=" * 78)
    print(summary_text)

    with open(OUTPUT_DIR / "run_summary_total.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "metrics_total.csv", index=False)

    print(f"\nOutput folder: {OUTPUT_DIR}")
    for name in ["run_summary_total.txt", "metrics_total.csv", "ae_full_ranking.csv",
                 "figure_total_insample.png", "figure_total_oos.png"]:
        print(f"Saved: {OUTPUT_DIR / name}")


def main():
    res, run_info, ae, dfp = run_replication()
    print_and_save_summary(res, run_info, ae, dfp)


if __name__ == "__main__":
    main()