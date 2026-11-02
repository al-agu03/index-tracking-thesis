"""
Capitalisation-weighted representation learning for small index-tracking portfolios.

At each quarterly formation date, on a five-year window: standardise returns;
fit a capitalisation-weighted autoencoder (N -> 16 tanh -> 4 -> N) or weighted
PCA, its unregularised linear counterpart; extract factors from the weighted
reconstruction; select stocks by forward selection on those factors; set
long-only, fully invested weights by quadratic programming; hold to the next
formation date. References: market cap, correlation with the benchmark and a
Kim-Kim-style scalar factor. The benchmark enters only the correlation
reference, the QP and the evaluation.

Reads the output of build_panel.py and crsp_delisting_full.csv from this folder.
See the README for the stages and the commands used in the thesis.

    python extension.py --stage test
"""

# Single-threaded linear algebra, set before NumPy loads: the summation order
# inside matrix products, and hence the results, depends on the thread count.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
from types import SimpleNamespace
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRADING_DAYS = 252

# ---- design ----------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [10, 20, 40, 60]
CAP_LOSS_MIX = [0.0, 0.5, 1.0]            # xi
LATENT_DIM = 4                            # K
HIDDEN = 16
LAMBDA_CANDIDATES = [0.0, 1e-5, 1e-4, 1e-3]
WINDOW_YEARS = 5

# ---- numerical rules -----------------------------------------------------------
MIN_RETURN_SD = 1e-8          # a stock whose window return sd is below is excluded
MIN_CAP_COVERAGE = 0.90       # share of window days with a market cap
QP_KKT_TOL = 1e-8            # absolute gradient residual, raw-return MSE units
QP_FEAS_TOL = 1e-10
RANK_TOL = 1e-6               # singular value below RANK_TOL x largest = missing
REDUNDANT_TOL = 1e-10         # residual norm^2 below this x original = redundant
HELD_TOL = 1e-10              # a QP weight above this counts as a holding
QP_RETRY_ITERS = 20000        # retry: more iterations ...
QP_RETRY_TOL = 1e-15          # ... and a finer stopping rule; same acceptance
QP_THIRD_ITERS = 200000       # third attempt, only if the retry fails

# Acquisitions with a contingent payment. The recorded terminal return is the
# payoff (it contains the cash consideration); the contingent right is excluded and
# the position becomes cash earning zero. The run stops unless the data show
# exactly these values (verify_resolved).
RESOLVED_IN_SPEC = {
    76149: dict(name="SWY", date="2015-01-30", ret=-0.005128, flags=("MV", "MU", "DA"),
                treatment="recorded return (implies 34.92); Casa Ley and PDC CVR excluded"),
    75107: dict(name="ABMD", date="2022-12-22", ret=-0.002677, flags=("MV", "D2", "DA"),
                treatment="recorded return (implies $380.00 cash); milestone CVR excluded"),
    19502: dict(name="WBA", date="2025-08-28", ret=-0.044240, flags=("MV", "D2", "DA"),
                treatment="recorded return (implies $11.45 cash); contingent right excluded"),
}

# SIVB and SBNY have no recorded delisting return. The position is carried
# unchanged from the last recorded valuation date (2023-03-10) to CRSP's next price
# after delisting (2023-03-28), when the loss next price / delisting-date price - 1
# is recognised; liquidation into cash at that price is then assumed, not a
# confirmed payment. The run stops unless the data show exactly these values
# (verify_delayed).
DELAYED_VALUATION = {
    11786: dict(name="SIVB", last_day="2023-03-10", last_price=39.37, last_flag="BA",
                last_return=-0.628725, terminal_row="2023-03-13",
                value_date="2023-03-28", value=0.40, value_flag="DP"),
    90090: dict(name="SBNY", last_day="2023-03-10", last_price=70.00, last_flag="TR",
                last_return=-0.228735, terminal_row="2023-03-13",
                value_date="2023-03-28", value=0.13, value_flag="DP"),
}
DELISTING_FILE = "crsp_delisting_full.csv"   # CRSP delisting table (StkDelists)

# ---- stages -------------------------------------------------------------------
STAGES = {"develop": (2010, 2015), "validate": (2016, 2019), "final": (2020, 2025)}


# ===========================================================================
# Panel
# ===========================================================================
def _read(name):
    df = pd.read_csv(HERE / name, index_col=0, parse_dates=True)
    df.columns = df.columns.astype(int)
    return df


def verify_resolved(unresolved, terminal_events, returns):
    """
    Check that the data for the three acquisitions match RESOLVED_IN_SPEC
    (terminal event, return, flags, no later observation); otherwise stop.
    """
    for p, r in RESOLVED_IN_SPEC.items():
        d = pd.Timestamp(r["date"])
        ev = terminal_events[(terminal_events.PERMNO == p) & (terminal_events.DlyCalDt == d)]
        problems = []
        if unresolved.get(p) != d:
            problems.append("not listed in unresolved_keys.csv on that date")
        if len(ev) != 1:
            problems.append("no single terminal event on that date")
        else:
            e = ev.iloc[0]
            if not np.isclose(e.DlyRet, r["ret"], rtol=0, atol=1e-12):
                problems.append(f"terminal return {e.DlyRet} differs from {r['ret']}")
            if (e.DlyRetMissFlg, e.DlyDistRetFlg, e.DlyPrcFlg) != r["flags"]:
                problems.append("flags differ from spec 11.1")
        col = returns[p] if p in returns.columns else pd.Series(dtype=float)
        if d not in col.index or not np.isclose(col.get(d), r["ret"], rtol=0, atol=1e-12):
            problems.append("returns.csv does not carry the recorded return on that date")
        elif col.dropna().index.max() != d:
            problems.append("observations after the termination date")
        if problems:
            raise SystemExit(f"{r['name']} ({p}) cannot be treated as resolved per spec "
                             f"11.1: {'; '.join(problems)}")


def verify_delayed(unresolved, terminal_events, delistings, returns):
    """
    Check that the panel and CRSP's delisting record for SIVB and SBNY match
    DELAYED_VALUATION; otherwise stop.
    """
    for p, v in DELAYED_VALUATION.items():
        problems = []
        term = pd.Timestamp(v["terminal_row"])
        if unresolved.get(p) != term:
            problems.append("not listed in unresolved_keys.csv on the terminal row date")
        ev = terminal_events[(terminal_events.PERMNO == p) & (terminal_events.DlyCalDt == term)]
        if len(ev) != 1 or not np.isclose(ev.iloc[0].DlyPrc, v["value"], rtol=0, atol=1e-9) \
                or ev.iloc[0].DlyRetMissFlg != "DG" or ev.iloc[0].DlyDistRetFlg != "NO" \
                or ev.iloc[0].DlyPrcFlg != v["value_flag"] or not np.isnan(ev.iloc[0].DlyRet):
            problems.append("terminal row differs")
        rec = delistings[delistings.permno == p]
        if len(rec) != 1:
            problems.append("no single CRSP delisting record")
        else:
            r = rec.iloc[0]
            checks = [r.delistingdt == pd.Timestamp(v["last_day"]),
                      np.isclose(r.deldtprc, v["last_price"], rtol=0, atol=1e-9),
                      r.deldtprcflg == v["last_flag"], r.delpaymenttype == "PRCF",
                      np.isnan(r.delret), r.delretmisstype == "DG",
                      r.delnextdt == pd.Timestamp(v["value_date"]),
                      r.delamtdt == pd.Timestamp(v["value_date"]),
                      np.isclose(r.delnextprc, v["value"], rtol=0, atol=1e-9),
                      r.delnextprcflg == v["value_flag"], r.deldivamt == 0,
                      r.deldistype == "NO"]
            if not all(checks):
                problems.append("CRSP delisting record differs")
        col = returns[p] if p in returns.columns else pd.Series(dtype=float)
        last = pd.Timestamp(v["last_day"])
        if not np.isclose(col.get(last, np.nan), v["last_return"], rtol=0, atol=1e-9):
            problems.append("returns.csv lacks the move to the delisting-date price")
        elif col.dropna().index.max() != last:
            problems.append("returns.csv has observations after the last trading day")
        if pd.Timestamp(v["value_date"]) not in returns.index:
            problems.append("valuation date is not a trading day of the panel")
        if problems:
            raise SystemExit(f"{v['name']} ({p}): delayed valuation refused: "
                             f"{'; '.join(problems)}")


def final_period_unresolved(panel):
    """Unresolved terminal payoffs dated inside the final evaluation period."""
    lo, hi = STAGES["final"]
    return {p: str(d.date()) for p, d in panel.unresolved.items() if lo <= d.year <= hi}


class Panel:
    """The built artefacts, with eligibility and the rebalance calendar."""

    def __init__(self, window_years=WINDOW_YEARS):
        self.returns = _read("returns.csv")
        self.caps = _read("caps.csv").reindex(columns=self.returns.columns)
        self.member = _read("membership.csv").reindex(
            columns=self.returns.columns).astype(bool)
        self.index = pd.read_csv(HERE / "index.csv", index_col=0,
                                 parse_dates=True)["totret"]
        # five calendar years of history; the number of trading days varies
        self.window_years = window_years
        self.min_days = int(window_years * 240)
        ex = pd.read_csv(HERE / "estimation_exceptions.csv", parse_dates=["DlyCalDt"])
        self.exceptions = set(zip(ex.PERMNO, ex.DlyCalDt))
        un = pd.read_csv(HERE / "unresolved_keys.csv", parse_dates=["DlyCalDt"])
        te = pd.read_csv(HERE / "terminal_events.csv", parse_dates=["DlyCalDt"])
        self.terminal_date = dict(zip(te.PERMNO, te.DlyCalDt))
        raw = dict(zip(un.PERMNO, un.DlyCalDt))
        verify_resolved(raw, te, self.returns)           # stops if the data differ
        dl = pd.read_csv(HERE / DELISTING_FILE,
                         parse_dates=["delistingdt", "delnextdt", "delamtdt", "deldlydt"])
        verify_delayed(raw, te, dl, self.returns)        # stops if the data differ
        # delayed valuations: the terminal event moves to the valuation date, with
        # the residual return to the post-delisting price recognised there
        self.terminal_return = {}
        for p, v in DELAYED_VALUATION.items():
            d = pd.Timestamp(v["value_date"])
            self.terminal_date[p] = d
            self.terminal_return[p] = v["value"] / v["last_price"] - 1.0
        self.unresolved = {p: d for p, d in raw.items()
                           if p not in RESOLVED_IN_SPEC and p not in DELAYED_VALUATION}

    def rebalance_dates(self, start_year, end_year):
        """First trading day of each quarter."""
        idx, out = self.returns.index, []
        for y in range(start_year, end_year + 1):
            for m in (1, 4, 7, 10):
                after = idx[idx >= pd.Timestamp(y, m, 1)]
                if len(after):
                    out.append(after[0])
        return pd.DatetimeIndex(sorted(set(out)))

    def last_day(self, year):
        idx = self.returns.index
        return idx[idx <= pd.Timestamp(year, 12, 31)][-1]

    def estimation_window(self, formation):
        """Dates used to fit, ending the day before formation."""
        lo = formation - pd.DateOffset(years=self.window_years)
        idx = self.returns.index
        dates = idx[(idx >= lo) & (idx < formation)]
        if len(dates) < self.min_days:
            raise ValueError(f"{formation.date()}: {len(dates)} days in the window, "
                             f"{self.min_days} required")
        return dates

    def eligible(self, formation):
        """
        Members on the formation date with a market cap that day and complete
        usable returns over the window, a market cap on at least MIN_CAP_COVERAGE
        of window days, and a window return sd of at least MIN_RETURN_SD. Returns
        the eligible list and the names excluded by the last two rules.
        """
        mem = self.member.loc[formation]
        caps = self.caps.loc[formation]
        names = [p for p in mem[mem].index if not np.isnan(caps[p])]
        dates = self.estimation_window(formation)
        complete = self.returns.loc[dates, names].notna().all()
        for p, d in self.exceptions:
            if p in complete.index and d in dates:
                complete[p] = False
        names = list(complete[complete].index)
        coverage = self.caps.loc[dates, names].notna().mean()
        low_cap = sorted(coverage[coverage < MIN_CAP_COVERAGE].index)
        names = [p for p in names if p not in set(low_cap)]
        sd = self.returns.loc[dates, names].std(ddof=0)
        flat = sorted(sd[sd < MIN_RETURN_SD].index)
        names = sorted(p for p in names if p not in set(flat))   # stable order
        return names, {"low_cap_coverage": low_cap, "flat_returns": flat}


# ===========================================================================
# Preprocessing and capitalisation weights
# ===========================================================================
def standardise(R):
    """Column z-scores with divisor T, on training-window statistics."""
    mu = R.mean(0)
    sd = R.std(0)
    return (R - mu) / sd, mu, sd


def cap_shares(panel, dates, names):
    """c_i: training-window average market cap, normalised to sum to one."""
    c = panel.caps.loc[dates, names].mean().values      # NaN-skipping mean
    return c / c.sum()


def reconstruction_weights(c, xi):
    """q_i(xi) = (1 - xi)/N + xi c_i; positive, finite, sum to one."""
    q = (1.0 - xi) / len(c) + xi * c
    if not (np.all(np.isfinite(q)) and np.all(q > 0) and abs(q.sum() - 1) < 1e-12):
        raise ValueError("reconstruction weights must be positive, finite, sum to 1")
    return q


# ===========================================================================
# Representations
# ===========================================================================
class FitFailed(RuntimeError):
    """The loss, a gradient or a parameter became non-finite."""


class WeightedAE:
    """
    N -> HIDDEN (tanh) -> latent_dim (linear) -> N (linear decoder), with biases.

        L = (1/T) sum_t sum_i q_i (X_ti - Xhat_ti)^2 + lam * sum_l ||W_l||_F^2

    over the three weight matrices, biases excluded; no factor 1/2, so the
    penalty's gradient is 2 lam W. Adam, full batch; the checkpoint with the
    lowest regularised training loss is restored. latent_dim=1 with equal q is
    the scalar-factor reference.
    """

    def __init__(self, n_features, q, latent_dim=LATENT_DIM, hidden=HIDDEN, lam=1e-4,
                 learning_rate=1e-2, epochs=4000, seed=0):
        self.q = np.asarray(q, dtype=float)
        self.lam, self.lr, self.epochs = lam, learning_rate, epochs
        rng = np.random.default_rng(seed)

        def glorot(fan_out, fan_in):
            a = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-a, a, size=(fan_out, fan_in))
        self.p = {"W1": glorot(hidden, n_features), "b1": np.zeros(hidden),
                  "W2": glorot(latent_dim, hidden), "b2": np.zeros(latent_dim),
                  "W3": glorot(n_features, latent_dim), "b3": np.zeros(n_features)}

    def forward(self, X, p=None):
        p = self.p if p is None else p
        A1 = np.tanh(X @ p["W1"].T + p["b1"])
        H = A1 @ p["W2"].T + p["b2"]                 # latent scores
        return A1, H, H @ p["W3"].T + p["b3"]

    def loss_and_grad(self, X):
        p, T = self.p, X.shape[0]
        A1, H, Xhat = self.forward(X)
        E = Xhat - X
        penalty = sum(float((p[k] ** 2).sum()) for k in ("W1", "W2", "W3"))
        loss = float((E ** 2 * self.q).sum() / T) + self.lam * penalty
        dX = (2.0 / T) * E * self.q
        g = {"W3": dX.T @ H + 2 * self.lam * p["W3"], "b3": dX.sum(0)}
        dH = dX @ p["W3"]
        g["W2"] = dH.T @ A1 + 2 * self.lam * p["W2"]
        g["b2"] = dH.sum(0)
        dZ1 = (dH @ p["W2"]) * (1.0 - A1 ** 2)
        g["W1"] = dZ1.T @ X + 2 * self.lam * p["W1"]
        g["b1"] = dZ1.sum(0)
        return loss, g

    def fit(self, X):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(x) for k, x in self.p.items()}
        best, best_p, best_epoch, last = np.inf, None, -1, np.nan
        self.history = []
        for t in range(1, self.epochs + 1):
            loss, g = self.loss_and_grad(X)
            if not (np.isfinite(loss) and all(np.all(np.isfinite(x)) for x in g.values())):
                raise FitFailed(f"non-finite loss or gradient at epoch {t}")
            if loss < best:                          # loss of the current parameters
                best, best_epoch = loss, t - 1
                best_p = {k: x.copy() for k, x in self.p.items()}
            for k in self.p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                self.p[k] -= self.lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
            if not all(np.all(np.isfinite(x)) for x in self.p.values()):
                raise FitFailed(f"non-finite parameters at epoch {t}")
            last = loss
            if t in (1, 500, 1000, 2000, 4000, 8000, 16000) or t == self.epochs:
                self.history.append((t, loss))
        final, _ = self.loss_and_grad(X)
        if final < best:
            best, best_epoch, best_p = final, self.epochs, {k: x.copy() for k, x in self.p.items()}
        if not np.isfinite(final):
            raise FitFailed("non-finite final loss")
        self.p = best_p
        self.diag = {"loss_min": best, "epoch_min": best_epoch, "loss_last": final,
                     "last_above_min": (final - best) / max(abs(best), 1e-12),
                     "epochs": self.epochs}
        return self

    def reconstruct(self, X):
        return self.forward(X)[2]

    def latent(self, X):
        return self.forward(X)[1]


def weighted_pca(X, q, K=LATENT_DIM):
    """
    Exact minimiser of sum_t sum_i q_i (X_ti - Xhat_ti)^2 over rank-K Xhat: the SVD
    of Y = X diag(sqrt q), mapped back. Columns are not re-standardised after
    scaling. Signs fixed so that each right singular vector's largest entry is
    positive, for determinism.
    """
    s = np.sqrt(q)
    U, S, Vt = np.linalg.svd(X * s, full_matrices=False)
    flip = np.sign(Vt[np.arange(len(S)), np.abs(Vt).argmax(1)])
    U, Vt = U * flip, Vt * flip[:, None]
    return (U[:, :K] * S[:K]) @ Vt[:K] / s


def selection_factors(Xhat, q, K=LATENT_DIM):
    """
    F = U_K Sigma_K from the SVD of the centred reconstruction scaled by sqrt(q).
    Independent of how the latent space is parametrised; preserves the
    importance of each direction (no whitening). Returns F restricted to its
    usable directions and the effective rank.
    """
    A = (Xhat - Xhat.mean(0)) * np.sqrt(q)
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    rank = int((S[:K] > RANK_TOL * S[0]).sum()) if S[0] > 0 else 0
    if rank == 0 or not np.all(np.isfinite(S)):
        raise FitFailed("reconstruction has no usable finite factor")
    return U[:, :rank] * S[:rank], rank


# ===========================================================================
# Selection
# ===========================================================================
def factor_error(X, F, cols):
    """E_fac(S) = min_B ||F - X_S B||_F^2 / (T K), by least squares (explicit)."""
    if not cols:
        return float((F ** 2).sum() / F.size)
    B = np.linalg.lstsq(X[:, cols], F, rcond=None)[0]
    return float(((F - X[:, cols] @ B) ** 2).sum() / F.size)


def forward_selection(X, F, n_max):
    """
    Greedy: each step adds the stock whose column most reduces E_fac, with the
    regression refitted on all selected columns (implemented by orthogonalising
    the candidates against the selected set). Ties go to the lowest PERMNO.
    Selections are nested across budgets for a given representation, xi and
    seed. Returns the ordered selection and E_fac after each step.
    """
    T, N = X.shape
    if F.ndim != 2 or F.shape[0] != T or F.shape[1] == 0 or not np.isfinite(F).all():
        raise FitFailed("invalid selection factors")
    if not 0 < n_max <= N:
        raise ValueError("stock budget exceeds eligible universe")
    Z, R = X.astype(float).copy(), F.astype(float).copy()
    norm0 = (Z ** 2).sum(0)
    order, errors = [], []
    for _ in range(n_max):
        norms = (Z ** 2).sum(0)
        ok = norms > REDUNDANT_TOL * norm0
        ok[order] = False
        if not ok.any():
            raise ValueError(f"no non-redundant candidate left after {len(order)} stocks")
        gain = np.full(N, -np.inf)
        C = Z[:, ok].T @ R
        gain[ok] = (C ** 2).sum(1) / norms[ok]
        j = int(np.argmax(gain))
        z = Z[:, j] / np.sqrt(norms[j])
        R -= np.outer(z, z @ R)
        Z -= np.outer(z, z @ Z)
        order.append(j)
        errors.append(float((R ** 2).sum() / F.size))
    return order, errors


def scalar_selection(X, h, n):
    """
    Kim-Kim-style reference: orient the scalar factor to correlate positively with
    the equally weighted average standardised return (fallback: with the first
    stock's return, by PERMNO, if that correlation is numerically zero); select the
    n stocks with the highest signed correlation with it.
    """
    h = h - h.mean()
    if not np.isfinite(h).all() or np.linalg.norm(h) <= 1e-12:
        raise FitFailed("constant or non-finite scalar factor")
    avg = X.mean(1)
    c0 = np.corrcoef(h, avg)[0, 1] if h.std() > 0 and avg.std() > 0 else 0.0
    if abs(c0) < 1e-12:
        c0 = np.corrcoef(h, X[:, 0])[0, 1]
    if c0 < 0:
        h = -h
    Xc = X - X.mean(0)
    corr = (Xc.T @ h) / (np.sqrt((Xc ** 2).sum(0)) * np.sqrt((h ** 2).sum()))
    return list(np.lexsort((np.arange(len(corr)), -corr))[:n])   # ties: lowest index


def marketcap_selection(c, n):
    """Largest training-window average market cap."""
    return list(np.lexsort((np.arange(len(c)), -c))[:n])


def correlation_selection(R, y, n):
    """Highest training-window return correlation with the benchmark."""
    Rc, yc = R - R.mean(0), y - y.mean()
    corr = (Rc.T @ yc) / (np.sqrt((Rc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()))
    return list(np.lexsort((np.arange(len(corr)), -corr))[:n])


# ===========================================================================
# Weighting, holding, metrics
# ===========================================================================
def simplex_projection(v):
    """Euclidean projection onto {w >= 0, sum w = 1} (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    c = np.cumsum(u) - 1.0
    rho = np.nonzero(u - c / np.arange(1, len(u) + 1) > 0)[0][-1]
    return np.maximum(v - c[rho] / (rho + 1.0), 0.0)


def long_only_qp(R, y, iters=4000, tol=1e-12, retry=True):
    """
    min ||R w - y||^2 / T subject to w >= 0, 1'w = 1, by projected gradient.
    Returns the weights and KKT diagnostics: at the optimum the gradient is equal
    across held names and no smaller on any excluded one.
    """
    T, N = R.shape
    G = R.T @ R * (2.0 / T)
    b = R.T @ y * (2.0 / T)
    largest = np.linalg.eigvalsh(G)[-1]
    if not np.isfinite(largest) or largest <= 0:
        raise FitFailed("invalid QP curvature")
    step = 1.0 / largest
    w = np.full(N, 1.0 / N)
    for it in range(iters):
        w_new = simplex_projection(w - step * (G @ w - b))
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w = w_new
    grad = G @ w - b
    held = w > HELD_TOL
    kkt = float(grad[held].max() - grad[held].min()) if held.any() else np.nan
    slack = float((grad[~held] - grad[held].min()).min()) if (~held).any() else 0.0
    diag = {"kkt_spread": kkt, "worst_excluded_slack": slack, "qp_iters": it + 1,
            "sum_residual": float(abs(w.sum() - 1)), "min_weight": float(w.min())}
    diag["qp_accepted"] = bool(np.isfinite(w).all() and np.isfinite(kkt)
        and kkt <= QP_KKT_TOL and slack >= -QP_KKT_TOL
        and diag["sum_residual"] <= QP_FEAS_TOL and diag["min_weight"] >= -QP_FEAS_TOL)
    if not diag["qp_accepted"]:
        if retry:
            # Retry with more iterations and a finer stopping rule, then with up to
            # QP_THIRD_ITERS; same acceptance test. qp_iters records the iterations
            # of the attempt returned, not the total.
            try:
                w, checked = long_only_qp(R, y, iters=max(QP_RETRY_ITERS, iters * 5),
                                          tol=min(QP_RETRY_TOL, tol), retry=False)
            except FitFailed:
                w, checked = long_only_qp(R, y, iters=QP_THIRD_ITERS,
                                          tol=min(QP_RETRY_TOL, tol), retry=False)
            checked["qp_retried"] = True
            return w, checked
        raise FitFailed(f"QP failed optimality/feasibility checks: {diag}")
    diag["qp_retried"] = False
    return w, diag


def reject_unresolved_holdings(panel, ids, weights, formation, end):
    """Stop if a held position has an unresolved terminal payoff in the period."""
    touched = [p for p, wi in zip(ids, weights) if wi > HELD_TOL and p in panel.unresolved
               and formation < panel.unresolved[p] <= end]
    if touched:
        raise FitFailed(f"{formation}: unresolved terminal payoff for {touched}; "
                        "resolve the data/accounting before using this period")
    return touched


def hold(panel, names, w0, start, end):
    """
    Buy and hold from the day after `start` to `end` with weights drifting. A
    terminating holding converts to cash at its terminal return; cash earns zero.
    A delayed valuation is carried unchanged until its valuation date, then
    converted (assumed liquidation); a holding period ending before that date is
    refused. Returns daily returns, end values per position and cash.
    """
    dates = panel.returns.index[(panel.returns.index > start) & (panel.returns.index <= end)]
    R = panel.returns.loc[dates, names].values
    events = {}
    residual = getattr(panel, "terminal_return", {})
    for j, p in enumerate(names):
        d = panel.terminal_date.get(p)
        if d is not None and start < d <= end:
            events.setdefault(d, []).append(j)
        if p in residual and d is not None and d > end and start < pd.Timestamp(
                DELAYED_VALUATION[p]["terminal_row"]) <= end:
            raise ValueError(f"{p}: delayed valuation falls after the holding period")
    units = np.asarray(w0, dtype=float).copy()
    cash, out = 0.0, []
    for i, d in enumerate(dates):
        r = R[i]
        live = ~np.isnan(r)
        before = units.sum() + cash
        units = np.where(live, units * (1.0 + np.nan_to_num(r)), units)
        for j in events.get(d, ()):
            units[j] *= 1.0 + residual.get(names[j], 0.0)   # delayed valuation only
            cash += units[j]
            units[j] = 0.0
        after = units.sum() + cash
        base = 1.0 if i == 0 else before
        out.append(after / base - 1.0 if base > 0 else 0.0)
    return (pd.Series(out, index=dates, name="port"),
            pd.Series(units, index=names), cash)


def metrics(port, bench):
    """TE (sample sd, as Phase A), MAR, daily active-return RMSE, compounded returns."""
    a = (port - bench).dropna()
    return {"TE": float(a.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "MAR": float(a.mean() * TRADING_DAYS),
            "RMSE_daily": float(np.sqrt((a ** 2).mean())),
            "cum_port": float((1 + port).prod() - 1),
            "cum_bench": float((1 + bench).prod() - 1),
            "days": len(a)}


def turnover(pre_values, pre_cash, new_names, new_w):
    """Traded notional against the drifted pre-trade portfolio (full sum)."""
    total = pre_values.sum() + pre_cash
    if total <= 0:
        return np.nan
    pre = pre_values / total
    post = pd.Series(new_w, index=new_names)
    both = pre.index.union(post.index)
    return float((post.reindex(both, fill_value=0.0) - pre.reindex(both, fill_value=0.0)).abs().sum())


# ===========================================================================
# One formation date: every strategy of the stage
# ===========================================================================
def strategy_key(method, xi, lam, seed, n):
    return f"{method}|xi={xi:g}|lam={lam:g}|seed={seed}|n={n}"


def run_formation(panel, formation, end, cfg, prev):
    """
    Fit, select, weight and hold every strategy for one formation date. `prev`
    maps strategy key -> (end values, cash) of the previous period, for turnover.
    Returns strategy rows, daily rows, fit rows and the new `prev` entries.
    """
    win = panel.estimation_window(formation)
    assert win.max() < formation                       # nothing from the holding period
    names, excluded = panel.eligible(formation)
    R = panel.returns.loc[win, names].values
    y = panel.index.loc[win].values
    X, _, _ = standardise(R)
    c = cap_shares(panel, win, names)
    bench = panel.index

    selections, fits = [], []        # (method, xi, lam, seed, {n: cols}, extra)

    def from_factors(method, xi, lam, seed, Xhat, q, extra):
        F, rank = selection_factors(Xhat, q)
        order, errs = forward_selection(X, F, max(BUDGETS))
        sel = {n: order[:n] for n in BUDGETS}
        efac = {n: errs[n - 1] for n in BUDGETS}
        selections.append((method, xi, lam, seed, sel,
                           dict(extra, effective_rank=rank, rank_deficient=rank < LATENT_DIM, efac=efac)))

    for xi in cfg["xis"]:
        q = reconstruction_weights(c, xi)
        if "wpca" in cfg["methods"] and (not cfg.get("selected_xi") or xi in cfg["selected_xi"]["wpca"].values()):
            from_factors("wpca", xi, np.nan, -1, weighted_pca(X, q), q, {})
        if "ae" in cfg["methods"] and (not cfg.get("selected_xi") or xi in cfg["selected_xi"]["ae"].values()):
            for lam in cfg["lams"]:
                for seed in cfg["seeds"]:
                    t0 = time.time()
                    ae = WeightedAE(len(names), q, lam=lam, learning_rate=cfg["lr"],
                                    epochs=cfg["epochs"], seed=seed)
                    try:
                        ae.fit(X)
                    except FitFailed as err:
                        fits.append(dict(formation=formation, method="ae", xi=xi, lam=lam,
                                         seed=seed, failed=True, error=str(err)))
                        raise FitFailed(f"{formation}: AE xi={xi}, lambda={lam}, seed={seed}: {err}") from err
                    fits.append(dict(formation=formation, method="ae", xi=xi, lam=lam,
                                     seed=seed, failed=False, seconds=time.time() - t0,
                                     **ae.diag))
                    from_factors("ae", xi, lam, seed, ae.reconstruct(X), q, {})
    if "scalar" in cfg["methods"]:
        q0 = reconstruction_weights(c, 0.0)
        for seed in cfg["seeds"]:
            t0 = time.time()
            ae = WeightedAE(len(names), q0, latent_dim=1, lam=cfg["scalar_lam"],
                            learning_rate=cfg["lr"], epochs=cfg["epochs"], seed=seed)
            try:
                ae.fit(X)
            except FitFailed as err:
                fits.append(dict(formation=formation, method="scalar", xi=0.0,
                                 lam=cfg["scalar_lam"], seed=seed, failed=True, error=str(err)))
                raise FitFailed(f"{formation}: scalar seed={seed}: {err}") from err
            fits.append(dict(formation=formation, method="scalar", xi=0.0,
                             lam=cfg["scalar_lam"], seed=seed, failed=False,
                             seconds=time.time() - t0, **ae.diag))
            h = ae.latent(X)[:, 0]
            selections.append(("scalar", 0.0, cfg["scalar_lam"], seed,
                               {n: scalar_selection(X, h, n) for n in BUDGETS}, {}))
    selections.append(("marketcap", np.nan, np.nan, -1,
                       {n: marketcap_selection(c, n) for n in BUDGETS}, {}))
    selections.append(("correlation", np.nan, np.nan, -1,
                       {n: correlation_selection(R, y, n) for n in BUDGETS}, {}))

    rows, daily, new_prev = [], [], {}
    for method, xi, lam, seed, sel, extra in selections:
        for n, cols in sel.items():
            policies = cfg.get("selected_xi")
            if policies and method in ("ae", "wpca") and xi != policies[method][str(n)]:
                continue
            if len(set(cols)) != n:
                raise ValueError(f"{method}: selection of {len(set(cols))} unique, not {n}")
            ids = [names[j] for j in cols]
            w, diag = long_only_qp(R[:, cols], y)
            key = strategy_key(method, xi, lam, seed, n)
            touched = reject_unresolved_holdings(panel, ids, w, formation, end)
            port, end_values, end_cash = hold(panel, ids, w, formation, end)
            rows.append(dict(
                formation=formation, end=end, key=key, method=method, xi=xi, lam=lam,
                seed=seed, n=n, eligible=len(names),
                excluded_low_cap=len(excluded["low_cap_coverage"]),
                excluded_flat=len(excluded["flat_returns"]),
                held=int((w > HELD_TOL).sum()), max_weight=float(w.max()),
                hhi=float((w ** 2).sum()),
                turnover=turnover(*prev[key], ids, w) if key in prev else np.nan,
                rank_deficient=extra.get("rank_deficient", False),
                effective_rank=extra.get("effective_rank", np.nan),
                efac=extra.get("efac", {}).get(n, np.nan),
                touches_unresolved=" ".join(map(str, touched)),
                selected=" ".join(map(str, ids)),
                weights=" ".join(f"{x:.10g}" for x in w),
                end_values=json.dumps({**{str(k): float(v) for k, v in end_values.items()},
                                       "_cash": float(end_cash)}),
                **diag))
            new_prev[key] = (end_values, end_cash)
            daily.append(pd.DataFrame({"formation": formation, "key": key,
                                       "date": port.index, "port": port.values,
                                       "bench": bench.reindex(port.index).values}))
    return rows, pd.concat(daily), fits, new_prev


# ===========================================================================
# Stage runner: resumable, fingerprinted
# ===========================================================================
DATA_FILES = [DELISTING_FILE, "returns.csv", "caps.csv", "membership.csv", "index.csv",
              "estimation_exceptions.csv", "unresolved_keys.csv", "terminal_events.csv"]


def fingerprint(cfg):
    h = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
         for f in DATA_FILES + [Path(__file__).name]}
    return json.dumps(dict(cfg, sha256=h), sort_keys=True, default=str)


def run_stage(panel, stage, cfg, years, out_name=None):
    out = HERE / (out_name or f"outputs_{stage}")
    out.mkdir(exist_ok=True)
    fp = fingerprint(dict(cfg, stage=stage, years=years))
    files = {k: out / f"{k}.csv" for k in ("strategies", "daily", "fits")}
    done_file, cfg_file = out / "done.txt", out / "config.json"
    if any(f.exists() for f in files.values()) and not (cfg_file.exists()
                                                        and cfg_file.read_text() == fp):
        raise SystemExit(f"{out.name} holds results of a different configuration, data "
                         f"or code; move it away to start afresh")
    cfg_file.write_text(fp)
    done = set(done_file.read_text().split()) if done_file.exists() else set()

    # drop rows of a formation that was interrupted half-way
    for f in files.values():
        if f.exists():
            df = pd.read_csv(f)
            keep = df[df.formation.astype(str).str[:10].isin(done)]
            if len(keep) < len(df):
                tmp = f.with_suffix(".tmp")
                keep.to_csv(tmp, index=False)
                os.replace(tmp, f)

    start_year, end_year = years
    dates = panel.rebalance_dates(start_year, end_year)
    ends = list(dates[1:]) + [panel.last_day(end_year)]
    prev = {}
    if files["strategies"].exists() and done:
        last = pd.read_csv(files["strategies"], parse_dates=["formation"])
        last = last[last.formation == last.formation.max()]
        for r in last.itertuples():
            ev = json.loads(r.end_values)
            cash = ev.pop("_cash")
            prev[r.key] = (pd.Series({int(k): v for k, v in ev.items()}), cash)

    print(f"stage {stage}: {len(dates)} formations {dates[0].date()} .. {dates[-1].date()}, "
          f"{len(done)} already done")
    for formation, end in zip(dates, ends):
        tag = str(formation.date())
        if tag in done:
            continue
        t0 = time.time()
        rows, daily, fits, prev_new = run_formation(panel, formation, end, cfg, prev)
        prev = prev_new
        for name, df in (("strategies", pd.DataFrame(rows)), ("daily", daily),
                         ("fits", pd.DataFrame(fits))):
            df.to_csv(files[name], mode="a", index=False, header=not files[name].exists())
        with open(done_file, "a") as fh:
            fh.write(tag + "\n")
        print(f"  {tag}  eligible {rows[0]['eligible']}  {len(rows)} strategies  "
              f"[{time.time() - t0:5.0f}s]", flush=True)
    check_coverage(out, panel, dates, ends, cfg)
    return summarise(out, stage)


def expected_keys(cfg):
    keys = set()
    for n in BUDGETS:
        for method in ("marketcap", "correlation"):
            keys.add(strategy_key(method, np.nan, np.nan, -1, n))
        for xi in cfg["xis"]:
            for method in ("ae", "wpca"):
                if method not in cfg["methods"]:
                    continue
                policy = cfg.get("selected_xi")
                if policy and xi != policy[method][str(n)]:
                    continue
                for lam in (cfg["lams"] if method == "ae" else [np.nan]):
                    for seed in (cfg["seeds"] if method == "ae" else [-1]):
                        keys.add(strategy_key(method, xi, lam, seed, n))
        if "scalar" in cfg["methods"]:
            for seed in cfg["seeds"]:
                keys.add(strategy_key("scalar", 0., cfg["scalar_lam"], seed, n))
    return keys


def check_coverage(out, panel, dates, ends, cfg):
    """No model selection on partial histories, missing seeds or duplicate rows."""
    strat = pd.read_csv(out / "strategies.csv", parse_dates=["formation"])
    daily = pd.read_csv(out / "daily.csv", parse_dates=["formation", "date"])
    keys = expected_keys(cfg)
    if strat.duplicated(["formation", "key"]).any() or daily.duplicated(["date", "key"]).any():
        raise FitFailed("duplicate strategy or return observations")
    if set(strat.formation) != set(dates) or set(daily.formation) != set(dates):
        raise FitFailed("formation coverage mismatch")
    for date, end in zip(dates, ends):
        if set(strat.loc[strat.formation == date, "key"]) != keys:
            raise FitFailed(f"{date}: incomplete strategy/seed coverage")
        expected = set(panel.returns.index[(panel.returns.index > date) & (panel.returns.index <= end)])
        block = daily[daily.formation == date]
        if set(block.key) != keys:
            raise FitFailed(f"{date}: missing return histories")
        for key, g in block.groupby("key"):
            if set(g.date) != expected or not np.isfinite(g[["port", "bench"]].values).all():
                raise FitFailed(f"{date}: incomplete or invalid returns for {key}")


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2))
    os.replace(tmp, path)


def load_final_policy(path, lr, epochs, lam):
    if not path.exists():
        raise ValueError("run complete validation first; selected_policy.json is missing")
    policy = json.loads(path.read_text())
    origin = policy["validation_config"]
    cfg = {k: v for k, v in origin.items() if k != "sha256"}
    if json.loads(fingerprint(cfg)) != origin:
        raise ValueError("validation policy was produced by different data or code; revalidate")
    if (lr, epochs, lam) != (origin["lr"], origin["epochs"], origin["lams"][0]):
        raise ValueError("final numerical settings must match validation")
    for method in ("ae", "wpca"):
        if set(policy["selected_xi"][method]) != set(map(str, BUDGETS)):
            raise ValueError("incomplete selected policy")
        if any(x not in CAP_LOSS_MIX for x in policy["selected_xi"][method].values()):
            raise ValueError("invalid selected xi")
    return policy


def summarise(out, stage):
    """Pooled metrics per strategy over the stage, then means over seeds."""
    strat = pd.read_csv(out / "strategies.csv")
    daily = pd.read_csv(out / "daily.csv", parse_dates=["date"])
    per = []
    for key, g in daily.groupby("key", sort=False):
        s = strat[strat.key == key]
        m = metrics(g.set_index("date").port, g.set_index("date").bench)
        per.append(dict(key=key, method=s.method.iloc[0], xi=s.xi.iloc[0],
                        lam=s.lam.iloc[0], seed=s.seed.iloc[0], n=s.n.iloc[0], **m,
                        turnover_mean=s.turnover.mean(), held_mean=s.held.mean(),
                        effective_rank_min=s.effective_rank.min(),
                        periods_touching_unresolved=int((s.touches_unresolved
                                                         .fillna("") != "").sum())))
    per = pd.DataFrame(per)
    per.to_csv(out / "summary_per_seed.csv", index=False)
    cols = ["TE", "MAR", "RMSE_daily", "turnover_mean", "held_mean"]
    grp = per.groupby(["method", "xi", "lam", "n"], dropna=False)
    summ = grp[cols].mean().join(grp.RMSE_daily.std().rename("RMSE_seed_sd")) \
                    .join(grp.size().rename("seeds")).reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nmean over seeds (RMSE_daily is the selection criterion):")
    print(summ.round(6).to_string(index=False))
    if per.periods_touching_unresolved.sum():
        print("\nNOTE: some periods hold a security with an unresolved terminal payoff "
              "(spec 11.1, 11.9); see strategies.csv, column touches_unresolved.")
    if stage == "develop":
        ae = summ[summ.method == "ae"]
        print("\nlambda_ext: mean RMSE_daily over xi, budgets and seeds")
        print(ae.groupby("lam").RMSE_daily.mean().round(7).to_string())
    if stage == "validate":
        print("\nxi chosen per budget (minimum mean RMSE_daily over seeds):")
        chosen = {}
        for method in ("ae", "wpca"):
            chosen[method] = {}
            s = summ[summ.method == method]
            for n, g in s.groupby("n"):
                b = g.sort_values(["RMSE_daily", "xi"]).iloc[0]
                chosen[method][str(int(n))] = float(b.xi)
                print(f"  {method:5s} n={n:3d}  xi={b.xi:g}  RMSE_daily={b.RMSE_daily:.6f}")
        atomic_json(out / "selected_policy.json", {
            "selected_xi": chosen,
            "validation_config": json.loads((out / "config.json").read_text()),
            "selection_criterion": "mean pooled daily active-return RMSE over five seeds"})
    return summ


# ===========================================================================
# Stage: convergence (learning rate, epochs) and timing
# ===========================================================================
def formation_data(panel, formation, xi):
    win = panel.estimation_window(formation)
    names, _ = panel.eligible(formation)
    X, _, _ = standardise(panel.returns.loc[win, names].values)
    q = reconstruction_weights(cap_shares(panel, win, names), xi)
    return X, q


def stage_time(panel, args):
    formation = panel.rebalance_dates(2015, 2015)[0]
    X, q = formation_data(panel, formation, 1.0)
    print(f"{formation.date()}: T={X.shape[0]} days, N={X.shape[1]} stocks")
    t0 = time.time()
    ae = WeightedAE(X.shape[1], q, lam=args.ae_weight_penalty if args.ae_weight_penalty is not None else 1e-4,
                    learning_rate=args.learning_rate or 1e-2,
                    epochs=args.epochs or 4000, seed=0).fit(X)
    t_ae = time.time() - t0
    t0 = time.time()
    F, rank = selection_factors(ae.reconstruct(X), q)
    forward_selection(X, F, max(BUDGETS))
    t_sel = time.time() - t0
    print(f"autoencoder fit, {ae.epochs} epochs: {t_ae:.1f}s ({1000 * t_ae / ae.epochs:.1f} ms/epoch)")
    print(f"factor extraction + forward selection to {max(BUDGETS)}: {t_sel:.2f}s")
    per_formation = (len(CAP_LOSS_MIX) * len(SEEDS) + len(SEEDS)) * t_ae
    print(f"validation estimate: {len(CAP_LOSS_MIX)} xi x {len(SEEDS)} seeds + "
          f"{len(SEEDS)} scalar fits = {per_formation / 60:.0f} min per formation, "
          f"{16 * per_formation / 3600:.1f} h for 16 formations")


def stage_converge(panel, args):
    """Loss trajectories on development formations; nothing is backtested."""
    out = HERE / "outputs_converge"
    out.mkdir(exist_ok=True)
    rates = args.rates or [1e-3, 3e-3, 1e-2]
    epochs = args.epochs or 8000
    lam = args.ae_weight_penalty if args.ae_weight_penalty is not None else 1e-4
    formations = [panel.rebalance_dates(y, y)[k] for y, k in ((2010, 0), (2012, 2), (2015, 0))]
    config = dict(rates=rates, epochs=epochs, lam=lam, seeds=[0],
                  formations=[str(x.date()) for x in formations], xis=[0., 1.])
    fp = fingerprint(config)
    config_file = out / "config.json"
    if any(out.iterdir()) and (not config_file.exists() or config_file.read_text() != fp):
        raise SystemExit("convergence output has incompatible or unversioned results; archive it first")
    config_file.write_text(fp)
    rows = []
    for formation in formations:
        for xi in (0.0, 1.0):
            X, q = formation_data(panel, formation, xi)
            for lr in rates:
                cache = out / f"fit_{formation.date()}_xi{xi:g}_lr{lr:.17g}.json"
                if cache.exists():
                    rows.append(json.loads(cache.read_text()))
                    continue
                t0 = time.time()
                try:
                    ae = WeightedAE(X.shape[1], q, lam=lam, learning_rate=lr,
                                    epochs=epochs, seed=0).fit(X)
                    F, rank = selection_factors(ae.reconstruct(X), q)
                except FitFailed as err:
                    r = dict(formation=str(formation.date()), xi=xi, lr=lr, failed=str(err))
                    atomic_json(cache, r)
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(out / "convergence.csv", index=False)
                    continue
                r = dict(formation=str(formation.date()), xi=xi, lr=lr, lam=lam,
                         seconds=round(time.time() - t0, 1), effective_rank=rank, **ae.diag)
                r.update({f"loss_at_{t}": l for t, l in ae.history})
                atomic_json(cache, r)
                rows.append(r)
                pd.DataFrame(rows).to_csv(out / "convergence.csv", index=False)
                print(f"  {formation.date()} xi={xi:g} lr={lr:g}: min loss {ae.diag['loss_min']:.6f} "
                      f"at epoch {ae.diag['epoch_min']}, last {ae.diag['loss_last']:.6f}, "
                      f"rank {rank}, {r['seconds']}s", flush=True)
    pd.DataFrame(rows).to_csv(out / "convergence.csv", index=False)
    print(f"\nwrote {out / 'convergence.csv'}")


# ===========================================================================
# Stage: tests
# ===========================================================================
def stage_test(panel):
    rng = np.random.default_rng(0)
    results = []

    def check(name, ok):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    # synthetic data with a factor structure
    T, N = 300, 30
    Xs = rng.normal(size=(T, 3)) @ rng.normal(size=(3, N)) + 0.5 * rng.normal(size=(T, N))
    Xs, _, _ = standardise(Xs)
    c = rng.uniform(1, 10, N)
    c /= c.sum()

    q0, q1 = reconstruction_weights(c, 0.0), reconstruction_weights(c, 1.0)
    check("reconstruction weights sum to one (xi = 0, 0.5, 1)",
          all(abs(reconstruction_weights(c, x).sum() - 1) < 1e-12 for x in CAP_LOSS_MIX))

    ae = WeightedAE(N, q0, lam=0.0, seed=0)
    loss, _ = ae.loss_and_grad(Xs)
    check("xi = 0 loss equals the ordinary elementwise MSE",
          abs(loss - ((ae.reconstruct(Xs) - Xs) ** 2).mean()) < 1e-12)

    ae = WeightedAE(N, q1, lam=1e-3, seed=1)
    _, g = ae.loss_and_grad(Xs)
    worst = 0.0
    for k in ae.p:
        idx = tuple(rng.integers(0, s) for s in ae.p[k].shape)
        old = ae.p[k][idx]
        ae.p[k][idx] = old + 1e-6
        lp, _ = ae.loss_and_grad(Xs)
        ae.p[k][idx] = old - 1e-6
        lm, _ = ae.loss_and_grad(Xs)
        ae.p[k][idx] = old
        num = (lp - lm) / 2e-6
        worst = max(worst, abs(num - g[k][idx]) / max(abs(num), 1e-8))
    check(f"autoencoder gradients match finite differences (worst rel. error {worst:.1e})",
          worst < 1e-5)

    Xh = weighted_pca(Xs, q1, 4)
    S = np.linalg.svd(Xs * np.sqrt(q1), compute_uv=False)
    obj = float(((Xs - Xh) ** 2 * q1).sum())
    other = float(((Xs - weighted_pca(Xs, q0, 4)) ** 2 * q1).sum())
    check("weighted PCA attains the weighted rank-K optimum (= discarded singular values)",
          abs(obj - (S[4:] ** 2).sum()) < 1e-8 and obj <= other + 1e-12)

    F1, r1 = selection_factors(Xh, q1)
    check("selection factors of weighted PCA are its principal-component scores",
          r1 == 4 and np.allclose(np.sort(np.diag(F1.T @ F1)), np.sort(S[:4] ** 2)))

    ae = WeightedAE(N, q1, lam=1e-4, epochs=300, seed=0).fit(Xs)
    M = rng.normal(size=(4, 4)) + 3 * np.eye(4)
    ae2 = WeightedAE(N, q1, seed=0)
    ae2.p = dict(ae.p, W2=M @ ae.p["W2"], b2=M @ ae.p["b2"], W3=ae.p["W3"] @ np.linalg.inv(M))
    Fa, _ = selection_factors(ae.reconstruct(Xs), q1)
    Fb, _ = selection_factors(ae2.reconstruct(Xs), q1)
    check("selection factors do not depend on the latent coordinates",
          np.allclose(np.abs(Fa), np.abs(Fb), atol=1e-8))

    order, errs = forward_selection(Xs, F1, 12)
    brute, sel = [], []
    for _ in range(12):
        cand = [i for i in range(N) if i not in sel]
        e = [factor_error(Xs, F1, sel + [i]) for i in cand]
        sel.append(cand[int(np.argmin(e))])
        brute.append(min(e))
    check("forward selection = explicit candidate-by-candidate least squares",
          order == sel and np.allclose(errs, brute))
    check("factor-reconstruction error does not increase as stocks are added",
          all(b <= a + 1e-12 for a, b in zip(errs, errs[1:])))
    check("each selection has exactly n unique stocks", len(set(order)) == 12)

    w, d = long_only_qp(rng.normal(size=(200, 8)) * 0.01, rng.normal(size=200) * 0.01)
    check("QP weights are non-negative and sum to one",
          w.min() >= 0 and abs(w.sum() - 1) < 1e-12 and d["kkt_spread"] < 1e-8)

    class Toy:          # two stocks, three days; stock 2 terminates on day 2
        returns = pd.DataFrame({1: [0.10, 0.00, -0.10], 2: [0.00, 0.50, np.nan]},
                               index=pd.date_range("2020-01-02", periods=3))
        terminal_date = {2: pd.Timestamp("2020-01-03")}
    port, endv, cash = hold(Toy, [1, 2], np.array([0.5, 0.5]),
                            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-04"))
    # day1: 0.55 + 0.5 = 1.05; day2: 0.55 + 0.75 (to cash) = 1.30; day3: 0.495 + 0.75
    hand = [0.05, 1.30 / 1.05 - 1, (0.495 + 0.75) / 1.30 - 1]
    check("buy-and-hold accounting matches a hand calculation (with a termination)",
          np.allclose(port.values, hand) and abs(cash - 0.75) < 1e-12)

    # real data: one formation, short training, full pipeline, twice
    formation = panel.rebalance_dates(2014, 2014)[0]
    cfg = dict(methods=["ae", "wpca", "scalar"], xis=[0.0, 1.0], lams=[1e-4], seeds=[0],
               lr=1e-2, epochs=150, scalar_lam=1e-4)
    end = panel.rebalance_dates(2014, 2014)[1]
    rows, daily, fits, _ = run_formation(panel, formation, end, cfg, {})
    rows2, _, _, _ = run_formation(panel, formation, end, cfg, {})
    df = pd.DataFrame(rows)
    check(f"real data {formation.date()}: every selection has n unique eligible stocks "
          f"({df.eligible.iloc[0]} eligible)",
          all(len(set(r["selected"].split())) == r["n"] for r in rows))
    check("real data: QP feasible for every strategy",
          (df.min_weight >= 0).all() and (df.sum_residual < 1e-10).all())
    check("real data: holding period starts after formation",
          (pd.to_datetime(daily.date) > formation).all())
    check("real data: identical settings and seeds reproduce the selections",
          [r["selected"] for r in rows] == [r["selected"] for r in rows2])
    safety_checks(panel, check)
    print(f"\n{sum(results)} of {len(results)} checks passed")
    return all(results)


def resolved_event_checks(panel, check, rejected):
    """Acquisitions: recorded return; SIVB and SBNY: delayed valuation."""
    check("no unresolved terminal payoff remains", panel.unresolved == {})
    idx = panel.returns.index
    for p, r in RESOLVED_IN_SPEC.items():
        d = pd.Timestamp(r["date"])
        start = idx[idx < d][-6]                         # hold over the last five days
        end = idx[idx > d][3] if (idx > d).sum() > 3 else idx[-1]
        port, units, cash = hold(panel, [p], np.array([1.0]), start, end)
        rets = panel.returns.loc[(idx > start) & (idx <= d), p].values
        check(f"{r['name']} {r['date']}: payoff is the recorded return "
              f"{r['ret']:+.4%}, then cash",
              np.isclose(port.loc[d], r["ret"], rtol=0, atol=1e-12)
              and np.isclose(cash, np.prod(1 + rets), rtol=0, atol=1e-12)
              and units.iloc[0] == 0.0 and (port.loc[port.index > d] == 0).all())
    for p, v in DELAYED_VALUATION.items():
        last, vd = pd.Timestamp(v["last_day"]), pd.Timestamp(v["value_date"])
        start, end = pd.Timestamp("2023-03-01"), pd.Timestamp("2023-04-03")
        port, units, cash = hold(panel, [p], np.array([1.0]), start, end)
        carried = np.prod(1 + panel.returns.loc[(idx > start) & (idx <= last), p].values)
        loss = v["value"] / v["last_price"] - 1
        between = port.loc[(port.index > last) & (port.index < vd)]
        check(f"{v['name']}: carried unchanged {last.date()}..{vd.date()}, loss "
              f"{loss:+.2%} on {vd.date()}, then cash",
              len(between) > 0 and (between == 0).all()
              and np.isclose(port.loc[vd], loss, rtol=0, atol=1e-12)
              and np.isclose(cash, carried * (1 + loss), rtol=0, atol=1e-12)
              and units.iloc[0] == 0.0 and (port.loc[port.index > vd] == 0).all())
        check(f"{v['name']}: no conversion on {v['terminal_row']}; the only terminal "
              f"event is the valuation on {vd.date()}",
              panel.terminal_date[p] == vd)
        check(f"{v['name']}: a holding period ending before {vd.date()} is refused",
              rejected(lambda p=p: hold(panel, [p], np.array([1.0]), start,
                                        pd.Timestamp("2023-03-27")), ValueError))
        check(f"{v['name']}: carried value corresponds to the delisting-date price "
              f"(last return {v['last_return']:+.6f})",
              np.isclose(panel.returns.at[last, p], v["last_return"], rtol=0, atol=1e-9))
    check("final stage no longer blocked", final_period_unresolved(panel) == {})
    fake = SimpleNamespace(unresolved={11786: pd.Timestamp("2023-03-13")})
    check("an unresolved payoff in a held position still stops the run",
          rejected(lambda: reject_unresolved_holdings(
              fake, [11786], [.5], pd.Timestamp("2023-01-03"), pd.Timestamp("2023-04-03"))))
    check("an unresolved payoff at zero weight does not block",
          reject_unresolved_holdings(fake, [11786], [0.], pd.Timestamp("2023-01-03"),
                                     pd.Timestamp("2023-04-03")) == [])
    te = pd.read_csv(HERE / "terminal_events.csv", parse_dates=["DlyCalDt"])
    un = pd.read_csv(HERE / "unresolved_keys.csv", parse_dates=["DlyCalDt"])
    raw = dict(zip(un.PERMNO, un.DlyCalDt))
    tampered = panel.returns.copy()
    tampered.loc[pd.Timestamp("2015-01-30"), 76149] = -0.5
    check("a spec 11.1 resolution is refused if the data differ",
          rejected(lambda: verify_resolved(raw, te, tampered), SystemExit))
    dl = pd.read_csv(HERE / DELISTING_FILE,
                     parse_dates=["delistingdt", "delnextdt", "delamtdt", "deldlydt"])
    bad = dl.copy()
    bad.loc[bad.permno == 11786, "delnextprc"] = 0.50
    check("the delayed valuation is refused if CRSP's record differs",
          rejected(lambda: verify_delayed(raw, te, bad, panel.returns), SystemExit))
    bad = dl.copy()
    bad.loc[bad.permno == 90090, "deldivamt"] = 0.13
    check("the delayed valuation is refused if a payment is recorded",
          rejected(lambda: verify_delayed(raw, te, bad, panel.returns), SystemExit))


def safety_checks(panel, check):
    """Regression checks for rejection, coverage, frozen policies and resume."""
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    def rejected(fn, error=FitFailed):
        try:
            fn()
        except error:
            return True
        return False

    rng = np.random.default_rng(123)
    X, _, _ = standardise(rng.normal(size=(40, 8)))
    q = np.full(8, 1 / 8)
    check("zero-rank reconstruction rejected",
          rejected(lambda: selection_factors(np.zeros_like(X), q)))
    check("constant scalar factor rejected",
          rejected(lambda: scalar_selection(X, np.ones(40), 3)))
    check("unconverged QP rejected",
          rejected(lambda: long_only_qp(X * .01, X[:, 0] * .01, iters=1, retry=False)))
    resolved_event_checks(panel, check, rejected)
    # QP retry: on a real case that needs it (market-cap selection of 60 on the
    # 2019-07-01 window; training data only), the retry must use more iterations
    # and the tighter stopping tolerance, and pass the unchanged acceptance test.
    f = pd.Timestamp("2019-07-01")
    win = panel.estimation_window(f)
    names, _ = panel.eligible(f)
    Rq, yq = panel.returns.loc[win, names].values, panel.index.loc[win].values
    cols = marketcap_selection(cap_shares(panel, win, names), 60)
    calls = []
    real_qp = long_only_qp
    def spy(*a, **k):
        calls.append(k)
        return real_qp(*a, **k)
    with patch.dict(globals(), long_only_qp=spy):
        wq, dq = spy(Rq[:, cols], yq)
    retry = calls[-1] if len(calls) > 1 else {}
    f3 = pd.Timestamp("2021-04-01")                    # a case that needs the third attempt
    names3, _ = panel.eligible(f3)
    win3 = panel.estimation_window(f3)
    R3, y3 = panel.returns.loc[win3, names3].values, panel.index.loc[win3].values
    A3 = R3[:, correlation_selection(R3, y3, 40)]
    second_fails = rejected(lambda: long_only_qp(A3, y3, iters=QP_RETRY_ITERS,
                                                 tol=QP_RETRY_TOL, retry=False))
    w3, d3 = long_only_qp(A3, y3)
    check("third QP attempt: the 2021-04-01 correlation n=40 case fails the retry but "
          "is accepted by the third attempt under the same tolerances",
          second_fails and d3["qp_retried"] and d3["qp_iters"] > QP_RETRY_ITERS
          and d3["kkt_spread"] <= QP_KKT_TOL and d3["worst_excluded_slack"] >= -QP_KKT_TOL
          and d3["sum_residual"] <= QP_FEAS_TOL and d3["min_weight"] >= -QP_FEAS_TOL)
    check("QP retry uses 20,000 iterations and tolerance 1e-15, same acceptance",
          dq.get("qp_retried") is True and retry.get("iters") == QP_RETRY_ITERS
          and retry.get("tol") == QP_RETRY_TOL and dq["kkt_spread"] <= QP_KKT_TOL
          and dq["sum_residual"] <= QP_FEAS_TOL and wq.min() >= -QP_FEAS_TOL)
    formation = panel.rebalance_dates(2014, 2014)[0]
    cfg = dict(methods=["ae"], xis=[0.], lams=[1e-4], seeds=[0],
               lr=.01, epochs=1, scalar_lam=1e-4)
    with patch.object(WeightedAE, "fit", side_effect=FitFailed("injected failure")):
        check("a failed AE stops the formation rather than disappearing",
              rejected(lambda: run_formation(panel, formation, formation, cfg, {})))

    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        out = Path(tmp)
        date = pd.Timestamp("2014-01-02")
        days = pd.date_range("2014-01-03", periods=2)
        toy = SimpleNamespace(returns=pd.DataFrame(index=days))
        config = dict(methods=[], xis=[], lams=[], seeds=[])
        with patch.dict(globals(), BUDGETS=[1]):
            keys = expected_keys(config)
            rows = [dict(formation=date, key=k) for k in keys]
            daily = [dict(formation=date, date=d, key=k, port=0., bench=0.)
                     for k in keys for d in days]
            pd.DataFrame(rows).to_csv(out / "strategies.csv", index=False)
            pd.DataFrame(daily).to_csv(out / "daily.csv", index=False)
            check_coverage(out, toy, [date], [days[-1]], config)
            pd.DataFrame(daily[:-1]).to_csv(out / "daily.csv", index=False)
            check("missing return dates rejected before summarising",
                  rejected(lambda: check_coverage(out, toy, [date], [days[-1]], config)))
            pd.DataFrame(daily).to_csv(out / "daily.csv", index=False)
            pd.DataFrame(rows[:-1]).to_csv(out / "strategies.csv", index=False)
            check("missing strategies rejected before summarising",
                  rejected(lambda: check_coverage(out, toy, [date], [days[-1]], config)))

        origin = dict(lr=.01, epochs=2, lams=[1e-4])
        policy = dict(validation_config=origin,
                      selected_xi={m: {str(n): .5 for n in BUDGETS} for m in ("ae", "wpca")})
        path = out / "policy.json"
        atomic_json(path, policy)
        with patch.dict(globals(), fingerprint=lambda cfg: json.dumps(cfg)):
            loaded = load_final_policy(path, .01, 2, 1e-4)
            check("final stage loads the saved validation policy", loaded == policy)
            check("final settings cannot differ from validation",
                  rejected(lambda: load_final_policy(path, .02, 2, 1e-4), ValueError))
        final_cfg = dict(methods=["ae", "wpca", "scalar"], xis=CAP_LOSS_MIX,
                         lams=[1e-4], seeds=SEEDS, scalar_lam=1e-4,
                         selected_xi=policy["selected_xi"])
        check("final strategy keys include only the selected xi per budget",
              len(expected_keys(final_cfg)) == len(BUDGETS) * (5 + 1 + 5 + 2))

    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        fake = SimpleNamespace(rebalance_dates=lambda y, _: pd.date_range(f"{y}-01-01", periods=4))
        args = SimpleNamespace(rates=[.001], epochs=2, ae_weight_penalty=1e-4)
        original_fit = WeightedAE.fit
        calls = []
        def interrupt_fit(ae, data):
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt()
            return original_fit(ae, data)
        with patch.dict(globals(), HERE=Path(tmp), fingerprint=lambda cfg: json.dumps(cfg),
                        formation_data=lambda *args: (X, q)):
            with patch.object(WeightedAE, "fit", interrupt_fit):
                try:
                    stage_converge(fake, args)
                except KeyboardInterrupt:
                    pass
            completed = len(list((Path(tmp) / "outputs_converge").glob("fit_*.json")))
            remaining = []
            def count_fit(ae, data):
                remaining.append(1)
                return original_fit(ae, data)
            with patch.object(WeightedAE, "fit", count_fit):
                stage_converge(fake, args)
            result = pd.read_csv(Path(tmp) / "outputs_converge" / "convergence.csv")
            check("convergence resumes completed fits after interruption",
                  completed == 2 and len(remaining) == 4 and len(result) == 6)



# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["test", "time", "converge", "develop", "validate", "final"])
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--ae-weight-penalty", type=float, help="lambda_ext")
    ap.add_argument("--rates", type=float, nargs="+", help="converge: learning rates")
    ap.add_argument("--seeds", type=int, nargs="+", help="develop: seeds (default 0 1)")
    ap.add_argument("--years", type=int, nargs=2, help="develop: first and last year")
    ap.add_argument("--lambdas", type=float, nargs="+",
                    help="develop: lambda_ext values other than the prespecified grid; "
                         "results go to outputs_develop_lambda-ext/")
    ap.add_argument("--spec-frozen", action="store_true",
                    help="final: confirm the specification is frozen")
    a = ap.parse_args()
    if a.epochs is not None and a.epochs <= 0:
        ap.error("--epochs must be positive")
    if a.learning_rate is not None and (not np.isfinite(a.learning_rate) or a.learning_rate <= 0):
        ap.error("--learning-rate must be positive and finite")
    if a.ae_weight_penalty is not None and (not np.isfinite(a.ae_weight_penalty) or a.ae_weight_penalty < 0):
        ap.error("--ae-weight-penalty must be finite and nonnegative")
    if a.rates and any(not np.isfinite(r) or r <= 0 for r in a.rates):
        ap.error("--rates must be positive and finite")
    if a.lambdas and (a.stage != "develop" or any(not np.isfinite(x) or x < 0 for x in a.lambdas)):
        ap.error("--lambdas is for --stage develop and must be finite and nonnegative")
    if a.seeds and (len(a.seeds) != len(set(a.seeds)) or min(a.seeds) < 0):
        ap.error("--seeds must be unique and nonnegative")

    panel = Panel()
    print(f"panel: {panel.returns.shape[0]} dates x {panel.returns.shape[1]} securities, "
          f"{panel.window_years}-year window\n")
    if a.stage == "test":
        raise SystemExit(0 if stage_test(panel) else 1)
    if a.stage == "time":
        return stage_time(panel, a)
    if a.stage == "converge":
        return stage_converge(panel, a)
    if a.learning_rate is None or a.epochs is None:
        ap.error("--learning-rate and --epochs are required (set them after --stage converge)")

    if a.stage == "develop":
        years = tuple(a.years) if a.years else (2013, 2015)
        if not (2010 <= years[0] <= years[1] <= 2015):
            ap.error("develop uses 2010-2015 only")
        lams = a.lambdas or LAMBDA_CANDIDATES
        cfg = dict(methods=["ae"], xis=CAP_LOSS_MIX, lams=lams,
                   seeds=a.seeds or [0, 1], lr=a.learning_rate, epochs=a.epochs,
                   scalar_lam=np.nan)
        # an extended grid is written to its own folder, so it never mixes with
        # the main grid
        extra = None if lams == LAMBDA_CANDIDATES else "outputs_develop_lambda-ext"
        return run_stage(panel, "develop", cfg, years, out_name=extra)

    if a.ae_weight_penalty is None:
        ap.error("--ae-weight-penalty is required (fixed on 2010-2015 by --stage develop)")
    if a.stage == "final" and not a.spec_frozen:
        ap.error("the final stage needs --spec-frozen: 2020-2025 is reserved until the "
                 "specification is frozen (spec 8)")
    cfg = dict(methods=["ae", "wpca", "scalar"], xis=CAP_LOSS_MIX,
               lams=[a.ae_weight_penalty], seeds=SEEDS, lr=a.learning_rate,
               epochs=a.epochs, scalar_lam=a.ae_weight_penalty)
    if a.stage == "final":
        policy = load_final_policy(HERE / "outputs_validate" / "selected_policy.json",
                                   a.learning_rate, a.epochs, a.ae_weight_penalty)
        cfg["selected_xi"] = policy["selected_xi"]
        pending = final_period_unresolved(panel)
        if pending:
            ap.error(f"resolve final-period terminal events before evaluation: {pending}")
    summ = run_stage(panel, a.stage, cfg, STAGES[a.stage])
    if a.stage == "final":
        strat = pd.read_csv(HERE / "outputs_final" / "strategies.csv")
        if (strat.touches_unresolved.fillna("") != "").any():
            raise SystemExit("holdings touch an unresolved terminal payoff (spec 11.1, "
                             "11.9): these results cannot be reported as final")


if __name__ == "__main__":
    main()
