"""
Trading-frequency sensitivity for the selections saved by extension.py.

Reweighting: the quarterly selection is kept and the QP weights are re-estimated
monthly or weekly on the latest five-year window. Reselection: stocks are
reselected only in January (annual) or January and July (semiannual), with the
QP weights re-estimated every quarter; stocks without a complete window are
dropped until the next reselection, so a portfolio can hold fewer candidates than
the budget. Representation models are not refitted. Covers the selected xi of the
autoencoder and weighted PCA, market cap and correlation.

Before reporting, it checks that the saved run matches the current code and data
and that the QP and buy-and-hold reproduce the saved weights and returns exactly.
It has no checkpoint-resume. Turnover at quarterly formations excludes the first
quarter, intra-quarter turnover does not, so the two means are not directly
additive.

    python frequency_sensitivity.py --test
    python frequency_sensitivity.py --period validate | final
"""
import extension as ex          # first: it fixes single-threaded linear algebra

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FREQUENCIES = ["monthly", "weekly"]
RESELECTION = {"reselect-annual": (1,), "reselect-semiannual": (1, 7)}   # months
SCOPE = ("ae", "wpca", "marketcap", "correlation")


# ===========================================================================
# Reweighting
# ===========================================================================
def rebalance_points(panel, start, end, freq):
    """First trading day of each month or calendar week strictly inside (start, end),
    excluding the formation's own month or week."""
    idx = panel.returns.index
    days = idx[(idx > start) & (idx < end)]
    if not len(days):
        return pd.DatetimeIndex([])
    code = "M" if freq == "monthly" else "W"
    per = days.to_period(code)
    first = ~pd.Series(per).duplicated().values
    return days[first & (per != pd.Timestamp(start).to_period(code))]


def window_complete(panel, d, names):
    """The estimation window for date d, and which names have complete, ordinary
    returns over it (the same rule as eligibility in extension.py)."""
    win = panel.estimation_window(d)
    ok = panel.returns.loc[win, names].notna().all().values
    bad = {p for p, day in panel.exceptions if day in win}
    return win, np.array([o and p not in bad for p, o in zip(names, ok)])


def hold_reweighted(panel, names, w0, start, end, points, trace=False):
    """
    extension.hold() with the QP re-estimated at the close of each date in
    `points`. Only names that traded that day, are still held and have a complete
    window take part; the others keep their units; idle cash is redeployed. With
    no points the arithmetic is that of hold(). Returns daily returns, end units,
    cash, summed intermediate turnover and the number of reweights.
    """
    dates = panel.returns.index[(panel.returns.index > start) & (panel.returns.index <= end)]
    R = panel.returns.loc[dates, names].values
    residual = getattr(panel, "terminal_return", {})
    events = {}
    for j, p in enumerate(names):
        d = panel.terminal_date.get(p)
        if d is not None and start < d <= end:
            events.setdefault(d, []).append(j)
        if p in residual and d is not None and d > end and start < pd.Timestamp(
                ex.DELAYED_VALUATION[p]["terminal_row"]) <= end:
            raise ValueError(f"{p}: delayed valuation falls after the holding period")
    pts = set(points)
    units = np.asarray(w0, dtype=float).copy()
    done = np.zeros(len(names), dtype=bool)
    cash, out, intra, n_rw, path = 0.0, [], 0.0, 0, []
    for i, d in enumerate(dates):
        r = R[i]
        live = ~np.isnan(r)
        before = units.sum() + cash
        units = np.where(live, units * (1.0 + np.nan_to_num(r)), units)
        for j in events.get(d, ()):
            units[j] *= 1.0 + residual.get(names[j], 0.0)
            cash += units[j]
            units[j] = 0.0
            done[j] = True
        after = units.sum() + cash
        base = 1.0 if i == 0 else before
        out.append(after / base - 1.0 if base > 0 else 0.0)
        if d in pts:
            win, full = window_complete(panel, d, names)
            trade = live & ~done & full
            if trade.any():
                total = units.sum() + cash
                alloc = units[trade].sum() + cash
                ids = [names[j] for j in np.where(trade)[0]]
                w, _ = ex.long_only_qp(panel.returns.loc[win, ids].values,
                                       panel.index.loc[win].values)
                new = np.zeros_like(units)
                new[trade] = alloc * w
                intra += float(np.abs(new[trade] - units[trade]).sum() / total) if total > 0 else 0.0
                units = np.where(trade, new, units)
                cash = 0.0
                n_rw += 1
        if trace:
            path.append(units.copy())
    res = (pd.Series(out, index=dates, name="port"), pd.Series(units, index=names),
           cash, intra, n_rw)
    return res + (pd.DataFrame(path, index=dates, columns=names),) if trace else res


# ===========================================================================
# Reading the main run
# ===========================================================================
def load_run(period):
    """The saved run, checked to come from the current extension.py and data."""
    src = HERE / f"outputs_{period}"
    cfg_text = (src / "config.json").read_text()
    stored = json.loads(cfg_text)
    if ex.fingerprint({k: v for k, v in stored.items() if k != "sha256"}) != cfg_text:
        raise SystemExit(f"{src.name} was produced by a different extension.py or data; "
                         f"rerun that stage first")
    policy = json.loads((HERE / "outputs_validate" / "selected_policy.json").read_text())
    # round-trip parsing, so saved floats are read back bit for bit
    strat = pd.read_csv(src / "strategies.csv", parse_dates=["formation", "end"],
                        float_precision="round_trip")
    daily = pd.read_csv(src / "daily.csv", parse_dates=["formation", "date"],
                        float_precision="round_trip")
    return strat, daily, policy["selected_xi"], cfg_text


def reselection_ids(panel, formation, ids):
    """The previously selected stocks that still have a complete, ordinary window
    at this formation (delisted or interrupted ones drop out until reselection)."""
    _, ok = window_complete(panel, formation, ids)
    return [p for p, o in zip(ids, ok) if o]


def in_scope(strat, selected_xi):
    keep = strat.method.isin(["marketcap", "correlation"])
    for m in ("ae", "wpca"):
        chosen = strat.n.astype(str).map(selected_xi[m])
        keep |= (strat.method == m) & np.isclose(strat.xi, chosen)
    return strat[keep].sort_values(["formation", "key"])


# ===========================================================================
def run(period):
    panel = ex.Panel()
    strat, daily, selected_xi, src_cfg = load_run(period)
    rows_in = in_scope(strat, selected_xi)
    saved = daily.set_index(["key", "date"]).port
    out_rows, out_daily, prev, last, kept = [], [], {}, None, {}
    for r in rows_in.itertuples():
        if r.formation != last:
            print(f"  {r.formation.date()}", flush=True)
            last = r.formation
        ids = [int(x) for x in r.selected.split()]
        win = panel.estimation_window(r.formation)
        w, _ = ex.long_only_qp(panel.returns.loc[win, ids].values, panel.index.loc[win].values)
        w_saved = np.array([float(x) for x in r.weights.split()])
        if not np.allclose(w, w_saved, rtol=0, atol=1e-9):
            raise SystemExit(f"{r.formation.date()} {r.key}: QP does not reproduce saved weights")
        port, _, _ = ex.hold(panel, ids, w, r.formation, r.end)
        ref = saved.loc[r.key].loc[port.index].values
        if not np.array_equal(port.values, ref):
            raise SystemExit(f"{r.formation.date()} {r.key}: buy and hold does not reproduce "
                             f"the saved daily returns")
        for freq in FREQUENCIES:
            key = f"{r.key}|freq={freq}"
            pts = rebalance_points(panel, r.formation, r.end, freq)
            vport, vend, vcash, vintra, nrw = hold_reweighted(panel, ids, w, r.formation,
                                                              r.end, pts)
            out_rows.append(dict(formation=r.formation, end=r.end, key=key, method=r.method,
                                 xi=r.xi, lam=r.lam, seed=r.seed, n=r.n, freq=freq,
                                 reweights=nrw, intra_turnover=vintra,
                                 turnover=ex.turnover(*prev[key], ids, w) if key in prev else np.nan))
            prev[key] = (vend, vcash)
            out_daily.append(pd.DataFrame({"formation": r.formation, "key": key,
                                           "date": vport.index, "port": vport.values,
                                           "bench": panel.index.reindex(vport.index).values}))
        for variant, months in RESELECTION.items():
            key = f"{r.key}|freq={variant}"
            if r.formation.month in months:              # reselection quarter = main run
                vids, vw = ids, w
                kept[key] = ids
            else:
                if key not in kept:
                    raise SystemExit(f"{key}: the period must start in a reselection month")
                vids = reselection_ids(panel, r.formation, kept[key])
                if not vids:
                    raise SystemExit(f"{r.formation.date()} {key}: no stock left to hold")
                vwin = panel.estimation_window(r.formation)
                vw, _ = ex.long_only_qp(panel.returns.loc[vwin, vids].values,
                                        panel.index.loc[vwin].values)
            vport, vend, vcash = ex.hold(panel, vids, vw, r.formation, r.end)
            out_rows.append(dict(formation=r.formation, end=r.end, key=key, method=r.method,
                                 xi=r.xi, lam=r.lam, seed=r.seed, n=r.n, freq=variant,
                                 reweights=0, intra_turnover=0.0, held=len(vids),
                                 turnover=ex.turnover(*prev[key], vids, vw) if key in prev else np.nan))
            prev[key] = (vend, vcash)
            out_daily.append(pd.DataFrame({"formation": r.formation, "key": key,
                                           "date": vport.index, "port": vport.values,
                                           "bench": panel.index.reindex(vport.index).values}))
    out = HERE / f"outputs_frequency_{period}"
    out.mkdir(exist_ok=True)
    strat_f = pd.DataFrame(out_rows)
    daily_f = pd.concat(out_daily)
    # the quarterly rows of the main run, for side-by-side reporting
    q = rows_in.assign(freq="quarterly", intra_turnover=0.0, reweights=0)
    pairs = set(zip(rows_in.key, rows_in.formation))          # same strategy-quarters
    dq = daily[[k in pairs for k in zip(daily.key, daily.formation)]]
    per = []
    for key, g in pd.concat([dq, daily_f]).groupby("key", sort=False):
        s = pd.concat([q, strat_f])
        s = s[s.key == key]
        m = ex.metrics(g.set_index("date").port, g.set_index("date").bench)
        per.append(dict(key=key, method=s.method.iloc[0], xi=s.xi.iloc[0], seed=s.seed.iloc[0],
                        n=s.n.iloc[0], freq=s.freq.iloc[0], **m,
                        turnover_mean=s.turnover.mean(), intra_turnover_mean=s.intra_turnover.mean(),
                        reweights_mean=s.reweights.mean()))
    per = pd.DataFrame(per)
    grp = per.groupby(["method", "xi", "n", "freq"], dropna=False)
    summ = grp[["TE", "MAR", "RMSE_daily", "turnover_mean", "intra_turnover_mean",
                "reweights_mean"]].mean().join(grp.size().rename("seeds")).reset_index()
    for name, df in (("strategies", strat_f), ("daily", daily_f),
                     ("summary_per_seed", per), ("summary", summ)):
        tmp = out / f"{name}.tmp"
        df.to_csv(tmp, index=False)
        tmp.replace(out / f"{name}.csv")
    (out / "config.json").write_text(json.dumps(dict(
        period=period, frequencies=FREQUENCIES, reselection=RESELECTION, scope=SCOPE,
        selected_xi=selected_xi,
        source_config_sha256=hashlib.sha256(src_cfg.encode()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        sort_keys=True, indent=2))
    pd.set_option("display.width", 200)
    print(f"reproduced {len(rows_in)} saved strategy-quarters exactly; "
          f"{len(strat_f)} variant strategy-quarters\n")
    label = summ.method + np.where(summ.xi.notna(), " xi=" + summ.xi.astype(str), "")
    t = summ.assign(config=label).pivot_table(index=["config", "freq"], columns="n",
                                               values="RMSE_daily") * 1e4
    print("daily active-return RMSE x 1e-4 (mean over seeds):")
    print(t.round(2).to_string())
    print(f"\nwrote {out}")


# ===========================================================================
def tests():
    panel = ex.Panel()
    results = []

    def check(name, ok):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    idx = panel.returns.index
    f, e = pd.Timestamp("2019-01-02"), pd.Timestamp("2019-04-01")
    m = rebalance_points(panel, f, e, "monthly")
    w = rebalance_points(panel, f, e, "weekly")
    check("monthly points: first trading day of February and March only",
          list(m) == [idx[idx >= "2019-02-01"][0], idx[idx >= "2019-03-01"][0]])
    check("weekly points: one per later calendar week, each its first trading day",
          len(w) == len(set(w.to_period("W"))) and all(
              d == idx[(idx.to_period("W") == d.to_period("W"))][0] for d in w)
          and f.to_period("W") not in set(w.to_period("W")) and (w < e).all())
    names, _ = panel.eligible(f)
    win = panel.estimation_window(f)
    ids = [names[j] for j in ex.marketcap_selection(ex.cap_shares(panel, win, names), 20)]
    w0, _ = ex.long_only_qp(panel.returns.loc[win, ids].values, panel.index.loc[win].values)
    a = ex.hold(panel, ids, w0, f, e)
    b = hold_reweighted(panel, ids, w0, f, e, [])
    check("without reweighting dates it reproduces extension.hold() exactly",
          a[0].equals(b[0]) and np.array_equal(a[1].values, b[1].values) and a[2] == b[2]
          and b[3] == 0.0 and b[4] == 0)
    port, units, cash, intra, nrw, path = hold_reweighted(panel, ids, w0, f, e, m, trace=True)
    win1, full = window_complete(panel, m[0], ids)
    w1, _ = ex.long_only_qp(panel.returns.loc[win1, ids].values, panel.index.loc[win1].values)
    after = path.loc[m[0]].values
    check("at a reweighting date the portfolio is reset to the QP on that window",
          full.all() and np.allclose(after / after.sum(), w1, rtol=0, atol=1e-12)
          and nrw == len(m) and intra > 0)
    check("reweighting conserves wealth (no costs)",
          np.isclose(after.sum(), np.prod(1 + port.loc[:m[0]].values), rtol=0, atol=1e-12))
    f2, e2 = pd.Timestamp("2023-01-03"), pd.Timestamp("2023-04-03")
    pair = [11786, 14593]                                  # SIVB and Apple
    pts = rebalance_points(panel, f2, e2, "weekly")
    p2, _, _, _, _, path2 = hold_reweighted(panel, pair, np.array([.5, .5]), f2, e2, pts,
                                            trace=True)
    sv = path2[11786]
    span = sv.loc["2023-03-10":"2023-03-27"]
    check("weekly reweights leave SIVB's carried position untouched until its valuation",
          sum(pd.Timestamp("2023-03-10") < d < pd.Timestamp("2023-03-28") for d in pts) >= 2
          and (span == span.iloc[0]).all() and sv.loc["2023-03-28"] == 0.0)
    v = ex.DELAYED_VALUATION[11786]
    loss = v["value"] / v["last_price"] - 1
    held, v_before = span.iloc[0], path2.loc["2023-03-27"].sum()
    apple = panel.returns.at[pd.Timestamp("2023-03-28"), 14593]
    expected = (v_before - held) * (1 + apple) + held * (1 + loss)
    check("SIVB's loss is recognised on 2023-03-28 within a reweighted portfolio",
          np.isclose(p2.loc["2023-03-28"], expected / v_before - 1, rtol=0, atol=1e-12))
    keep = reselection_ids(panel, pd.Timestamp("2023-04-03"), [11786, 14593])
    check("reselection variants drop a stock that delisted since the last selection",
          keep == [14593])
    keep = reselection_ids(panel, f, ids)
    check("reselection variants keep stocks that still have a complete window", keep == ids)
    fake = pd.DataFrame(dict(method=["ae", "ae", "wpca", "marketcap", "scalar"],
                             xi=[1.0, 0.5, 0.5, np.nan, 0.0], n=[10, 10, 20, 10, 10],
                             formation=pd.Timestamp("2016-01-04"), key=list("abcde")))
    sel = {"ae": {str(n): 1.0 for n in ex.BUDGETS},
           "wpca": {**{str(n): 1.0 for n in ex.BUDGETS}, "20": 0.5}}
    check("scope: selected xi only, references kept, scalar excluded",
          list(in_scope(fake, sel).key) == ["a", "c", "d"])
    print(f"\n{sum(results)} of {len(results)} checks passed")
    return all(results)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true")
    g.add_argument("--period", choices=["validate", "final"])
    a = ap.parse_args()
    if a.test:
        raise SystemExit(0 if tests() else 1)
    run(a.period)


if __name__ == "__main__":
    main()
