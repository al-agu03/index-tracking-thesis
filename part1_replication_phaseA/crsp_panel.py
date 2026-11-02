"""
Build panel_total.csv from the CRSP daily stock extract: total-return levels, one
column per security (compounded DlyRet), with the S&P 500 total-return index
(^SP500TR, downloaded with yfinance) in the last column, SP500.

    python crsp_panel.py --csv <extract>.csv

--price-return builds price-return levels from the extract instead; it is not
used in the thesis.
"""

import argparse
from pathlib import Path

import pandas as pd

SPAN = ("2010-01-01", "2015-12-31")     # must match FULL_SPAN in the model scripts
INDEX_SYMBOL = "^SP500TR"


def load(path, ret_col):
    df = pd.read_csv(path, parse_dates=["DlyCalDt"], low_memory=False)
    for c in ("PERMNO", "DlyCalDt", "Ticker", ret_col):
        if c not in df.columns:
            raise SystemExit(f"column {c} missing; found: {', '.join(df.columns)}")
    for c in ("DlyPrc", "DlyRet", "DlyRetx", "DlyFacPrc", "DisDivAmt"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"{len(df)} rows, {df.PERMNO.nunique()} securities, "
          f"{df.DlyCalDt.min().date()} to {df.DlyCalDt.max().date()}")

    # A security paying several distributions on one day appears once per
    # distribution. Duplicates are merged after checking that DlyPrc, DlyRet,
    # DlyRetx and DlyFacPrc agree; the distributions are summed.
    key = ["PERMNO", "DlyCalDt"]
    if df.duplicated(subset=key).any():
        g = df[df.duplicated(subset=key, keep=False)].groupby(key)
        cols = [c for c in ("DlyPrc", "DlyRet", "DlyRetx", "DlyFacPrc") if c in df]
        bad = [c for c in cols if g[c].nunique(dropna=False).max() > 1]
        if bad:
            raise SystemExit(f"duplicated security-days disagree on {', '.join(bad)}")
        print(f"{g.ngroups} security-days carry several distributions; merged")
        if "DisDivAmt" in df.columns:
            tot = df.groupby(key, as_index=False).DisDivAmt.sum(min_count=1)
            df = (df.drop_duplicates(subset=key).drop(columns="DisDivAmt")
                    .merge(tot, on=key))
        else:
            df = df.drop_duplicates(subset=key)
    return df


def check(df):
    """Diagnostic: compare DlyRet and DlyRetx with the returns implied by price,
    price factor and distributions. Large discrepancies need investigating."""
    if not {"DlyPrc", "DlyFacPrc", "DlyRetx", "DlyRet", "DisDivAmt"} <= set(df.columns):
        return
    c = df.sort_values(["PERMNO", "DlyCalDt"])
    prev = c.groupby("PERMNO").DlyPrc.shift().abs()
    adj = c.DlyPrc.abs() * c.DlyFacPrc
    for label, implied, ref in (("DlyRetx", adj / prev - 1, c.DlyRetx),
                                ("DlyRet", (adj + c.DisDivAmt.fillna(0)) / prev - 1,
                                 c.DlyRet)):
        err = (implied - ref).abs()
        q = err.quantile(0.9999)
        off = c.loc[err > 1e-4, "Ticker"].dropna().nunique()
        print(f"  {label}: {q:.1e} at the 99.99th percentile"
              f"{'' if q < 1e-5 else '  <- investigate'}"
              f"{f', {off} securities differ on a corporate action' if off else ''}")


def benchmark(df, price_return, start):
    if price_return:
        sp = df.groupby("DlyCalDt").sprtrn.first().sort_index()
        return (1.0 + pd.to_numeric(sp, errors="coerce")).cumprod()
    import yfinance as yf
    d = yf.download(INDEX_SYMBOL, start=start, end="2016-01-01",
                    auto_adjust=True, progress=False)
    if d.empty:
        raise SystemExit(f"could not download {INDEX_SYMBOL}; "
                         f"retry or use --price-return")
    s = d["Close"]
    s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--price-return", action="store_true",
                    help="both sides without dividends, from the extract alone")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    path = Path(args.csv)
    path = path if path.is_absolute() else here / path
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    ret_col = "DlyRetx" if args.price_return else "DlyRet"
    df = load(path, ret_col)
    check(df)

    # PERMNO is the key; each security is named by its ticker at the start of
    # the study period.
    span = df[df.DlyCalDt.between(*SPAN)].dropna(subset=["Ticker"])
    name = span.sort_values("DlyCalDt").groupby("PERMNO").Ticker.first()
    if name.duplicated().any():
        raise SystemExit(f"shared start-of-period ticker:\n{name[name.duplicated()]}")

    wide = df.pivot_table(index="DlyCalDt", columns="PERMNO", values=ret_col)
    wide = wide[[p for p in name.index if p in wide.columns]]
    levels = (1.0 + wide).cumprod().where(wide.notna())   # NaN outside each history
    levels.columns = [name[p] for p in levels.columns]

    win = levels.loc[SPAN[0]:SPAN[1]]
    complete = [c for c in win.columns if win[c].notna().all()]
    short = [c for c in win.columns if c not in complete]
    print(f"\n{len(win)} trading days in {SPAN[0][:4]}-{SPAN[1][:4]}: "
          f"{len(complete)} complete series, {len(short)} terminating early")
    pd.DataFrame({
        "ticker": levels.columns,
        "permno": [p for p in name.index if p in wide.columns],
        "complete": [c in complete for c in levels.columns],
        "last_observation": [win[c].last_valid_index() for c in levels.columns],
    }).sort_values(["complete", "last_observation"]).to_csv(
        here / "crsp_coverage.csv", index=False)
    print("every exclusion is a security that stopped trading; see crsp_coverage.csv")

    bench = benchmark(df, args.price_return, str(levels.index.min().date()))
    w = bench.loc[SPAN[0]:SPAN[1]]
    print(f"benchmark compounds to {w.iloc[-1] / w.iloc[0] - 1:+.1%} over the span "
          f"({'price' if args.price_return else 'total'} return)")

    out = levels.assign(SP500=bench.reindex(levels.index))
    out = out[out.SP500.notna()].dropna(how="all")
    out.to_csv(here / "panel_total.csv")
    print(f"wrote panel_total.csv: {len(out)} days, {out.shape[1] - 1} securities")


if __name__ == "__main__":
    main()
