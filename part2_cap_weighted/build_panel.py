"""
Build the Part 2 panel from three CRSP extracts in this folder:
daily_crsp_2005_2025.csv, membership_intervals.csv, index_crsp_sp500_totret.csv.

Writes returns.csv, caps.csv, index.csv, membership.csv, terminal_events.csv,
coverage_gaps.csv, payoffs_to_resolve.csv, unresolved_keys.csv,
estimation_exceptions.csv, reconstruction.csv and build_report.txt.

estimation_exceptions.csv lists returns excluded from estimation because they are
not ordinary daily returns (multi-period and delisting returns). They stay in
returns.csv, so their effect still enters portfolio accounting.

    python build_panel.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAILY = HERE / "daily_crsp_2005_2025.csv"
MEMBER = HERE / "membership_intervals.csv"
INDEX = HERE / "index_crsp_sp500_totret.csv"

# Index-level quantities are defined only where the universe is the whole
# index. The universe is every security that was a member at some point in
# 2010-2025, so earlier years miss names that left before 2010.
IN_SCOPE = pd.Timestamp("2010-01-01")

# DlyRetDurFlg taxonomy, per CRSP's duration dictionary.
#   D1-D4  one trading day spanning one to four calendar days
#   DU     one trading day spanning five or more calendar days
#   DD     delisting return (a terminal payoff)
#   P1-P9  multi-period return
#   MR     missing return
# Any other code stops the build.
MISSING_RETURN = {"MR"}
MULTI_DAY = {f"P{i}" for i in range(1, 10)}
DELISTING_DURATION = {"DD"}
ORDINARY = {"D1", "D2", "D3", "D4", "DU"}
KNOWN_DURATION = ORDINARY | DELISTING_DURATION | MULTI_DAY | MISSING_RETURN

# Numeric but not an ordinary daily observation: kept for portfolio accounting,
# excluded from estimation.
NOT_DAILY = MULTI_DAY | DELISTING_DURATION

KEYS = ["PERMNO", "DlyCalDt"]
NUMERIC = ["DlyRet", "DlyCap", "DlyPrc"]
FLAGS = ["DlyPrcFlg", "TradingStatusFlg", "DlyRetMissFlg", "DlyRetDurFlg",
         "DlyDistRetFlg"]
COLS = KEYS + NUMERIC + FLAGS

lines = []


def note(s=""):
    lines.append(s)
    print(s)


def load():
    """
    Read the extract, keeping flag strings intact.

    `NA` is a flag value meaning nothing was flagged. Pandas converts it to a
    missing value by default, which erases the distinction between a flag that
    says "clean" and a flag that is absent, so missing values are declared
    explicitly and the numeric columns coerced afterwards.
    """
    df = pd.read_csv(DAILY, usecols=COLS, parse_dates=["DlyCalDt"],
                     dtype={"PERMNO": "int32"}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    note(f"daily file: {len(df):,} rows, {df.PERMNO.nunique()} securities, "
         f"{df.DlyCalDt.min().date()} to {df.DlyCalDt.max().date()}")
    note(f"  flag values retained: " + ", ".join(
        f"{c}={df[c].nunique()}" for c in FLAGS))
    unknown = set(df.DlyRetDurFlg.dropna().unique()) - KNOWN_DURATION
    if unknown:
        raise SystemExit(f"unrecognised DlyRetDurFlg codes: {sorted(unknown)}. "
                         f"Classify them before treating them as returns.")
    return df


def deduplicate(df):
    """
    One row per security-day.

    The extract repeats some keys. Averaging them, as pivot_table would do
    silently, is only harmless while the repeats agree; so agreement is checked
    and a disagreement stops the build rather than being averaged away.
    """
    dup = df.duplicated(subset=KEYS, keep=False)
    if not dup.any():
        note("duplicate keys: none")
        return df
    g = df[dup].groupby(KEYS)
    checked = NUMERIC + FLAGS            # every retained non-key column
    conflict = [k for k, x in g
                if any(x[f].nunique(dropna=False) > 1 for f in checked)]
    note(f"duplicate keys: {g.ngroups} keys, {dup.sum() - g.ngroups} excess rows, "
         f"{len(conflict)} with conflicting values "
         f"(checked {len(checked)} columns)")
    if conflict:
        raise SystemExit(f"conflicting duplicates, first few: {conflict[:5]}")
    return df.drop_duplicates(subset=KEYS, keep="first")


def classify_events(df):
    """
    Separate terminal flagged observations from interruptions in coverage.

    A flagged observation is terminal, within this extract, only when the
    security never reappears in it. Five securities carry status flags mid
    history and resume afterwards; treating those as exits would remove years
    of data each. "Terminal" is a statement about the extract, not a claim
    that the security ceased to exist.
    """
    last_obs = df.groupby("PERMNO").DlyCalDt.max()
    flagged = df[df.TradingStatusFlg.isin(["D", "X"])].copy()
    flagged["is_last"] = flagged.DlyCalDt == flagged.PERMNO.map(last_obs)

    keep = ["PERMNO", "DlyCalDt", "TradingStatusFlg", "DlyRetMissFlg",
            "DlyDistRetFlg", "DlyRet", "DlyPrc", "DlyPrcFlg"]
    terminal = flagged[flagged.is_last][keep].copy()

    note(f"terminal events: {len(terminal)} securities "
         f"({(terminal.TradingStatusFlg == 'D').sum()} D, "
         f"{(terminal.TradingStatusFlg == 'X').sum()} X)")
    return terminal.sort_values("DlyCalDt"), coverage_gaps(df, last_obs)


def coverage_gaps(df, last_obs):
    """
    Consecutive runs of suspended coverage, one row per run.

    Nearly every flagged observation is followed by another flagged one, so the
    next recorded date is not the date coverage resumed. Runs are grouped and
    reported as intervals, with the first active date and the first date
    carrying a usable return recorded separately: the first active observation
    carries RA and no return, so coverage resuming is not the same as the
    return series resuming.
    """
    s = df.sort_values(KEYS).reset_index(drop=True)
    flagged = s.TradingStatusFlg.isin(["D", "X"])
    s["is_last"] = s.DlyCalDt == s.PERMNO.map(last_obs)
    inside = flagged & ~s.is_last
    if not inside.any():
        note("coverage gaps: none")
        return pd.DataFrame()

    # a new run starts wherever the previous row was not part of this run
    newrun = inside & ~(inside.shift(1, fill_value=False)
                        & s.PERMNO.eq(s.PERMNO.shift(1)))
    s.loc[inside, "run"] = newrun[inside].cumsum()

    numeric = s.DlyRet.notna() & ~s.DlyRetDurFlg.isin(MISSING_RETURN)
    rows = []
    for run, g in s[inside].groupby("run"):
        p = g.PERMNO.iloc[0]
        after = s[(s.PERMNO == p) & (s.DlyCalDt > g.DlyCalDt.max())]
        active = after[after.TradingStatusFlg == "A"]
        valid = after[numeric.reindex(after.index, fill_value=False)]
        rows.append((p, g.DlyCalDt.min(), g.DlyCalDt.max(), len(g),
                     g.TradingStatusFlg.iloc[0],
                     active.DlyCalDt.min() if len(active) else pd.NaT,
                     valid.DlyCalDt.min() if len(valid) else pd.NaT))
    gaps = pd.DataFrame(rows, columns=[
        "PERMNO", "gap_start", "gap_end", "observations", "status",
        "first_active_date", "first_valid_return_date"])
    note(f"coverage gaps: {len(gaps)} runs over {gaps.PERMNO.nunique()} "
         f"securities, {gaps.observations.sum()} flagged observations")
    for r in gaps.itertuples(index=False):
        note(f"  {r.PERMNO} {r.gap_start.date()} to {r.gap_end.date()} "
             f"({r.observations} obs, {r.status}) -> active "
             f"{r.first_active_date.date()}, first usable return "
             f"{r.first_valid_return_date.date()}")
    return gaps.sort_values("gap_start")


def payoffs_to_resolve(terminal):
    """
    Terminal rows whose payoff the extract does not settle.

    Two kinds. No return at all, where DlyRetMissFlg is DG -- a delisting price
    more than ten periods from the delisting date, so DlyPrc is a delisting
    price (DlyPrcFlg = DP) but not an executable liquidation value on the
    terminal date. And a numeric return flagged MV, missing corporate action
    value, so the figure may not be the whole payoff.

    Written out as keys as well as a report: the experiment must refuse to
    value a held position from an unresolved payoff, not warn about it.
    """
    unresolved = terminal[terminal.DlyRet.isna()
                          | terminal.DlyRetMissFlg.eq("MV")].copy()
    unresolved["reason"] = np.where(
        unresolved.DlyRet.isna(), "no return (" + unresolved.DlyRetMissFlg + ")",
        "numeric return flagged MV")
    note(f"terminal payoffs needing a stated rule: {len(unresolved)}")
    for r in unresolved.itertuples(index=False):
        note(f"  {r.PERMNO} {r.DlyCalDt.date()} status={r.TradingStatusFlg} "
             f"prc={r.DlyPrc} ({r.DlyPrcFlg}) ret={r.DlyRet} -- {r.reason}")
    return unresolved


def panel(df, field, valid):
    wide = df[valid].pivot(index="DlyCalDt", columns="PERMNO", values=field)
    wide.index.name = "date"
    return wide


def membership_matrix(dates, permnos):
    """
    Boolean dates x PERMNO: member that day.

    Built interval by interval. Eleven securities left the index and later
    returned; collapsing them to one span would count them as members
    throughout the gap.
    """
    iv = pd.read_csv(MEMBER, parse_dates=["MbrStartDt", "MbrEndDt"])
    m = pd.DataFrame(False, index=dates, columns=permnos)
    for p, s, e in iv.itertuples(index=False):
        if p in m.columns:
            m.loc[(dates >= s) & (dates <= e), p] = True
    spells = iv.PERMNO.value_counts()
    note(f"membership: {len(iv)} intervals, {iv.PERMNO.nunique()} securities, "
         f"{(spells > 1).sum()} with more than one spell")
    return m


def check_calendars(ret_dates, cap_dates, idx_dates):
    """
    Compare the three source calendars before intersecting them.

    Taking an intersection first would drop a date silently, after which a
    count of surviving dates proves nothing. Any benchmark date missing from
    the panel inside the evaluation period stops the build; a gap outside it
    is reported and tolerated, since those years are estimation history.
    """
    panel = ret_dates.union(cap_dates)
    missing = idx_dates.difference(panel)
    extra = panel.difference(idx_dates)
    note(f"calendars: {len(ret_dates)} return dates, {len(cap_dates)} cap dates, "
         f"{len(idx_dates)} benchmark dates")
    note(f"  benchmark dates absent from the panel: {len(missing)}")
    note(f"  panel dates absent from the benchmark: {len(extra)}")
    bad = missing[missing >= IN_SCOPE]
    if len(bad):
        raise SystemExit(f"{len(bad)} benchmark dates in scope are missing from "
                         f"the panel, first few: {[str(d.date()) for d in bad[:5]]}")
    if len(missing):
        note(f"  (all outside the in-scope period, earliest "
             f"{missing.min().date()}, latest {missing.max().date()})")

    dates = panel.intersection(idx_dates)
    first = dates[dates >= IN_SCOPE][0]
    if dates.get_loc(first) == 0:
        raise SystemExit(f"first in-scope date {first.date()} has no preceding "
                         f"date, so it cannot be weighted on lagged caps")
    note(f"  first in-scope date {first.date()}, preceded by "
         f"{dates[dates.get_loc(first) - 1].date()}")
    return dates


def reconstruct(returns, caps, index, member):
    """
    Rebuild the index on every in-scope date, not a sample.

    The target is the cap-weighted total return of its own members, weighted by
    the previous day's capitalisations, so it should reproduce. Reported as a
    full distribution: a sampled check passes while hundreds of days are wrong.
    """
    dates = returns.index[returns.index >= IN_SCOPE]
    pos = returns.index.get_indexer(dates) - 1
    out = []
    for d, i in zip(dates, pos):
        prev = returns.index[i]
        w = caps.loc[prev].where(member.loc[d])
        r = returns.loc[d]
        ok = w.notna() & r.notna()
        den = float(w[ok].sum()) if ok.any() else 0.0
        ref = float(index.loc[d])
        if not ok.any() or not np.isfinite(den) or den <= 0:
            raise SystemExit(
                f"{d.date()}: cannot weight the cross-section "
                f"(usable names {int(ok.sum())}, weight total {den}). "
                f"An in-scope date must be reconstructible.")
        got = float((w[ok] * r[ok]).sum() / den)
        if not (np.isfinite(got) and np.isfinite(ref)):
            raise SystemExit(f"{d.date()}: non-finite benchmark "
                             f"(crsp {ref}, rebuilt {got})")
        out.append((d, ref, got, got - ref, int(ok.sum()),
                    int(member.loc[d].sum())))
    if len(out) != len(dates):
        raise SystemExit(f"expected {len(dates)} in-scope dates, got {len(out)}")
    # names_used < members means the cross-section was incomplete that day. It
    # does not identify why: an entrant on its first observation and an
    # established member with absent data both produce it.
    rec = pd.DataFrame(out, columns=["date", "crsp", "rebuilt", "diff",
                                     "names_used", "members"]).set_index("date")
    a = rec["diff"].abs()
    note(f"reconstruction over {len(rec)} in-scope dates:")
    note(f"  median |diff| {a.median():.2e}   p99 {a.quantile(0.99):.2e}   "
         f"max {a.max():.2e} ({a.max() * 1e4:.2f} bp)")
    for t in (1e-7, 1e-6, 1e-5, 1e-4):
        note(f"  days |diff| > {t:.0e}: {(a > t).sum()}")
    note(f"  annualised sd of residual {rec['diff'].std(ddof=1) * 252**0.5:.2e}, "
         f"mean {rec['diff'].mean() * 252:+.2e}")
    inc = rec.names_used < rec.members
    note(f"  dates with incomplete constituent coverage: {inc.sum()}")
    note(f"    |diff| on those dates: median {a[inc].median():.2e}, "
         f"max {a[inc].max():.2e}")
    note(f"    |diff| elsewhere:      median {a[~inc].median():.2e}, "
         f"max {a[~inc].max():.2e}")
    note("  the residual is not fully explained; see SPECIFICATION.md 11.4")
    worst = rec.loc[a.sort_values(ascending=False).index[:8]]
    note("  worst dates:")
    for r in worst.itertuples():
        note(f"    {r.Index.date()}  crsp {r.crsp:+.6f}  rebuilt {r.rebuilt:+.6f}  "
             f"diff {r.diff:+.2e}  used {r.names_used}/{r.members}")
    return rec


def completeness(returns, member):
    note("universe completeness (members with a return / members, by year):")
    for y in range(returns.index.year.min(), returns.index.year.max() + 1):
        d = returns.index[returns.index.year == y][-1]
        mem = int(member.loc[d].sum())
        held = int((member.loc[d] & returns.loc[d].notna()).sum())
        flag = "" if mem >= 490 else "   <- universe incomplete"
        note(f"  {y}  {held:3d} / {mem:3d}{flag}")


def main():
    df = deduplicate(load())

    numeric = df.DlyRet.notna() & ~df.DlyRetDurFlg.isin(MISSING_RETURN)
    one_day = numeric & ~df.DlyRetDurFlg.isin(NOT_DAILY)
    has_cap = df.DlyCap.notna()
    note(f"returns: {numeric.sum():,} numeric, of which "
         f"{(numeric & df.DlyRetDurFlg.isin(MULTI_DAY)).sum()} multi-period and "
         f"{(numeric & df.DlyRetDurFlg.isin(DELISTING_DURATION)).sum()} "
         f"delisting returns -- excluded from estimation, kept for accounting")
    note(f"capitalisations: {has_cap.sum():,} observed, of which "
         f"{(has_cap & ~numeric).sum()} on days with no usable return")

    returns = panel(df, "DlyRet", numeric)
    caps = panel(df, "DlyCap", has_cap)
    exceptions = df.loc[numeric & ~one_day,
                        ["PERMNO", "DlyCalDt", "DlyRetDurFlg", "DlyRet"]].copy()
    exceptions["reason"] = np.where(
        exceptions.DlyRetDurFlg.isin(MULTI_DAY), "multi-period return",
        "delisting return")

    terminal, gaps = classify_events(df)
    unresolved = payoffs_to_resolve(terminal)

    index = (pd.read_csv(INDEX, parse_dates=["DlyCalDt"])
               .set_index("DlyCalDt")["DlyTotRet"].rename("totret"))
    index.index.name = "date"

    dates = check_calendars(returns.index, caps.index, index.index)
    returns = returns.reindex(dates)
    caps = caps.reindex(dates).reindex(columns=returns.columns)
    index = index.loc[dates]
    note(f"panel: {returns.shape[0]} dates x {returns.shape[1]} securities")

    member = membership_matrix(dates, returns.columns)
    completeness(returns, member)
    rec = reconstruct(returns, caps, index, member)

    returns.to_csv(HERE / "returns.csv")
    caps.to_csv(HERE / "caps.csv")
    index.to_csv(HERE / "index.csv")
    member.to_csv(HERE / "membership.csv")
    terminal.to_csv(HERE / "terminal_events.csv", index=False)
    gaps.to_csv(HERE / "coverage_gaps.csv", index=False)
    unresolved.to_csv(HERE / "payoffs_to_resolve.csv", index=False)
    unresolved[["PERMNO", "DlyCalDt"]].to_csv(
        HERE / "unresolved_keys.csv", index=False)
    exceptions.to_csv(HERE / "estimation_exceptions.csv", index=False)
    rec.to_csv(HERE / "reconstruction.csv")
    (HERE / "build_report.txt").write_text("\n".join(lines) + "\n")
    note()
    note("returns.csv holds every numeric return, so portfolio accounting keeps "
         "real changes\nin wealth. estimation_exceptions.csv lists the cells that "
         "are not one-day\nobservations; the estimation code drops those, the "
         "accounting code does not.")


if __name__ == "__main__":
    main()
