# Deep learning for partial index tracking

Code for a master's thesis on tracking the S&P 500 with a small number of stocks.

- **Part 1** replicates the autoencoder-based smart-indexing example of Heaton,
  Polson and Witte, re-estimates it with cross-validated model selection (Phase A),
  and separates the contribution of stock selection from that of the mapping.
- **Part 2** selects stocks that jointly reproduce factors from a
  capitalisation-weighted autoencoder or weighted PCA, and weights them with a
  long-only quadratic programme.

All models are written in NumPy.

```
part1_replication_phaseA/
  crsp_panel.py              CRSP extract -> panel_total.csv (total-return levels)
  hpw_replication_total.py   replication of the HPW example
  phaseA_deep_frontier.py    Phase A
  ablation_phaseA.py         selection rules x mappings
part2_cap_weighted/
  build_panel.py             CRSP extracts -> returns, caps, membership, index
  extension.py               representations, selection, QP, backtest stages
  frequency_sensitivity.py   reweighting and reselection variants
```

## Data

The data come from CRSP and require access to the relevant CRSP datasets through
WRDS. Neither the licensed raw data nor the panels derived from them are included.
Put each file in the folder of the script that reads it.

**Part 1**

- CRSP daily stock file, June 2009 to December 2015, S&P 500 constituents on
  4 January 2010, any file name: `PERMNO, Ticker, DlyCalDt, DlyRet`, plus
  `DlyPrc, DlyRetx, DlyFacPrc, DisDivAmt` for the consistency check.
- `Daily cap.csv`, same securities and period: `PERMNO, Ticker, DlyCalDt, DlyCap`.

The benchmark is the S&P 500 total-return index (`^SP500TR`), downloaded from Yahoo
Finance by `crsp_panel.py`. The option `--price-return` (which also needs `sprtrn`)
is not part of the thesis workflow.

**Part 2**

- `daily_crsp_2005_2025.csv`: CRSP daily stock file, 2005-2025, for every security
  that was an S&P 500 member at some point in 2010-2025: `PERMNO, DlyCalDt, DlyRet,
  DlyCap, DlyPrc, DlyPrcFlg, TradingStatusFlg, DlyRetMissFlg, DlyRetDurFlg,
  DlyDistRetFlg`.
- `membership_intervals.csv`: `PERMNO, MbrStartDt, MbrEndDt`.
- `index_crsp_sp500_totret.csv`: CRSP value-weighted index of the S&P 500 universe,
  `DlyCalDt, DlyTotRet`.
- `crsp_delisting_full.csv`: CRSP delisting table for the same securities:
  `permno, delistingdt, deldtprc, deldtprcflg, delpaymenttype, delret,
  delretmisstype, delnextdt, delnextprc, delnextprcflg, delamtdt, deldivamt,
  deldistype, deldlydt`.

The Part 2 benchmark is CRSP's value-weighted index, not the official S&P 500.

## Requirements

```
pip install -r requirements.txt
```

The requirements give minimum versions only; the exact environment of the thesis
runs was not recorded. The model scripts force single-threaded linear algebra, so
a run repeats exactly on the same machine and library versions; other versions
can give slightly different results.

## Reproducing the results

Run times below are approximate.

### Part 1

```
cd part1_replication_phaseA
python crsp_panel.py --csv "<daily stock extract>.csv"
python hpw_replication_total.py

python phaseA_deep_frontier.py --ae-beta 0 --n 60 --max-size 60
python phaseA_deep_frontier.py --n 60 --max-size 60
python ablation_phaseA.py --n 60 --ae-beta 0 --phaseA-dir phaseA_outputs_v2_beta0_n60_max60
python ablation_phaseA.py --n 60 --phaseA-dir phaseA_outputs_v2_n60_max60
```

The first Phase A run is the main specification (no sparsity penalty); the second
uses HPW's sparsity weight. Sixty stocks were chosen by inspecting the validation
frontier. Each Phase A run and each ablation takes several hours and resumes from
its saved progress if rerun with the same command.

### Part 2

```
cd part2_cap_weighted
python build_panel.py
python extension.py --stage test
python extension.py --stage converge
python extension.py --stage develop  --learning-rate 1e-3 --epochs 4000
python extension.py --stage develop  --learning-rate 1e-3 --epochs 4000 --lambdas 3e-3 1e-2
python extension.py --stage validate --learning-rate 1e-3 --epochs 4000 --ae-weight-penalty 1e-3
python extension.py --stage final    --learning-rate 1e-3 --epochs 4000 --ae-weight-penalty 1e-3 --spec-frozen
python frequency_sensitivity.py --period validate
python frequency_sensitivity.py --period final
```

- `converge` writes loss diagnostics for three learning rates to
  `outputs_converge/`; the learning rate and epochs were chosen from them by hand.
  Completed fits are reused if it is rerun.
- `develop` reports portfolio performance for each candidate weight penalty in
  `outputs_develop/`; with `--lambdas` the extra candidates go to
  `outputs_develop_lambda-ext/`. The two runs are not merged; the penalty was
  chosen by comparing them and is passed explicitly to the next commands.
- `validate` selects xi for each method and budget and saves it in
  `outputs_validate/selected_policy.json`. `final` runs only those specifications.
  `--spec-frozen` records the intention to evaluate the final period once; the flag
  cannot itself show that results were unseen.
- `develop`, `validate` and `final` resume completed formation dates and write
  selections, weights, daily returns and summaries to `outputs_<stage>/`;
  `validate` and `final` take a few hours each. `test` and `time` save nothing.
- `frequency_sensitivity.py` needs the corresponding `outputs_<period>/`, writes
  `outputs_frequency_<period>/` and does not resume.

Several scripts store hashes of their own source and data. The comments in this
version were edited after the thesis runs, so its hashes differ from those saved
with the original outputs; the calculations are unchanged.

## License

The code is released under the MIT licence (see `LICENSE`). The licence does not
cover CRSP or any other third-party data.
