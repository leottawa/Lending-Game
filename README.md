# Lending Game: Team HECredit (Lender 2)

HEC Paris, MSc Data Science & AI for Business 2026-27, *Introduction to Finance for Data Scientists*.
Team: Leo Burtin, Ion Goursaud de Merlis, Scott Mayette, Julia Rouzou.

We price 10,000-euro consumer loans against two rival lenders who see private signals we do not.
Round 1: 2nd of 12 lenders, EUR 62.5M profit, 38.4% market share. Round 2 submitted on 23 October 2026.
The method and results are in the report, [`report/HECredit_Stage2_report.pdf`](report/HECredit_Stage2_report.pdf).

## Contents

| Path | What it is |
| --- | --- |
| `stage1_pricing.py` | Round 1 pricing: spline GAM default model on all three signals, expected profit integrated over the rivals' unseen signals and a prior on their pricing |
| `stage2/stage2_pricing.py` | Round 2 model: best response to the rival offers observed in Round 1, winner's-curse regression on those offers, cross-fitted backtest (`--validate`) |
| `stage2/ch_model.py` | Level-k (cognitive hierarchy) prediction of the rivals' reaction, backtest of Table 2, and the submitted Round 2 offers (`--final`) |
| `stage2/analysis/` | Round 1 diagnosis and robustness scripts, with the logs of the backtests |
| `report/` | Stage 2 report: LaTeX source, PDF and figure |
| `submissions/` | Offers actually submitted: Round 1 (rebuilt from the diagnostic file) and Round 2 |
| `data/` | Course data (simulated): past loans, applications for both rounds, Round 1 diagnostic file |

## Reproduce

Requirements: Python 3 with pandas, numpy, scipy, scikit-learn (matplotlib for the figure). Run from the repository root.

```bash
python3 stage1_pricing.py data                     # Round 1 offers, written to data/
python3 stage2/stage2_pricing.py data --validate   # cross-fitted backtest on Round 1
python3 stage2/ch_model.py stage2 data             # level-k prediction and the backtest in Table 2
python3 stage2/ch_model.py stage2 data --final     # Round 2 offers, written to data/
```

The Stage 1 script reproduces the submitted Round 1 rates up to small numerical differences between library versions.
To rebuild the report, run `pdflatex HECredit_Stage2_report.tex` twice inside `report/`.
