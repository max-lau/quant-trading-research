# Quant Trading Research Project

An end-to-end, AI-driven quantitative trading pipeline, built as a structured
learning project covering the skills systematic funds screen for: data
engineering, statistics, backtesting discipline, ML alpha research, and risk
management.

**Status:** in progress — see milestones below.

## Why this project exists

Most portfolio projects in this space fail for one of two reasons: they skip
transaction costs (making every backtest look artificially good), or they
never address survivorship/look-ahead bias. This project treats both as
first-class concerns from the start, not an afterthought.

## Milestones

- [x] **M0 — Foundations**: environment, market structure notes, vectorization
  discipline (`notes/m0_vectorization_check.py`)
- [x] **M1 — Data engineering**: real OHLCV pipeline (`src/fetch_real_data.py`),
  a synthetic data generator with an injected delisting event for testing
  (`src/simulate_data.py`), automated data-quality checks
  (`src/data_quality.py`), and a quantified demonstration of survivorship bias
  (`notes/m1_survivorship_demo.py` — naive backtests overstated returns by
  ~119 points in the synthetic test case).
- [ ] **M2 — Statistics**: return distributions, fat tails, volatility
  clustering, stationarity.
- [ ] **M3 — Classic strategies & honest backtesting**: momentum, mean
  reversion, walk-forward validation, cost-adjusted performance.
- [ ] **M4 — ML alpha models**: engineered features, triple-barrier labeling,
  gradient-boosted trees, purged cross-validation, SHAP explainability.
- [ ] **M5 — NLP/LLM features**: sentiment and event extraction from
  unstructured text as model inputs.
- [ ] **M6 — Risk management**: position sizing, VaR/CVaR, drawdown control.
- [ ] **M7 — Execution**: paper trading via Alpaca, live monitoring.
- [ ] **M8 — Demo & writeup**: interactive dashboard, research summary.

## Setup

```bash
git clone <this-repo>
cd quant-project
pip install -r requirements.txt
python src/fetch_real_data.py   # pulls real OHLCV via yfinance -> data/real_ohlcv.csv
python src/data_quality.py      # coverage / gap / outlier report
```

## Known limitations (stated deliberately, not hidden)

- `fetch_real_data.py` uses yfinance's current-day ticker list — it does **not**
  correct for survivorship bias in a real universe (see `simulate_data.py`
  and the survivorship demo for how that bias is measured and why it matters).
- The business-day calendar used for gap detection doesn't account for
  exchange holidays precisely; a handful of flagged "gaps" are just market
  closures (Thanksgiving, Good Friday, etc.), not data defects.
- Data through 2025 is used for learning purposes; no forward-looking claims
  are made about live performance.

## Stack

Python, pandas, NumPy, scikit-learn, LightGBM (upcoming), yfinance, Alpaca
(upcoming for paper trading).
