# BTC 15m Order-Flow Mean Reversion — Validation Report

**Deliverable:** [`btc_mean_reversion.py`](btc_mean_reversion.py) — one file containing the
data loader, signal logic, position sizing, backtest, and the full validation harness.

```bash
pip install numpy pandas
python btc_mean_reversion.py --run-all              # real data if network allows, else synthetic
python btc_mean_reversion.py --run-all --source synthetic --mr-kappa 0.0   # null control
```

## Strategy

- **Signal:** z-score of taker buy/sell flow imbalance (window 96 bars) + z-score of price
  stretch from a 96-bar EMA anchor (std over 192 bars).
- **Entry:** stretch z ≤ −z_entry **and** imbalance z ≤ −imb_entry → long the flush
  (symmetric short on blow-offs).
- **Exit:** stretch z reverts through 0, or blow-out stop at |z| ≥ 4, or 96-bar time stop.
- **Sizing:** 40% annualized vol target from trailing 96-bar realized vol, capped at 1×.
- **Execution:** signals at bar close, position applied to the *next* bar (no lookahead);
  7.5 bps per side (fee + slippage), i.e. 15 bps round trip.

## ⚠️ Critical caveat: data provenance

This session ran in a network-restricted environment. **Every market-data host was blocked
by policy** (Binance REST, Binance Vision, Bybit, OKX, Kraken, Coinbase, CryptoCompare —
all returned proxy CONNECT denials). The loader implements and prefers those real sources
with CSV caching, but here it fell through to a clearly-labeled **synthetic generator**
(regime-switching vol, Student-t(4) fat tails, AR(1) order flow contemporaneously coupled
to returns, and a configurable ornstein-uhlenbeck mean-reversion strength `mr_kappa`).

**Therefore no result below is evidence of a real-world edge.** The runs validate that the
pipeline and harness are correct, honest, and able to detect a dead edge. To get real
answers, run `--run-all` on a machine with exchange access (or drop Binance 15m kline CSVs
into `data/<ASSET>_15m.csv`).

Two further honesty notes about the thesis itself, independent of environment:

1. The requested signal was **order-book imbalance**. Free historical L2 book snapshots do
   not exist at 2-year depth; the implementation uses **taker buy/sell flow imbalance**
   (available in Binance klines) as the standard proxy, and an intrabar-pressure proxy on
   Bybit. That is a weaker signal than true book imbalance and this substitution is itself
   a risk to the thesis.
2. 15 bps round trip is realistic for a taker on spot; at 15m frequency costs are the most
   likely killer of this edge on real data. The null control below shows exactly what that
   death looks like.

## Run A — synthetic with weak built-in mean reversion (`mr_kappa=0.012`)

Pipeline demonstration. The generator embeds mean reversion by construction, so passing is
expected and **claims nothing** about real BTC.

| Test | Result |
|---|---|
| Baseline (in-sample, default params) | Sharpe +3.72, +218.9%, 465 trades, 67.1% win, maxDD −9.7% |
| Walk-forward (5 anchored folds) | OOS Sharpe +4.88 / +2.94 / +2.39 / +3.29 → survives |
| 30% holdout (touched once) | Sharpe +3.46, +75.6%, 293 trades, 71.0% win, maxDD −8.8% |
| ±25% parameter sweep (11 perturbations) | All positive Sharpe (3.69–4.24) |
| Cross-asset ETH / SOL / XRP | Sharpe +5.36 / +5.12 / +4.88 |
| Monte Carlo (1000 resamples, holdout trades) | median +106.7%, p05 +49.0%, prob(loss) 0.0% |

Consensus walk-forward params: anchor=96, z_entry=1.5, imb_entry=0.5.

## Run B — null control (`mr_kappa=0.0`): **the edge dies everywhere, as it must**

Same harness, synthetic data with the mean-reversion component removed (pure
regime-switching random walk). A harness that cannot kill a false edge is worthless;
this one kills it in **all five tests**, unfiltered:

| Test | Result |
|---|---|
| Baseline (in-sample) | Sharpe −1.31, −39.5%, maxDD −48.4% |
| Walk-forward | **EDGE DIES** — all folds negative OOS (−0.59, −1.37, −0.26, −2.37); median ≤ 0 |
| 30% holdout | **EDGE DIES** — Sharpe −1.37, −11.3%, 33.3% win |
| ±25% sweep | **9/11 perturbations Sharpe ≤ 0** (the 2 positives are noise, not robustness) |
| Cross-asset | **EDGE DIES on ETH (−0.71), SOL (−0.47), XRP (−1.40)** |
| Monte Carlo | **prob(total loss) = 81%**, median −10.1%, p05 −23.6% |

Note the walk-forward optimizer still found train Sharpes up to +2.28 on noise — classic
overfitting that the OOS folds then exposed. That is the harness doing its job.

## Bottom line

- The code is complete and runnable; the harness demonstrably detects both a live edge and
  a dead one, and hides nothing.
- **No claim of a real BTC edge is made or possible from this environment.** On real data,
  the prior should be skeptical: 15m mean reversion with a 15 bps round trip is exactly
  the regime where costs and the flow-proxy substitution kill edges — rerun Run A's
  command with network access and believe whatever the verdict block prints.
