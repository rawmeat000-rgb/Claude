#!/usr/bin/env python3
"""
Dual Momentum (relative + absolute) rotation strategy + honest validation.

Why this design (built directly from the momentum_strategy.py findings)
-----------------------------------------------------------------------
That test showed single-asset time-series momentum's real edge is NOT raw
return but DRAWDOWN CONTROL / surviving bear markets — and that a bull-only
backtest can't show it. Dual Momentum (Gary Antonacci) leans into exactly that:

    RELATIVE momentum : each rebalance, rank the universe and hold the STRONGEST
                        asset(s) — capture leadership, rotate as it changes.
    ABSOLUTE momentum : if the chosen asset's own trend is negative, hold CASH
                        instead — this is the bear-market circuit breaker.

So it profits from dispersion (picking the right coin) AND from stepping aside
in broad downturns. We therefore judge it on RISK-ADJUSTED terms — Sharpe,
Sortino, Calmar (return/maxDD) — versus an equal-weight buy-and-hold basket,
and we explicitly measure behaviour DURING bear markets.

This file: multi-asset data (full bull+bear cycles, synthetic fallback loudly
labeled), signals, rotation backtest, and validation — walk-forward, 30%
holdout, +/-25% sweep, block-bootstrap Monte Carlo, an explicit bear-market
regime test, and a --null control that must fail.

Usage
-----
    python dual_momentum.py                # single backtest
    python dual_momentum.py --run-all      # full validation suite
    python dual_momentum.py --run-all --null   # null control (no trend/dispersion)

Dependencies: numpy, pandas.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

DPY = 365
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ASSET_BETA = {"BTC": 1.0, "ETH": 1.15, "SOL": 1.5, "XRP": 1.25}
ASSET_START = {"BTC": 30000.0, "ETH": 2000.0, "SOL": 100.0, "XRP": 0.6}


@dataclass(frozen=True)
class Params:
    lookbacks: tuple = (30, 90, 180)   # blended momentum horizons, days
    top_n: int = 1                     # how many assets to hold
    rebalance_d: int = 7               # days between rebalances (weekly)
    abs_filter: bool = True            # absolute-momentum cash switch
    trend_len: int = 100               # SMA for absolute-momentum filter
    target_vol: float = 0.50           # portfolio vol target (annualized)
    vol_win: int = 30
    max_leverage: float = 1.0
    cash_yield: float = 0.02           # annual yield on cash (stable/T-bill proxy)
    fee_bps: float = 10.0
    slip_bps: float = 5.0


# ----------------------------------------------------------------------------
# Multi-asset data with FULL cycles (synthetic fallback, loudly labeled)
# ----------------------------------------------------------------------------

def make_panel(n_days: int, seed: int, trend_strength: float = 1.0) -> pd.DataFrame:
    """
    Wide daily close panel for the 4 assets. A shared crypto 'market' factor has
    regime-switching drift including SUSTAINED BEAR markets (so the absolute
    filter matters), plus persistent per-asset relative strength (so rotation
    matters). trend_strength=0 -> NULL CONTROL: no market trend, no persistent
    dispersion; dual momentum then has nothing to exploit and MUST fail.

    LOUD CAVEAT: structure is built in here; profit validates the pipeline, not
    a live edge.
    """
    rng = np.random.default_rng(seed)
    base_vol = 0.60 / np.sqrt(DPY)
    # shared market regime chain: 0=bull,1=chop,2=bear (persistent, real bears)
    trans = np.array([[0.975, 0.015, 0.010],
                      [0.020, 0.960, 0.020],
                      [0.030, 0.020, 0.950]])
    mkt_mu = np.array([0.9, 0.0, -1.2]) * base_vol * trend_strength
    state = 0
    mkt = np.empty(n_days)
    volm = 1.0
    for t in range(n_days):
        state = rng.choice(3, p=trans[state]) if t else 0
        volm = 0.9 * volm + 0.1 * (rng.random() * 1.4 + 0.5)
        mkt[t] = mkt_mu[state] + base_vol * volm * rng.standard_t(4) / np.sqrt(2.0)

    # persistent per-asset relative strength (AR(1)) -> rotation has signal
    rel = {a: 0.0 for a in ASSETS}
    prices = {a: [ASSET_START[a]] for a in ASSETS}
    for t in range(1, n_days):
        for a in ASSETS:
            rel[a] = 0.95 * rel[a] + 0.31 * rng.standard_normal()
            r = (ASSET_BETA[a] * mkt[t]
                 + rel[a] * 0.6 * base_vol * trend_strength
                 + base_vol * 0.7 * rng.standard_normal())
            prices[a].append(prices[a][-1] * np.exp(r))

    idx = pd.date_range(end=pd.Timestamp.now("UTC").normalize(), periods=n_days, freq="D")
    df = pd.DataFrame({a: prices[a] for a in ASSETS}, index=idx)
    df.attrs["source"] = f"SYNTHETIC(seed={seed}, trend={trend_strength})"
    return df


def load_panel(years: float = 8.0, source: str = "auto", trend_strength: float = 1.0,
               verbose: bool = True) -> pd.DataFrame:
    n_days = int(years * DPY)
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "panel_1d.csv")
    if source != "synthetic" and os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        df.attrs["source"] = "cache"
        if verbose:
            print(f"[data] {len(df)} daily bars x {len(ASSETS)} assets from cache")
        return df
    df = make_panel(n_days, seed=101, trend_strength=trend_strength)
    if verbose:
        print(f"[data] {len(df)} daily bars x {len(ASSETS)} assets {df.attrs['source']}")
        print("[data] SYNTHETIC — validates pipeline, NOT a live edge")
    return df


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------

def momentum_scores(prices: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Blended trailing-return momentum per asset (higher = stronger)."""
    score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for L in p.lookbacks:
        score = score + (prices / prices.shift(L) - 1.0)
    return score / len(p.lookbacks)


def abs_ok(prices: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Absolute-momentum gate: price above its own trend SMA."""
    sma = prices.rolling(p.trend_len, min_periods=p.trend_len // 2).mean()
    return prices > sma


# ----------------------------------------------------------------------------
# Backtest (weekly rotation, cash switch, vol targeting)
# ----------------------------------------------------------------------------

def _metrics(daily_ret: pd.Series, per_year=DPY) -> dict:
    r = daily_ret.fillna(0.0)
    eq = (1 + r).cumprod()
    tot = float(eq.iloc[-1] - 1)
    yrs = len(r) / per_year
    ann = float((1 + tot) ** (1 / yrs) - 1) if yrs > 0 else 0.0
    sd = r.std()
    sharpe = float(r.mean() / sd * np.sqrt(per_year)) if sd > 0 else 0.0
    downside = r[r < 0].std()
    sortino = float(r.mean() / downside * np.sqrt(per_year)) if downside > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    calmar = float(ann / abs(dd)) if dd < 0 else 0.0
    return {"ann": ann, "sharpe": sharpe, "sortino": sortino, "maxdd": dd,
            "calmar": calmar, "total": tot}


@dataclass
class BTResult:
    ret: pd.Series
    bench_ret: pd.Series
    m: dict
    bm: dict
    n_trades: int
    cash_frac: float

    def summary(self) -> str:
        return (f"annRet={self.m['ann']:+7.2%}  sharpe={self.m['sharpe']:+5.2f}  "
                f"sortino={self.m['sortino']:+5.2f}  maxDD={self.m['maxdd']:7.2%}  "
                f"calmar={self.m['calmar']:5.2f}  cash={self.cash_frac:4.0%}  "
                f"|| B&H sharpe={self.bm['sharpe']:+5.2f} maxDD={self.bm['maxdd']:7.2%} "
                f"calmar={self.bm['calmar']:4.2f}")

    def beats_bh(self) -> bool:
        # the honest bar: better risk-adjusted return AND shallower drawdown
        return self.m["sharpe"] > self.bm["sharpe"] and self.m["maxdd"] > self.bm["maxdd"]


def backtest(prices: pd.DataFrame, p: Params) -> BTResult:
    scores = momentum_scores(prices, p)
    gate = abs_ok(prices, p)
    rets = prices.pct_change()
    ew_bench = rets.mean(axis=1)                       # equal-weight buy&hold
    cash_daily = (1 + p.cash_yield) ** (1 / DPY) - 1

    # portfolio vol scalar from equal-weight realized vol
    rv = ew_bench.rolling(p.vol_win, min_periods=p.vol_win // 2).std() * np.sqrt(DPY)
    vol_scalar = (p.target_vol / rv.replace(0, np.nan)).clip(upper=p.max_leverage)

    dates = prices.index
    weights = pd.DataFrame(0.0, index=dates, columns=list(prices.columns) + ["CASH"])
    cur = pd.Series(0.0, index=list(prices.columns) + ["CASH"])
    n_trades = 0
    warm = max(p.lookbacks) + 1

    for i, d in enumerate(dates):
        if i >= warm and i % p.rebalance_d == 0:
            s = scores.loc[d]
            ranked = s.sort_values(ascending=False)
            picks = list(ranked.index[:p.top_n])
            new = pd.Series(0.0, index=cur.index)
            slot = (vol_scalar.loc[d] if not np.isnan(vol_scalar.loc[d]) else 0.0) / p.top_n
            for a in picks:
                if p.abs_filter and not bool(gate.loc[d, a]):
                    new["CASH"] += slot                # circuit breaker -> cash
                else:
                    new[a] += slot
            new["CASH"] += max(0.0, 1.0 - slot * p.top_n)  # residual to cash
            if (new - cur).abs().sum() > 1e-9:
                n_trades += 1
            cur = new
        weights.loc[d] = cur.values

    held = weights.shift(1).fillna(0.0)                # next-day execution
    asset_pnl = (held[list(prices.columns)] * rets).sum(axis=1)
    cash_pnl = held["CASH"] * cash_daily
    turn = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = turn * (p.fee_bps + p.slip_bps) / 1e4
    port = asset_pnl + cash_pnl - cost

    cash_frac = float(held["CASH"].mean())
    return BTResult(port, ew_bench, _metrics(port), _metrics(ew_bench), n_trades, cash_frac)


def run(prices: pd.DataFrame, p: Params) -> BTResult:
    return backtest(prices, p)


# ----------------------------------------------------------------------------
# Validation harness
# ----------------------------------------------------------------------------

GRID = {"trend_len": [50, 100, 150], "top_n": [1, 2], "target_vol": [0.40, 0.50, 0.60]}
MIN_TRADES = 8


def _grid(base: Params):
    for tl in GRID["trend_len"]:
        for tn in GRID["top_n"]:
            for tv in GRID["target_vol"]:
                yield replace(base, trend_len=tl, top_n=tn, target_vol=tv)


def _score(r: BTResult) -> float:
    return (r.m["sharpe"] - r.bm["sharpe"]) if r.n_trades >= MIN_TRADES else -np.inf


def walk_forward(prices: pd.DataFrame, base: Params, n_folds: int = 5, verbose=True):
    n = len(prices)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    rows, picks = [], []
    for k in range(1, n_folds):
        train, test = prices.iloc[:edges[k]], prices.iloc[edges[k]:edges[k + 1]]
        best, best_p = -np.inf, base
        for cand in _grid(base):
            s = _score(run(train, cand))
            if s > best:
                best, best_p = s, cand
        r = run(test, best_p)
        picks.append(best_p)
        rows.append({"fold": k, "picked": f"tl={best_p.trend_len},top={best_p.top_n},tv={best_p.target_vol}",
                     "oos_sharpe": round(r.m["sharpe"], 2), "bh_sharpe": round(r.bm["sharpe"], 2),
                     "oos_calmar": round(r.m["calmar"], 2), "beats_bh": r.beats_bh()})
        if verbose:
            print(f"[wf] fold {k}: {rows[-1]['picked']}  OOS {r.summary()}")
    keys = [(q.trend_len, q.top_n, q.target_vol) for q in picks]
    modal = max(set(keys), key=keys.count)
    consensus = replace(base, trend_len=modal[0], top_n=modal[1], target_vol=modal[2])
    excess = [r["oos_sharpe"] - r["bh_sharpe"] for r in rows]
    verdict = ("EDGE DIES: no risk-adjusted edge over B&H OOS (median excess Sharpe <= 0)"
               if np.median(excess) <= 0 else
               "fragile: some folds fail to beat B&H" if min(excess) <= 0 else
               "survives: beats B&H risk-adjusted across folds")
    return pd.DataFrame(rows), consensus, verdict


def param_sweep(prices: pd.DataFrame, p: Params, verbose=True) -> pd.DataFrame:
    specs = {
        "trend_len": lambda f: replace(p, trend_len=max(10, int(round(p.trend_len * f)))),
        "target_vol": lambda f: replace(p, target_vol=p.target_vol * f),
        "rebalance_d": lambda f: replace(p, rebalance_d=max(1, int(round(p.rebalance_d * f)))),
        "lookbacks": lambda f: replace(p, lookbacks=tuple(max(5, int(round(L * f))) for L in p.lookbacks)),
    }
    br = run(prices, p)
    rows = [{"param": "(base)", "mult": 1.0, "sharpe": round(br.m["sharpe"], 2),
             "excess": round(br.m["sharpe"] - br.bm["sharpe"], 2), "calmar": round(br.m["calmar"], 2)}]
    for name, mk in specs.items():
        for f in (0.75, 1.25):
            r = run(prices, mk(f))
            rows.append({"param": name, "mult": f, "sharpe": round(r.m["sharpe"], 2),
                         "excess": round(r.m["sharpe"] - r.bm["sharpe"], 2),
                         "calmar": round(r.m["calmar"], 2)})
    out = pd.DataFrame(rows)
    if verbose:
        print(out.to_string(index=False))
    return out


def bear_test(prices: pd.DataFrame, p: Params, verbose=True) -> dict:
    """Behaviour when the equal-weight basket is in a >20% drawdown."""
    r = run(prices, p)
    bench_eq = (1 + r.bench_ret.fillna(0)).cumprod()
    in_bear = (bench_eq / bench_eq.cummax() - 1) < -0.20
    if in_bear.sum() < 20:
        return {"note": "insufficient bear days in this dataset"}
    strat_bear = float((1 + r.ret[in_bear]).prod() - 1)
    bench_bear = float((1 + r.bench_ret[in_bear]).prod() - 1)
    out = {"bear_days": int(in_bear.sum()), "strat_ret_in_bear": strat_bear,
           "bench_ret_in_bear": bench_bear, "avg_cash_in_bear": float(0.0)}
    if verbose:
        print(f"   bear days: {out['bear_days']}  |  strategy {strat_bear:+.1%}  "
              f"vs basket {bench_bear:+.1%} during bear markets")
    return out


def monte_carlo(daily_ret: pd.Series, n_runs=1000, seed=0) -> dict:
    r = daily_ret.dropna().to_numpy()
    if len(r) < 30:
        return {"error": "too few obs"}
    rng = np.random.default_rng(seed)
    block, n = 14, len(r)
    finals, dds = np.empty(n_runs), np.empty(n_runs)
    for i in range(n_runs):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n - block)
            idx.extend(range(s, s + block))
        x = r[np.array(idx[:n])]
        eq = np.cumprod(1 + x)
        finals[i] = eq[-1] - 1
        dds[i] = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"runs": n_runs, "median_return": float(np.median(finals)),
            "p05_return": float(np.percentile(finals, 5)),
            "prob_loss": float((finals <= 0).mean()),
            "median_maxdd": float(np.median(dds))}


def run_all(source: str, years: float, trend_strength: float) -> list[str]:
    print("=" * 80)
    print("DUAL MOMENTUM (relative + absolute) — FULL VALIDATION SUITE")
    if trend_strength == 0.0:
        print("NULL CONTROL: no market trend, no persistent dispersion. Must fail.")
    print("=" * 80)
    base = Params()
    failures = []

    prices = load_panel(years=years, source=source, trend_strength=trend_strength)
    if "SYNTHETIC" in prices.attrs.get("source", ""):
        print("\n*** DATA IS SYNTHETIC (network-restricted). Validates pipeline, ***")
        print("*** not the live edge. Swap in real daily bars for a real verdict.***\n")

    split = int(len(prices) * 0.70)
    ins, hold = prices.iloc[:split], prices.iloc[split:]
    print(f"[split] in-sample {ins.index[0]:%Y-%m-%d}..{ins.index[-1]:%Y-%m-%d} | "
          f"holdout {hold.index[0]:%Y-%m-%d}..{hold.index[-1]:%Y-%m-%d}")

    print("\n--- 0. Baseline (in-sample) vs equal-weight buy&hold -----------------")
    br = run(ins, base)
    print("   " + br.summary())
    print(f"   beats B&H (better Sharpe AND drawdown)? {br.beats_bh()}")
    if not br.beats_bh():
        failures.append("baseline: no risk-adjusted edge over equal-weight B&H")

    print("\n--- 1. Walk-forward, 5 folds (metric = excess Sharpe over B&H) -------")
    wf, tuned, verdict = walk_forward(ins, base)
    print(f"[wf] consensus: trend_len={tuned.trend_len}, top_n={tuned.top_n}, target_vol={tuned.target_vol}")
    print(f"[wf] verdict: {verdict}")
    if "DIES" in verdict:
        failures.append(f"walk-forward: {verdict}")

    print("\n--- 2. 30% out-of-sample holdout (tuned, touched once) ---------------")
    hr = run(hold, tuned)
    print("   " + hr.summary())
    if not hr.beats_bh():
        failures.append(f"holdout: no risk-adjusted edge (sharpe {hr.m['sharpe']:+.2f} vs B&H {hr.bm['sharpe']:+.2f})")
        print("   >>> NO RISK-ADJUSTED EDGE ON HOLDOUT <<<")
    if hr.m["ann"] <= 0:
        failures.append(f"holdout: negative absolute return ({hr.m['ann']:+.2%})")

    print("\n--- 3. Parameter sweep +/-25% (in-sample) ----------------------------")
    sw = param_sweep(ins, tuned)
    neg = int((sw["excess"] <= 0).sum())
    if neg:
        failures.append(f"sweep: {neg}/{len(sw)} perturbations lack edge over B&H")
        print(f"   >>> {neg}/{len(sw)} perturbations fail to beat B&H <<<")
    else:
        print("   all +/-25% perturbations keep positive excess Sharpe")

    print("\n--- 4. Bear-market regime test (the reason to run momentum) ----------")
    bt = bear_test(hold, tuned)
    if "strat_ret_in_bear" in bt and bt["strat_ret_in_bear"] < bt["bench_ret_in_bear"]:
        failures.append("bear test: strategy did WORSE than basket during bear markets")
        print("   >>> FAILED: strategy did not protect in bear markets <<<")

    print("\n--- 5. Monte Carlo: 1000 block-bootstrap resamples (holdout) ---------")
    mc = monte_carlo(hr.ret, 1000, seed=42)
    for k, v in mc.items():
        print(f"   {k:>15}: {v:.4f}" if isinstance(v, float) else f"   {k:>15}: {v}")
    if isinstance(mc.get("prob_loss"), float) and mc["prob_loss"] > 0.30:
        failures.append(f"monte carlo: prob(loss)={mc['prob_loss']:.0%} > 30%")
        print("   >>> MONTE CARLO: HIGH LOSS PROBABILITY <<<")

    print("\n" + "=" * 80)
    print("VERDICT — tests where the edge dies or is fragile (unfiltered):")
    if failures:
        for f in failures:
            print(f"  X {f}")
    else:
        print("  (none — dual momentum survived every test on this dataset)")
    print("=" * 80)
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--null", action="store_true", help="null control: no trend/dispersion")
    ap.add_argument("--source", default="auto", choices=["auto", "synthetic"])
    ap.add_argument("--years", type=float, default=8.0)
    args = ap.parse_args()

    trend = 0.0 if args.null else 1.0
    if args.run_all:
        run_all(args.source, args.years, trend)
    else:
        prices = load_panel(years=args.years, source=args.source, trend_strength=trend)
        r = run(prices, Params())
        print("Dual momentum, default params, full sample:")
        print("   " + r.summary())
        print(f"   beats equal-weight B&H (Sharpe AND drawdown)? {r.beats_bh()}")


if __name__ == "__main__":
    main()
