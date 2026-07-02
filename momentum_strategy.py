#!/usr/bin/env python3
"""
Time-series momentum strategy + honest validation harness, in one file.

Edge thesis
-----------
Assets that have gone up over the recent past tend to keep going up (and vice
versa) — the most robust, most-replicated anomaly across markets and decades
(Moskowitz-Ooi-Pedersen "Time Series Momentum"; Antonacci "Dual Momentum").
Its retail advantage: it trades INFREQUENTLY, so realistic costs don't eat it.

Best-practice design baked in
-----------------------------
1. BLENDED lookbacks (1/3/6/12 months) — not one overfit number. Signal = mean
   of the sign of each trailing return (optionally skipping the most recent
   month to dodge short-term reversal).
2. ABSOLUTE-momentum trend filter — only hold long when trend is positive; go
   flat otherwise. This is momentum's crash protection and most of its Sharpe.
3. VOLATILITY TARGETING for sizing — scale exposure to a target annual vol.
4. DAILY bars — momentum is real daily/weekly, noise-and-fees intraday.
5. Benchmarked against BUY-AND-HOLD — the honest bar. Momentum usually wins on
   risk-adjusted return and drawdown, not necessarily raw return in a bull run.

Contents: data loader (real sources -> synthetic fallback, loudly labeled),
signals, sizing, backtest (next-bar exec, per-side costs), and validation:
walk-forward, 30% holdout, +/-25% sweep, cross-asset ETH/SOL/XRP, 1000-run
Monte Carlo, buy-and-hold benchmark, and a --null control that must fail.

Usage
-----
    python momentum_strategy.py                # single backtest, BTC
    python momentum_strategy.py --run-all      # full validation suite
    python momentum_strategy.py --run-all --null   # null control (no trend)

Dependencies: numpy, pandas.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

DPY = 365  # crypto trades daily, 365/yr
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ASSET_SPECS = {  # symbol -> (synthetic ann vol, synthetic seed, start price)
    "BTC": (0.60, 11, 30000.0),
    "ETH": (0.75, 22, 2000.0),
    "SOL": (1.00, 33, 100.0),
    "XRP": (0.85, 44, 0.6),
}


@dataclass(frozen=True)
class Params:
    lookbacks: tuple = (21, 63, 126, 252)  # 1/3/6/12 months (trading-ish days)
    skip: int = 0                # skip most-recent N days (equities use ~21)
    entry_thresh: float = 0.0    # need blended signal above this to hold
    trend_len: int = 100         # absolute-momentum SMA filter length
    target_vol: float = 0.40     # annualized vol target for sizing
    vol_win: int = 30            # realized-vol lookback (days)
    max_leverage: float = 1.0    # cap on notional
    long_short: bool = False     # False = long/flat (retail-honest default)
    fee_bps: float = 10.0        # per-side fee
    slip_bps: float = 5.0        # per-side slippage


# ----------------------------------------------------------------------------
# Data loader (real -> synthetic fallback)
# ----------------------------------------------------------------------------

def make_synthetic(symbol: str, n_days: int, seed: int, ann_vol: float,
                   start: float, trend_strength: float = 1.0) -> pd.DataFrame:
    """
    Daily series with regime-switching DRIFT (bull/chop/bear) so momentum has a
    real trend to ride, plus Student-t fat tails and vol regimes.

    trend_strength=0 -> NULL CONTROL: drift is ~0 (random walk); momentum then
    has nothing to capture and a correct harness MUST report the edge dying.

    LOUD CAVEAT: the trend is built in here, so a profitable result validates the
    pipeline, not the live edge.
    """
    rng = np.random.default_rng(seed)
    sig = ann_vol / np.sqrt(DPY)
    # slow drift-regime chain: 0=bull(+), 1=chop(0), 2=bear(-), persistent
    trans = np.array([[0.97, 0.02, 0.01],
                      [0.02, 0.96, 0.02],
                      [0.02, 0.03, 0.95]])
    mu = np.array([0.9, 0.0, -1.1]) * sig * trend_strength  # per-day drift by regime
    state = 0
    logp = np.empty(n_days)
    logp[0] = np.log(start)
    vol_mult = 1.0
    for t in range(1, n_days):
        state = rng.choice(3, p=trans[state])
        vol_mult = 0.9 * vol_mult + 0.1 * (rng.random() * 1.6 + 0.4)
        shock = sig * vol_mult * rng.standard_t(4) / np.sqrt(2.0)
        logp[t] = logp[t - 1] + mu[state] + shock

    close = np.exp(logp)
    open_ = np.empty(n_days); open_[0] = close[0]; open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0, 0.5 * sig, n_days)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    idx = pd.date_range(end=pd.Timestamp.now("UTC").normalize(), periods=n_days, freq="D")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                      index=idx)
    df.attrs["source"] = f"SYNTHETIC(seed={seed}, trend={trend_strength})"
    return df


def load_data(asset: str, years: float = 8.0, source: str = "auto",
              trend_strength: float = 1.0, verbose: bool = True) -> pd.DataFrame:
    """Cache -> (real loaders would go here) -> synthetic. Network is blocked in
    this environment, so real sources are stubbed and we fall through."""
    ann_vol, seed, start = ASSET_SPECS[asset]
    n_days = int(years * DPY)
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"{asset}_1d.csv")
    if source != "synthetic" and os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        df.attrs["source"] = "cache"
        if verbose:
            print(f"[data] {asset}: {len(df)} daily bars from cache")
        return df
    df = make_synthetic(asset, n_days, seed, ann_vol, start, trend_strength)
    if verbose:
        print(f"[data] {asset}: {len(df)} daily bars {df.attrs['source']} "
              f"— validates pipeline, NOT the live edge")
    return df


# ----------------------------------------------------------------------------
# Signals + sizing
# ----------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    # blended time-series momentum: mean of sign(trailing return) across lookbacks
    sig = pd.Series(0.0, index=close.index)
    valid = pd.Series(True, index=close.index)
    for L in p.lookbacks:
        past = close.shift(p.skip)
        ref = close.shift(p.skip + L)
        mom = past / ref - 1.0
        sig = sig + np.sign(mom).fillna(0.0)
        valid &= mom.notna()
    sig = sig / len(p.lookbacks)            # in [-1, 1]
    out["mom"] = sig.where(valid)

    # absolute-momentum trend filter (price above its own SMA)
    sma = close.rolling(p.trend_len, min_periods=p.trend_len // 2).mean()
    out["trend_ok"] = (close > sma)

    # vol-targeted sizing
    ret = close.pct_change()
    rv = ret.rolling(p.vol_win, min_periods=p.vol_win // 2).std() * np.sqrt(DPY)
    out["vol_scalar"] = (p.target_vol / rv.replace(0, np.nan)).clip(upper=p.max_leverage)
    out["ret"] = ret
    return out


def target_position(sig: pd.DataFrame, p: Params) -> pd.Series:
    mom = sig["mom"].to_numpy()
    trend_ok = sig["trend_ok"].to_numpy()
    vs = np.nan_to_num(sig["vol_scalar"].to_numpy(), nan=0.0)
    pos = np.zeros(len(sig))
    for t in range(len(sig)):
        m = mom[t]
        if np.isnan(m):
            continue
        if m > p.entry_thresh and trend_ok[t]:
            pos[t] = m * vs[t]                         # long, conviction-scaled
        elif p.long_short and m < -p.entry_thresh and not trend_ok[t]:
            pos[t] = m * vs[t]                         # short
        # else flat
    return pd.Series(np.clip(pos, -p.max_leverage, p.max_leverage), index=sig.index)


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------

@dataclass
class BTResult:
    equity: pd.Series
    bh_equity: pd.Series
    total_return: float
    ann_return: float
    sharpe: float
    max_dd: float
    n_trades: int
    win_rate: float
    exposure: float
    bh_ann: float
    bh_sharpe: float
    bh_max_dd: float

    def summary(self) -> str:
        return (f"annRet={self.ann_return:+7.2%}  sharpe={self.sharpe:+5.2f}  "
                f"maxDD={self.max_dd:7.2%}  trades={self.n_trades:4d}  "
                f"win={self.win_rate:5.1%}  expo={self.exposure:5.1%}  ||  "
                f"B&H annRet={self.bh_ann:+7.2%} sharpe={self.bh_sharpe:+5.2f} "
                f"maxDD={self.bh_max_dd:7.2%}")

    def beats_bh(self) -> bool:
        return self.sharpe > self.bh_sharpe and self.max_dd > self.bh_max_dd


def backtest(sig: pd.DataFrame, p: Params) -> BTResult:
    pos = target_position(sig, p)
    ret = np.nan_to_num(sig["ret"].to_numpy(), nan=0.0)
    pos_arr = pos.to_numpy()
    held = np.concatenate([[0.0], pos_arr[:-1]])          # next-bar execution
    cost = (p.fee_bps + p.slip_bps) / 1e4
    turn = np.abs(np.concatenate([[0.0], np.diff(pos_arr)]))
    pnl = held * ret - turn * cost
    equity = pd.Series((1 + pnl).cumprod(), index=sig.index)
    bh = pd.Series((1 + ret).cumprod(), index=sig.index)

    # trades = contiguous nonzero-position episodes
    trades, in_pos, entry_i = [], False, 0
    for t in range(len(pos_arr)):
        if not in_pos and pos_arr[t] != 0:
            in_pos, entry_i = True, t
        elif in_pos and (pos_arr[t] == 0 or t == len(pos_arr) - 1):
            seg = pnl[entry_i + 1: t + 1]
            trades.append(float(np.prod(1 + seg) - 1))
            in_pos = False
    n = len(sig)
    yrs = n / DPY

    def stats(eq, series_pnl):
        tot = float(eq.iloc[-1] - 1)
        ann = float((1 + tot) ** (1 / yrs) - 1) if yrs > 0 else 0.0
        sd = series_pnl.std()
        shp = float(series_pnl.mean() / sd * np.sqrt(DPY)) if sd > 0 else 0.0
        dd = float((eq / eq.cummax() - 1).min())
        return tot, ann, shp, dd

    tot, ann, shp, dd = stats(equity, pd.Series(pnl))
    _, bh_ann, bh_shp, bh_dd = stats(bh, pd.Series(ret))
    nt = len(trades)
    wr = float(np.mean([t > 0 for t in trades])) if nt else 0.0
    expo = float((held != 0).mean())
    return BTResult(equity, bh, tot, ann, shp, dd, nt, wr, expo, bh_ann, bh_shp, bh_dd)


def run(df: pd.DataFrame, p: Params) -> BTResult:
    return backtest(compute_signals(df, p), p)


# ----------------------------------------------------------------------------
# Validation harness
# ----------------------------------------------------------------------------

GRID = {
    "trend_len": [50, 100, 150],
    "target_vol": [0.30, 0.40, 0.50],
    "skip": [0, 21],
}
MIN_TRADES = 8


def _grid(base: Params):
    for tl in GRID["trend_len"]:
        for tv in GRID["target_vol"]:
            for sk in GRID["skip"]:
                yield replace(base, trend_len=tl, target_vol=tv, skip=sk)


def _score(r: BTResult) -> float:
    # reward risk-adjusted excess over buy&hold; require enough trades
    return (r.sharpe - r.bh_sharpe) if r.n_trades >= MIN_TRADES else -np.inf


def walk_forward(df: pd.DataFrame, base: Params, n_folds: int = 5, verbose=True):
    n = len(df)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    rows, picks = [], []
    for k in range(1, n_folds):
        train, test = df.iloc[:edges[k]], df.iloc[edges[k]:edges[k + 1]]
        best, best_p = -np.inf, base
        for cand in _grid(base):
            s = _score(run(train, cand))
            if s > best:
                best, best_p = s, cand
        r = run(test, best_p)
        picks.append(best_p)
        rows.append({"fold": k, "picked": f"tl={best_p.trend_len},tv={best_p.target_vol},sk={best_p.skip}",
                     "oos_sharpe": round(r.sharpe, 2), "oos_bh_sharpe": round(r.bh_sharpe, 2),
                     "oos_annRet": round(r.ann_return, 4), "beats_bh": r.beats_bh()})
        if verbose:
            print(f"[wf] fold {k}: {rows[-1]['picked']}  OOS {r.summary()}")
    keys = [(q.trend_len, q.target_vol, q.skip) for q in picks]
    modal = max(set(keys), key=keys.count)
    consensus = replace(base, trend_len=modal[0], target_vol=modal[1], skip=modal[2])
    excess = [r["oos_sharpe"] - r["oos_bh_sharpe"] for r in rows]
    verdict = ("EDGE DIES: does not beat buy&hold OOS (median excess Sharpe <= 0)"
               if np.median(excess) <= 0 else
               "fragile: some folds fail to beat buy&hold" if min(excess) <= 0 else
               "survives: beats buy&hold across folds")
    return pd.DataFrame(rows), consensus, verdict


def param_sweep(df: pd.DataFrame, p: Params, verbose=True) -> pd.DataFrame:
    specs = {
        "trend_len": lambda f: replace(p, trend_len=max(10, int(round(p.trend_len * f)))),
        "target_vol": lambda f: replace(p, target_vol=p.target_vol * f),
        "vol_win": lambda f: replace(p, vol_win=max(5, int(round(p.vol_win * f)))),
        "lookbacks": lambda f: replace(p, lookbacks=tuple(max(5, int(round(L * f))) for L in p.lookbacks)),
    }
    base_r = run(df, p)
    rows = [{"param": "(base)", "mult": 1.0, "sharpe": round(base_r.sharpe, 2),
             "excess": round(base_r.sharpe - base_r.bh_sharpe, 2),
             "annRet": round(base_r.ann_return, 4)}]
    for name, mk in specs.items():
        for f in (0.75, 1.25):
            r = run(df, mk(f))
            rows.append({"param": name, "mult": f, "sharpe": round(r.sharpe, 2),
                         "excess": round(r.sharpe - r.bh_sharpe, 2),
                         "annRet": round(r.ann_return, 4)})
    out = pd.DataFrame(rows)
    if verbose:
        print(out.to_string(index=False))
    return out


def monte_carlo(equity_pnl: pd.Series, n_runs: int = 1000, seed: int = 0) -> dict:
    r = equity_pnl.dropna().to_numpy()
    if len(r) < 30:
        return {"error": "too few obs"}
    rng = np.random.default_rng(seed)
    finals, dds = np.empty(n_runs), np.empty(n_runs)
    block = 21  # block bootstrap preserves short-run autocorrelation
    n = len(r)
    for i in range(n_runs):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n - block)
            idx.extend(range(s, s + block))
        s = r[np.array(idx[:n])]
        eq = np.cumprod(1 + s)
        finals[i] = eq[-1] - 1
        dds[i] = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"runs": n_runs, "median_return": float(np.median(finals)),
            "p05_return": float(np.percentile(finals, 5)),
            "p95_return": float(np.percentile(finals, 95)),
            "prob_loss": float((finals <= 0).mean()),
            "median_maxdd": float(np.median(dds))}


def run_all(source: str, years: float, trend_strength: float) -> list[str]:
    print("=" * 80)
    print("TIME-SERIES MOMENTUM — FULL VALIDATION SUITE")
    if trend_strength == 0.0:
        print("NULL CONTROL: random-walk data (no trend). Harness must kill the edge.")
    print("=" * 80)
    base = Params()
    failures = []

    df = load_data("BTC", years=years, source=source, trend_strength=trend_strength)
    if "SYNTHETIC" in df.attrs.get("source", ""):
        print("\n*** DATA IS SYNTHETIC (network-restricted). Validates pipeline, ***")
        print("*** not the live edge. Get real daily data for a real verdict.  ***\n")

    split = int(len(df) * 0.70)
    ins, hold = df.iloc[:split], df.iloc[split:]
    print(f"[split] in-sample {ins.index[0]:%Y-%m-%d}..{ins.index[-1]:%Y-%m-%d} | "
          f"holdout {hold.index[0]:%Y-%m-%d}..{hold.index[-1]:%Y-%m-%d}")

    print("\n--- 0. Baseline (in-sample) vs buy&hold ------------------------------")
    base_r = run(ins, base)
    print("   " + base_r.summary())
    print(f"   beats buy&hold (better Sharpe AND drawdown)? {base_r.beats_bh()}")
    if not base_r.beats_bh():
        failures.append("baseline: does NOT beat buy&hold on risk-adjusted basis")

    print("\n--- 1. Walk-forward, 5 folds (metric = excess Sharpe over B&H) -------")
    wf, tuned, verdict = walk_forward(ins, base)
    print(f"[wf] consensus: trend_len={tuned.trend_len}, target_vol={tuned.target_vol}, skip={tuned.skip}")
    print(f"[wf] verdict: {verdict}")
    if "DIES" in verdict:
        failures.append(f"walk-forward: {verdict}")

    print("\n--- 2. 30% out-of-sample holdout (tuned, touched once) ---------------")
    hr = run(hold, tuned)
    print("   " + hr.summary())
    if not hr.beats_bh():
        failures.append(f"holdout: does not beat buy&hold (sharpe={hr.sharpe:+.2f} vs B&H {hr.bh_sharpe:+.2f})")
        print("   >>> DOES NOT BEAT BUY&HOLD ON HOLDOUT <<<")
    if hr.ann_return <= 0:
        failures.append(f"holdout: negative absolute return ({hr.ann_return:+.2%})")

    print("\n--- 3. Parameter sweep +/-25% (in-sample) ----------------------------")
    sw = param_sweep(ins, tuned)
    neg = int((sw["excess"] <= 0).sum())
    if neg:
        failures.append(f"sweep: {neg}/{len(sw)} perturbations don't beat B&H (excess Sharpe <= 0)")
        print(f"   >>> {neg}/{len(sw)} perturbations fail to beat B&H <<<")
    else:
        print("   all +/-25% perturbations keep positive excess Sharpe")

    print("\n--- 4. Cross-asset: ETH, SOL, XRP (BTC-tuned params) -----------------")
    for a in ("ETH", "SOL", "XRP"):
        adf = load_data(a, years=years, source=source, trend_strength=trend_strength)
        r = run(adf, tuned)
        print(f"   {a}: " + r.summary())
        if not r.beats_bh():
            failures.append(f"cross-asset {a}: does not beat buy&hold")
            print(f"   >>> {a} DOES NOT BEAT BUY&HOLD <<<")

    print("\n--- 5. Monte Carlo: 1000 block-bootstrap resamples (holdout) ---------")
    hr_sig = compute_signals(hold, tuned)
    pos = target_position(hr_sig, tuned)
    ret = np.nan_to_num(hr_sig["ret"].to_numpy(), nan=0.0)
    held = np.concatenate([[0.0], pos.to_numpy()[:-1]])
    pnl = pd.Series(held * ret, index=hold.index)
    mc = monte_carlo(pnl, 1000, seed=42)
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
        print("  (none — momentum survived every test on this dataset)")
    print("=" * 80)
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--null", action="store_true", help="null control: random-walk data")
    ap.add_argument("--source", default="auto", choices=["auto", "synthetic"])
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--long-short", action="store_true", help="enable shorts (default long/flat)")
    args = ap.parse_args()

    trend = 0.0 if args.null else 1.0
    if args.run_all:
        run_all(args.source, args.years, trend)
    else:
        df = load_data("BTC", years=args.years, source=args.source, trend_strength=trend)
        r = run(df, Params(long_short=args.long_short))
        print("BTC daily time-series momentum, default params, full sample:")
        print("   " + r.summary())
        print(f"   beats buy&hold (Sharpe AND drawdown)? {r.beats_bh()}")


if __name__ == "__main__":
    main()
