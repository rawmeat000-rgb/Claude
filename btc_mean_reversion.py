#!/usr/bin/env python3
"""
BTC 15m mean-reversion strategy + full validation harness, in one file.

Edge thesis
-----------
Mean reversion using (a) an order-flow imbalance z-score (taker buy vs sell
volume) and (b) price stretch from a moving anchor (EMA). Enter when price is
extremely stretched *and* order flow is extremely one-sided in the same
direction (capitulation / blow-off); exit when price reverts to the anchor,
or on a blow-out stop, or on a time stop.

Contents
--------
1. Data loader      : Binance REST -> Binance Vision zips -> Bybit -> synthetic
                      fallback (loudly labeled). CSV cache. Taker-buy volume is
                      used for imbalance where the source provides it; otherwise
                      an intrabar pressure proxy is used and flagged.
2. Signal logic     : imbalance z-score + stretch z-score state machine.
3. Position sizing  : volatility targeting, capped at 1x notional.
4. Backtest         : next-bar execution (no lookahead), per-side fees+slippage,
                      reports total/annualized return, #trades, win rate,
                      annualized Sharpe, max drawdown, exposure.
5. Validation harness (run with --run-all):
   - anchored walk-forward across 5 folds on the first 70% of data
   - 30% chronological out-of-sample holdout, touched exactly once
   - parameter sweep of +/-25% around the selected parameters
   - cross-asset test on ETH, SOL, XRP with the BTC-tuned parameters
   - 1000-run Monte Carlo resample of trade returns

Honesty contract: every test prints PASS/FAIL/FRAGILE flags; nothing is
filtered. If the edge dies somewhere, it is printed and returned.

Usage
-----
    python btc_mean_reversion.py                 # single backtest, BTC
    python btc_mean_reversion.py --run-all       # full validation suite
    python btc_mean_reversion.py --source synthetic --seed 7 --run-all

Dependencies: numpy, pandas (loader additionally uses stdlib urllib/zipfile).
"""

from __future__ import annotations

import argparse
import io
import json
import os

import time
import zipfile
import urllib.request
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

BARS_PER_DAY = 96                      # 15m bars, 24/7 market
BARS_PER_YEAR = BARS_PER_DAY * 365    # 35,040
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ASSET_SPECS = {
    # symbol -> (binance symbol, bybit symbol, synthetic ann vol, synthetic seed)
    "BTC": ("BTCUSDT", "BTCUSDT", 0.55, 101),
    "ETH": ("ETHUSDT", "ETHUSDT", 0.70, 202),
    "SOL": ("SOLUSDT", "SOLUSDT", 0.95, 303),
    "XRP": ("XRPUSDT", "XRPUSDT", 0.85, 404),
}


@dataclass(frozen=True)
class Params:
    anchor_len: int = 96        # EMA anchor span, bars (1 day)
    stretch_win: int = 192      # window for stretch z-score std (2 days)
    imb_win: int = 96           # window for imbalance z-score
    z_entry: float = 2.0        # stretch z threshold to enter
    imb_entry: float = 1.0      # imbalance z threshold to enter
    z_exit: float = 0.0         # exit when stretch z reverts through this
    z_stop: float = 4.0         # blow-out stop on stretch z
    max_hold: int = 96          # time stop, bars
    target_vol: float = 0.40    # annualized vol target for sizing
    vol_win: int = 96           # realized-vol lookback for sizing
    fee_bps: float = 5.0        # per-side fee
    slip_bps: float = 2.5       # per-side slippage


# ----------------------------------------------------------------------------
# 1. Data loader
# ----------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mr-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _load_binance_rest(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Binance klines include taker-buy base volume (true flow imbalance)."""
    rows, cur = [], start_ms
    while cur < end_ms:
        url = ("https://api.binance.com/api/v3/klines"
               f"?symbol={symbol}&interval=15m&startTime={cur}&endTime={end_ms}&limit=1000")
        batch = json.loads(_http_get(url))
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][6] + 1
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_vol", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    return _finish_kline_frame(df)


def _load_binance_vision(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Monthly zip archives from data.binance.vision (same schema as REST)."""
    frames = []
    month = pd.Timestamp(start.year, start.month, 1)
    while month <= end:
        url = (f"https://data.binance.vision/data/spot/monthly/klines/"
               f"{symbol}/15m/{symbol}-15m-{month:%Y-%m}.zip")
        try:
            raw = _http_get(url, timeout=60)
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                with z.open(z.namelist()[0]) as f:
                    frames.append(pd.read_csv(f, header=None, names=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_vol", "n_trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"]))
        except Exception:
            pass  # missing month: skip
        month = month + pd.offsets.MonthBegin(1)
    if not frames:
        raise RuntimeError("binance vision: no data")
    return _finish_kline_frame(pd.concat(frames, ignore_index=True))


def _finish_kline_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("empty kline frame")
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "taker_buy": df["taker_buy_base"].astype(float),
    })
    out = out.drop_duplicates("ts").sort_values("ts").set_index("ts")
    out["imb_raw"] = np.where(out["volume"] > 0,
                              2.0 * out["taker_buy"] / out["volume"] - 1.0, 0.0)
    out.attrs["source"] = "binance"
    return out


def _load_bybit(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Bybit spot klines lack taker split -> intrabar pressure proxy for flow."""
    rows, cur = [], start_ms
    while cur < end_ms:
        url = ("https://api.bybit.com/v5/market/kline?category=spot"
               f"&symbol={symbol}&interval=15&start={cur}&end={end_ms}&limit=1000")
        batch = json.loads(_http_get(url))["result"]["list"]
        if not batch:
            break
        rows.extend(batch)  # bybit returns newest-first
        cur = int(batch[0][0]) + 1
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df = df.astype({"ts": np.int64, "open": float, "high": float, "low": float,
                    "close": float, "volume": float})
    out = df.assign(ts=pd.to_datetime(df["ts"], unit="ms", utc=True)) \
            .drop_duplicates("ts").sort_values("ts").set_index("ts")
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["imb_raw"] = ((out["close"] - out["open"]) / rng).fillna(0.0).clip(-1, 1)
    out["taker_buy"] = out["volume"] * (out["imb_raw"] + 1) / 2
    out.attrs["source"] = "bybit(proxy-imbalance)"
    return out[["open", "high", "low", "close", "volume", "taker_buy", "imb_raw"]]


def make_synthetic(symbol: str, n_bars: int, seed: int, ann_vol: float,
                   mr_kappa: float = 0.012) -> pd.DataFrame:
    """
    Synthetic 15m series calibrated to crypto stylized facts:
    - 2-state Markov volatility regimes, Student-t(4) innovations (fat tails)
    - weak OU pull of log-price toward its own EMA (mr_kappa per bar)
    - order flow contemporaneously correlated with returns, AR(1) persistent
    - volume lognormal, correlated with |return|

    LOUD CAVEAT: this exists so the pipeline is runnable offline. A weak
    mean-reversion component is built in by construction, so profitable results
    on synthetic data validate the *code path*, not the real-world edge.
    """
    rng = np.random.default_rng(seed)
    sig_bar = ann_vol / np.sqrt(BARS_PER_YEAR)
    # volatility regime chain: 0=calm(0.7x), 1=stressed(1.9x)
    state = np.zeros(n_bars, dtype=int)
    p_up, p_dn = 0.004, 0.03
    for t in range(1, n_bars):
        u = rng.random()
        state[t] = (1 if u < p_up else 0) if state[t - 1] == 0 else (0 if u < p_dn else 1)
    mult = np.where(state == 0, 0.7, 1.9)
    shocks = rng.standard_t(4, n_bars) / np.sqrt(2.0)  # unit-variance t(4)

    logp = np.empty(n_bars)
    logp[0] = np.log(50000.0 if symbol == "BTC" else 100.0)
    anchor = logp[0]
    alpha = 2.0 / (96 + 1)
    drift = 0.10 / BARS_PER_YEAR  # mild positive drift
    rets = np.zeros(n_bars)
    for t in range(1, n_bars):
        eps = sig_bar * mult[t] * shocks[t]
        rets[t] = drift - mr_kappa * (logp[t - 1] - anchor) + eps
        logp[t] = logp[t - 1] + rets[t]
        anchor += alpha * (logp[t] - anchor)

    close = np.exp(logp)
    open_ = np.empty(n_bars); open_[0] = close[0]; open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0, 0.4 * sig_bar * mult, n_bars)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick

    volume = np.exp(rng.normal(0, 0.6, n_bars)) * (1 + 40 * np.abs(rets) / sig_bar * 0.1) * 100
    imb = np.zeros(n_bars)
    for t in range(1, n_bars):
        imb[t] = 0.85 * imb[t - 1] + 0.6 * np.tanh(rets[t] / (2 * sig_bar)) + rng.normal(0, 0.25)
    imb = np.clip(imb, -0.99, 0.99)
    taker_buy = volume * (imb + 1) / 2

    idx = pd.date_range(end=pd.Timestamp.now('UTC').floor("15min"), periods=n_bars,
                        freq="15min", tz="UTC")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": volume, "taker_buy": taker_buy, "imb_raw": imb},
                      index=idx)
    df.attrs["source"] = f"SYNTHETIC(seed={seed})"
    return df


def load_data(asset: str, years: float = 2.0, source: str = "auto",
              seed_offset: int = 0, mr_kappa: float = 0.012,
              verbose: bool = True) -> pd.DataFrame:
    """Load 15m candles for `asset`. Cache -> Binance -> Vision -> Bybit -> synthetic."""
    bsym, ysym, ann_vol, seed = ASSET_SPECS[asset]
    n_bars = int(years * BARS_PER_YEAR)
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"{asset}_15m.csv")

    if source != "synthetic" and os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        df.attrs["source"] = "cache"
        if verbose:
            print(f"[data] {asset}: {len(df)} bars from cache")
        return df

    end = pd.Timestamp.now('UTC').floor("15min")
    start = end - pd.Timedelta(minutes=15 * n_bars)
    if source in ("auto", "binance", "bybit"):
        loaders = {
            "binance": [lambda: _load_binance_rest(bsym, int(start.timestamp() * 1000), int(end.timestamp() * 1000)),
                        lambda: _load_binance_vision(bsym, start, end)],
            "bybit": [lambda: _load_bybit(ysym, int(start.timestamp() * 1000), int(end.timestamp() * 1000))],
        }
        order = loaders["binance"] + loaders["bybit"] if source == "auto" else loaders[source]
        for fn in order:
            try:
                df = fn()
                df.to_csv(cache)
                if verbose:
                    print(f"[data] {asset}: {len(df)} bars from {df.attrs['source']}")
                return df
            except Exception as e:
                if verbose:
                    print(f"[data] {asset}: loader failed ({type(e).__name__}: {e}); trying next")

    df = make_synthetic(asset, n_bars, seed + seed_offset, ann_vol, mr_kappa=mr_kappa)
    if verbose:
        print(f"[data] {asset}: {len(df)} bars SYNTHETIC fallback "
              f"(ann_vol={ann_vol:.0%}, seed={seed + seed_offset}, kappa={mr_kappa}) — "
              f"results validate the pipeline, NOT the live edge")
    return df


# ----------------------------------------------------------------------------
# 2. Signals
# ----------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """All rolling stats use data up to and including bar t; positions derived
    from bar-t signals are applied to bar t+1 returns (no lookahead)."""
    out = df.copy()
    close = out["close"]
    anchor = close.ewm(span=p.anchor_len, adjust=False).mean()
    dev = close - anchor
    dev_sd = dev.rolling(p.stretch_win, min_periods=p.stretch_win // 2).std()
    out["stretch_z"] = dev / dev_sd.replace(0, np.nan)

    imb = out["imb_raw"]
    mu = imb.rolling(p.imb_win, min_periods=p.imb_win // 2).mean()
    sd = imb.rolling(p.imb_win, min_periods=p.imb_win // 2).std()
    out["imb_z"] = (imb - mu) / sd.replace(0, np.nan)

    ret = close.pct_change()
    rv = ret.rolling(p.vol_win, min_periods=p.vol_win // 2).std() * np.sqrt(BARS_PER_YEAR)
    out["size"] = (p.target_vol / rv.replace(0, np.nan)).clip(upper=1.0)
    out["ret"] = ret
    return out


# ----------------------------------------------------------------------------
# 3+4. Position sizing + backtest
# ----------------------------------------------------------------------------

@dataclass
class BTResult:
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame          # entry_i, exit_i, side, size, net_ret
    total_return: float
    ann_return: float
    sharpe: float
    max_dd: float
    n_trades: int
    win_rate: float
    exposure: float

    def summary(self) -> str:
        return (f"return={self.total_return:+8.2%}  ann={self.ann_return:+7.2%}  "
                f"trades={self.n_trades:4d}  win={self.win_rate:6.1%}  "
                f"sharpe={self.sharpe:+6.2f}  maxDD={self.max_dd:7.2%}  "
                f"exposure={self.exposure:5.1%}")


def backtest(sig: pd.DataFrame, p: Params) -> BTResult:
    n = len(sig)
    stretch = sig["stretch_z"].to_numpy()
    imbz = sig["imb_z"].to_numpy()
    size = np.nan_to_num(sig["size"].to_numpy(), nan=0.0)
    ret = np.nan_to_num(sig["ret"].to_numpy(), nan=0.0)

    pos = np.zeros(n)           # target position decided at close of bar t
    entry_bar = -1
    trades = []                 # (entry_i, exit_i, side, size)
    cur = 0.0

    for t in range(n):
        sz, iz = stretch[t], imbz[t]
        if np.isnan(sz) or np.isnan(iz):
            pos[t] = cur
            continue
        if cur == 0.0:
            if sz <= -p.z_entry and iz <= -p.imb_entry:
                cur = size[t]; entry_bar = t            # long the flush
            elif sz >= p.z_entry and iz >= p.imb_entry:
                cur = -size[t]; entry_bar = t           # short the blow-off
        elif cur > 0:
            if sz >= -p.z_exit or sz <= -p.z_stop or (t - entry_bar) >= p.max_hold:
                trades.append((entry_bar, t, 1, cur)); cur = 0.0
        else:
            if sz <= p.z_exit or sz >= p.z_stop or (t - entry_bar) >= p.max_hold:
                trades.append((entry_bar, t, -1, -cur)); cur = 0.0
        pos[t] = cur
    if cur != 0.0:
        trades.append((entry_bar, n - 1, int(np.sign(cur)), abs(cur)))

    held = np.concatenate([[0.0], pos[:-1]])            # bar-t pnl uses bar t-1 target
    cost = (p.fee_bps + p.slip_bps) / 1e4
    turn = np.abs(np.concatenate([[0.0], np.diff(pos)]))
    pnl = held * ret - turn * cost
    equity = pd.Series((1 + pnl).cumprod(), index=sig.index)

    rows = []
    for e, x, side, tsz in trades:
        # trade pnl accrues on bars e+1..x+? (position held from bar e+1 through
        # bar after exit decision); use the pnl stream between entries for net ret
        seg = pnl[e + 1: min(x + 2, n)]
        rows.append({"entry": sig.index[e], "exit": sig.index[x], "side": side,
                     "size": tsz, "net_ret": float(np.prod(1 + seg) - 1),
                     "bars": x - e})
    tdf = pd.DataFrame(rows)

    years = n / BARS_PER_YEAR
    total = float(equity.iloc[-1] - 1)
    annr = float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0
    sd = pnl.std()
    sharpe = float(pnl.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    nt = len(tdf)
    wr = float((tdf["net_ret"] > 0).mean()) if nt else 0.0
    expo = float((held != 0).mean())
    return BTResult(equity, pd.Series(pos, index=sig.index), tdf,
                    total, annr, sharpe, dd, nt, wr, expo)


def run(df: pd.DataFrame, p: Params) -> BTResult:
    return backtest(compute_signals(df, p), p)


# ----------------------------------------------------------------------------
# 5. Validation harness
# ----------------------------------------------------------------------------

GRID = {
    "anchor_len": [48, 96, 192],
    "z_entry": [1.5, 2.0, 2.5],
    "imb_entry": [0.5, 1.0, 1.5],
}
MIN_TRADES = 20


def _grid_iter(base: Params):
    for a in GRID["anchor_len"]:
        for z in GRID["z_entry"]:
            for i in GRID["imb_entry"]:
                yield replace(base, anchor_len=a, stretch_win=2 * a, z_entry=z,
                              imb_entry=i)


def _score(r: BTResult) -> float:
    return r.sharpe if r.n_trades >= MIN_TRADES else -np.inf


def walk_forward(df: pd.DataFrame, base: Params, n_folds: int = 5, verbose=True):
    """Anchored walk-forward on `df`: train on folds 1..k, test on fold k+1."""
    n = len(df)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    fold_rows, picks = [], []
    for k in range(1, n_folds):
        train, test = df.iloc[:edges[k]], df.iloc[edges[k]:edges[k + 1]]
        best, best_p = -np.inf, base
        for cand in _grid_iter(base):
            s = _score(run(train, cand))
            if s > best:
                best, best_p = s, cand
        r = run(test, best_p)
        picks.append(best_p)
        fold_rows.append({"fold": k, "train_bars": len(train), "test_bars": len(test),
                          "picked": f"anchor={best_p.anchor_len},z={best_p.z_entry},imb={best_p.imb_entry}",
                          "train_sharpe": round(best, 2), "oos_sharpe": round(r.sharpe, 2),
                          "oos_return": round(r.total_return, 4), "oos_trades": r.n_trades,
                          "oos_win": round(r.win_rate, 3)})
        if verbose:
            print(f"[wf] fold {k}: picked {fold_rows[-1]['picked']}  "
                  f"train_sharpe={best:+.2f}  OOS {r.summary()}")
    # consensus params: modal (anchor, z, imb) among fold picks
    keys = [(q.anchor_len, q.z_entry, q.imb_entry) for q in picks]
    modal = max(set(keys), key=keys.count)
    consensus = replace(base, anchor_len=modal[0], stretch_win=2 * modal[0],
                        z_entry=modal[1], imb_entry=modal[2])
    oos_sharpes = [r["oos_sharpe"] for r in fold_rows]
    verdict = ("EDGE DIES in walk-forward (median OOS Sharpe <= 0)"
               if np.median(oos_sharpes) <= 0 else
               "fragile in walk-forward (some folds negative)"
               if min(oos_sharpes) <= 0 else "survives walk-forward")
    return pd.DataFrame(fold_rows), consensus, verdict


def param_sweep(df: pd.DataFrame, p: Params, verbose=True) -> pd.DataFrame:
    """One-at-a-time +/-25% perturbation of each key parameter."""
    specs = {
        "anchor_len": lambda f: replace(p, anchor_len=max(8, int(round(p.anchor_len * f))),
                                        stretch_win=max(16, int(round(p.stretch_win * f)))),
        "z_entry": lambda f: replace(p, z_entry=p.z_entry * f),
        "imb_entry": lambda f: replace(p, imb_entry=p.imb_entry * f),
        "max_hold": lambda f: replace(p, max_hold=max(4, int(round(p.max_hold * f)))),
        "imb_win": lambda f: replace(p, imb_win=max(16, int(round(p.imb_win * f)))),
    }
    rows = []
    base_r = run(df, p)
    rows.append({"param": "(base)", "mult": 1.0, "sharpe": round(base_r.sharpe, 2),
                 "return": round(base_r.total_return, 4), "trades": base_r.n_trades})
    for name, mk in specs.items():
        for f in (0.75, 1.25):
            r = run(df, mk(f))
            rows.append({"param": name, "mult": f, "sharpe": round(r.sharpe, 2),
                         "return": round(r.total_return, 4), "trades": r.n_trades})
    out = pd.DataFrame(rows)
    if verbose:
        print(out.to_string(index=False))
    return out


def monte_carlo(trades: pd.DataFrame, n_runs: int = 1000, seed: int = 0) -> dict:
    """Resample trade net returns with replacement; distribution of outcomes."""
    if trades.empty:
        return {"error": "no trades"}
    r = trades["net_ret"].to_numpy()
    rng = np.random.default_rng(seed)
    totals, dds = np.empty(n_runs), np.empty(n_runs)
    for i in range(n_runs):
        s = rng.choice(r, size=len(r), replace=True)
        eq = np.cumprod(1 + s)
        totals[i] = eq[-1] - 1
        dds[i] = (eq / np.maximum.accumulate(eq) - 1).min()
    return {
        "runs": n_runs, "n_trades": len(r),
        "median_return": float(np.median(totals)),
        "p05_return": float(np.percentile(totals, 5)),
        "p95_return": float(np.percentile(totals, 95)),
        "prob_loss": float((totals <= 0).mean()),
        "median_maxdd": float(np.median(dds)),
        "p95_maxdd": float(np.percentile(dds, 5)),  # 5th pct = worst tail
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def run_all(source: str, years: float, seed_offset: int,
            mr_kappa: float = 0.012) -> list[str]:
    print("=" * 78)
    print("BTC 15m ORDER-FLOW MEAN REVERSION — FULL VALIDATION SUITE")
    if source == "synthetic" and mr_kappa == 0.0:
        print("NULL CONTROL: synthetic data with NO mean-reversion component.")
        print("A correct harness must report the edge dying below.")
    print("=" * 78)
    base = Params()
    failures = []

    df = load_data("BTC", years=years, source=source, seed_offset=seed_offset,
                   mr_kappa=mr_kappa)
    if "SYNTHETIC" in df.attrs.get("source", ""):
        print("\n*** DATA IS SYNTHETIC (network-restricted environment). ***\n"
            "*** All numbers below validate the pipeline, not the live edge. ***\n")

    split = int(len(df) * 0.70)
    ins, holdout = df.iloc[:split], df.iloc[split:]
    print(f"[split] in-sample: {ins.index[0]:%Y-%m-%d} .. {ins.index[-1]:%Y-%m-%d} "
          f"({len(ins)} bars) | holdout (30%): {holdout.index[0]:%Y-%m-%d} .. "
          f"{holdout.index[-1]:%Y-%m-%d} ({len(holdout)} bars)")

    print("\n--- 0. Baseline backtest, default params, full in-sample -------------")
    base_r = run(ins, base)
    print("   " + base_r.summary())

    print("\n--- 1. Anchored walk-forward, 5 folds, in-sample only ----------------")
    wf_tbl, tuned, wf_verdict = walk_forward(ins, base)
    print(f"[wf] consensus params: anchor={tuned.anchor_len}, z_entry={tuned.z_entry}, "
          f"imb_entry={tuned.imb_entry}")
    print(f"[wf] verdict: {wf_verdict}")
    if "DIES" in wf_verdict:
        failures.append(f"walk-forward: {wf_verdict}")

    print("\n--- 2. 30% out-of-sample holdout (touched once, tuned params) --------")
    hold_r = run(holdout, tuned)
    print("   " + hold_r.summary())
    if hold_r.sharpe <= 0 or hold_r.total_return <= 0:
        failures.append(f"holdout: EDGE DIES (sharpe={hold_r.sharpe:+.2f}, "
                        f"return={hold_r.total_return:+.2%})")
        print("   >>> EDGE DIES ON HOLDOUT <<<")
    elif hold_r.sharpe < 0.5:
        failures.append(f"holdout: weak (sharpe={hold_r.sharpe:+.2f} < 0.5)")

    print("\n--- 3. Parameter sweep +/-25% around tuned params (in-sample) --------")
    sw = param_sweep(ins, tuned)
    neg = int((sw["sharpe"] <= 0).sum())
    if neg > 0:
        msg = f"param sweep: {neg}/{len(sw)} perturbations have Sharpe <= 0"
        failures.append(msg)
        print(f"   >>> {msg} <<<")
    else:
        print("   all +/-25% perturbations keep positive Sharpe")

    print("\n--- 4. Cross-asset test: ETH, SOL, XRP with BTC-tuned params ---------")
    for a in ("ETH", "SOL", "XRP"):
        adf = load_data(a, years=years, source=source, seed_offset=seed_offset,
                        mr_kappa=mr_kappa)
        r = run(adf, tuned)
        print(f"   {a}: " + r.summary())
        if r.sharpe <= 0 or r.total_return <= 0:
            failures.append(f"cross-asset {a}: EDGE DIES (sharpe={r.sharpe:+.2f})")
            print(f"   >>> EDGE DIES ON {a} <<<")

    print("\n--- 5. Monte Carlo: 1000 resamples of holdout trade returns ----------")
    src_trades = hold_r.trades if len(hold_r.trades) >= 10 else run(df, tuned).trades
    which = "holdout" if len(hold_r.trades) >= 10 else "full-sample (holdout had <10 trades)"
    mc = monte_carlo(src_trades, n_runs=1000, seed=42)
    print(f"   basis: {which} trades")
    for k, v in mc.items():
        print(f"   {k:>15}: {v:.4f}" if isinstance(v, float) else f"   {k:>15}: {v}")
    if isinstance(mc.get("prob_loss"), float) and mc["prob_loss"] > 0.20:
        failures.append(f"monte carlo: prob(total loss)={mc['prob_loss']:.0%} > 20%")
        print("   >>> MONTE CARLO: UNACCEPTABLE LOSS PROBABILITY <<<")

    print("\n" + "=" * 78)
    print("VERDICT — tests where the edge dies or is fragile (unfiltered):")
    if failures:
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("  (none — edge survived every test on this dataset)")
    print("=" * 78)
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-all", action="store_true", help="full validation suite")
    ap.add_argument("--source", default="auto",
                    choices=["auto", "binance", "bybit", "synthetic"])
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0, help="synthetic seed offset")
    ap.add_argument("--mr-kappa", type=float, default=0.012,
                    help="synthetic mean-reversion strength; 0 = null control "
                         "(harness must then report the edge dying)")
    args = ap.parse_args()

    if args.run_all:
        run_all(args.source, args.years, args.seed, mr_kappa=args.mr_kappa)
    else:
        df = load_data("BTC", years=args.years, source=args.source, seed_offset=args.seed)
        r = run(df, Params())
        print("BTC 15m mean reversion, default params, full sample:")
        print("   " + r.summary())


if __name__ == "__main__":
    main()
