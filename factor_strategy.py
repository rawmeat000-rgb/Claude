#!/usr/bin/env python3
"""
Six-model factor strategy + honest validation harness, in one file.

Turns the "multi-strategy stock bot" screener (Lynch / Piotroski / Minervini /
Graham / Buffett / O'Neil) from a point-in-time checklist into a RANKABLE,
BACKTESTABLE cross-sectional equity strategy, and then tries hard to kill it.

The idea
--------
Each rebalance date, every stock in the universe gets six model scores, grouped
into three well-known factor families:

    VALUE    = Lynch (low PEG)        + Graham (price below Graham number)
    QUALITY  = Buffett (high ROE)     + Piotroski (high F-Score)
    MOMENTUM = Minervini (trend pass) + O'Neil (12m relative strength)

Each raw metric is cross-sectionally z-scored (sign-adjusted so higher = better),
averaged into a composite, and we go long the top-K names equal-weight, rebalance
monthly. Benchmark = equal-weight the whole universe. The question the harness
answers: does the composite beat the benchmark out-of-sample, or is it noise?

**READ THIS — the fundamental-data trap**
Correct fundamental backtesting requires POINT-IN-TIME data: each stock's ROE,
F-Score, PEG, etc. AS THEY WERE KNOWN on each historical date. Using today's
fundamentals on past prices is lookahead + survivorship bias and produces a
beautiful fake edge. This environment has no market access, so this file runs on
a clearly-labeled SYNTHETIC panel that validates the code and the harness — NOT
a real-world edge. To get real answers, supply a point-in-time panel CSV (see
--panel) sourced from a provider that timestamps fundamentals.

Usage
-----
    python factor_strategy.py                 # single backtest, synthetic panel
    python factor_strategy.py --run-all       # full validation suite
    python factor_strategy.py --run-all --null   # null control (no real signal)

Dependencies: numpy, pandas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class Params:
    top_k: int = 10           # number of names held long
    rebalance_m: int = 1      # months between rebalances
    w_value: float = 1.0      # composite weights on the three families
    w_quality: float = 1.0
    w_momentum: float = 1.0
    use_gate: bool = False    # screener-style hard gate before ranking
    gate_fscore: float = 6.0
    gate_roe: float = 0.15
    cost_bps: float = 20.0    # round-trip cost per name traded, bps


# ----------------------------------------------------------------------------
# Synthetic point-in-time panel (loudly labeled)
# ----------------------------------------------------------------------------

def make_synthetic_panel(n_stocks: int = 60, n_years: int = 12, seed: int = 7,
                         premia=(0.9, 0.7, 1.1), verbose: bool = True) -> pd.DataFrame:
    """
    Long-format monthly panel with columns:
        date, ticker, price, fwd_ret, peg, graham_ratio, roe, fscore, eps_growth
    plus latent truth used only for generation.

    Construction embeds a genuine (but noisy) link from value/quality/momentum
    exposure to forward return, scaled by `premia`. Set premia=(0,0,0) for the
    null control: fundamentals and past returns then carry NO predictive power
    and a correct harness MUST report the edge dying.

    LOUD CAVEAT: because the link is built in here, a profitable result validates
    the pipeline, not the real market.
    """
    rng = np.random.default_rng(seed)
    n_m = n_years * MONTHS_PER_YEAR
    dates = pd.date_range("2013-01-01", periods=n_m, freq="MS")
    pv, pq, pm = premia

    # persistent latent exposures per stock (AR(1))
    val = rng.standard_normal(n_stocks)
    qual = rng.standard_normal(n_stocks)
    mom = rng.standard_normal(n_stocks)
    beta = rng.uniform(0.7, 1.3, n_stocks)
    price = np.full(n_stocks, 100.0)

    rows = []
    for t, d in enumerate(dates):
        mkt = rng.normal(0.007, 0.045)                      # monthly market return
        z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
        # forward return depends on CURRENT exposures (known at t) + noise
        fwd = (beta * mkt
               + (pv * z(val) + pq * z(qual) + pm * z(mom)) / 100.0
               + rng.normal(0, 0.05, n_stocks))
        # reported fundamentals = monotonic transforms of exposures + noise
        peg = np.clip(1.5 - 0.5 * z(val) + rng.normal(0, 0.3, n_stocks), 0.1, 6.0)
        graham_ratio = np.clip(1.0 - 0.15 * z(val) + rng.normal(0, 0.15, n_stocks), 0.2, 3.0)
        roe = np.clip(0.15 + 0.05 * z(qual) + rng.normal(0, 0.03, n_stocks), -0.2, 0.6)
        fscore = np.clip(np.round(5 + 1.6 * z(qual) + rng.normal(0, 1.0, n_stocks)), 0, 9)
        eps_growth = np.clip(0.15 + 0.10 * z(qual) + 0.05 * z(mom) + rng.normal(0, 0.05, n_stocks), -0.5, 1.0)

        for i in range(n_stocks):
            rows.append((d, f"S{i:02d}", price[i], fwd[i], peg[i], graham_ratio[i],
                         roe[i], int(fscore[i]), eps_growth[i]))
        # advance price and drift exposures
        price = price * (1 + fwd)
        val = 0.92 * val + 0.39 * rng.standard_normal(n_stocks)
        qual = 0.92 * qual + 0.39 * rng.standard_normal(n_stocks)
        mom = 0.85 * mom + 0.53 * rng.standard_normal(n_stocks)

    df = pd.DataFrame(rows, columns=["date", "ticker", "price", "fwd_ret", "peg",
                                     "graham_ratio", "roe", "fscore", "eps_growth"])
    df.attrs["source"] = f"SYNTHETIC(seed={seed}, premia={premia})"
    if verbose:
        tag = "NULL CONTROL" if premia == (0, 0, 0) else "demo"
        print(f"[panel] {tag}: {n_stocks} stocks x {n_m} months, {df.attrs['source']}")
        print("[panel] SYNTHETIC — validates pipeline/harness, NOT a live edge")
    return df


# ----------------------------------------------------------------------------
# Signals: six models -> three factors -> composite rank
# ----------------------------------------------------------------------------

def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)

def _trailing_return(price_wide: pd.DataFrame, k: int) -> pd.DataFrame:
    return price_wide / price_wide.shift(k) - 1.0

def compute_scores(panel: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Add per-date cross-sectional factor z-scores and a composite. Momentum
    (Minervini/O'Neil) is derived only from price history known at each date."""
    df = panel.copy()
    price_wide = df.pivot(index="date", columns="ticker", values="price")

    # MOMENTUM inputs from price only (no lookahead)
    r12 = _trailing_return(price_wide, 12)                 # O'Neil relative strength
    ma10 = price_wide.rolling(10, min_periods=6).mean()
    ma20 = price_wide.rolling(20, min_periods=10).mean()
    hi12 = price_wide.rolling(12, min_periods=6).max()
    lo12 = price_wide.rolling(12, min_periods=6).min()
    # simplified Minervini trend template (count of conditions, 0..5)
    trend = ((price_wide > ma10).astype(float)
             + (price_wide > ma20).astype(float)
             + (ma10 > ma20).astype(float)
             + (price_wide >= 0.85 * hi12).astype(float)
             + (price_wide >= 1.25 * lo12).astype(float))

    def long_of(wide, name):
        return wide.reset_index().melt(id_vars="date", var_name="ticker", value_name=name)

    df = df.merge(long_of(r12, "rs12"), on=["date", "ticker"], how="left")
    df = df.merge(long_of(trend, "trend"), on=["date", "ticker"], how="left")

    g = df.groupby("date", group_keys=False)
    # sign-adjust so higher = better; PEG & graham_ratio are "lower is better"
    df["z_value"] = g.apply(lambda x: (_z(-x["peg"]) + _z(-x["graham_ratio"])) / 2)
    df["z_quality"] = g.apply(lambda x: (_z(x["roe"]) + _z(x["fscore"])) / 2)
    df["z_momentum"] = g.apply(lambda x: (_z(x["rs12"]) + _z(x["trend"])) / 2)

    df["composite"] = (p.w_value * df["z_value"]
                       + p.w_quality * df["z_quality"]
                       + p.w_momentum * df["z_momentum"]) / (p.w_value + p.w_quality + p.w_momentum)
    return df


# ----------------------------------------------------------------------------
# Backtest: monthly long top-K vs equal-weight benchmark
# ----------------------------------------------------------------------------

@dataclass
class BTResult:
    port: pd.Series            # strategy monthly returns
    bench: pd.Series           # benchmark monthly returns
    active: pd.Series          # port - bench
    ann_ret: float
    ann_bench: float
    ann_active: float
    sharpe: float              # of ACTIVE (excess vs benchmark)
    info_ratio: float
    max_dd: float              # of strategy equity
    hit_rate: float            # fraction of months beating benchmark
    n_reb: int
    avg_turnover: float

    def summary(self) -> str:
        return (f"annRet={self.ann_ret:+7.2%}  bench={self.ann_bench:+7.2%}  "
                f"active={self.ann_active:+7.2%}  IR={self.info_ratio:+5.2f}  "
                f"maxDD={self.max_dd:7.2%}  hit={self.hit_rate:5.1%}  reb={self.n_reb}")


def backtest(scored: pd.DataFrame, p: Params) -> BTResult:
    dates = sorted(scored["date"].unique())
    prev_holds: set[str] = set()
    port_r, bench_r, active_r, turns = [], [], [], []

    for t, d in enumerate(dates):
        if t % p.rebalance_m != 0:
            continue
        day = scored[scored["date"] == d].dropna(subset=["composite", "fwd_ret"])
        if len(day) < p.top_k:
            continue
        elig = day
        if p.use_gate:
            elig = day[(day["fscore"] >= p.gate_fscore) & (day["roe"] >= p.gate_roe)]
            if len(elig) < p.top_k:
                elig = day  # gate too tight this month; fall back to full ranking
        picks = elig.nlargest(p.top_k, "composite")
        holds = set(picks["ticker"])

        gross = picks["fwd_ret"].mean()
        traded = len(holds.symmetric_difference(prev_holds))
        cost = traded / (2 * p.top_k) * (p.cost_bps / 1e4)
        port_r.append(gross - cost)
        bench_r.append(day["fwd_ret"].mean())
        active_r.append(port_r[-1] - bench_r[-1])
        turns.append(traded / (2 * p.top_k))
        prev_holds = holds

    idx = pd.RangeIndex(len(port_r))
    port = pd.Series(port_r, index=idx)
    bench = pd.Series(bench_r, index=idx)
    active = pd.Series(active_r, index=idx)
    per_year = MONTHS_PER_YEAR / p.rebalance_m

    def annualize(s):
        return float((1 + s).prod() ** (per_year / max(len(s), 1)) - 1)

    eq = (1 + port).cumprod()
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    a_sd = active.std()
    ir = float(active.mean() / a_sd * np.sqrt(per_year)) if a_sd > 0 else 0.0
    p_sd = port.std()
    sharpe = float(port.mean() / p_sd * np.sqrt(per_year)) if p_sd > 0 else 0.0
    hit = float((active > 0).mean()) if len(active) else 0.0
    return BTResult(port, bench, active, annualize(port), annualize(bench),
                    annualize(active), sharpe, ir, dd, hit, len(port_r),
                    float(np.mean(turns)) if turns else 0.0)


def run(panel: pd.DataFrame, p: Params) -> BTResult:
    return backtest(compute_scores(panel, p), p)


# ----------------------------------------------------------------------------
# Validation harness
# ----------------------------------------------------------------------------

def quantile_spread(scored: pd.DataFrame, n_q: int = 5) -> list[float]:
    """Mean forward return by composite quintile — a real signal is monotonic."""
    means = [[] for _ in range(n_q)]
    for d, day in scored.dropna(subset=["composite", "fwd_ret"]).groupby("date"):
        if len(day) < n_q * 2:
            continue
        q = pd.qcut(day["composite"], n_q, labels=False, duplicates="drop")
        for qi in range(n_q):
            sel = day["fwd_ret"][q == qi]
            if len(sel):
                means[qi].append(sel.mean())
    return [float(np.mean(m)) * MONTHS_PER_YEAR if m else float("nan") for m in means]


def param_sweep(panel: pd.DataFrame, p: Params) -> pd.DataFrame:
    specs = {
        "top_k": lambda f: replace(p, top_k=max(3, int(round(p.top_k * f)))),
        "rebalance_m": lambda f: replace(p, rebalance_m=max(1, int(round(p.rebalance_m * f)))),
        "w_value": lambda f: replace(p, w_value=p.w_value * f),
        "w_quality": lambda f: replace(p, w_quality=p.w_quality * f),
        "w_momentum": lambda f: replace(p, w_momentum=p.w_momentum * f),
        "cost_bps": lambda f: replace(p, cost_bps=p.cost_bps * f),
    }
    rows = [{"param": "(base)", "mult": 1.0, **_row(run(panel, p))}]
    for name, mk in specs.items():
        for f in (0.75, 1.25):
            rows.append({"param": name, "mult": f, **_row(run(panel, mk(f)))})
    return pd.DataFrame(rows)


def _row(r: BTResult) -> dict:
    return {"active": round(r.ann_active, 4), "IR": round(r.info_ratio, 2),
            "maxDD": round(r.max_dd, 3)}


def monte_carlo(active: pd.Series, n_runs: int = 1000, seed: int = 0) -> dict:
    if len(active) < 5:
        return {"error": "too few periods"}
    a = active.to_numpy()
    rng = np.random.default_rng(seed)
    finals = np.empty(n_runs)
    for i in range(n_runs):
        finals[i] = np.prod(1 + rng.choice(a, size=len(a), replace=True)) - 1
    return {"runs": n_runs, "median_active_total": float(np.median(finals)),
            "p05": float(np.percentile(finals, 5)), "p95": float(np.percentile(finals, 95)),
            "prob_underperform": float((finals <= 0).mean())}


def run_all(panel: pd.DataFrame, base: Params) -> list[str]:
    print("=" * 78)
    print("SIX-MODEL FACTOR STRATEGY — VALIDATION SUITE")
    print("=" * 78)
    failures = []
    if "SYNTHETIC" in panel.attrs.get("source", ""):
        print("\n*** SYNTHETIC PANEL — no lookahead-safe REAL fundamentals here. ***")
        print("*** Numbers validate the pipeline/harness, not a live edge.    ***\n")

    dates = sorted(panel["date"].unique())
    split = int(len(dates) * 0.70)
    ins = panel[panel["date"].isin(dates[:split])]
    oos = panel[panel["date"].isin(dates[split:])]
    print(f"[split] in-sample {dates[0]:%Y-%m}..{dates[split-1]:%Y-%m} | "
          f"holdout {dates[split]:%Y-%m}..{dates[-1]:%Y-%m}")

    print("\n--- 0. Baseline (in-sample) ------------------------------------------")
    base_r = run(ins, base)
    print("   " + base_r.summary())
    if base_r.ann_active <= 0:
        failures.append(f"baseline: no active return (+{base_r.ann_active:.2%})")

    print("\n--- 1. Quintile monotonicity (does more signal = more return?) -------")
    qs = quantile_spread(compute_scores(ins, base))
    print("   Q1..Q5 annualized fwd return: " + "  ".join(f"{q:+.2%}" for q in qs))
    if not (qs[-1] > qs[0]):
        failures.append("quintile: top quintile does NOT beat bottom (signal not monotonic)")
        print("   >>> SIGNAL NOT MONOTONIC — top quintile fails to beat bottom <<<")

    print("\n--- 2. 30% out-of-sample holdout (touched once) ----------------------")
    oos_r = run(oos, base)
    print("   " + oos_r.summary())
    if oos_r.ann_active <= 0 or oos_r.info_ratio <= 0:
        failures.append(f"holdout: EDGE DIES (active={oos_r.ann_active:+.2%}, IR={oos_r.info_ratio:+.2f})")
        print("   >>> EDGE DIES ON HOLDOUT <<<")
    elif oos_r.info_ratio < 0.3:
        failures.append(f"holdout: weak (IR={oos_r.info_ratio:+.2f} < 0.3)")

    print("\n--- 3. Parameter sweep +/-25% (in-sample) ----------------------------")
    sw = param_sweep(ins, base)
    print(sw.to_string(index=False))
    neg = int((sw["active"] <= 0).sum())
    if neg:
        failures.append(f"sweep: {neg}/{len(sw)} perturbations have active return <= 0")
        print(f"   >>> {neg}/{len(sw)} perturbations non-positive <<<")

    print("\n--- 4. Per-year stability (in-sample) --------------------------------")
    scored_ins = compute_scores(ins, base)
    yrs = sorted({d.year for d in dates[:split]})
    neg_years = 0
    for y in yrs:
        yp = scored_ins[scored_ins["date"].dt.year == y]
        r = backtest(yp, base)
        flag = "" if r.ann_active > 0 else "  <-- negative"
        print(f"   {y}: active={r.ann_active:+7.2%}  IR={r.info_ratio:+5.2f}{flag}")
        neg_years += r.ann_active <= 0
    if neg_years > len(yrs) / 2:
        failures.append(f"stability: {neg_years}/{len(yrs)} years non-positive")

    print("\n--- 5. Sub-universe robustness (random split of the universe) --------")
    tickers = sorted(panel["ticker"].unique())
    rng = np.random.default_rng(1)
    perm = rng.permutation(tickers)
    for half_name, half in [("A", perm[:len(perm)//2]), ("B", perm[len(perm)//2:])]:
        r = run(panel[panel["ticker"].isin(half)], replace(base, top_k=min(base.top_k, len(half)//3)))
        print(f"   universe {half_name} ({len(half)} names): " + r.summary())
        if r.ann_active <= 0:
            failures.append(f"sub-universe {half_name}: EDGE DIES (active={r.ann_active:+.2%})")
            print(f"   >>> EDGE DIES ON SUB-UNIVERSE {half_name} <<<")

    print("\n--- 6. Monte Carlo: 1000 resamples of holdout active returns ---------")
    mc = monte_carlo(oos_r.active, 1000, seed=42)
    for k, v in mc.items():
        print(f"   {k:>22}: {v:.4f}" if isinstance(v, float) else f"   {k:>22}: {v}")
    if isinstance(mc.get("prob_underperform"), float) and mc["prob_underperform"] > 0.35:
        failures.append(f"monte carlo: prob(underperform)={mc['prob_underperform']:.0%} > 35%")
        print("   >>> MONTE CARLO: HIGH UNDERPERFORMANCE PROBABILITY <<<")

    print("\n" + "=" * 78)
    print("VERDICT — tests where the edge dies or is fragile (unfiltered):")
    if failures:
        for f in failures:
            print(f"  X {f}")
    else:
        print("  (none — edge survived every test on this panel)")
    print("=" * 78)
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--null", action="store_true",
                    help="null control: synthetic panel with NO real signal")
    ap.add_argument("--panel", help="CSV of a POINT-IN-TIME panel to use instead of synthetic")
    ap.add_argument("--gate", action="store_true", help="apply screener-style F/ROE hard gate")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.panel:
        panel = pd.read_csv(args.panel, parse_dates=["date"])
        panel.attrs["source"] = f"CSV:{args.panel}"
        print(f"[panel] loaded {len(panel)} rows from {args.panel}")
    else:
        premia = (0.0, 0.0, 0.0) if args.null else (0.9, 0.7, 1.1)
        panel = make_synthetic_panel(seed=args.seed, premia=premia)

    base = Params(use_gate=args.gate)
    if args.run_all:
        run_all(panel, base)
    else:
        r = run(panel, base)
        print("Six-model composite, top-10 monthly, full panel:")
        print("   " + r.summary())


if __name__ == "__main__":
    main()
