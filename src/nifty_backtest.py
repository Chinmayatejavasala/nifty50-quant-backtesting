"""Nifty 50 backtesting: feature engineering, rule-based and ML signals, walk-forward evaluation.

All strategies are long/flat on the Nifty 50 index. A signal computed at the close of day t
is traded at the close of day t+1 (one-day lag), so no look-ahead bias enters the backtest.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- data
def load_data(ticker="^NSEI", start="2010-01-01", end=None, csv=None, synthetic=False):
    """Return a DataFrame with columns Open, High, Low, Close, Volume indexed by date.

    synthetic=True generates fake prices. It exists ONLY to smoke-test the code when the
    internet is unavailable; never report results from it.
    """
    if synthetic:
        return _synthetic_prices(start, end)
    if csv is not None:
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("No data returned. Check internet access or pass csv=...")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def _synthetic_prices(start, end, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end or "2024-12-31")
    n = len(idx)
    vol = np.where(rng.random(n) < 0.02, 0.02, 0.009)
    ret = rng.normal(0.0004, vol)
    close = 8000 * np.exp(np.cumsum(ret))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close,
         "Volume": rng.integers(1e8, 3e8, n)},
        index=idx,
    )


# --------------------------------------------------------------------------- features
def rsi(close: pd.Series, window=14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["ret"] = c.pct_change()
    out["sma_10"] = c.rolling(10).mean()
    out["sma_50"] = c.rolling(50).mean()
    out["sma_ratio"] = out["sma_10"] / out["sma_50"] - 1
    out["rsi_14"] = rsi(c, 14)
    out["zscore_20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    out["vol_20"] = out["ret"].rolling(20).std()
    out["mom_5"] = c.pct_change(5)
    out["mom_20"] = c.pct_change(20)
    out["vol_chg"] = out["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    return out


FEATURES = ["sma_ratio", "rsi_14", "zscore_20", "vol_20", "mom_5", "mom_20", "ret"]


# --------------------------------------------------------------------------- signals
def signal_ma_cross(df: pd.DataFrame) -> pd.Series:
    """Long when the 10-day SMA is above the 50-day SMA, else flat."""
    return (df["sma_10"] > df["sma_50"]).astype(float).where(df["sma_50"].notna())


def signal_mean_reversion(df: pd.DataFrame, entry=-1.0, exit_=0.0) -> pd.Series:
    """Enter long when the 20-day Z-score falls below `entry`, exit when it recovers to `exit_`."""
    z = df["zscore_20"].to_numpy()
    pos = np.full(len(z), np.nan)
    state = 0.0
    for i, v in enumerate(z):
        if np.isnan(v):
            continue
        if state == 0.0 and v < entry:
            state = 1.0
        elif state == 1.0 and v >= exit_:
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=df.index)


def signal_xgboost_walk_forward(df: pd.DataFrame, min_train_years=3, retrain_every=TRADING_DAYS,
                                threshold=0.5, seed=42) -> pd.Series:
    """Expanding-window walk-forward XGBoost predicting next-day direction.

    The model is retrained every `retrain_every` days using only data available before the
    test block. Long when predicted P(up) > threshold, else flat.
    """
    from xgboost import XGBClassifier

    data = df.dropna(subset=FEATURES).copy()
    data["target"] = (data["ret"].shift(-1) > 0).astype(int)
    data = data.iloc[:-1]  # last row has no known next-day return
    min_train = int(min_train_years * TRADING_DAYS)
    pos = pd.Series(np.nan, index=df.index)

    for start in range(min_train, len(data), retrain_every):
        train = data.iloc[:start]
        test = data.iloc[start:start + retrain_every]
        model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, random_state=seed, eval_metric="logloss")
        model.fit(train[FEATURES], train["target"])
        proba = model.predict_proba(test[FEATURES])[:, 1]
        pos.loc[test.index] = (proba > threshold).astype(float)
    return pos


# --------------------------------------------------------------------------- backtest
def run_backtest(df: pd.DataFrame, signal: pd.Series, cost_bps=5.0) -> pd.DataFrame:
    """Daily strategy returns. Position is lagged one day; costs charged on every position change."""
    position = signal.shift(1)
    turnover = position.diff().abs().fillna(0)
    strat_ret = position * df["ret"] - turnover * cost_bps / 1e4
    return pd.DataFrame({"position": position, "strategy_ret": strat_ret})


def metrics(returns: pd.Series, position: pd.Series | None = None, rf=0.0) -> dict:
    r = returns.dropna()
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    ann_ret = equity.iloc[-1] ** (1 / years) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() - rf / TRADING_DAYS) / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    active = r[position.reindex(r.index) > 0] if position is not None else r
    win_rate = (active > 0).mean() if len(active) else np.nan
    trades = int(position.diff().abs().sum() / 2) if position is not None else 0
    return {"Ann. Return": ann_ret, "Ann. Vol": ann_vol, "Sharpe": sharpe,
            "Max Drawdown": max_dd, "Win Rate (active days)": win_rate, "Round Trips": trades}


def evaluate_all(df: pd.DataFrame, cost_bps=5.0):
    """Run every strategy over the same out-of-sample window and return (metrics table, returns frame)."""
    df = add_features(df)
    signals = {
        "MA Crossover": signal_ma_cross(df),
        "Mean Reversion": signal_mean_reversion(df),
        "XGBoost (walk-forward)": signal_xgboost_walk_forward(df),
    }
    # common evaluation window = where the walk-forward model has predictions
    first = signals["XGBoost (walk-forward)"].first_valid_index()
    window = df.index[df.index >= first]

    rets, rows = {}, {}
    bh = df["ret"].loc[window].copy()
    bh.iloc[0] = 0.0
    rets["Buy and Hold"] = bh
    rows["Buy and Hold"] = metrics(bh, pd.Series(1.0, index=bh.index))
    for name, sig in signals.items():
        bt = run_backtest(df, sig, cost_bps).loc[window]
        rets[name] = bt["strategy_ret"].fillna(0)
        rows[name] = metrics(rets[name], bt["position"].fillna(0))
    return pd.DataFrame(rows).T, pd.DataFrame(rets)


def plot_results(returns: pd.DataFrame, path="results/equity_curves.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    equity = (1 + returns).cumprod()
    dd = equity / equity.cummax() - 1
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]})
    equity.plot(ax=ax[0])
    ax[0].set_title("Out-of-sample equity curves (growth of 1)")
    ax[0].set_ylabel("Equity")
    ax[0].grid(alpha=0.3)
    dd.plot(ax=ax[1], legend=False)
    ax[1].set_title("Drawdown")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="Nifty 50 backtest")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None, help="Optional CSV with Date,Open,High,Low,Close,Volume")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--synthetic", action="store_true", help="Smoke-test on fake data (NOT for reporting)")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    df = load_data(start=a.start, end=a.end, csv=a.csv, synthetic=a.synthetic)
    table, rets = evaluate_all(df, a.cost_bps)
    os.makedirs(a.out, exist_ok=True)
    table.to_csv(os.path.join(a.out, "metrics.csv"))
    plot_results(rets, os.path.join(a.out, "equity_curves.png"))
    print(f"Data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    print(f"Out-of-sample window: {rets.index[0].date()} to {rets.index[-1].date()}")
    print(table.round(3).to_string())
    if a.synthetic:
        print("\n*** SYNTHETIC DATA: smoke test only, these numbers mean nothing. ***")


if __name__ == "__main__":
    main()
