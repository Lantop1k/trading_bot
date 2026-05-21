"""Synthetic OHLC data and technical features for ML strategy demo."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# --- ML training targets (labels_three_way) ---------------------------------
# We label each bar by the close-to-close return over the next N bars vs a symmetric threshold.
# Previously defaults were horizon=3 and thresh≈0.15%, which left almost everything in the “neutral”
# band → models learned HOLD most of the time.
#
# Wider horizon = evaluate a longer stretch of price action (in bar units); cumulative return over
# many bars crosses ±threshold more often → fewer HOLD labels in training (more BUY/SELL).
# Slightly lower threshold also pushes more directional labels (can add noise — tune as needed).
#
# Adjust for your chart timeframe (e.g. M15: 48 bars ≈ 12 hours; H1: 48 bars ≈ 2 days).
LABEL_HORIZON_BARS = 48
LABEL_RETURN_THRESHOLD = 0.001  # 0.10% absolute forward return vs threshold


def _seed_from_symbol(symbol: str) -> int:
    h = hashlib.sha256(symbol.upper().encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def generate_synthetic_ohlc(symbol: str, bars: int = 220) -> pd.DataFrame:
    """Deterministic pseudo-price series per symbol (offline demo, not live MT5)."""
    rng = np.random.default_rng(_seed_from_symbol(symbol))
    # Geometric random walk with mild trend
    ret = rng.normal(0.00015, 0.008, bars)
    close = 1.0 + np.cumsum(ret) * 50
    high = close + np.abs(rng.normal(0, 0.002, bars)) * close
    low = close - np.abs(rng.normal(0, 0.002, bars)) * close
    open_ = np.roll(close, 1)
    open_[0] = close[0] - rng.normal(0, 0.001) * close[0]
    # Ensure OHLC consistency
    ohlc = np.column_stack([open_, high, low, close])
    for i in range(bars):
        hi, lo = max(ohlc[i, 0], ohlc[i, 3]), min(ohlc[i, 0], ohlc[i, 3])
        ohlc[i, 1] = max(ohlc[i, 1], hi, ohlc[i, 3])
        ohlc[i, 2] = min(ohlc[i, 2], lo, ohlc[i, 3])
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=bars, freq="h")
    return pd.DataFrame(
        {"open": ohlc[:, 0], "high": ohlc[:, 1], "low": ohlc[:, 2], "close": ohlc[:, 3]},
        index=idx,
    )


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Technical-style features for sklearn models."""
    out = pd.DataFrame(index=df.index)
    c = df["close"].astype(float)
    h, low = df["high"].astype(float), df["low"].astype(float)
    o = df["open"].astype(float)

    out["ret_1"] = c.pct_change()
    out["ret_5"] = c.pct_change(5)
    out["hl_range"] = (h - low) / (c + 1e-12)
    out["body"] = (c - o) / (o + 1e-12)
    out["ma7_ratio"] = c / (c.rolling(7).mean() + 1e-12) - 1.0
    out["ma21_ratio"] = c / (c.rolling(21).mean() + 1e-12) - 1.0
    # RSI-like (14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    out["rsi_norm"] = (100 - (100 / (1 + rs))) / 100.0
    out["vol_10"] = out["ret_1"].rolling(10).std()

    return out.replace([np.inf, -np.inf], np.nan).bfill().fillna(0.0)


def labels_three_way(
    df: pd.DataFrame,
    horizon: int | None = None,
    thresh: float | None = None,
) -> pd.Series:
    """Future return labels: 0=SELL, 1=HOLD, 2=BUY.

    Uses :data:`LABEL_HORIZON_BARS` and :data:`LABEL_RETURN_THRESHOLD` when arguments are omitted.
    """
    hz = LABEL_HORIZON_BARS if horizon is None else horizon
    th = LABEL_RETURN_THRESHOLD if thresh is None else thresh
    c = df["close"].astype(float)
    fut = c.shift(-hz) / c - 1.0
    y = pd.Series(1, index=df.index, dtype=int)
    y = y.mask(fut > th, 2)
    y = y.mask(fut < -th, 0)
    return y


ACTION_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}
