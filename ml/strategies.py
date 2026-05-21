"""ML + classic technical strategies for recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.data_features import ACTION_NAMES, compute_features, labels_three_way

# Use only the most recent N rows for fitting — full history makes /api/analyze feel stuck.
_ML_FIT_MAX_ROWS = 400

# Commit to HOLD only when model P(HOLD) exceeds this; otherwise compare BUY vs SELL only.
_HOLD_COMMIT_THRESHOLD = 0.70


def _action_from_three_way_probs(probs: dict[str, float]) -> tuple[str, int, float]:
    """Map class probabilities to an action.

    If P(HOLD) > 70%, use HOLD. Otherwise ignore HOLD and pick **BUY** or **SELL**
    whichever has higher probability (ties → BUY).
    """
    sell = float(probs.get("SELL", 0) or 0)
    hold = float(probs.get("HOLD", 0) or 0)
    buy = float(probs.get("BUY", 0) or 0)
    if hold > _HOLD_COMMIT_THRESHOLD:
        return "HOLD", 1, hold
    if buy >= sell:
        return "BUY", 2, buy
    return "SELL", 0, sell


def _clamp_int(x: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(x)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _clamp_float(x: Any, default: float, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _strategy_params_for_key(strategy_key: str, raw: dict[str, Any] | None) -> dict[str, Any]:
    """Return validated per-strategy parameters with defaults."""
    p = raw or {}
    if strategy_key == "ma_crossover":
        fast = _clamp_int(p.get("sma_fast"), 20, 2, 400)
        slow = _clamp_int(p.get("sma_slow"), 50, 3, 600)
        if fast >= slow:
            slow = min(600, fast + 1)
        return {"sma_fast": fast, "sma_slow": slow}
    if strategy_key == "ema_crossover":
        fast = _clamp_int(p.get("ema_fast"), 12, 2, 300)
        slow = _clamp_int(p.get("ema_slow"), 26, 3, 500)
        if fast >= slow:
            slow = min(500, fast + 1)
        return {"ema_fast": fast, "ema_slow": slow}
    if strategy_key == "rsi_reversion":
        period = _clamp_int(p.get("rsi_period"), 14, 2, 200)
        low = _clamp_float(p.get("rsi_low"), 30.0, 1.0, 49.0)
        high = _clamp_float(p.get("rsi_high"), 70.0, 51.0, 99.0)
        if low >= high:
            low = min(49.0, high - 1.0)
        return {"rsi_period": period, "rsi_low": low, "rsi_high": high}
    if strategy_key == "bollinger_reversion":
        period = _clamp_int(p.get("bb_period"), 20, 5, 300)
        stdev = _clamp_float(p.get("bb_std"), 2.0, 0.5, 6.0)
        return {"bb_period": period, "bb_std": stdev}
    if strategy_key == "macd_signal":
        fast = _clamp_int(p.get("macd_fast"), 12, 2, 200)
        slow = _clamp_int(p.get("macd_slow"), 26, 3, 400)
        signal = _clamp_int(p.get("macd_signal"), 9, 2, 200)
        if fast >= slow:
            slow = min(400, fast + 1)
        return {"macd_fast": fast, "macd_slow": slow, "macd_signal": signal}
    if strategy_key == "breakout_20":
        lookback = _clamp_int(p.get("donchian_period"), 20, 2, 500)
        return {"donchian_period": lookback}
    return {}


def _make_models() -> dict[str, Any]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=48,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=45,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=42,
        ),
        "mlp_neural": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.12,
        ),
    }


STRATEGY_LABELS = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm_rbf": "SVM (RBF)",
    "logistic_regression": "Logistic Regression",
    "mlp_neural": "Neural Network (MLP)",
    "ma_crossover": "SMA Crossover (20/50)",
    "ema_crossover": "EMA Crossover (12/26)",
    "rsi_reversion": "RSI Mean Reversion (14)",
    "bollinger_reversion": "Bollinger Reversion (20, 2.0)",
    "macd_signal": "MACD Signal (12,26,9)",
    "breakout_20": "Donchian Breakout (20)",
}

# Dropdown / API order: **simple → complex** (rule-based first, then ML from linear to deep).
STRATEGY_KEYS: list[str] = [
    "ma_crossover",
    "ema_crossover",
    "rsi_reversion",
    "bollinger_reversion",
    "macd_signal",
    "breakout_20",
    "logistic_regression",
    "svm_rbf",
    "random_forest",
    "gradient_boosting",
    "mlp_neural",
]


@dataclass
class PredictionResult:
    action: str
    action_code: int
    confidence: float
    probs: dict[str, float]
    note: str


def _probs_from_signal(score: float, hold_band: float = 0.2) -> tuple[dict[str, float], int, float]:
    """Map signal score to SELL/HOLD/BUY probabilities."""
    s = max(-1.0, min(1.0, float(score)))
    intensity = abs(s)
    hold = max(0.05, 1.0 - intensity / max(hold_band, 1e-9))
    directional = 1.0 - hold
    if s >= 0:
        buy = directional
        sell = 0.0
    else:
        sell = directional
        buy = 0.0
    total = sell + hold + buy
    probs = {"SELL": sell / total, "HOLD": hold / total, "BUY": buy / total}
    action, action_code, confidence = _action_from_three_way_probs(probs)
    return probs, action_code, confidence


def _predict_non_ml(
    ohlc: pd.DataFrame,
    strategy_key: str,
    strategy_params: dict[str, Any] | None = None,
) -> tuple[dict[str, float], int, float, str]:
    c = ohlc["close"].astype(float)
    h = ohlc["high"].astype(float)
    low = ohlc["low"].astype(float)
    params = _strategy_params_for_key(strategy_key, strategy_params)

    if strategy_key == "ma_crossover":
        fast = int(params["sma_fast"])
        slow = int(params["sma_slow"])
        if len(c) < slow + 2:
            probs = {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33}
            _, code, conf = _action_from_three_way_probs(probs)
            return probs, code, conf, f"Not enough bars for SMA({fast}/{slow})."
        ma_fast = c.rolling(fast).mean().iloc[-1]
        ma_slow = c.rolling(slow).mean().iloc[-1]
        score = (ma_fast - ma_slow) / (abs(ma_slow) + 1e-12)
        probs, code, conf = _probs_from_signal(score, hold_band=0.0025)
        return probs, code, conf, f"SMA({fast})-SMA({slow}) gap: {score:.5f}"

    if strategy_key == "ema_crossover":
        fast = int(params["ema_fast"])
        slow = int(params["ema_slow"])
        if len(c) < slow + 2:
            probs = {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33}
            _, code, conf = _action_from_three_way_probs(probs)
            return probs, code, conf, f"Not enough bars for EMA({fast}/{slow})."
        ema_fast = c.ewm(span=fast, adjust=False).mean().iloc[-1]
        ema_slow = c.ewm(span=slow, adjust=False).mean().iloc[-1]
        score = (ema_fast - ema_slow) / (abs(ema_slow) + 1e-12)
        probs, code, conf = _probs_from_signal(score, hold_band=0.002)
        return probs, code, conf, f"EMA({fast})-EMA({slow}) gap: {score:.5f}"

    if strategy_key == "rsi_reversion":
        period = int(params["rsi_period"])
        low_thr = float(params["rsi_low"])
        high_thr = float(params["rsi_high"])
        if len(c) < period + 2:
            probs = {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33}
            _, code, conf = _action_from_three_way_probs(probs)
            return probs, code, conf, f"Not enough bars for RSI({period})."
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        last_rsi = float(rsi.iloc[-1])
        if last_rsi < low_thr:
            score = (low_thr - last_rsi) / max(low_thr, 1e-9)
        elif last_rsi > high_thr:
            score = -(last_rsi - high_thr) / max(100.0 - high_thr, 1e-9)
        else:
            score = 0.0
        probs, code, conf = _probs_from_signal(score, hold_band=0.35)
        return probs, code, conf, f"RSI({period})={last_rsi:.2f}, thresholds {low_thr:.1f}/{high_thr:.1f}"

    if strategy_key == "bollinger_reversion":
        period = int(params["bb_period"])
        bb_std = float(params["bb_std"])
        if len(c) < period + 2:
            probs = {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33}
            _, code, conf = _action_from_three_way_probs(probs)
            return probs, code, conf, f"Not enough bars for Bollinger({period},{bb_std:.2f})."
        ma = c.rolling(period).mean()
        sd = c.rolling(period).std()
        upper = ma + bb_std * sd
        lower = ma - bb_std * sd
        last = float(c.iloc[-1])
        up = float(upper.iloc[-1])
        lo = float(lower.iloc[-1])
        mid = float(ma.iloc[-1])
        if last > up:
            score = -min(1.0, (last - up) / (abs(mid) + 1e-12) * 25)
        elif last < lo:
            score = min(1.0, (lo - last) / (abs(mid) + 1e-12) * 25)
        else:
            score = 0.0
        probs, code, conf = _probs_from_signal(score, hold_band=0.25)
        return probs, code, conf, f"Close={last:.5f}, BB({period},{bb_std:.2f}) upper={up:.5f}, lower={lo:.5f}"

    if strategy_key == "macd_signal":
        fast = int(params["macd_fast"])
        slow = int(params["macd_slow"])
        signal_p = int(params["macd_signal"])
        if len(c) < slow + signal_p + 2:
            probs = {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33}
            _, code, conf = _action_from_three_way_probs(probs)
            return probs, code, conf, f"Not enough bars for MACD({fast},{slow},{signal_p})."
        ema_fast = c.ewm(span=fast, adjust=False).mean()
        ema_slow = c.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_p, adjust=False).mean()
        hist = macd - signal
        score = float(hist.iloc[-1]) / (abs(float(c.iloc[-1])) + 1e-12) * 100
        probs, code, conf = _probs_from_signal(score, hold_band=0.08)
        return probs, code, conf, f"MACD({fast},{slow},{signal_p}) histogram normalized: {score:.5f}"

    if strategy_key == "breakout_20":
        lookback = int(params["donchian_period"])
        if len(c) < lookback + 2:
            probs = {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33}
            _, code, conf = _action_from_three_way_probs(probs)
            return probs, code, conf, f"Not enough bars for Donchian({lookback})."
        hh = h.rolling(lookback).max().shift(1).iloc[-1]
        ll = low.rolling(lookback).min().shift(1).iloc[-1]
        last = float(c.iloc[-1])
        if last > hh:
            score = min(1.0, (last - hh) / (abs(last) + 1e-12) * 120)
        elif last < ll:
            score = -min(1.0, (ll - last) / (abs(last) + 1e-12) * 120)
        else:
            score = 0.0
        probs, code, conf = _probs_from_signal(score, hold_band=0.2)
        return probs, code, conf, f"Donchian({lookback}) levels high={hh:.5f}, low={ll:.5f}"

    probs = {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33}
    _, code, conf = _action_from_three_way_probs(probs)
    return probs, code, conf, "Unknown technical strategy."


def train_predict(
    ohlc: pd.DataFrame,
    strategy_key: str,
    lot_amount: float,
    mt5_criteria: str,
    strategy_params: dict[str, Any] | None = None,
) -> PredictionResult:
    """Predict with either ML classifier or technical-rule strategy."""
    models = _make_models()
    if strategy_key not in STRATEGY_LABELS:
        strategy_key = STRATEGY_KEYS[0]

    if strategy_key not in models:
        probs_map, code, conf, strategy_note = _predict_non_ml(
            ohlc,
            strategy_key,
            strategy_params=strategy_params,
        )
        action = ACTION_NAMES[code]
        note_parts = [
            f"Strategy: {STRATEGY_LABELS.get(strategy_key, strategy_key)}",
            strategy_note,
            f"Requested lot/context: {lot_amount}",
            f"HOLD only if P(HOLD)>{_HOLD_COMMIT_THRESHOLD:.0%}; else max(BUY,SELL).",
        ]
        if mt5_criteria.strip():
            note_parts.append(f"Your criteria noted ({len(mt5_criteria)} chars).")
        return PredictionResult(
            action=action,
            action_code=code,
            confidence=conf,
            probs=probs_map,
            note=" | ".join(note_parts),
        )

    X_full = compute_features(ohlc)
    y_full = labels_three_way(ohlc)
    valid = ~(X_full.isna().any(axis=1) | y_full.isna())
    X = X_full.loc[valid]
    y = y_full.loc[valid]

    if len(X) < 40:
        return PredictionResult(
            action="HOLD",
            action_code=1,
            confidence=0.0,
            probs={"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33},
            note="Insufficient bars for reliable ML fit.",
        )

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", models[strategy_key]),
        ]
    )

    X_train, X_pred = X.iloc[:-1], X.iloc[-1:]
    y_train = y.iloc[:-1]
    if len(X_train) > _ML_FIT_MAX_ROWS:
        X_train = X_train.iloc[-_ML_FIT_MAX_ROWS :]
        y_train = y_train.iloc[-_ML_FIT_MAX_ROWS :]

    try:
        clf.fit(X_train.values, y_train.values)
        proba = clf.predict_proba(X_pred.values)[0]
        classes = clf.classes_
    except Exception as exc:  # noqa: BLE001
        return PredictionResult(
            action="HOLD",
            action_code=1,
            confidence=0.0,
            probs={"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33},
            note=f"Model fit failed ({exc!s}). Try another strategy or more data.",
        )

    base_probs = {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
    for c, p in zip(classes, proba):
        base_probs[ACTION_NAMES[int(c)]] = float(p)
    probs_map = base_probs

    action, code, conf = _action_from_three_way_probs(probs_map)
    note_parts = [
        f"Strategy: {STRATEGY_LABELS.get(strategy_key, strategy_key)}",
        f"Requested lot/context: {lot_amount}",
        f"HOLD only if P(HOLD)>{_HOLD_COMMIT_THRESHOLD:.0%}; else max(BUY,SELL).",
    ]
    if mt5_criteria.strip():
        note_parts.append(f"Your criteria noted ({len(mt5_criteria)} chars).")

    return PredictionResult(
        action=action,
        action_code=code,
        confidence=conf,
        probs=probs_map,
        note=" | ".join(note_parts),
    )


def ohlc_to_candles_json(df: pd.DataFrame) -> list[dict]:
    """Format for lightweight-charts candlestick series."""
    out = []
    for ts, row in df.iterrows():
        t = int(ts.timestamp()) if hasattr(ts, "timestamp") else int(pd.Timestamp(ts).timestamp())
        out.append(
            {
                "time": t,
                "open": round(float(row["open"]), 5),
                "high": round(float(row["high"]), 5),
                "low": round(float(row["low"]), 5),
                "close": round(float(row["close"]), 5),
            }
        )
    return out
