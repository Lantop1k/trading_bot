"""MetaTrader 5 terminal bridge: live rates, account info, market execution.

Requires Windows + MetaTrader 5 terminal installed; Python package: MetaTrader5.
Uses the account already logged in the terminal unless MT5_LOGIN/PASSWORD/SERVER are set.
"""

from __future__ import annotations

import os
import threading
import time
import math
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent

import pandas as pd

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    mt5 = None

_initialized = False
# MetaTrader5 terminal RPC is not safe under concurrent calls from Flask threads.
_mt5_lock = threading.RLock()

TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def package_installed() -> bool:
    return mt5 is not None


def _tf_constant(name: str):
    if not mt5:
        return None
    key = TIMEFRAME_MAP.get(name.upper())
    if not key:
        return None
    return getattr(mt5, key, None)


def ensure_mt5() -> tuple[bool, str]:
    """Initialize terminal connection once."""
    global _initialized
    if not package_installed():
        return False, "MetaTrader5 Python package missing. Install: pip install MetaTrader5"

    if _initialized:
        try:
            info = mt5.terminal_info()
            if info is not None:
                return True, "connected"
        except Exception:
            pass
        shutdown_mt5()

    path = os.environ.get("MT5_TERMINAL_PATH") or os.environ.get("MT5_PATH")
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    kwargs: dict[str, Any] = {}
    if path:
        kwargs["path"] = path
    if login and password and server:
        kwargs["login"] = int(login)
        kwargs["password"] = password
        kwargs["server"] = server

    if not mt5.initialize(**kwargs):
        err = mt5.last_error()
        return False, f"MT5 initialize failed: {err}. Open MT5 and log in, or set MT5_PATH / credentials."

    _initialized = True
    return True, "connected"


def shutdown_mt5() -> None:
    global _initialized
    if mt5 and _initialized:
        mt5.shutdown()
    _initialized = False


def _symbol_select_retry(sym: str) -> tuple[bool, str]:
    """Must run under _mt5_lock. Retries transient (-1, 'Terminal: Call failed') from concurrent RPC."""
    sym = sym.upper().strip()
    si0 = mt5.symbol_info(sym)
    if si0 is None:
        return (
            False,
            f"Symbol '{sym}' not found. Open MetaTrader 5, add the instrument to Market Watch, "
            "and use the broker's exact name (suffix may differ, e.g. EURUSD vs EURUSD.a).",
        )
    last_err = None
    for attempt in range(6):
        if mt5.symbol_select(sym, True):
            return True, "ok"
        last_err = mt5.last_error()
        if attempt < 5:
            time.sleep(0.12 + 0.08 * attempt)
            code = last_err[0] if last_err else None
            if code == -1:
                shutdown_mt5()
                ok_r, msg_r = ensure_mt5()
                if not ok_r:
                    return False, f"Reconnect failed: {msg_r}"
    return False, f"symbol_select failed for {sym}: {last_err}"


def _is_forex_spot_symbol(name: str) -> bool:
    """True if MT5 classifies the symbol as Forex / Forex no leverage (excludes stocks, CFD metals, crypto)."""
    si = mt5.symbol_info(name)
    if si is None:
        return False
    mode = int(getattr(si, "trade_calc_mode", -1))
    # SYMBOL_TRADE_CALC_MODE_FOREX = 0, FOREX_NO_LEVERAGE = 1 (MQL5 enum)
    return mode in (0, 1)


def estimate_spread_cost_deposit(symbol: str, volume: float = 1.0) -> float | None:
    """Approximate one-way spread crossing cost (|ask−bid|) in **account deposit currency** for ``volume`` lots.

    Uses ``trade_tick_size`` and ``trade_tick_value``. Returns ``None`` if MT5 data is unavailable.
    """
    if not package_installed() or mt5 is None:
        return None
    sym = symbol.upper().strip()
    with _mt5_lock:
        ok, _ = ensure_mt5()
        if not ok:
            return None
        ok_sel, _ = _symbol_select_retry(sym)
        if not ok_sel:
            return None
        tick = mt5.symbol_info_tick(sym)
        si = mt5.symbol_info(sym)
        if tick is None or si is None:
            return None
        ask, bid = float(tick.ask), float(tick.bid)
        spread_price = abs(ask - bid)
        ts = float(getattr(si, "trade_tick_size", 0) or 0)
        tv = float(getattr(si, "trade_tick_value", 0) or 0)
        if ts <= 0 or tv <= 0:
            return None
        ticks = spread_price / ts
        return float(ticks * tv * float(volume))


def filter_pairs_by_max_spread_cost(
    pairs: list[tuple[str, str]],
    max_cost: float | None,
    volume: float = 1.0,
) -> tuple[list[tuple[str, str]], int]:
    """Drop symbols whose estimated spread cost exceeds ``max_cost`` (deposit currency).

    Keeps symbols when spread cannot be estimated (MT5 offline / missing ticks).
    ``max_cost`` ``None`` or ``<= 0`` disables filtering.
    """
    if max_cost is None or max_cost <= 0:
        return pairs, 0
    if not package_installed():
        return pairs, 0
    kept: list[tuple[str, str]] = []
    excluded = 0
    for code, label in pairs:
        cost = estimate_spread_cost_deposit(code, volume)
        if cost is not None and cost > max_cost:
            excluded += 1
            continue
        kept.append((code, label))
    return kept, excluded


def market_watch_instruments(
    limit: int | None = 96,
    forex_only: bool = False,
    max_spread_cost_usd: float | None = 10.0,
    volume_for_spread_check: float = 1.0,
) -> tuple[list[tuple[str, str]], str]:
    """Symbols visible in MetaTrader 5 Market Watch: ``(name, label)`` tuples.

    Label uses the terminal instrument description when present. Sorted by symbol name.
    ``limit`` caps how many instruments are returned (full ML scans are expensive).

    If ``forex_only`` is True, only symbols MT5 reports as **Forex** (``trade_calc_mode`` 0 or 1)
    are included — stocks, indices, metals, and crypto are excluded.

    If ``max_spread_cost_usd`` is set (default 10), symbols with estimated **one-way** spread cost in
    deposit currency above that value (for ``volume_for_spread_check`` lots, usually 1.0) are omitted
    before applying ``limit``.

    Returns ``([], error_message)`` if the package is missing, MT5 is not connected, or the call fails.
    """
    if not package_installed():
        return [], "MetaTrader5 package not installed"
    with _mt5_lock:
        ok, err = ensure_mt5()
        if not ok:
            return [], err
        syms = mt5.symbols_get()
        if syms is None:
            return [], f"symbols_get failed: {mt5.last_error()}"
        pairs: list[tuple[str, str]] = []
        for s in syms:
            if not getattr(s, "visible", False):
                continue
            name = s.name
            if forex_only and not _is_forex_spot_symbol(name):
                continue
            desc = getattr(s, "description", None)
            label = (str(desc).strip() if desc else "") or name
            pairs.append((name, label))
        pairs.sort(key=lambda x: x[0])
    if max_spread_cost_usd is not None and max_spread_cost_usd > 0:
        pairs, _ = filter_pairs_by_max_spread_cost(
            pairs,
            max_spread_cost_usd,
            volume_for_spread_check,
        )
    if limit is not None and limit > 0 and len(pairs) > limit:
        pairs = pairs[:limit]
    return pairs, ""


def rates_to_ohlc_df(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    return df[["open", "high", "low", "close"]].astype(float)


def _atr14(ohlc: pd.DataFrame) -> float | None:
    """Last ATR(14) from OHLC, or None if insufficient data."""
    if ohlc is None or len(ohlc) < 16:
        return None
    h = ohlc["high"].astype(float)
    low = ohlc["low"].astype(float)
    c = ohlc["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([h - low, (h - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    if atr is None or pd.isna(atr):
        return None
    return float(atr)


def _round_for_symbol(symbol: str, x: float) -> float:
    """Round price for typical instrument classes."""
    s = symbol.upper()
    if "JPY" in s:
        return round(x, 3)
    if any(k in s for k in ("XAU", "XAG", "GOLD", "SILVER")):
        return round(x, 2)
    if "BTC" in s or "ETH" in s:
        return round(x, 2) if x > 500 else round(x, 4)
    if any(k in s for k in ("US500", "US30", "US100", "NAS100", "SPX")):
        return round(x, 2)
    return round(x, 5)


def _recommend_from_analysis_ohlc(symbol: str, action: str, ohlc: pd.DataFrame) -> dict[str, Any]:
    """Non-zero SL/TP from the same OHLC used for ML (last close + ATR / range)."""
    last_close = float(ohlc["close"].iloc[-1])
    atr = _atr14(ohlc) if len(ohlc) >= 16 else None
    if atr is not None and atr > 0:
        stop_dist = max(1.5 * atr, 0.0005 * last_close)
    else:
        tail = ohlc.tail(min(12, len(ohlc)))
        rng = float(tail["high"].max() - tail["low"].min())
        stop_dist = max(0.45 * rng, 0.0015 * last_close, last_close * 1e-7)
    tp_dist = 2.0 * stop_dist
    rf = lambda v: _round_for_symbol(symbol, v)
    if action == "BUY":
        entry = last_close
        sl = entry - stop_dist
        tp = entry + tp_dist
    else:
        entry = last_close
        sl = entry + stop_dist
        tp = entry - tp_dist
    return {
        "sl": rf(sl),
        "tp": rf(tp),
        "deviation": 25,
        "entry": rf(entry),
        "stop_distance_price": rf(stop_dist),
        "risk_reward": "1:2 (from analysis bars — verify on MT5 before live execution)",
        "rationale": (
            "SL/TP come from the same bars as this analysis (ATR or recent range vs last close). "
            "Increase or reduce before submitting. When MT5 is connected, live bid/ask levels are used automatically if available."
        ),
    }


def recommend_order_params(symbol: str, action: str, ohlc: pd.DataFrame | None) -> dict[str, Any]:
    """Suggest SL/TP (prices) and deviation (points). Prefer live MT5; else analysis OHLC.

    For BUY/SELL, SL and TP are non-zero whenever enough OHLC rows exist.
    """
    action = (action or "").strip().upper()
    empty: dict[str, Any] = {
        "sl": 0.0,
        "tp": 0.0,
        "deviation": 20,
        "entry": None,
        "stop_distance_price": None,
        "risk_reward": "n/a",
        "rationale": "",
    }
    if action not in ("BUY", "SELL"):
        empty["rationale"] = (
            "Signal is HOLD — no automatic SL/TP. Run Analyze again if you want directional levels."
        )
        return empty

    # 1) Live MT5 quote + ATR from analysis OHLC
    if package_installed() and mt5 is not None:
        with _mt5_lock:
            ok, _ = ensure_mt5()
            if ok:
                sym = symbol.upper().strip()
                ok_sym, _ = _symbol_select_retry(sym)
                if ok_sym:
                    tick = mt5.symbol_info_tick(sym)
                    si = mt5.symbol_info(sym)
                    if tick is not None and si is not None:
                        point = float(si.point)
                        digits = int(si.digits)
                        spread_pts = int(si.spread) if si.spread else 1
                        spread_price = spread_pts * point
                        stops_level_pts = int(getattr(si, "trade_stops_level", 0) or 0)
                        min_dist = stops_level_pts * point + point

                        atr = _atr14(ohlc) if ohlc is not None and len(ohlc) >= 16 else None
                        if atr is not None and atr > 0:
                            stop_dist = max(1.5 * atr, 2.5 * spread_price, min_dist)
                        else:
                            stop_dist = max(3.0 * spread_price, 10 * point, min_dist)

                        tp_dist = 2.0 * stop_dist
                        if action == "BUY":
                            entry = float(tick.ask)
                            sl = entry - stop_dist
                            tp = entry + tp_dist
                        else:
                            entry = float(tick.bid)
                            sl = entry + stop_dist
                            tp = entry - tp_dist

                        def rpx(x: float) -> float:
                            return round(x, digits)

                        dev = int(max(20, min(300, spread_pts * 2 + 10)))
                        atr_txt = f"{atr:.5f}" if atr else "n/a"
                        return {
                            "sl": rpx(sl),
                            "tp": rpx(tp),
                            "deviation": dev,
                            "entry": rpx(entry),
                            "stop_distance_price": round(stop_dist, digits),
                            "risk_reward": "1:2 (live bid/ask + analysis ATR)",
                            "rationale": (
                                f"Live quote + ATR(14)={atr_txt}, spread={spread_pts} pts. "
                                f"Stop distance ~ max(1.5xATR, 2.5xspread, broker min stop). "
                                f"TP at 2x stop distance - edit tighter/wider as you prefer."
                            ),
                        }

    # 2) Fallback: same OHLC as ML model (always fills non-zero SL/TP when possible)
    if ohlc is not None and len(ohlc) >= 5:
        return _recommend_from_analysis_ohlc(symbol, action, ohlc)

    empty["rationale"] = (
        "Not enough bars to estimate SL/TP — change timeframe or symbol and Analyze again."
    )
    return empty


def symbol_metrics(symbol: str) -> dict[str, Any] | None:
    if not package_installed():
        return None
    with _mt5_lock:
        ok, _ = ensure_mt5()
        if not ok:
            return None
        sym = symbol.upper().strip()
        ok_sym, _ = _symbol_select_retry(sym)
        if not ok_sym:
            return None
        si = mt5.symbol_info(sym)
        tick = mt5.symbol_info_tick(sym)
        if si is None or tick is None:
            return None
        return {
            "point": float(si.point),
            "digits": int(si.digits),
            "spread": int(si.spread) if si.spread else None,
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "volume_min": float(si.volume_min),
            "volume_max": float(si.volume_max),
            "volume_step": float(si.volume_step),
        }


def get_rates_dataframe(symbol: str, timeframe: str, count: int = 500) -> tuple[pd.DataFrame | None, str]:
    if not package_installed():
        return None, "MetaTrader5 not installed"

    tf = _tf_constant(timeframe)
    if tf is None:
        return None, f"Unknown timeframe: {timeframe}"

    sym = symbol.upper().strip()
    with _mt5_lock:
        ok, msg = ensure_mt5()
        if not ok:
            return None, msg

        ok_sym, err_sym = _symbol_select_retry(sym)
        if not ok_sym:
            return None, err_sym

        rates = mt5.copy_rates_from_pos(sym, tf, 0, count)
        if rates is None or len(rates) == 0:
            return None, f"No rates for {sym}: {mt5.last_error()}"

        return rates_to_ohlc_df(rates), "ok"


def account_snapshot() -> dict[str, Any] | None:
    if not package_installed():
        return None
    with _mt5_lock:
        ok, _ = ensure_mt5()
        if not ok:
            return None
        ai = mt5.account_info()
        if ai is None:
            return None
        return {
            "login": ai.login,
            "server": ai.server,
            "balance": round(ai.balance, 2),
            "equity": round(ai.equity, 2),
            "margin_free": round(ai.margin_free, 2),
            "currency": ai.currency,
            "trade_allowed": bool(ai.trade_allowed),
        }


def terminal_snapshot() -> dict[str, Any] | None:
    if not package_installed():
        return None
    with _mt5_lock:
        ok, _ = ensure_mt5()
        if not ok:
            return None
        t = mt5.terminal_info()
        if t is None:
            return None
        return {"connected": t.connected, "name": t.name}


def _order_check_ok(chk: Any) -> bool:
    if chk is None:
        return False
    rc = int(chk.retcode)
    done = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
    return rc in (0, done)


def _decimals_from_step(step: float) -> int:
    s = f"{step:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".", 1)[1])


def _normalize_volume_for_symbol(si: Any, volume: float) -> tuple[float, bool, str]:
    """Normalize user volume to broker constraints: [min, max] and step grid."""
    vmin = float(getattr(si, "volume_min", 0.01) or 0.01)
    vmax = float(getattr(si, "volume_max", 100.0) or 100.0)
    vstep = float(getattr(si, "volume_step", 0.01) or 0.01)
    if volume <= 0:
        return 0.0, False, f"Volume must be > 0 (symbol min {vmin}, step {vstep})."

    v = min(max(float(volume), vmin), vmax)
    units = (v - vmin) / vstep
    k = int(math.floor(units + 0.5))
    norm = vmin + k * vstep
    norm = min(max(norm, vmin), vmax)
    decimals = _decimals_from_step(vstep)
    norm = round(norm, decimals)
    changed = abs(norm - float(volume)) > (10 ** (-(decimals + 2)))
    msg = ""
    if changed:
        msg = (
            f"Volume adjusted from {float(volume)} to {norm} "
            f"(min={vmin}, max={vmax}, step={vstep})."
        )
    return norm, changed, msg


def _normalize_stops_for_symbol(
    si: Any,
    buy: bool,
    price: float,
    sl: float,
    tp: float,
) -> tuple[float, float, bool, str, int]:
    """Normalize SL/TP against broker min stop/freeze distance."""
    point = float(getattr(si, "point", 0.0) or 0.0)
    digits = int(getattr(si, "digits", 5) or 5)
    stop_pts = int(getattr(si, "trade_stops_level", 0) or 0)
    freeze_pts = int(getattr(si, "trade_freeze_level", 0) or 0)
    min_pts = max(stop_pts, freeze_pts, 1)
    min_dist = max(min_pts * point, point * 2.0)
    changed = False

    n_sl = float(sl) if float(sl) > 0 else 0.0
    n_tp = float(tp) if float(tp) > 0 else 0.0
    pad = point

    if buy:
        if n_sl > 0:
            max_sl = price - min_dist - pad
            if n_sl >= max_sl:
                n_sl = max_sl
                changed = True
        if n_tp > 0:
            min_tp = price + min_dist + pad
            if n_tp <= min_tp:
                n_tp = min_tp
                changed = True
    else:
        if n_sl > 0:
            min_sl = price + min_dist + pad
            if n_sl <= min_sl:
                n_sl = min_sl
                changed = True
        if n_tp > 0:
            max_tp = price - min_dist - pad
            if n_tp >= max_tp:
                n_tp = max_tp
                changed = True

    if n_sl > 0:
        n_sl = round(n_sl, digits)
    if n_tp > 0:
        n_tp = round(n_tp, digits)

    msg = ""
    if changed:
        msg = (
            f"Stops adjusted to broker limits (min distance {min_pts} points). "
            f"SL={n_sl or 0}, TP={n_tp or 0}."
        )
    return n_sl, n_tp, changed, msg, min_pts


def _latest_position_ticket(symbol: str) -> int | None:
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    latest = max(positions, key=lambda p: int(getattr(p, "time_msc", 0) or getattr(p, "time", 0)))
    return int(latest.ticket)


def _attach_stops_after_fill(
    symbol: str,
    buy: bool,
    si: Any,
    target_sl: float,
    target_tp: float,
) -> tuple[bool, float, float, str]:
    """Try setting SL/TP after deal execution (TRADE_ACTION_SLTP)."""
    if target_sl <= 0 and target_tp <= 0:
        return True, 0.0, 0.0, "No SL/TP requested."

    ticket = _latest_position_ticket(symbol)
    if ticket is None:
        return False, float(target_sl), float(target_tp), "Could not find open position ticket to attach stops."

    done_rc = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
    invalid_stops = int(getattr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10016))
    last_msg = "Failed to attach stops."
    cur_sl, cur_tp = float(target_sl), float(target_tp)

    for mult in (1.0, 1.5, 2.0, 3.0):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            last_msg = "No tick while attaching stops."
            break
        px = float(tick.ask) if buy else float(tick.bid)
        # widen distances gradually by scaling away from entry
        raw_sl = cur_sl
        raw_tp = cur_tp
        if mult > 1.0:
            if buy:
                if raw_sl > 0:
                    raw_sl = px - abs(px - raw_sl) * mult
                if raw_tp > 0:
                    raw_tp = px + abs(raw_tp - px) * mult
            else:
                if raw_sl > 0:
                    raw_sl = px + abs(raw_sl - px) * mult
                if raw_tp > 0:
                    raw_tp = px - abs(px - raw_tp) * mult
        nsl, ntp, _, _, min_pts = _normalize_stops_for_symbol(si, buy, px, raw_sl, raw_tp)
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": float(nsl),
            "tp": float(ntp),
        }
        chk = mt5.order_check(req)
        if chk is not None and int(chk.retcode) not in (0, done_rc, invalid_stops):
            last_msg = f"order_check SLTP retcode={chk.retcode}"
            continue
        res = mt5.order_send(req)
        if res is not None and int(res.retcode) == done_rc:
            return True, float(nsl), float(ntp), f"Stops attached after fill on position {ticket}."
        if res is not None:
            last_msg = f"Attach stops failed: {res.retcode} {res.comment}"
        else:
            last_msg = f"Attach stops failed: {mt5.last_error()}"

    return False, float(cur_sl), float(cur_tp), last_msg


def _select_filling_mode(symbol: str, request: dict[str, Any]) -> int:
    """Pick type_filling that the symbol/broker accept (avoids retcode 10030).

    symbol_info.filling_mode uses MQL5 *symbol* bits (not ORDER_FILLING enum values):
    SYMBOL_FILLING_FOK=1, IOC=2, RETURN=4 OR'ed together.
    """
    si = mt5.symbol_info(symbol)
    if si is None:
        return mt5.ORDER_FILLING_IOC

    fm = int(si.filling_mode)
    # MQL5 symbol bitmask — do not confuse with mt5.ORDER_FILLING_* integers
    sym_fok, sym_ioc, sym_ret = 1, 2, 4
    # Try IOC first (common for FX), then FOK, then RETURN
    pairs: list[tuple[int, Any]] = []
    if fm & sym_ioc:
        pairs.append((sym_ioc, mt5.ORDER_FILLING_IOC))
    if fm & sym_fok:
        pairs.append((sym_fok, mt5.ORDER_FILLING_FOK))
    if fm & sym_ret:
        pairs.append((sym_ret, mt5.ORDER_FILLING_RETURN))

    for _, order_fill in pairs:
        req = {**request, "type_filling": order_fill}
        chk = mt5.order_check(req)
        if _order_check_ok(chk):
            return int(order_fill)

    for _, order_fill in pairs:
        return int(order_fill)

    for order_fill in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        req = {**request, "type_filling": order_fill}
        chk = mt5.order_check(req)
        if _order_check_ok(chk):
            return int(order_fill)

    return int(mt5.ORDER_FILLING_IOC)


def execute_market_order(
    symbol: str,
    volume: float,
    buy: bool,
    sl: float = 0.0,
    tp: float = 0.0,
    deviation: int = 20,
    magic: int = 704050,
    comment: str = "Flask ML assistant",
) -> tuple[bool, str, dict[str, Any]]:
    """Send a market DEAL. sl/tp in price (not points). Use 0 to omit."""
    if not package_installed():
        return False, "MetaTrader5 not installed", {}

    sym = symbol.upper().strip()
    with _mt5_lock:
        ok, msg = ensure_mt5()
        if not ok:
            return False, msg, {}

        ok_sym, err_sym = _symbol_select_retry(sym)
        if not ok_sym:
            return False, err_sym, {}

        ai = mt5.account_info()
        if ai is not None and not ai.trade_allowed:
            return (
                False,
                "Automated trading may be disabled in MT5: Tools -> Options -> Expert Advisors -> "
                "Allow algorithmic trading. Also confirm the account allows trading.",
                {},
            )

        si = mt5.symbol_info(sym)
        if si is None:
            return False, "symbol_info failed", {}

        if not si.visible:
            return False, "symbol not visible", {}

        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            return False, "no tick", {}

        norm_volume, volume_changed, volume_msg = _normalize_volume_for_symbol(si, float(volume))
        if norm_volume <= 0:
            return False, volume_msg or "invalid volume", {}

        order_type = mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL
        price = tick.ask if buy else tick.bid
        norm_sl, norm_tp, stops_changed, stops_msg, min_stop_pts = _normalize_stops_for_symbol(
            si, buy, float(price), float(sl), float(tp)
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": sym,
            "volume": float(norm_volume),
            "type": order_type,
            "price": price,
            "sl": float(norm_sl),
            "tp": float(norm_tp),
            "deviation": int(deviation),
            "magic": magic,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
        }
        request["type_filling"] = _select_filling_mode(sym, request)

        done_rc = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
        invalid_fill = int(getattr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030))
        invalid_stops = int(getattr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10016))
        executed_without_stops = False
        attach_stops_ok = False
        attach_stops_msg = ""

        result = mt5.order_send(request)
        if result is None:
            return False, f"order_send None: {mt5.last_error()}", {}

        # 10030 Unsupported filling mode — retry other ORDER_FILLING types
        if int(result.retcode) == invalid_fill:
            for alt in (
                mt5.ORDER_FILLING_IOC,
                mt5.ORDER_FILLING_FOK,
                mt5.ORDER_FILLING_RETURN,
            ):
                if int(alt) == int(request["type_filling"]):
                    continue
                req2 = {**request, "type_filling": int(alt)}
                result = mt5.order_send(req2)
                if result is not None and int(result.retcode) == done_rc:
                    break

        # 10016 Invalid stops — recalc once with fresh tick and stricter broker distance
        if int(result.retcode) == invalid_stops and (float(norm_sl) > 0 or float(norm_tp) > 0):
            tick2 = mt5.symbol_info_tick(sym)
            if tick2 is not None:
                price2 = tick2.ask if buy else tick2.bid
                sl2, tp2, _, _, _ = _normalize_stops_for_symbol(
                    si, buy, float(price2), float(norm_sl), float(norm_tp)
                )
                req2 = {**request, "price": float(price2), "sl": float(sl2), "tp": float(tp2)}
                result = mt5.order_send(req2)
                norm_sl, norm_tp = sl2, tp2
            # Some brokers reject SL/TP on market deal for CFDs/stocks. Fallback: open first, attach later.
            if int(result.retcode) == invalid_stops:
                req3 = {**request, "sl": 0.0, "tp": 0.0}
                result3 = mt5.order_send(req3)
                if result3 is not None and int(result3.retcode) == done_rc:
                    result = result3
                    executed_without_stops = True
                    attach_stops_ok, a_sl, a_tp, attach_stops_msg = _attach_stops_after_fill(
                        sym, buy, si, float(norm_sl), float(norm_tp)
                    )
                    norm_sl, norm_tp = a_sl, a_tp

        out = {
            "retcode": result.retcode,
            "deal": result.deal,
            "order": result.order,
            "volume": result.volume,
            "requested_volume": float(volume),
            "normalized_volume": float(norm_volume),
            "volume_changed": bool(volume_changed),
            "requested_sl": float(sl),
            "requested_tp": float(tp),
            "normalized_sl": float(norm_sl),
            "normalized_tp": float(norm_tp),
            "stops_changed": bool(stops_changed),
            "min_stop_points": int(min_stop_pts),
            "executed_without_stops": bool(executed_without_stops),
            "stops_attached_after_fill": bool(attach_stops_ok),
            "stops_attach_message": attach_stops_msg,
            "price": result.price,
            "comment": result.comment,
        }

        if int(result.retcode) != done_rc:
            extra_parts = []
            if volume_msg:
                extra_parts.append(volume_msg)
            if stops_msg:
                extra_parts.append(stops_msg)
            if int(result.retcode) == invalid_stops:
                extra_parts.append(
                    "Broker still rejected stops (10016). Increase SL/TP distance or set SL/TP to 0 and place stops after fill."
                )
            extra = f" | {' '.join(extra_parts)}" if extra_parts else ""
            return False, f"Order rejected: {result.retcode} {result.comment}{extra}", out

        success_msg = "Order executed"
        success_parts = []
        if volume_msg:
            success_parts.append(volume_msg)
        if stops_msg:
            success_parts.append(stops_msg)
        if success_parts:
            success_msg += f" | {' '.join(success_parts)}"
        if executed_without_stops:
            if attach_stops_ok:
                success_msg += " | Market order filled first, then SL/TP attached successfully."
            else:
                success_msg += f" | Market order filled without SL/TP. {attach_stops_msg}"
        return True, success_msg, out


def _close_position_market(
    sym: str,
    position: Any,
    deviation: int,
    magic: int,
    comment: str,
) -> tuple[bool, str, dict[str, Any]]:
    """Close one open position by ticket (market). Caller must hold ``_mt5_lock``."""
    if not mt5:
        return False, "MT5 missing", {}
    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        return False, "no tick", {}
    pos_buy = int(getattr(mt5, "POSITION_TYPE_BUY", 0))
    pt = int(position.type)
    if pt == pos_buy:
        order_type = mt5.ORDER_TYPE_SELL
        price = float(tick.bid)
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = float(tick.ask)

    ticket = int(position.ticket)
    vol = float(position.volume)

    req: dict[str, Any] = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym,
        "volume": vol,
        "type": order_type,
        "position": ticket,
        "price": price,
        "deviation": int(deviation),
        "magic": magic,
        "comment": (comment[:24] + " X")[:31],
        "type_time": mt5.ORDER_TIME_GTC,
    }
    req["type_filling"] = _select_filling_mode(sym, req)

    done_rc = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
    invalid_fill = int(getattr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030))

    result = mt5.order_send(req)
    if result is None:
        return False, f"order_send None: {mt5.last_error()}", {}

    if int(result.retcode) == invalid_fill:
        for alt in (
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_RETURN,
        ):
            if int(alt) == int(req["type_filling"]):
                continue
            req2 = {**req, "type_filling": int(alt)}
            result = mt5.order_send(req2)
            if result is not None and int(result.retcode) == done_rc:
                break

    out = {
        "retcode": result.retcode if result else None,
        "deal": result.deal if result else None,
        "order": result.order if result else None,
        "ticket_closed": ticket,
        "volume": vol,
    }
    if result is None or int(result.retcode) != done_rc:
        return (
            False,
            f"Close failed: {result.retcode if result else '?'} {result.comment if result else ''}",
            out,
        )
    return True, f"Closed position {ticket}", out


def execute_market_order_with_flip(
    symbol: str,
    volume: float,
    buy: bool,
    sl: float = 0.0,
    tp: float = 0.0,
    deviation: int = 20,
    magic: int = 704050,
    comment: str = "Flask ML assistant",
) -> tuple[bool, str, dict[str, Any]]:
    """Place a market order after aligning with existing positions.

    - **Opposite** exposure (short vs long signal): closes conflicting positions first, then opens.
    - **Same side** already open: skips a new entry (no pyramiding from auto-scan).
    - **Flat**: opens normally.

    Returns ``details`` with ``new_order_placed``, ``skipped_same_side``, ``closed_positions``.
    """
    if not package_installed():
        return False, "MetaTrader5 not installed", {}

    sym = symbol.upper().strip()
    desired_buy = buy
    flip_details: dict[str, Any] = {
        "new_order_placed": False,
        "skipped_same_side": False,
        "closed_positions": [],
    }

    with _mt5_lock:
        ok, msg = ensure_mt5()
        if not ok:
            return False, msg, flip_details

        ok_sym, err_sym = _symbol_select_retry(sym)
        if not ok_sym:
            return False, err_sym, flip_details

        pos_buy = int(getattr(mt5, "POSITION_TYPE_BUY", 0))

        raw = mt5.positions_get(symbol=sym)
        positions: list[Any] = list(raw) if raw else []

        to_close: list[Any] = []
        for p in positions:
            pt = int(p.type)
            is_buy = pt == pos_buy
            if desired_buy and not is_buy:
                to_close.append(p)
            elif not desired_buy and is_buy:
                to_close.append(p)

        for p in to_close:
            c_ok, c_msg, c_out = _close_position_market(sym, p, deviation, magic, comment)
            flip_details["closed_positions"].append(
                {"ticket": int(p.ticket), "ok": c_ok, "message": c_msg, "details": c_out}
            )
            if not c_ok:
                return (
                    False,
                    f"Could not close opposite position {p.ticket}: {c_msg}",
                    flip_details,
                )
            time.sleep(0.05)

        raw2 = mt5.positions_get(symbol=sym)
        remaining: list[Any] = list(raw2) if raw2 else []
        for p in remaining:
            pt = int(p.type)
            is_buy = pt == pos_buy
            if desired_buy and is_buy:
                flip_details["skipped_same_side"] = True
                return (
                    True,
                    "Already have a BUY position for this symbol — skipped new entry.",
                    flip_details,
                )
            if not desired_buy and not is_buy:
                flip_details["skipped_same_side"] = True
                return (
                    True,
                    "Already have a SELL position for this symbol — skipped new entry.",
                    flip_details,
                )

    ok2, msg2, open_out = execute_market_order(
        sym,
        volume,
        desired_buy,
        sl=sl,
        tp=tp,
        deviation=deviation,
        magic=magic,
        comment=comment,
    )
    flip_details["open_order"] = open_out
    flip_details["new_order_placed"] = bool(ok2)
    # Merge user-facing message
    if flip_details["closed_positions"]:
        cls = len(flip_details["closed_positions"])
        msg2 = f"Closed {cls} opposite leg(s). {msg2}" if ok2 else msg2
    return ok2, msg2, flip_details


def live_trading_allowed() -> bool:
    """True if env LIVE_TRADING_ENABLED is set, or project file LIVE_TRADING_ENABLED.txt contains 1/true/yes/on."""
    env = os.environ.get("LIVE_TRADING_ENABLED", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    flag = _PROJECT_ROOT / "LIVE_TRADING_ENABLED.txt"
    if not flag.is_file():
        return False
    try:
        line = (
            flag.read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
            .lstrip("\ufeff")
            .split("\n", 1)[0]
            .strip()
            .lower()
        )
        return line in ("1", "true", "yes", "on")
    except OSError:
        return False
