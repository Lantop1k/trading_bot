"""Flask MT5 ML strategy assistant — live rates via MetaTrader5 when available."""

from __future__ import annotations

import os
import time
import json
import threading as _threading
from collections import Counter

from flask import Flask, jsonify, render_template, request

from instrument_presets import PRESET_SYMBOLS, preset_forex_pairs
from ml.data_features import generate_synthetic_ohlc
from ml.instructions import build_trading_instructions
from ml.strategies import STRATEGY_KEYS, STRATEGY_LABELS, ohlc_to_candles_json, train_predict
from mt5_bridge import (
    account_snapshot,
    execute_market_order,
    execute_market_order_with_flip,
    filter_pairs_by_max_spread_cost,
    get_rates_dataframe,
    live_trading_allowed,
    market_watch_instruments,
    package_installed,
    recommend_order_params,
    symbol_metrics,
    terminal_snapshot,
    _mt5_lock,
    ensure_mt5,
    _close_position_market,
)

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

app = Flask(__name__)

# ---- Profit target background monitor (per-trade) ----
_pt_lock = _threading.Lock()
_pt_value = 0.0
_pt_active = False
_pt_stop_evt = _threading.Event()
_pt_thread = None


def _pt_per_trade_worker():
    """Background thread: checks every 0.5s and closes each position that hits the target."""
    while not _pt_stop_evt.is_set():
        try:
            with _pt_lock:
                target = _pt_value
                active = _pt_active
            if active and target > 0 and mt5 is not None:
                # get positions first without holding the lock long
                positions_to_close = []
                with _mt5_lock:
                    ok, _ = ensure_mt5()
                    if ok:
                        positions = mt5.positions_get()
                        if positions:
                            for pos in positions:
                                pos_profit = float(getattr(pos, "profit", 0))
                                if pos_profit >= target:
                                    positions_to_close.append(pos)
                # close each one individually with fresh lock acquisition
                for pos in positions_to_close:
                    for attempt in range(3):
                        try:
                            with _mt5_lock:
                                ok, _ = ensure_mt5()
                                if ok:
                                    _close_position_market(
                                        pos.symbol, pos, 25, 704050, "Per-trade PT hit"
                                    )
                            break
                        except Exception:
                            time.sleep(0.1)
        except Exception:
            pass
        _pt_stop_evt.wait(0.5)


def _start_pt_thread():
    global _pt_thread
    _pt_stop_evt.clear()
    if _pt_thread is None or not _pt_thread.is_alive():
        _pt_thread = _threading.Thread(target=_pt_per_trade_worker, daemon=True)
        _pt_thread.start()


def unique_symbols() -> list[tuple[str, str]]:
    """Unique symbols for dropdown and preset instrument scan, sorted alphabetically by symbol code."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for code, label in PRESET_SYMBOLS:
        c = code.strip().upper()
        if c in seen:
            continue
        seen.add(c)
        out.append((c, label))
    out.sort(key=lambda row: row[0])
    return out


TIMEFRAMES = [
    ("M1", "M1"),
    ("M5", "M5"),
    ("M15", "M15"),
    ("M30", "M30"),
    ("H1", "H1"),
    ("H4", "H4"),
    ("D1", "D1"),
]


def _fetch_ohlc(symbol: str, timeframe: str, bars: int = 500):
    """Return (DataFrame, data_source, error_or_empty)."""
    if package_installed():
        df, err = get_rates_dataframe(symbol, timeframe, count=bars)
        if df is not None and len(df) > 0:
            return df, "mt5_live", ""
    df = generate_synthetic_ohlc(symbol, bars=min(bars, 500))
    return df, "synthetic", "MT5 unavailable or symbol failed — using synthetic OHLC for ML only."


def _parse_scan_params_from_source(src: dict) -> dict:
    """Normalize instrument-scan parameters from request.args or JSON body."""
    timeframe = str(src.get("timeframe", "M15") or "M15").strip().upper() or "M15"
    strategy = str(src.get("strategy", STRATEGY_KEYS[0]) or "").strip()
    if strategy not in STRATEGY_KEYS:
        strategy = STRATEGY_KEYS[0]
    try:
        amount = float(src.get("amount", 0.1))
    except (TypeError, ValueError):
        amount = 0.1
    criteria = str(src.get("criteria", "") or "")

    instruments_mode = str(src.get("instruments", "market_watch") or "market_watch").strip().lower()
    if instruments_mode not in ("market_watch", "preset"):
        instruments_mode = "market_watch"

    fo = src.get("forex_only", True)
    if isinstance(fo, bool):
        forex_only = fo
    else:
        forex_only = str(fo).strip().lower() in ("1", "true", "yes", "on")

    try:
        limit = int(src.get("limit", 96))
    except (TypeError, ValueError):
        limit = 96
    limit = max(1, min(limit, 400))

    try:
        max_spread_usd = float(
            src.get("max_spread_usd", os.environ.get("MAX_SPREAD_USD", "10")),
        )
    except (TypeError, ValueError):
        max_spread_usd = 10.0
    max_spread_usd = max(0.0, max_spread_usd)

    try:
        max_trades = int(src.get("max_trades", 8))
    except (TypeError, ValueError):
        max_trades = 8
    max_trades = max(1, min(max_trades, 200))

    raw_strategy_params = src.get("strategy_params")
    if not isinstance(raw_strategy_params, dict):
        raw_json = src.get("strategy_params_json")
        if isinstance(raw_json, str) and raw_json.strip():
            try:
                parsed = json.loads(raw_json)
                raw_strategy_params = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                raw_strategy_params = {}
        else:
            raw_strategy_params = {}

    return {
        "timeframe": timeframe,
        "strategy": strategy,
        "amount": amount,
        "criteria": criteria,
        "instruments_mode": instruments_mode,
        "forex_only": forex_only,
        "limit": limit,
        "max_spread_usd": max_spread_usd,
        "max_trades": max_trades,
        "strategy_params": raw_strategy_params,
    }


def _resolve_instrument_pairs(p: dict) -> tuple[list[tuple[str, str]], str | None, str]:
    """Choose instrument list (same logic as /api/scan_symbols)."""
    limit = p["limit"]
    instruments_mode = p["instruments_mode"]
    forex_only = p["forex_only"]
    max_spread_usd = p["max_spread_usd"]
    spread_cap = max_spread_usd if max_spread_usd > 0 else None

    scan_note: str | None = None
    instruments_source = instruments_mode

    def _cap(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return pairs[:limit] if len(pairs) > limit else pairs

    if instruments_mode == "market_watch":
        pairs, mw_err = market_watch_instruments(
            limit=limit,
            forex_only=forex_only,
            max_spread_cost_usd=spread_cap,
        )
        if not pairs:
            parts: list[str] = []
            if mw_err:
                parts.append(mw_err)
            if forex_only:
                parts.append(
                    "No forex symbols in Market Watch (or MT5 unavailable) — using the app forex preset list. "
                    "Add FX pairs to Market Watch or disable Forex only to include stocks and other classes."
                )
                scan_note = " ".join(parts)
                pairs = preset_forex_pairs()
                instruments_source = "preset_forex_fallback"
            else:
                parts.append(
                    "Market Watch is empty or unavailable — using the full app preset list instead. "
                    "Add instruments to Market Watch in MT5 (View → Market Watch) for a full terminal scan."
                )
                scan_note = " ".join(parts)
                pairs = unique_symbols()
                instruments_source = "preset_fallback"
            if spread_cap is not None:
                pairs, n_fb = filter_pairs_by_max_spread_cost(pairs, spread_cap, 1.0)
                if n_fb:
                    fb_note = (
                        f"Removed {n_fb} instrument(s) with spread cost > {max_spread_usd} "
                        "(deposit currency, ~1.0 lot)."
                    )
                    scan_note = (scan_note + " " + fb_note) if scan_note else fb_note
            pairs = _cap(pairs)
    else:
        if forex_only:
            pairs = preset_forex_pairs()
        else:
            pairs = unique_symbols()
        if spread_cap is not None:
            pairs, n_drop = filter_pairs_by_max_spread_cost(pairs, spread_cap, 1.0)
            if n_drop:
                extra = (
                    f"Removed {n_drop} instrument(s) with estimated spread cost "
                    f"> {max_spread_usd} (deposit currency, ~1.0 lot)."
                )
                scan_note = (scan_note + " " + extra) if scan_note else extra
        pairs = _cap(pairs)

    return pairs, scan_note, instruments_source


def _score_instrument(
    code: str,
    label: str,
    timeframe: str,
    strategy: str,
    amount: float,
    criteria: str,
    strategy_params: dict[str, object] | None = None,
) -> tuple[dict, object | None]:
    """One scan row + OHLC for SL/TP (or None on error)."""
    try:
        ohlc, data_source, fetch_warn = _fetch_ohlc(code, timeframe, bars=500)
        result = train_predict(
            ohlc,
            strategy,
            amount,
            criteria,
            strategy_params=strategy_params,
        )
        row = {
            "symbol": code,
            "label": label,
            "action": result.action,
            "action_code": result.action_code,
            "confidence": round(result.confidence, 4),
            "probs": {k: round(v, 4) for k, v in result.probs.items()},
            "data_source": data_source,
            "fetch_warn": fetch_warn or None,
            "row_error": None,
        }
        return row, ohlc
    except Exception as row_exc:
        row = {
            "symbol": code,
            "label": label,
            "action": "HOLD",
            "action_code": 0,
            "confidence": 0.0,
            "probs": {"SELL": 0.0, "HOLD": 1.0, "BUY": 0.0},
            "data_source": "error",
            "fetch_warn": str(row_exc),
            "row_error": str(row_exc),
        }
        return row, None


@app.route("/")
def index():
    return render_template(
        "index.html",
        symbols=unique_symbols(),
        strategies=[(k, STRATEGY_LABELS[k]) for k in STRATEGY_KEYS],
        timeframes=TIMEFRAMES,
        live_trading_env=live_trading_allowed(),
    )


@app.route("/api/mt5/status")
def api_mt5_status():
    snap = account_snapshot()
    term = terminal_snapshot()
    return jsonify(
        {
            "package_installed": package_installed(),
            "connected": snap is not None,
            "account": snap,
            "terminal": term,
            "live_trading_enabled": live_trading_allowed(),
        }
    )


@app.route("/api/live_rates")
def api_live_rates():
    symbol = request.args.get("symbol", "EURUSD").strip().upper()
    timeframe = request.args.get("timeframe", "M15").strip().upper()
    try:
        count = min(int(request.args.get("count", 400)), 2000)
    except ValueError:
        count = 400

    if not package_installed():
        return jsonify({"ok": False, "error": "MetaTrader5 not installed", "candles": []}), 200

    df, msg = get_rates_dataframe(symbol, timeframe, count=count)
    if df is None:
        return jsonify({"ok": False, "error": msg, "candles": []}), 200

    candles = ohlc_to_candles_json(df)
    return jsonify({"ok": True, "symbol": symbol, "timeframe": timeframe, "candles": candles})


@app.route("/api/scan_strategies")
def api_scan_strategies():
    """Run every registered strategy on the same OHLC; used by the live multi-strategy panel."""
    symbol = request.args.get("symbol", "EURUSD").strip().upper() or "EURUSD"
    timeframe = request.args.get("timeframe", "M15").strip().upper() or "M15"
    try:
        amount = float(request.args.get("amount", 0.1))
    except (TypeError, ValueError):
        amount = 0.1
    criteria = str(request.args.get("criteria", ""))
    scan_params = _parse_scan_params_from_source(request.args.to_dict(flat=True))
    strategy_params = scan_params.get("strategy_params") or {}

    ohlc, data_source, fetch_warn = _fetch_ohlc(symbol, timeframe, bars=500)
    rows: list[dict] = []
    for key in STRATEGY_KEYS:
        result = train_predict(ohlc, key, amount, criteria, strategy_params=strategy_params)
        rows.append(
            {
                "key": key,
                "label": STRATEGY_LABELS.get(key, key),
                "action": result.action,
                "action_code": result.action_code,
                "confidence": round(result.confidence, 4),
                "probs": {k: round(v, 4) for k, v in result.probs.items()},
            }
        )

    votes = Counter(r["action"] for r in rows)
    consensus = votes.most_common(1)[0][0] if rows else "HOLD"
    total = len(rows)
    majority_count = votes.most_common(1)[0][1] if rows else 0

    return jsonify(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "amount": amount,
            "data_source": data_source,
            "fetch_warn": fetch_warn or None,
            "consensus": consensus,
            "consensus_count": majority_count,
            "consensus_total": total,
            "vote_counts": dict(votes),
            "strategies": rows,
            "updated_at": time.time(),
        }
    )


@app.route("/api/scan_symbols")
def api_scan_symbols():
    """Run the selected strategy on each instrument; recommends BUY / SELL / HOLD per row."""
    try:
        return _api_scan_symbols_core()
    except Exception as e:
        app.logger.exception("api_scan_symbols failed")
        return jsonify(
            {
                "ok": False,
                "error": str(e),
                "timeframe": request.args.get("timeframe", "M15"),
                "strategy": "",
                "strategy_label": "",
                "symbols": [],
                "vote_counts": {},
                "consensus": "HOLD",
                "consensus_count": 0,
                "consensus_total": 0,
                "data_summary": "error",
                "mt5_symbol_count": 0,
                "synthetic_symbol_count": 0,
                "instruments": "",
                "instrument_limit": 0,
                "scan_note": None,
                "updated_at": time.time(),
            }
        )


def _api_scan_symbols_core():
    """Inner implementation for /api/scan_symbols."""
    p = _parse_scan_params_from_source(request.args.to_dict(flat=True))
    timeframe = p["timeframe"]
    strategy = p["strategy"]
    amount = p["amount"]
    criteria = p["criteria"]
    max_spread_usd = p["max_spread_usd"]
    limit = p["limit"]
    forex_only = p["forex_only"]
    strategy_params = p.get("strategy_params") or {}

    pairs, scan_note, instruments_source = _resolve_instrument_pairs(p)

    rows: list[dict] = []
    mt5_n = 0
    synth_n = 0
    for code, label in pairs:
        row, _ohlc = _score_instrument(
            code,
            label,
            timeframe,
            strategy,
            amount,
            criteria,
            strategy_params=strategy_params,
        )
        rows.append(row)
        if not row.get("row_error"):
            if row["data_source"] == "mt5_live":
                mt5_n += 1
            else:
                synth_n += 1

    votes = Counter(r["action"] for r in rows if not r.get("row_error"))
    top = votes.most_common(1)
    consensus = top[0][0] if top else "HOLD"
    majority_count = top[0][1] if top else 0
    if synth_n == 0:
        data_summary = "mt5_live"
    elif mt5_n == 0:
        data_summary = "synthetic"
    else:
        data_summary = "mixed"

    return jsonify(
        {
            "ok": True,
            "timeframe": timeframe,
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
            "amount": amount,
            "forex_only": forex_only,
            "max_spread_usd": max_spread_usd,
            "instruments": instruments_source,
            "instrument_limit": limit,
            "scan_note": scan_note,
            "data_summary": data_summary,
            "mt5_symbol_count": mt5_n,
            "synthetic_symbol_count": synth_n,
            "consensus": consensus,
            "consensus_count": majority_count,
            "consensus_total": len(rows),
            "vote_counts": dict(votes),
            "symbols": rows,
            "updated_at": time.time(),
        }
    )


@app.route("/api/mt5/auto_execute_scan", methods=["POST"])
def api_mt5_auto_execute_scan():
    """Execute market orders for every BUY / SELL in the current instrument scan."""
    if not live_trading_allowed():
        msg = (
            "Live trading is disabled. Set environment variable LIVE_TRADING_ENABLED=1, "
            "or create LIVE_TRADING_ENABLED.txt with 1 in the project folder, then restart the app."
        )
        return jsonify({"ok": False, "error": msg, "message": msg}), 403

    data = request.get_json(silent=True) or {}
    if data.get("confirm") != "AUTO_EXECUTE":
        return (
            jsonify(
                {
                    "ok": False,
                    "error": 'Send JSON {"confirm": "AUTO_EXECUTE"} to acknowledge automated orders.',
                }
            ),
            400,
        )

    p = _parse_scan_params_from_source(data)
    pairs, _scan_note, _src = _resolve_instrument_pairs(p)
    timeframe = p["timeframe"]
    strategy = p["strategy"]
    amount = p["amount"]
    criteria = p["criteria"]
    strategy_params = p.get("strategy_params") or {}
    max_trades = p["max_trades"]

    results: list[dict] = []
    executed_ok = 0
    candidates: list[dict] = []
    for code, label in pairs:
        row, ohlc = _score_instrument(
            code,
            label,
            timeframe,
            strategy,
            amount,
            criteria,
            strategy_params=strategy_params,
        )
        if row.get("row_error"):
            results.append(
                {
                    "symbol": code,
                    "ok": False,
                    "skipped": True,
                    "reason": "row_error",
                    "message": row.get("row_error"),
                }
            )
            continue
        action = str(row.get("action") or "HOLD").upper()
        if action not in ("BUY", "SELL"):
            results.append({"symbol": code, "skipped": True, "reason": "hold_or_flat", "action": action})
            continue
        if row.get("data_source") != "mt5_live":
            results.append(
                {
                    "symbol": code,
                    "skipped": True,
                    "reason": "not_mt5_live",
                    "action": action,
                    "message": "No live MT5 prices for this symbol — order not sent.",
                }
            )
            continue
        if ohlc is None:
            results.append({"symbol": code, "ok": False, "skipped": True, "reason": "no_ohlc"})
            continue

        candidates.append(
            {
                "symbol": code,
                "label": label,
                "action": action,
                "confidence": float(row.get("confidence") or 0.0),
                "ohlc": ohlc,
            }
        )

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    selected = candidates[:max_trades]
    overflow = candidates[max_trades:]
    for item in overflow:
        results.append(
            {
                "symbol": item["symbol"],
                "label": item["label"],
                "action": item["action"],
                "confidence": item["confidence"],
                "skipped": True,
                "reason": "max_trades_limit",
                "message": f"Skipped by max_trades={max_trades} (lower confidence rank).",
            }
        )

    for item in selected:
        code = item["symbol"]
        label = item["label"]
        action = item["action"]
        ohlc = item["ohlc"]
        rec = recommend_order_params(code, action, ohlc)
        try:
            sl = float(rec.get("sl") or 0)
            tp = float(rec.get("tp") or 0)
            deviation = int(rec.get("deviation") or 25)
        except (TypeError, ValueError):
            sl, tp, deviation = 0.0, 0.0, 25
        if sl <= 0 or tp <= 0:
            results.append(
                {
                    "symbol": code,
                    "action": action,
                    "ok": False,
                    "skipped": True,
                    "reason": "no_stops",
                    "message": "Could not build SL/TP for this symbol.",
                }
            )
            continue

        buy = action == "BUY"
        ok, msg, flip_details = execute_market_order_with_flip(
            code,
            amount,
            buy,
            sl=sl,
            tp=tp,
            deviation=deviation,
            comment="Auto ML scan",
        )
        open_part = flip_details.get("open_order") if isinstance(flip_details, dict) else None
        results.append(
            {
                "symbol": code,
                "label": label,
                "action": action,
                "ok": ok,
                "message": msg,
                "details": open_part,
                "flip": flip_details,
                "confidence": item["confidence"],
                "sl": sl,
                "tp": tp,
                "deviation": deviation,
            }
        )
        if flip_details.get("new_order_placed"):
            executed_ok += 1
        time.sleep(0.15)

    return jsonify(
        {
            "ok": True,
            "executed_ok": executed_ok,
            "results": results,
            "timeframe": timeframe,
            "strategy": strategy,
            "amount": amount,
            "max_trades": max_trades,
            "eligible_trades": len(candidates),
            "updated_at": time.time(),
        }
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    symbol = str(data.get("symbol", "EURUSD")).strip().upper() or "EURUSD"
    timeframe = str(data.get("timeframe", "M15")).strip().upper() or "M15"
    try:
        amount = float(data.get("amount", 0.1))
    except (TypeError, ValueError):
        amount = 0.1
    criteria = str(data.get("criteria", ""))
    strategy = str(data.get("strategy", STRATEGY_KEYS[0])).strip()
    strategy_params = data.get("strategy_params")
    if not isinstance(strategy_params, dict):
        strategy_params = {}

    ohlc, data_source, fetch_warn = _fetch_ohlc(symbol, timeframe, bars=400)
    result = train_predict(
        ohlc,
        strategy,
        amount,
        criteria,
        strategy_params=strategy_params,
    )

    note_extra = []
    if fetch_warn:
        note_extra.append(fetch_warn)
    if data_source == "mt5_live":
        note_extra.append(f"OHLC from MT5 ({timeframe}), last bar updates with terminal.")
    else:
        note_extra.append("Chart/ML may use synthetic data until MT5 connects and symbol is valid.")

    full_note = result.note
    if note_extra:
        full_note = full_note + " | " + " ".join(note_extra)

    candles = ohlc_to_candles_json(ohlc)
    metrics = symbol_metrics(symbol) if package_installed() else None
    instructions = build_trading_instructions(
        result.action,
        symbol,
        amount,
        result.confidence,
        timeframe,
        data_source,
        metrics,
        criteria,
    )

    order_rec = recommend_order_params(symbol, result.action, ohlc)

    return jsonify(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "amount": amount,
            "data_source": data_source,
            "candles": candles,
            "prediction": {
                "action": result.action,
                "action_code": result.action_code,
                "confidence": round(result.confidence, 4),
                "probs": {k: round(v, 4) for k, v in result.probs.items()},
                "note": full_note,
            },
            "instructions": instructions,
            "symbol_metrics": metrics,
            "order_recommendation": order_rec,
        }
    )


@app.route("/api/mt5/close_all", methods=["POST"])
def api_mt5_close_all():
    if not live_trading_allowed():
        return jsonify({"ok": False, "error": "Live trading disabled"}), 403

    data = request.get_json(silent=True) or {}
    if data.get("confirm") != "CLOSE_ALL":
        return jsonify({"ok": False, "error": "Send confirm: CLOSE_ALL"}), 400

    if mt5 is None:
        return jsonify({"ok": False, "error": "MetaTrader5 package not installed"}), 500

    closed = 0
    errors = []
    with _mt5_lock:
        ok, msg = ensure_mt5()
        if not ok:
            return jsonify({"ok": False, "error": msg}), 500
        positions = mt5.positions_get()
        if not positions:
            return jsonify({"ok": True, "message": "No open positions to close.", "closed": 0})
        for pos in positions:
            sym = pos.symbol
            ok_c, msg_c, _ = _close_position_market(sym, pos, 25, 704050, "Profit target hit")
            if ok_c:
                closed += 1
            else:
                errors.append(f"{sym}: {msg_c}")

    msg_out = f"Closed {closed} position(s)."
    if errors:
        msg_out += " Errors: " + "; ".join(errors)
    return jsonify({"ok": len(errors) == 0, "message": msg_out, "closed": closed, "errors": errors})


@app.route("/api/mt5/profit_target/set", methods=["POST"])
def api_set_profit_target():
    global _pt_value, _pt_active
    data = request.get_json(silent=True) or {}
    try:
        target = float(data.get("target", 0))
    except (TypeError, ValueError):
        target = 0.0
    active = bool(data.get("active", False))
    with _pt_lock:
        _pt_value = target
        _pt_active = active and target > 0
    if _pt_active:
        _start_pt_thread()
    return jsonify({"ok": True, "target": _pt_value, "active": _pt_active})


@app.route("/api/mt5/profit_target/status")
def api_profit_target_status():
    with _pt_lock:
        return jsonify({"target": _pt_value, "active": _pt_active})


@app.route("/api/mt5/execute", methods=["POST"])
def api_mt5_execute():
    if not live_trading_allowed():
        msg = (
            "Live trading is disabled. Set environment variable LIVE_TRADING_ENABLED=1, "
            "or create LIVE_TRADING_ENABLED.txt with 1 in the project folder, then restart the app."
        )
        return jsonify({"ok": False, "error": msg, "message": msg}), 403

    data = request.get_json(silent=True) or {}
    if data.get("confirm") != "EXECUTE":
        return jsonify({"ok": False, "error": 'Send JSON {"confirm": "EXECUTE"} to acknowledge real orders.'}), 400

    symbol = str(data.get("symbol", "")).strip().upper()
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400

    try:
        volume = float(data.get("volume", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid volume"}), 400

    action = str(data.get("action", "")).strip().upper()
    if action not in ("BUY", "SELL"):
        return jsonify({"ok": False, "error": "action must be BUY or SELL"}), 400

    try:
        sl = float(data.get("sl", 0) or 0)
        tp = float(data.get("tp", 0) or 0)
        deviation = int(data.get("deviation", 20))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid sl/tp/deviation"}), 400

    buy = action == "BUY"
    ok, msg, details = execute_market_order(
        symbol,
        volume,
        buy,
        sl=sl,
        tp=tp,
        deviation=deviation,
    )
    return jsonify({"ok": ok, "message": msg, "details": details})


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true")
    app.run(
        debug=debug,
        use_reloader=False,
        threaded=True,
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
    )