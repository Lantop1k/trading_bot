"""Human-readable trading instructions from ML output and context."""

from __future__ import annotations

from typing import Any


def build_trading_instructions(
    action: str,
    symbol: str,
    volume: float,
    confidence: float,
    timeframe: str,
    data_source: str,
    metrics: dict[str, Any] | None,
    criteria: str,
) -> dict[str, Any]:
    """Structured checklist and narrative for the trader."""
    spread_txt = ""
    price_ctx = ""
    if metrics:
        spr = metrics.get("spread")
        bid = metrics.get("bid")
        ask = metrics.get("ask")
        if spr is not None:
            spread_txt = f"Broker-reported spread (points): {spr}"
        if bid is not None and ask is not None:
            price_ctx = f"Bid {bid} | Ask {ask}"

    steps: list[str] = []
    if action == "BUY":
        steps = [
            "Confirm trend/context aligns with your criteria (below).",
            "Check spread and session liquidity before entering long.",
            "Use the Execute button only if volume matches your risk plan.",
            "Consider stop-loss below recent swing low; take-profit per your R:R.",
        ]
    elif action == "SELL":
        steps = [
            "Confirm trend/context aligns with your criteria (below).",
            "Check spread and session liquidity before entering short.",
            "Use the Execute button only if volume matches your risk plan.",
            "Consider stop-loss above recent swing high; take-profit per your R:R.",
        ]
    else:
        steps = [
            "No directional market entry suggested — stay flat or manage existing positions.",
            "Re-analyze after new candles form or if volatility regime changes.",
            "If you must trade, reduce size and widen discretion; ML confidence is not mandatory edge.",
        ]

    risk = [
        "Past performance of the model does not predict future results.",
        "Slippage, gaps, and spread widening can exceed backtests.",
        "Never risk more than your account rules allow on a single signal.",
    ]

    checklist = [
        f"Symbol {symbol} | Timeframe {timeframe} | Bars source: {data_source}",
        f"Suggested direction: {action} | Model confidence: {confidence:.1%}",
        spread_txt or "Spread: (connect MT5 for live spread)",
        price_ctx or "Prices: (connect MT5 for live quotes)",
        f"Order size reference (lots): {volume}",
    ]
    if criteria.strip():
        checklist.append("Your criteria noted — apply them before any execution.")

    return {
        "summary": f"{action} bias on {symbol} ({timeframe}) from ML — verify manually.",
        "checklist": checklist,
        "steps": steps,
        "risk": risk,
        "execute_eligible": action in ("BUY", "SELL"),
    }
