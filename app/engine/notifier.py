"""Notifications Telegram Markdown de RwaMarket.

Silence gracieux : si `TELEGRAM_BOT_TOKEN` ou `TELEGRAM_CHAT_ID` est absent, les
fonctions retournent simplement False sans lever d'exception ni bloquer l'app.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

TELEGRAM_API = "https://api.telegram.org"
TELEGRAM_LIMIT = 3900
BRUSSELS_TZ = ZoneInfo("Europe/Brussels")


def _config() -> tuple[str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return token, chat_id


def telegram_enabled() -> bool:
    token, chat_id = _config()
    return bool(token and chat_id)


def _md_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    for char in ("_", "*", "[", "]", "`"):
        text = text.replace(char, f"\\{char}")
    return text


def _format_number(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return f"{number:,.{digits}f}{suffix}"


def _format_signed(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return f"{number:+,.{digits}f}{suffix}"


def _split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Découpe un message en morceaux sous la limite Telegram."""
    pieces: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return pieces


async def _post_message(text: str, parse_mode: str | None = "Markdown") -> bool:
    token, chat_id = _config()
    if not token or not chat_id:
        return False

    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{TELEGRAM_API}/bot{token}/sendMessage",
                json=payload,
            )
            if response.status_code == 200:
                try:
                    body = response.json()
                except (ValueError, json.JSONDecodeError):
                    body = {}
                return bool(body.get("ok", True))
    except Exception:
        return False
    return False


async def send_telegram_message(text: str, parse_mode: str | None = "Markdown") -> bool:
    """Envoie un message Telegram (Markdown par défaut), sans jamais lever d'erreur."""
    if not text or not telegram_enabled():
        return False

    chunks = _split_message(text)
    sent_any = False
    for chunk in chunks:
        ok = await _post_message(chunk, parse_mode=parse_mode)
        if not ok and parse_mode:
            ok = await _post_message(chunk, parse_mode=None)
        sent_any = sent_any or ok
    return sent_any


async def send_trade_alert(
    action: str,
    position: dict,
    trade: dict | None = None,
    account: dict | None = None,
    extra: str | None = None,
) -> bool:
    """Alerte d'ouverture ou de clôture d'un trade Paper Trading."""
    act = str(action or "").upper()
    title = "RwaMarket - Paper Trade Ouvert" if act == "OPEN" else "RwaMarket - Paper Trade Cloture"

    lines = [f"*{_md_escape(title)}*", ""]
    lines.append(f"*Symbole :* `{_md_escape(position.get('symbol'))}`")
    lines.append(f"*Marche :* {_md_escape(position.get('market_type'))}")
    lines.append(f"*Sens :* {_md_escape(position.get('side'))}")
    lines.append(f"*Entree :* {_format_number(position.get('entry_price'), 6)}")
    lines.append(f"*Quantite :* {_format_number(position.get('quantity'), 8)}")
    if position.get("stop_loss") is not None:
        lines.append(f"*Stop-Loss :* {_format_number(position.get('stop_loss'), 6)}")
    if position.get("take_profit") is not None:
        lines.append(f"*Take-Profit :* {_format_number(position.get('take_profit'), 6)}")

    if trade:
        lines.append("")
        lines.append(f"*Sortie :* {_format_number(trade.get('exit_price'), 6)}")
        lines.append(f"*PnL :* {_format_signed(trade.get('pnl_usd'), 2, ' $')}")
        lines.append(f"*PnL % :* {_format_signed(trade.get('pnl_pct'), 2, ' %')}")
        lines.append(f"*Raison :* {_md_escape(trade.get('exit_reason'))}")

    if account:
        lines.append("")
        lines.append(f"*Solde cash :* {_format_number(account.get('balance'), 2, ' $')}")
        lines.append(f"*PnL realise :* {_format_signed(account.get('pnl_usd'), 2, ' $')}")

    reasoning = position.get("reasoning_summary")
    if reasoning:
        lines.append("")
        lines.append(f"*Raisonnement :* {_md_escape(str(reasoning)[:700])}")
    if extra:
        lines.append("")
        lines.append(_md_escape(extra)[:900])

    return await send_telegram_message("\n".join(lines))


def _format_analysis_line(analysis: dict) -> str:
    symbol = _md_escape(analysis.get("symbol"))
    verdict = _md_escape(analysis.get("verdict") or "WATCH")
    confidence = _format_number(analysis.get("confidence"), 0, "%")
    plan = analysis.get("trade_plan") or {}
    entry = _format_number(plan.get("entry_price"), 4)
    stop = _format_number(plan.get("stop_loss"), 4)
    tp1 = _format_number(plan.get("take_profit_1"), 4)
    return f"- `{symbol}` : *{verdict}* ({confidence}) - Entree {entry} | SL {stop} | TP1 {tp1}"


async def send_morning_brief(
    screener_results: list[dict] | None = None,
    analyses: list[dict] | None = None,
    portfolio: dict | None = None,
    session_label: str = "Morning Brief",
) -> bool:
    """Envoie le récapitulatif matinal/après-midi formaté en Markdown."""
    now = datetime.now(BRUSSELS_TZ)
    lines = [
        f"*RwaMarket - {_md_escape(session_label)}*",
        f"_{now.strftime('%d/%m/%Y %H:%M')} (Bruxelles)_",
        "",
    ]

    results = screener_results or []
    lines.append("*Screener - Meilleures opportunites*")
    if results:
        for index, item in enumerate(results[:5], start=1):
            symbol = _md_escape(item.get("symbol"))
            score = _format_number(item.get("surge_score"), 1)
            grade = _md_escape(item.get("grade"))
            market = _md_escape(item.get("market_type"))
            price = _format_number(item.get("price"), 4)
            volume = _format_number(item.get("volume_ratio"), 2, "x")
            lines.append(
                f"{index}. `{symbol}` - *{score}/100* ({grade}, {market}) - "
                f"prix {price}, volume {volume}"
            )
    else:
        lines.append("- Aucun resultat exploitable pour le moment.")

    selected = analyses or []
    if selected:
        lines.append("")
        lines.append("*Analyses pre-calculees*")
        for analysis in selected[:3]:
            lines.append(_format_analysis_line(analysis))

    summary = (portfolio or {}).get("summary") if isinstance(portfolio, dict) else None
    if summary:
        lines.append("")
        lines.append("*Portefeuille Paper*")
        lines.append(f"- Equity : {_format_number(summary.get('equity_usd'), 2, ' $')}")
        lines.append(f"- Cash : {_format_number(summary.get('cash_available_usd'), 2, ' $')}")
        lines.append(f"- PnL realise : {_format_signed(summary.get('realized_pnl_usd'), 2, ' $')}")
        lines.append(f"- PnL latent : {_format_signed(summary.get('unrealized_pnl_usd'), 2, ' $')}")
        lines.append(f"- Win Rate : {_format_number(summary.get('win_rate_pct'), 1, ' %')}")
        lines.append(f"- Drawdown max : {_format_number(summary.get('max_drawdown_pct'), 2, ' %')}")

    lines.append("")
    lines.append("_Analyse educative, pas un conseil financier._")
    return await send_telegram_message("\n".join(lines))
