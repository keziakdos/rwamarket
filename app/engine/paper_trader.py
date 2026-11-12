"""Moteur Paper Trading de RwaMarket.

Inspiré d'AgenticTrading : portefeuille simulé, ouverture Long/Short,
surveillance automatique des Stop-Loss / Take-Profit, archivage des trades,
mise à jour de la mémoire et alertes Telegram.

Comptabilité retenue (cohérente avec `database.create_position`) :
- l'ouverture réserve le notionnel (`entry_price * quantity`) sur le cash ;
- la clôture restitue le notionnel réservé et applique le PnL réalisé.
"""

from __future__ import annotations

import asyncio
import math
import os
from datetime import UTC, datetime
from typing import Any

from app import database as db
from app.agents.memory import update_trading_memory
from app.engine import data_fetcher as fetcher
from app.engine import notifier

PROFILE_DEFAULTS = {
    "conservative": {
        "position_size_pct": 2.0,
        "stop_pct": 2.0,
        "tp_pct": 4.0,
        "atr_multiplier": 2.0,
        "rr": 2.0,
    },
    "moderate": {
        "position_size_pct": 5.0,
        "stop_pct": 3.0,
        "tp_pct": 6.0,
        "atr_multiplier": 1.6,
        "rr": 2.0,
    },
    "aggressive": {
        "position_size_pct": 10.0,
        "stop_pct": 5.0,
        "tp_pct": 10.0,
        "atr_multiplier": 1.3,
        "rr": 2.4,
    },
}

# Runner : une fois +1R atteint, le stop passe au point mort puis suit le prix.
RUNNER_TRIGGER_R = float(os.getenv("RUNNER_TRIGGER_R", "1.0"))
RUNNER_TRAIL_PCT = float(os.getenv("RUNNER_TRAIL_PCT", "2.5"))
# Pyramid : ajout uniquement sur position gagnante, plafonné à une fraction du notionnel initial.
PYRAMID_MIN_PROFIT_R = float(os.getenv("PYRAMID_MIN_PROFIT_R", "1.0"))
PYRAMID_MAX_ADD_FRACTION = float(os.getenv("PYRAMID_MAX_ADD_FRACTION", "0.5"))


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def _normalize_profile(risk_profile: str | None) -> str:
    profile = (risk_profile or os.getenv("DEFAULT_RISK_PROFILE", "moderate")).strip().lower()
    return profile if profile in PROFILE_DEFAULTS else "moderate"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _normalize_side(side: str | None) -> str | None:
    value = (side or "").strip().lower()
    mapping = {
        "buy": "long",
        "achat": "long",
        "long": "long",
        "sell": "short",
        "vente": "short",
        "short": "short",
    }
    return mapping.get(value)


def _round_quantity(quantity: float) -> float:
    if abs(quantity) >= 1:
        return round(quantity, 6)
    return round(quantity, 8)


def _round_price(price: float) -> float:
    if abs(price) >= 100:
        return round(price, 2)
    if abs(price) >= 1:
        return round(price, 4)
    return round(price, 6)


def _estimate_equity() -> float:
    account = db.get_paper_account()
    reserved = 0.0
    for position in db.get_open_positions():
        reserved += _as_float(
            position.get("reserved_cash"),
            _as_float(position.get("entry_price")) * _as_float(position.get("quantity")),
        )
    return _as_float(account.get("balance")) + reserved


def _default_stops(
    entry_price: float,
    side: str,
    risk_profile: str,
    atr: float | None = None,
) -> tuple[float, float]:
    params = PROFILE_DEFAULTS[_normalize_profile(risk_profile)]
    if atr and atr > 0:
        stop_distance = params["atr_multiplier"] * atr
        tp_distance = stop_distance * params["rr"]
    else:
        stop_distance = entry_price * (params["stop_pct"] / 100.0)
        tp_distance = entry_price * (params["tp_pct"] / 100.0)

    if side == "long":
        return (
            _round_price(entry_price - stop_distance),
            _round_price(entry_price + tp_distance),
        )
    return (
        _round_price(entry_price + stop_distance),
        _round_price(entry_price - tp_distance),
    )


async def get_current_price(
    symbol: str,
    market_type: str | None = None,
) -> tuple[float | None, dict | None]:
    """Prix actuel d'un symbole ; retourne (prix, snapshot) ou (None, None)."""
    market = (market_type or fetcher.detect_market_type(symbol)).lower()
    normalized = fetcher.normalize_symbol(symbol, market)
    try:
        if market == "crypto":
            snapshot = await fetcher.fetch_crypto_price(normalized)
        else:
            snapshot = await fetcher.fetch_stock_quote(normalized)
        price = _as_float((snapshot or {}).get("price"), 0.0)
        if price > 0:
            return price, snapshot
    except Exception:
        pass

    try:
        df = await fetcher.fetch_ohlcv(normalized, timeframe="1d", limit=5, market_type=market)
        if df is not None and not getattr(df, "empty", True):
            price = _as_float(df["Close"].iloc[-1], 0.0)
            if price > 0:
                return price, {
                    "symbol": normalized,
                    "market_type": market,
                    "price": price,
                    "source": "ohlcv_fallback",
                    "fetched_at": datetime.now(UTC).isoformat(),
                }
    except Exception:
        pass
    return None, None


async def _fetch_prices_for_positions(positions: list[dict]) -> dict[tuple[str, str], float]:
    """Déduplique les appels réseau par couple (symbole, marché)."""
    unique: dict[tuple[str, str], str] = {}
    for position in positions:
        symbol = str(position.get("symbol") or "").upper()
        market = str(position.get("market_type") or "").lower()
        if symbol:
            unique[(symbol, market)] = symbol

    async def _one(key: tuple[str, str], symbol: str) -> tuple[tuple[str, str], float | None]:
        price, _ = await get_current_price(symbol, key[1])
        return key, price

    results = await asyncio.gather(
        *[_one(key, symbol) for key, symbol in unique.items()],
        return_exceptions=True,
    )

    prices: dict[tuple[str, str], float] = {}
    for result in results:
        if isinstance(result, Exception):
            continue
        key, price = result
        if price is not None:
            prices[key] = price
    return prices


def _unrealized(position: dict, current_price: float) -> tuple[float, float]:
    entry = _as_float(position.get("entry_price"))
    quantity = _as_float(position.get("quantity"))
    side = position.get("side")
    if side == "short":
        pnl = (entry - current_price) * quantity
    else:
        pnl = (current_price - entry) * quantity
    notional = entry * quantity
    pnl_pct = (pnl / notional * 100.0) if notional else 0.0
    return round(pnl, 8), round(pnl_pct, 4)


def _profit_in_r(position: dict, current_price: float) -> float:
    """Retourne le profit latent exprimé en multiples de risque initial (R)."""
    entry = _as_float(position.get("entry_price"))
    stop = _as_float(position.get("stop_loss"))
    side = position.get("side")
    risk_per_unit = abs(entry - stop)
    # Une fois le stop au point mort (ou au-delà), on conserve une référence de
    # risque fixe basée sur le profil pour continuer à suivre le runner.
    if risk_per_unit <= 0 or (side == "long" and stop >= entry) or (
        side == "short" and stop <= entry
    ):
        profile = _normalize_profile(position.get("risk_profile"))
        risk_per_unit = entry * (PROFILE_DEFAULTS[profile]["stop_pct"] / 100.0)
    if risk_per_unit <= 0 or entry <= 0:
        return 0.0
    if side == "short":
        return (entry - current_price) / risk_per_unit
    return (current_price - entry) / risk_per_unit


def _runner_stop_update(position: dict, current_price: float) -> tuple[float | None, float]:
    """Calcule le nouveau stop de suivi (runner) inspiré de PRISM-Insight.

    Dès que le trade atteint ``RUNNER_TRIGGER_R``, le stop est remonté au point
    mort puis suit le prix d'un pourcentage fixe. Retourne (nouveau_stop, R).
    """
    entry = _as_float(position.get("entry_price"))
    stop = _as_float(position.get("stop_loss"), 0.0)
    side = position.get("side")
    profit_r = _profit_in_r(position, current_price)
    if side not in {"long", "short"} or stop <= 0 or entry <= 0:
        return None, profit_r
    if profit_r < RUNNER_TRIGGER_R:
        return None, profit_r

    trail_distance = current_price * (RUNNER_TRAIL_PCT / 100.0)
    if side == "long":
        candidate = max(entry, current_price - trail_distance)
        if candidate > stop:
            return _round_price(candidate), profit_r
    else:
        candidate = min(entry, current_price + trail_distance)
        if candidate < stop:
            return _round_price(candidate), profit_r
    return None, profit_r


async def pyramid_position(
    position_id: int,
    add_quantity: float | None = None,
    add_position_pct: float | None = None,
    reasoning_summary: str | None = None,
) -> dict:
    """Renforce (pyramide) une position gagnante sans dépasser les garde-fous PRISM.

    Conditions : position ouverte, profit >= ``PYRAMID_MIN_PROFIT_R`` et ajout
    plafonné à ``PYRAMID_MAX_ADD_FRACTION`` du notionnel initial. Le prix
    d'entrée devient le prix moyen pondéré et le stop est remonté au point mort.
    """
    position = db.get_position(position_id)
    if position is None:
        raise ValueError("Position introuvable.")
    if position["status"] != "open":
        raise ValueError("Impossible de renforcer une position clôturée.")

    symbol = position["symbol"]
    market = position["market_type"]
    side = position.get("side")
    current_price, _ = await get_current_price(symbol, market)
    if current_price is None:
        raise ValueError(f"Prix actuel introuvable pour {symbol}.")

    profit_r = _profit_in_r(position, current_price)
    if profit_r < PYRAMID_MIN_PROFIT_R:
        raise ValueError(
            f"Pyramide refusée : profit {profit_r:.2f}R < seuil {PYRAMID_MIN_PROFIT_R:.2f}R."
        )

    entry = _as_float(position.get("entry_price"))
    quantity = _as_float(position.get("quantity"))
    reserved = _as_float(position.get("reserved_cash"), entry * quantity)
    max_add_notional = reserved * PYRAMID_MAX_ADD_FRACTION

    if add_quantity is not None and _as_float(add_quantity) > 0:
        add_qty = _as_float(add_quantity)
    else:
        pct = _as_float(add_position_pct, 50.0)
        pct = max(5.0, min(pct, 100.0))
        add_qty = (max_add_notional * (pct / 100.0)) / current_price

    add_notional = add_qty * current_price
    if add_notional <= 0:
        raise ValueError("Quantité d'ajout invalide.")
    if add_notional > max_add_notional:
        add_qty = max_add_notional / current_price
        add_notional = add_qty * current_price

    account = db.get_paper_account()
    cash = _as_float(account.get("balance"))
    if add_notional > cash:
        add_qty = (cash * 0.98) / current_price
        add_notional = add_qty * current_price
    if add_qty <= 0:
        raise ValueError("Cash insuffisant pour renforcer la position.")

    new_quantity = quantity + add_qty
    new_entry = ((entry * quantity) + (current_price * add_qty)) / new_quantity
    old_stop = _as_float(position.get("stop_loss"), 0.0)
    if side == "long":
        new_stop = max(old_stop, entry)
    else:
        new_stop = min(old_stop, entry) if old_stop > 0 else entry

    now = db.utcnow_iso()
    note = (reasoning_summary or "Renforcement pyramide sur position gagnante.").strip()
    merged_reasoning = position.get("reasoning_summary") or ""
    merged_reasoning = (
        f"{merged_reasoning}\n[PYRAMID {now[:10]}] {note}" if merged_reasoning else f"[PYRAMID] {note}"
    )

    with db.get_connection() as conn:
        conn.execute(
            """
            UPDATE paper_positions
            SET entry_price = ?, quantity = ?, reserved_cash = reserved_cash + ?,
                stop_loss = ?, reasoning_summary = ?
            WHERE id = ?
            """,
            (
                _round_price(new_entry),
                _round_quantity(new_quantity),
                add_notional,
                _round_price(new_stop),
                merged_reasoning[-4000:],
                int(position_id),
            ),
        )
        conn.execute(
            "UPDATE paper_account SET balance = balance - ?, updated_at = ? WHERE id = ?",
            (add_notional, now, account["id"]),
        )

    updated = db.get_position(position_id) or {}
    await notifier.send_trade_alert(
        "PYRAMID",
        updated,
        account=db.get_paper_account(),
        extra=f"Ajout de {_round_quantity(add_qty)} @ {_round_price(current_price)} ({profit_r:.2f}R).",
    )
    return {
        "success": True,
        "position": updated,
        "account": db.get_paper_account(),
        "add_quantity": _round_quantity(add_qty),
        "add_price": _round_price(current_price),
        "profit_r": round(profit_r, 2),
        "new_entry_price": _round_price(new_entry),
        "new_stop_loss": _round_price(new_stop),
    }


# ---------------------------------------------------------------------------
# Ouverture de position
# ---------------------------------------------------------------------------
async def open_position(
    symbol: str,
    side: str | None = None,
    quantity: float | None = None,
    entry_price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    risk_profile: str = "moderate",
    reasoning_summary: str | None = None,
    market_type: str | None = None,
    position_size_pct: float | None = None,
    trade_plan: dict | None = None,
) -> dict:
    """Ouvre une position simulée, manuellement ou depuis une analyse IA."""
    market = (market_type or fetcher.detect_market_type(symbol)).lower()
    normalized_symbol = fetcher.normalize_symbol(symbol, market)
    profile = _normalize_profile(risk_profile)
    plan = trade_plan or {}

    normalized_side = _normalize_side(side)
    if normalized_side is None:
        plan_side = str(plan.get("direction") or "").lower()
        if plan_side in {"long", "short"}:
            normalized_side = plan_side
    if normalized_side is None:
        raise ValueError("Le sens de la position doit être 'long' ou 'short'.")

    entry = _as_float(entry_price if entry_price is not None else plan.get("entry_price"), 0.0)
    if entry <= 0:
        current_price, _ = await get_current_price(normalized_symbol, market)
        if current_price is None:
            raise ValueError(f"Prix d'entrée introuvable pour {normalized_symbol}.")
        entry = current_price

    if stop_loss is None and plan.get("stop_loss") is not None:
        stop_loss = _as_float(plan.get("stop_loss"))
    if take_profit is None:
        plan_tp = plan.get("take_profit_1", plan.get("take_profit"))
        if plan_tp is not None:
            take_profit = _as_float(plan_tp)

    if position_size_pct is None and plan.get("position_size_pct") is not None:
        position_size_pct = _as_float(plan.get("position_size_pct"))

    if stop_loss is None or take_profit is None:
        default_stop, default_tp = _default_stops(entry, normalized_side, profile)
        stop_loss = default_stop if stop_loss is None else stop_loss
        take_profit = default_tp if take_profit is None else take_profit

    stop_loss = _as_float(stop_loss, 0.0)
    take_profit = _as_float(take_profit, 0.0)
    if normalized_side == "long":
        if stop_loss >= entry:
            raise ValueError("Pour un Long, le stop-loss doit être sous le prix d'entrée.")
        if take_profit <= entry:
            raise ValueError("Pour un Long, le take-profit doit être au-dessus du prix d'entrée.")
    else:
        if stop_loss <= entry:
            raise ValueError("Pour un Short, le stop-loss doit être au-dessus du prix d'entrée.")
        if take_profit >= entry:
            raise ValueError("Pour un Short, le take-profit doit être sous le prix d'entrée.")

    account = db.get_paper_account()
    cash = _as_float(account.get("balance"))
    if cash <= 0:
        raise ValueError("Solde Paper Trading épuisé.")

    adjusted_quantity = False
    if quantity is not None and _as_float(quantity) > 0:
        qty = _as_float(quantity)
    else:
        pct = _as_float(position_size_pct, PROFILE_DEFAULTS[profile]["position_size_pct"])
        pct = max(0.1, min(pct, 25.0))
        equity = _estimate_equity()
        notional = equity * (pct / 100.0)
        qty = notional / entry if entry > 0 else 0.0

    if qty <= 0:
        raise ValueError("Quantité invalide.")
    cost = qty * entry
    if cost > cash:
        qty = (cash * 0.98) / entry
        adjusted_quantity = True
    if qty <= 0:
        raise ValueError("Solde insuffisant pour ouvrir cette position.")

    order_summary = (reasoning_summary or plan.get("rationale") or "Ouverture manuelle").strip()
    position = db.create_position(
        symbol=normalized_symbol,
        market_type=market,
        side=normalized_side,
        entry_price=_round_price(entry),
        quantity=_round_quantity(qty),
        stop_loss=_round_price(stop_loss),
        take_profit=_round_price(take_profit),
        risk_profile=profile,
        reasoning_summary=order_summary,
    )
    updated_account = db.get_paper_account()
    await notifier.send_trade_alert(
        "OPEN",
        position,
        account=updated_account,
        extra="Quantité ajustée au cash disponible." if adjusted_quantity else None,
    )
    return {
        "success": True,
        "position": position,
        "account": updated_account,
        "quantity_adjusted": adjusted_quantity,
        "message": f"Position {normalized_side} ouverte sur {normalized_symbol}.",
    }


async def open_position_from_plan(
    symbol: str,
    trade_plan: dict,
    verdict: str | None = None,
    risk_profile: str = "moderate",
    reasoning_summary: str | None = None,
    market_type: str | None = None,
    quantity: float | None = None,
) -> dict:
    """Ouvre un trade en un clic depuis le plan du ChiefStrategyAgent."""
    if not isinstance(trade_plan, dict) or not trade_plan:
        raise ValueError("Plan de trade invalide.")

    side = None
    plan_direction = str(trade_plan.get("direction") or "").lower()
    normalized_verdict = (verdict or "").strip().upper()
    if plan_direction in {"long", "short"}:
        side = plan_direction
    elif normalized_verdict in {"BUY", "SELL"}:
        side = "long" if normalized_verdict == "BUY" else "short"

    if side is None:
        raise ValueError(
            "Le verdict n'est pas actionnable (BUY ou SELL requis) pour ouvrir un trade."
        )

    summary = reasoning_summary or trade_plan.get("rationale")
    return await open_position(
        symbol=symbol,
        side=side,
        quantity=quantity,
        risk_profile=risk_profile,
        reasoning_summary=summary,
        market_type=market_type,
        trade_plan=trade_plan,
    )


# ---------------------------------------------------------------------------
# Clôture
# ---------------------------------------------------------------------------
def _lesson_for_exit(position: dict, trade: dict) -> str:
    reason = str(trade.get("exit_reason") or "manual")
    pnl_pct = _as_float(trade.get("pnl_pct"))
    if "stop" in reason:
        return (
            f"Stop touché sur {position.get('symbol')} ({pnl_pct:+.2f}%) : "
            "vérifier la taille du stop et le timing d'entrée."
        )
    if "take" in reason or "tp" in reason:
        return (
            f"Take-profit atteint sur {position.get('symbol')} ({pnl_pct:+.2f}%) : "
            "conserver le processus de sortie progressif."
        )
    return (
        f"Clôture manuelle de {position.get('symbol')} ({pnl_pct:+.2f}%) : "
        "documenter la raison pour calibrer les prochains setups."
    )


async def close_position_manual(
    position_id: int,
    exit_price: float | None = None,
    exit_reason: str = "manual",
    lesson_learned: str | None = None,
) -> dict:
    """Clôture manuelle d'une position, avec mémoire et alerte Telegram."""
    position = db.get_position(position_id)
    if position is None:
        raise ValueError("Position introuvable.")
    if position.get("status") != "open":
        raise ValueError("Cette position est déjà clôturée.")

    price = _as_float(exit_price, 0.0)
    if price <= 0:
        current_price, _ = await get_current_price(
            position["symbol"], position.get("market_type")
        )
        if current_price is None:
            raise ValueError("Prix de clôture introuvable.")
        price = current_price

    _, estimated_pnl_pct = _unrealized(position, price)
    trade = db.close_position(
        position_id=position_id,
        exit_price=price,
        exit_reason=exit_reason,
        lesson_learned=lesson_learned
        or _lesson_for_exit(position, {"exit_reason": exit_reason, "pnl_pct": estimated_pnl_pct}),
    )
    memory_result = await update_trading_memory(
        symbol=position["symbol"],
        market_type=position.get("market_type"),
        trade_closed=True,
    )
    account = db.get_paper_account()
    await notifier.send_trade_alert("CLOSE", position, trade=trade, account=account)
    return {
        "success": True,
        "trade": trade,
        "memory": memory_result,
        "account": account,
    }


async def check_and_update_positions() -> dict:
    """Vérifie SL/TP de toutes les positions ouvertes et clôture automatiquement."""
    positions = db.get_open_positions()
    if not positions:
        return {
            "checked": 0,
            "closed": [],
            "open": [],
            "errors": [],
            "account": db.get_paper_account(),
            "stats": db.get_performance_stats(),
            "checked_at": datetime.now(UTC).isoformat(),
        }

    prices = await _fetch_prices_for_positions(positions)
    closed: list[dict] = []
    still_open: list[dict] = []
    errors: list[dict] = []
    closed_pairs: set[tuple[str, str]] = set()

    for position in positions:
        key = (
            str(position.get("symbol") or "").upper(),
            str(position.get("market_type") or "").lower(),
        )
        current_price = prices.get(key)
        if current_price is None:
            errors.append(
                {
                    "position_id": position.get("id"),
                    "symbol": position.get("symbol"),
                    "error": "Prix indisponible",
                }
            )
            continue

        entry = _as_float(position.get("entry_price"))
        stop_loss = _as_float(position.get("stop_loss"), 0.0)
        take_profit = _as_float(position.get("take_profit"), 0.0)
        side = position.get("side")
        trigger: str | None = None

        # Runner exit : sécurise les gains dès +1R puis laisse courir avec un stop suiveur.
        runner_stop, profit_r = _runner_stop_update(position, current_price)
        runner_updated = False
        if runner_stop is not None:
            db.update_position_stops(int(position["id"]), stop_loss=runner_stop)
            position["stop_loss"] = runner_stop
            stop_loss = runner_stop
            runner_updated = True
        trailing_active = bool(
            stop_loss > 0
            and entry > 0
            and (
                (side == "long" and stop_loss >= entry)
                or (side == "short" and stop_loss <= entry)
            )
        )

        if side == "long":
            if stop_loss > 0 and current_price <= stop_loss:
                trigger = "runner_exit" if trailing_active else "stop_loss"
            elif take_profit > 0 and current_price >= take_profit:
                trigger = "take_profit"
        else:
            if stop_loss > 0 and current_price >= stop_loss:
                trigger = "runner_exit" if trailing_active else "stop_loss"
            elif take_profit > 0 and current_price <= take_profit:
                trigger = "take_profit"

        unrealized_pnl, unrealized_pct = _unrealized(position, current_price)
        if trigger is None:
            still_open.append(
                {
                    **position,
                    "current_price": _round_price(current_price),
                    "unrealized_pnl_usd": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pct,
                    "profit_r": round(profit_r, 2),
                    "runner_active": trailing_active,
                    "runner_stop_updated": runner_updated,
                }
            )
            continue

        lesson = _lesson_for_exit(
            position,
            {"exit_reason": trigger, "pnl_pct": unrealized_pct},
        )
        trade = db.close_position(
            position_id=int(position["id"]),
            exit_price=current_price,
            exit_reason=trigger,
            lesson_learned=lesson,
        )
        closed_pairs.add(key)
        closed.append(
            {
                "trade": trade,
                "entry_price": entry,
                "trigger_price": _round_price(current_price),
                "trigger": trigger,
            }
        )
        await notifier.send_trade_alert(
            "CLOSE",
            position,
            trade=trade,
            account=db.get_paper_account(),
        )

    memory_updates = []
    for symbol, market in closed_pairs:
        try:
            memory_updates.append(
                await update_trading_memory(
                    symbol=symbol,
                    market_type=market,
                    trade_closed=True,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "symbol": symbol,
                    "market_type": market,
                    "error": f"Mémoire non mise à jour : {exc}",
                }
            )

    return {
        "checked": len(positions),
        "closed": closed,
        "open": still_open,
        "memory_updates": memory_updates,
        "errors": errors,
        "account": db.get_paper_account(),
        "stats": db.get_performance_stats(),
        "checked_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Statistiques du portefeuille
# ---------------------------------------------------------------------------
async def get_portfolio_state(
    include_history: bool = True,
    history_limit: int = 50,
) -> dict:
    """Retourne l'état complet du portefeuille avec prix live et statistiques."""
    positions = db.get_open_positions()
    prices = await _fetch_prices_for_positions(positions) if positions else {}

    enriched_positions: list[dict] = []
    unrealized_total = 0.0
    reserved_total = 0.0

    for position in positions:
        key = (
            str(position.get("symbol") or "").upper(),
            str(position.get("market_type") or "").lower(),
        )
        current_price = prices.get(key, _as_float(position.get("entry_price")))
        pnl, pnl_pct = _unrealized(position, current_price)
        unrealized_total += pnl

        entry = _as_float(position.get("entry_price"))
        quantity = _as_float(position.get("quantity"))
        reserved = _as_float(position.get("reserved_cash"), entry * quantity)
        reserved_total += reserved

        stop_loss = _as_float(position.get("stop_loss"), 0.0)
        take_profit = _as_float(position.get("take_profit"), 0.0)
        distance_sl = (
            (current_price - stop_loss) / current_price * 100.0
            if stop_loss > 0 and current_price > 0
            else None
        )
        distance_tp = (
            (take_profit - current_price) / current_price * 100.0
            if take_profit > 0 and current_price > 0
            else None
        )

        enriched_positions.append(
            {
                **position,
                "current_price": _round_price(current_price),
                "market_value": round(current_price * quantity, 4),
                "unrealized_pnl_usd": round(pnl, 4),
                "unrealized_pnl_pct": pnl_pct,
                "distance_to_stop_loss_pct": round(distance_sl, 2) if distance_sl is not None else None,
                "distance_to_take_profit_pct": (
                    round(distance_tp, 2) if distance_tp is not None else None
                ),
                "price_stale": key not in prices,
            }
        )

    account = db.get_paper_account()
    stats = db.get_performance_stats()
    cash = _as_float(account.get("balance"))
    realized = _as_float(stats.get("total_pnl_usd"))
    equity = cash + reserved_total + unrealized_total
    initial = _as_float(account.get("initial_balance"))

    summary = {
        "equity_usd": round(equity, 4),
        "cash_available_usd": round(cash, 4),
        "cash_reserved_usd": round(reserved_total, 4),
        "realized_pnl_usd": round(realized, 4),
        "unrealized_pnl_usd": round(unrealized_total, 4),
        "total_pnl_usd": round(realized + unrealized_total, 4),
        "total_pnl_pct": round(
            ((equity / initial) - 1.0) * 100.0, 4
        )
        if initial > 0
        else 0.0,
        "win_rate_pct": _as_float(stats.get("win_rate_pct")),
        "profit_factor": stats.get("profit_factor"),
        "max_drawdown_pct": _as_float(stats.get("max_drawdown_pct")),
        "sharpe_ratio": _as_float(stats.get("sharpe_ratio")),
        "total_trades": int(stats.get("total_trades") or 0),
        "wins": int(stats.get("wins") or 0),
        "losses": int(stats.get("losses") or 0),
        "open_positions_count": len(enriched_positions),
        "best_trade_usd": _as_float(stats.get("best_trade_usd")),
        "worst_trade_usd": _as_float(stats.get("worst_trade_usd")),
    }

    payload: dict[str, Any] = {
        "account": account,
        "summary": summary,
        "positions": enriched_positions,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    if include_history:
        payload["history"] = db.get_trades_history(limit=history_limit)
    payload["memory"] = db.get_all_trading_memory(limit=50)
    return payload
