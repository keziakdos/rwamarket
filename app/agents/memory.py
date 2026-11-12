"""Mémoire de trading auto-apprenante de RwaMarket.

Inspiré de `compress_trading_memory.py` de PRISM-Insight : les trades clôturés
sont analysés pour produire exactement trois règles d'ajustement concrètes,
stockées dans `trading_memory`. Le LLM rapide (`FAST_MODEL`) n'est sollicité
que lorsqu'un nouveau trade a réellement été clôturé ; sinon, des règles
déterministes sont générées à partir des statistiques et des sorties.

`get_memory_context()` fournit au stratège final un résumé compact des
performances passées, des règles apprises et des pièges à éviter.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from typing import Any

from openai import AsyncOpenAI

from app import database as db
from app.engine.data_fetcher import detect_market_type, normalize_symbol

DEFAULT_FAST_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
LLM_TIMEOUT_SECONDS = float(os.getenv("FAST_LLM_TIMEOUT_SECONDS", "18"))
MAX_TRADES_FOR_MEMORY = 30


# ---------------------------------------------------------------------------
# Client LLM et parsing JSON
# ---------------------------------------------------------------------------
def _env_api_key() -> str:
    key = os.getenv("LLM_API_KEY", "").strip()
    if not key:
        return ""
    lowered = key.lower()
    if lowered in {"none", "null", "changeme", "your-key", "sk-your-key"}:
        return ""
    if "your_api_key" in lowered or "replace_me" in lowered:
        return ""
    return key


def build_llm_client() -> AsyncOpenAI | None:
    """Construit le client OpenAI-compatible si une clé exploitable existe."""
    api_key = _env_api_key()
    if not api_key:
        return None
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 45.0, "max_retries": 1}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return AsyncOpenAI(**kwargs)
    except Exception:
        return None


def _parse_json_object(text: str | None) -> dict | None:
    """Extrait un objet JSON d'une réponse LLM, même entourée de fences Markdown."""
    if not text:
        return None
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def call_llm_json(
    client: AsyncOpenAI | None,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 350,
    temperature: float = 0.2,
    timeout: float | None = None,
) -> dict | None:
    """Appelle un LLM et retourne un objet JSON, ou None en cas d'échec."""
    if client is None:
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    wait_seconds = timeout or LLM_TIMEOUT_SECONDS

    response = None
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                response_format={"type": "json_object"},
                **base_kwargs,
            ),
            timeout=wait_seconds,
        )
    except Exception:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(**base_kwargs),
                timeout=wait_seconds,
            )
        except Exception:
            return None

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    return _parse_json_object(content)


# ---------------------------------------------------------------------------
# Analyse des trades clôturés
# ---------------------------------------------------------------------------
def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _is_stop_exit(trade: dict) -> bool:
    reason = str(trade.get("exit_reason") or "").lower()
    return any(token in reason for token in ("stop", "sl_", "sl ", "stoploss", "stopped"))


def _is_take_profit_exit(trade: dict) -> bool:
    reason = str(trade.get("exit_reason") or "").lower()
    return any(token in reason for token in ("take_profit", "takeprofit", "tp1", "tp2", "target"))


def _trade_digest(trade: dict) -> dict:
    """Réduit un trade à l'essentiel pour le prompt et les diagnostics."""
    return {
        "closed_at": trade.get("closed_at"),
        "side": trade.get("side"),
        "entry_price": _as_float(trade.get("entry_price")),
        "exit_price": _as_float(trade.get("exit_price")),
        "pnl_pct": round(_as_float(trade.get("pnl_pct")), 3),
        "pnl_usd": round(_as_float(trade.get("pnl_usd")), 3),
        "exit_reason": trade.get("exit_reason"),
        "lesson_learned": trade.get("lesson_learned"),
    }


def _memory_diagnostics(trades: list[dict], stats: dict) -> dict:
    recent = trades[:MAX_TRADES_FOR_MEMORY]
    wins = [trade for trade in recent if _as_float(trade.get("pnl_usd")) > 0]
    losses = [trade for trade in recent if _as_float(trade.get("pnl_usd")) < 0]
    stop_exits = [trade for trade in recent if _is_stop_exit(trade)]
    take_profit_exits = [trade for trade in recent if _is_take_profit_exit(trade)]

    long_losses = [trade for trade in losses if trade.get("side") == "long"]
    short_losses = [trade for trade in losses if trade.get("side") == "short"]

    avg_win = sum(_as_float(t.get("pnl_pct")) for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(_as_float(t.get("pnl_pct")) for t in losses) / len(losses) if losses else 0.0

    last_three = recent[:3]
    last_three_pnls = [_as_float(t.get("pnl_pct")) for t in last_three]

    return {
        "total_trades": len(recent),
        "wins": len(wins),
        "losses": len(losses),
        "stop_exits": len(stop_exits),
        "take_profit_exits": len(take_profit_exits),
        "long_losses": len(long_losses),
        "short_losses": len(short_losses),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "last_three_pnl_pct": [round(value, 3) for value in last_three_pnls],
        "win_rate_pct": _as_float(stats.get("win_rate_pct")),
        "total_pnl_usd": _as_float(stats.get("total_pnl_usd")),
        "max_drawdown_pct": _as_float(stats.get("max_drawdown_pct")),
        "profit_factor": stats.get("profit_factor"),
    }


def _deterministic_rules(symbol: str, market_type: str, diagnostics: dict) -> list[str]:
    """Produit exactement trois règles concrètes à partir des statistiques."""
    stop_exits = diagnostics["stop_exits"]
    losses = diagnostics["losses"]
    wins = diagnostics["wins"]
    avg_win = diagnostics["avg_win_pct"]
    avg_loss = diagnostics["avg_loss_pct"]
    win_rate = diagnostics["win_rate_pct"]
    last_three = diagnostics["last_three_pnl_pct"]

    if stop_exits > 0 or (losses and abs(avg_loss) > abs(avg_win)):
        rule_stop = (
            f"Placer un stop-loss initial à 1,5x ATR et l'exécuter sans exception : "
            f"{stop_exits} sortie(s) par stop ont coûté en moyenne {abs(avg_loss):.2f}%."
        )
    else:
        rule_stop = (
            "Conserver un stop-loss à 1,5x ATR et le remonter au point mort dès que "
            "le trade atteint +1R, afin de ne jamais transformer un gain en perte."
        )

    consecutive_losses = len(last_three) >= 2 and all(value <= 0 for value in last_three[:2])
    if consecutive_losses:
        rule_sizing = (
            f"Après deux pertes consécutives sur {symbol}, réduire la taille de position "
            "de 50% et n'augmenter à nouveau qu'après un trade gagnant confirmé."
        )
    elif avg_win and abs(avg_win) < abs(avg_loss) * 1.5:
        rule_sizing = (
            f"Exiger un rapport risque/rendement d'au moins 2:1 : la perte moyenne actuelle "
            f"({avg_loss:.2f}%) est trop proche du gain moyen ({avg_win:.2f}%)."
        )
    else:
        rule_sizing = (
            f"Dimensionner chaque position pour risquer au maximum 1% du capital, avec un "
            f"objectif de gain moyen d'au moins {max(abs(avg_win), 2.0):.2f}% par trade."
        )

    if diagnostics["take_profit_exits"] > 0 and wins:
        rule_exit = (
            "Sécuriser 50% de la position à TP1 (1,5R à 2R) puis laisser courir le reste "
            "vers TP2 avec un stop suiveur sous le dernier plus-bas structurel."
        )
    elif win_rate < 45 and diagnostics["total_trades"] >= 3:
        rule_exit = (
            f"Le win rate de {win_rate:.1f}% est faible sur {market_type} : ne prendre que "
            "les setups alignés EMA9/EMA20 avec volume > 1,5x la moyenne 20 périodes."
        )
    else:
        rule_exit = (
            "Ne pas entrer avant une publication macro ou un événement majeur : attendre "
            "la confirmation du volume et la clôture de la première bougie post-événement."
        )

    rules = [rule_stop, rule_sizing, rule_exit]
    if max(abs(avg_loss), abs(avg_win)) < 3 and diagnostics["total_trades"] >= 2:
        rules[2] = (
            "Les mouvements moyens sont inférieurs à 3% : privilégier des objectifs courts "
            "et sortir dès que le momentum faiblit plutôt que d'attendre une tendance majeure."
        )

    if diagnostics["short_losses"] > diagnostics["long_losses"]:
        rules[0] = (
            "Réduire les positions short non confirmées par la structure : les pertes short "
            "dominantes indiquent des ventes prématurées contre la tendance de fond. "
            "Stop obligatoire à 1,5x ATR."
        )

    unique: list[str] = []
    for rule in rules:
        if rule not in unique:
            unique.append(rule)
    while len(unique) < 3:
        unique.append(
            "Conserver une journalisation systématique du contexte d'entrée et de sortie "
            "pour valider ou invalider cette règle lors des prochains trades."
        )
    return unique[:3]


def _rules_to_text(rules: list[str]) -> str:
    return "\n".join(f"{index}. {rule.strip()}" for index, rule in enumerate(rules[:3], start=1))


async def _llm_rules(
    client: AsyncOpenAI | None,
    symbol: str,
    market_type: str,
    trades: list[dict],
    diagnostics: dict,
) -> tuple[list[str] | None, dict | None]:
    if client is None or not trades:
        return None, None

    model = os.getenv("FAST_MODEL", DEFAULT_FAST_MODEL)
    system_prompt = (
        "Tu es un compresseur de mémoire de trading. Analyse des trades clôturés et "
        "extrais des règles actionnables, concrètes et vérifiables. Réponds uniquement "
        "avec un objet JSON valide, sans texte autour."
    )
    user_prompt = json.dumps(
        {
            "instruction": (
                "Produis exactement 3 règles d'ajustement concrètes pour les prochains "
                "trades. Chaque règle doit mentionner une action précise (stop, taille, "
                "entrée, sortie) et être justifiée par les données. Ajoute les pièges "
                "récurrents observés. Reste factuel, ne prédit aucun prix."
            ),
            "symbol": symbol,
            "market_type": market_type,
            "statistiques": diagnostics,
            "trades_recents": [_trade_digest(trade) for trade in trades[:15]],
            "schema": {
                "rules": ["règle 1", "règle 2", "règle 3"],
                "pitfalls": ["piège 1", "piège 2"],
                "dominant_pattern": "motif dominant",
                "risk_note": "note de risque",
            },
        },
        ensure_ascii=False,
        default=str,
    )

    payload = await call_llm_json(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=350,
        temperature=0.2,
    )
    if not payload:
        return None, None

    raw_rules = payload.get("rules")
    if isinstance(raw_rules, str):
        raw_rules = [raw_rules]
    if not isinstance(raw_rules, list):
        return None, payload

    rules = [str(rule).strip() for rule in raw_rules if str(rule).strip()]
    if len(rules) < 3:
        return None, payload
    return rules[:3], payload


def _memory_looks_compressed(memory: dict | None) -> bool:
    """Vérifie que la mémoire contient bien les 3 règles numérotées de Phase 2."""
    raw = (memory or {}).get("compressed_rules") or ""
    numbered = [
        line
        for line in str(raw).splitlines()
        if re.match(r"^\s*\d+[.)]\s+\S", line)
    ]
    return len(numbered) >= 3


def _new_trade_since_memory(memory: dict | None, trades: list[dict]) -> bool:
    """Détecte si un trade a été clôturé après la dernière compression."""
    if not trades:
        return False
    if memory is None:
        return True
    if not _memory_looks_compressed(memory):
        return True
    if len(trades) > int(memory.get("total_trades") or 0):
        return True

    latest = str(trades[0].get("closed_at") or "")
    updated = str(memory.get("updated_at") or "")
    return bool(latest and updated and latest > updated)


async def update_trading_memory(
    symbol: str,
    market_type: str | None = None,
    trade_closed: bool | None = None,
    force_llm: bool = False,
    client: AsyncOpenAI | None = None,
) -> dict:
    """Analyse les derniers trades et met à jour les 3 règles mémorisées.

    Le LLM rapide n'est appelé que si `trade_closed` est vrai (ou détecté
    automatiquement) et si un client est disponible. Sinon, les règles sont
    calculées de façon déterministe.
    """
    market = (market_type or detect_market_type(symbol)).lower()
    normalized = normalize_symbol(symbol, market)

    trades = db.get_trades_history(symbol=normalized, market_type=market, limit=MAX_TRADES_FOR_MEMORY)
    if not trades:
        return {
            "updated": False,
            "reason": "aucun_trade_cloture",
            "symbol": normalized,
            "market_type": market,
            "rules": [],
            "used_llm": False,
        }

    memory = db.get_trading_memory(normalized, market)
    if trade_closed is None:
        trade_closed = _new_trade_since_memory(memory, trades)

    stats = db.get_performance_stats(symbol=normalized, market_type=market)
    diagnostics = _memory_diagnostics(trades, stats)
    deterministic_rules = _deterministic_rules(normalized, market, diagnostics)

    llm_client = client if client is not None else (build_llm_client() if (trade_closed or force_llm) else None)
    used_llm = False
    llm_payload: dict | None = None
    rules = deterministic_rules

    if (trade_closed or force_llm) and llm_client is not None:
        llm_rules, llm_payload = await _llm_rules(
            llm_client, normalized, market, trades, diagnostics
        )
        if llm_rules:
            rules = llm_rules
            used_llm = True

    updated_memory = db.upsert_trading_memory(
        symbol=normalized,
        market_type=market,
        compressed_rules=_rules_to_text(rules),
        win_rate=_as_float(stats.get("win_rate_pct")),
        total_trades=int(stats.get("total_trades") or len(trades)),
    )

    return {
        "updated": True,
        "symbol": normalized,
        "market_type": market,
        "trade_closed": bool(trade_closed),
        "used_llm": used_llm,
        "rules": rules,
        "diagnostics": diagnostics,
        "llm_pitfalls": (llm_payload or {}).get("pitfalls") or [],
        "llm_pattern": (llm_payload or {}).get("dominant_pattern"),
        "memory": updated_memory,
    }


async def update_all_trading_memories(
    force_llm: bool = False,
    client: AsyncOpenAI | None = None,
) -> list[dict]:
    """Met à jour la mémoire de tous les couples (symbole, marché) connus."""
    trades = db.get_trades_history(limit=100000)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for trade in trades:
        key = (str(trade["symbol"]).upper(), str(trade["market_type"]).lower())
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    results = []
    for symbol, market_type in pairs:
        results.append(
            await update_trading_memory(
                symbol=symbol,
                market_type=market_type,
                force_llm=force_llm,
                client=client,
            )
        )
    return results


async def compress_trading_memory(
    symbol: str,
    market_type: str | None = None,
    force_llm: bool = False,
) -> dict:
    """Alias explicite inspiré de PRISM-Insight pour déclencher la compression."""
    return await update_trading_memory(
        symbol=symbol,
        market_type=market_type,
        force_llm=force_llm,
    )


# ---------------------------------------------------------------------------
# Contexte mémoire pour le juge final
# ---------------------------------------------------------------------------
def _parse_rules(memory: dict | None) -> list[str]:
    raw = (memory or {}).get("compressed_rules") or ""
    rules: list[str] = []
    for line in str(raw).splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            rules.append(cleaned)
    return rules


def _build_pitfalls(
    diagnostics: dict,
    stats: dict,
    market_type: str,
    risk_profile: str,
) -> list[str]:
    pitfalls: list[str] = []
    total = int(stats.get("total_trades") or 0)

    if total == 0:
        pitfalls.extend(
            [
                (
                    "Aucun historique sur ce symbole : éviter les positions surdimensionnées "
                    "tant que la stratégie n'a pas été validée par au moins 5 trades."
                ),
                "Ne pas confondre un signal technique isolé avec une confirmation multi-agents.",
                "Toujours définir le stop-loss avant l'entrée et vérifier le R:R minimal.",
            ]
        )
        if market_type == "crypto":
            pitfalls.append("La crypto évolue 24/7 : éviter les entrées impulsives sur faible volume.")
        else:
            pitfalls.append("Attention aux gaps d'ouverture et aux publications de résultats.")
        return pitfalls

    win_rate = _as_float(stats.get("win_rate_pct"))
    avg_loss = diagnostics.get("avg_loss_pct", 0.0)
    avg_win = diagnostics.get("avg_win_pct", 0.0)

    if diagnostics.get("stop_exits", 0) > 0:
        pitfalls.append(
            f"{diagnostics['stop_exits']} trade(s) sorti(s) au stop : ne jamais déplacer "
            "un stop contre la position ni trader sans stop."
        )
    if total >= 3 and win_rate < 45:
        pitfalls.append(
            f"Win rate faible ({win_rate:.1f}%) : exiger une confirmation de volume et "
            "d'alignement EMA avant toute entrée."
        )
    if avg_loss and abs(avg_loss) >= abs(avg_win):
        pitfalls.append(
            f"Les pertes moyennes ({avg_loss:.2f}%) dominent les gains moyens "
            f"({avg_win:.2f}%) : couper les pertes plus tôt et viser un R:R >= 2."
        )
    if diagnostics.get("short_losses", 0) > diagnostics.get("long_losses", 0):
        pitfalls.append(
            "Les shorts perdants dominent : éviter de vendre contre la tendance de fond "
            "sans cassure structurelle confirmée."
        )
    if diagnostics.get("last_three_pnl_pct") and all(
        value <= 0 for value in diagnostics["last_three_pnl_pct"][:2]
    ):
        pitfalls.append(
            "Deux derniers trades perdants : réduire temporairement la taille et attendre "
            "un setup de qualité A avant de reprendre le risque normal."
        )

    if risk_profile == "conservative":
        pitfalls.append("Profil conservateur : refuser les setups sous 60% de confiance et limiter le risque à 1%.")
    elif risk_profile == "aggressive":
        pitfalls.append("Profil agressif : la taille peut être supérieure, mais jamais sans stop ATR.")

    if not pitfalls:
        pitfalls.append(
            "Aucun piège majeur détecté : maintenir la discipline de stop et de sizing."
        )
    return pitfalls[:6]


def get_memory_context(
    symbol: str,
    market_type: str | None = None,
    risk_profile: str = "moderate",
) -> dict:
    """Retourne performances, règles et pièges à injecter dans le prompt du juge."""
    market = (market_type or detect_market_type(symbol)).lower()
    normalized = normalize_symbol(symbol, market)

    memory = db.get_trading_memory(normalized, market)
    stats = db.get_performance_stats(symbol=normalized, market_type=market)
    trades = db.get_trades_history(symbol=normalized, market_type=market, limit=MAX_TRADES_FOR_MEMORY)
    diagnostics = _memory_diagnostics(trades, stats)
    rules = _parse_rules(memory)
    risk = (risk_profile or "moderate").lower()
    pitfalls = _build_pitfalls(diagnostics, stats, market, risk)

    performance = {
        "total_trades": int(stats.get("total_trades") or 0),
        "win_rate_pct": _as_float(stats.get("win_rate_pct")),
        "total_pnl_usd": _as_float(stats.get("total_pnl_usd")),
        "avg_pnl_pct": _as_float(stats.get("avg_pnl_pct")),
        "avg_win_pct": diagnostics["avg_win_pct"],
        "avg_loss_pct": diagnostics["avg_loss_pct"],
        "best_trade_usd": _as_float(stats.get("best_trade_usd")),
        "worst_trade_usd": _as_float(stats.get("worst_trade_usd")),
        "profit_factor": stats.get("profit_factor"),
        "max_drawdown_pct": _as_float(stats.get("max_drawdown_pct")),
        "sharpe_ratio": _as_float(stats.get("sharpe_ratio")),
        "stop_exits": diagnostics["stop_exits"],
        "take_profit_exits": diagnostics["take_profit_exits"],
    }

    if performance["total_trades"] == 0:
        summary = (
            f"Aucun trade clôturé pour {normalized} ({market}). "
            "Le juge doit rester prudent, exiger une confirmation multi-agents et "
            "appliquer un sizing standard."
        )
    else:
        summary = (
            f"{performance['total_trades']} trade(s) clôturé(s) sur {normalized} ({market}) : "
            f"win rate {performance['win_rate_pct']:.1f}%, PnL {performance['total_pnl_usd']:+.2f}$, "
            f"drawdown max {performance['max_drawdown_pct']:.1f}%. "
            f"Sorties stop: {performance['stop_exits']}, sorties TP: {performance['take_profit_exits']}."
        )

    recent_trades = [_trade_digest(trade) for trade in trades[:5]]
    recent_lessons = [
        trade["lesson_learned"]
        for trade in trades[:8]
        if trade.get("lesson_learned")
    ]

    lines = [
        f"MÉMOIRE DE TRADING — {normalized} ({market})",
        f"Performance: {summary}",
    ]
    if rules:
        lines.append("Règles apprises:")
        for index, rule in enumerate(rules[:3], start=1):
            lines.append(f"{index}. {rule}")
    else:
        lines.append("Règles apprises: aucune règle consolidée pour l'instant.")
    if pitfalls:
        lines.append("Pièges à éviter:")
        for pitfall in pitfalls:
            lines.append(f"- {pitfall}")
    if recent_lessons:
        lines.append("Dernières leçons:")
        for lesson in recent_lessons[:4]:
            lines.append(f"- {lesson}")

    return {
        "symbol": normalized,
        "market_type": market,
        "has_memory": memory is not None,
        "summary": summary,
        "performance": performance,
        "rules": rules,
        "pitfalls": pitfalls,
        "recent_trades": recent_trades,
        "recent_lessons": recent_lessons,
        "memory_updated_at": (memory or {}).get("updated_at"),
        "prompt_context": "\n".join(lines),
    }
