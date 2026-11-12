"""Orchestrateur multi-agents basse latence et low-cost de RwaMarket.

Flux SSE (`stream_market_analysis`) :
0. Cache SQLite (< TTL) -> rejoue tous les événements immédiatement.
1. Données + indicateurs déterministes -> événement `quant`.
2. Trois agents Flash en parallèle -> `agent_tech`, `agent_macro`, `agent_sentiment`.
3. ChiefStrategyAgent sur le modèle de raisonnement -> `synthesis`.

Si aucune clé LLM exploitable n'est configurée, chaque agent bascule sur une
sortie déterministe intelligente : le flux ne lève jamais d'exception.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from datetime import UTC, datetime
from collections.abc import AsyncGenerator
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from openai import AsyncOpenAI

from app import database as db
from app.agents.memory import (
    build_llm_client,
    call_llm_json,
    evaluate_buy_gates,
    get_memory_context,
)
from app.engine import data_fetcher as fetcher
from app.engine import quant_tools as quant

DEFAULT_FAST_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_REASONING_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
RISK_PROFILES = {"conservative", "moderate", "aggressive"}
ANALYSIS_MODES = {"auto", "research", "analytical"}
DEFAULT_ANALYSIS_MODE = "auto"
MODE_DIRECTIVES = {
    "auto": (
        "Mode AUTO : équilibre rapidité et profondeur. Priorise les signaux les plus "
        "actionnables sans surcharger l'analyse."
    ),
    "research": (
        "Mode RESEARCH : privilégie la découverte de catalyseurs, le contexte macro, "
        "les actualités et les flux de sentiment. Explore les scénarios alternatifs."
    ),
    "analytical": (
        "Mode ANALYTICAL : privilégie la rigueur quantitative, les niveaux clés, les "
        "scénarios d'invalidation et la cohérence mathématique du plan de trade."
    ),
}
VALID_VERDICTS = {"BUY", "SELL", "HOLD", "WATCH"}
VALID_BIASES = {"bullish", "bearish", "neutral"}
FAST_MAX_TOKENS = 350
CHIEF_MAX_TOKENS = 1200
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "20"))
CHIEF_TIMEOUT_SECONDS = float(os.getenv("CHIEF_LLM_TIMEOUT_SECONDS", "45"))

PROFILE_PARAMS = {
    "conservative": {
        "atr_multiplier": 2.0,
        "tp1_rr": 1.5,
        "tp2_rr": 3.0,
        "base_position_pct": 2.0,
        "allocation_multiplier": 0.5,
        "max_position_pct": 5.0,
        "confidence_floor": 60.0,
    },
    "moderate": {
        "atr_multiplier": 1.6,
        "tp1_rr": 2.0,
        "tp2_rr": 3.6,
        "base_position_pct": 5.0,
        "allocation_multiplier": 1.0,
        "max_position_pct": 10.0,
        "confidence_floor": 50.0,
    },
    "aggressive": {
        "atr_multiplier": 1.3,
        "tp1_rr": 2.4,
        "tp2_rr": 4.5,
        "base_position_pct": 10.0,
        "allocation_multiplier": 1.5,
        "max_position_pct": 20.0,
        "confidence_floor": 45.0,
    },
}


# ---------------------------------------------------------------------------
# Utilitaires généraux
# ---------------------------------------------------------------------------
def sse_event(event: str, payload: Any) -> str:
    """Formate un événement Server-Sent Events avec une charge JSON."""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {data}\n\n"


def _fast_model() -> str:
    return os.getenv("FAST_MODEL", DEFAULT_FAST_MODEL).strip() or DEFAULT_FAST_MODEL


def _reasoning_model() -> str:
    return (
        os.getenv("REASONING_MODEL", DEFAULT_REASONING_MODEL).strip()
        or DEFAULT_REASONING_MODEL
    )


def _normalize_risk_profile(risk_profile: str | None) -> str:
    profile = (risk_profile or "moderate").strip().lower()
    return profile if profile in RISK_PROFILES else "moderate"


def _normalize_mode(mode: str | None) -> str:
    value = (mode or DEFAULT_ANALYSIS_MODE).strip().lower()
    return value if value in ANALYSIS_MODES else DEFAULT_ANALYSIS_MODE


def _mode_directive(mode: str) -> str:
    return MODE_DIRECTIVES.get(_normalize_mode(mode), MODE_DIRECTIVES[DEFAULT_ANALYSIS_MODE])


def _normalize_timeframe(timeframe: str | None) -> str:
    value = (timeframe or "1d").strip()
    allowed = {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    }
    if value in allowed:
        return value
    for candidate in allowed:
        if candidate.lower() == value.lower():
            return candidate
    return "1d"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _clamp(value: Any, low: float, high: float, default: float = 0.0) -> float:
    number = _as_float(value, default)
    return max(low, min(high, number))


def _round_price(value: float) -> float:
    magnitude = abs(value)
    if magnitude >= 100:
        return round(value, 2)
    if magnitude >= 1:
        return round(value, 4)
    return round(value, 6)


def _string_list(value: Any, limit: int = 6, max_length: int = 320) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip(" -•\t") for part in value.splitlines() if part.strip()]
        if len(parts) <= 1:
            parts = [value.strip()]
    elif isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if str(item).strip()]
    else:
        parts = [str(value).strip()]
    return [part[:max_length] for part in parts if part][:limit]


def _normalize_bias(value: Any, fallback: str = "neutral") -> str:
    bias = str(value or "").strip().lower()
    mapping = {
        "bull": "bullish",
        "haussier": "bullish",
        "positive": "bullish",
        "up": "bullish",
        "bear": "bearish",
        "baissier": "bearish",
        "negative": "bearish",
        "down": "bearish",
        "neutre": "neutral",
        "mixed": "neutral",
        "sideways": "neutral",
    }
    bias = mapping.get(bias, bias)
    return bias if bias in VALID_BIASES else fallback


def _normalize_verdict(value: Any, fallback: str = "WATCH") -> str:
    verdict = str(value or "").strip().upper()
    mapping = {
        "LONG": "BUY",
        "ACHAT": "BUY",
        "SHORT": "SELL",
        "VENTE": "SELL",
        "ATTENDRE": "WATCH",
        "CONSERVER": "HOLD",
    }
    verdict = mapping.get(verdict, verdict)
    return verdict if verdict in VALID_VERDICTS else fallback


def _agent_error_payload(agent_key: str, error: str) -> dict:
    return {
        "agent": agent_key,
        "bias": "neutral",
        "score": 50.0,
        "confidence": 25.0,
        "key_points": [],
        "risks": [f"Agent indisponible : {error[:200]}"],
        "summary": "Analyse déterministe indisponible pour cet agent.",
        "source": "error",
        "error": error[:400],
    }


def _normalize_agent_output(
    payload: dict | None,
    fallback: dict,
    agent_key: str,
) -> dict:
    if not isinstance(payload, dict):
        return fallback

    result = dict(fallback)
    result["bias"] = _normalize_bias(payload.get("bias", payload.get("sentiment")), fallback["bias"])
    result["score"] = round(_clamp(payload.get("score"), 0, 100, fallback["score"]), 2)
    result["confidence"] = round(
        _clamp(payload.get("confidence"), 0, 100, fallback["confidence"]), 2
    )
    key_points = _string_list(payload.get("key_points"), 5)
    risks = _string_list(payload.get("risks"), 5)
    if key_points:
        result["key_points"] = key_points
    if risks:
        result["risks"] = risks
    summary = str(payload.get("summary") or "").strip()
    if summary:
        result["summary"] = summary[:600]
    result["agent"] = agent_key
    result["source"] = "llm"
    return result


def _compact_indicators(quant_payload: dict) -> dict:
    indicators = quant_payload.get("indicators") or {}
    return {
        "trend": indicators.get("trend"),
        "trend_score": indicators.get("trend_score"),
        "momentum_pct": indicators.get("momentum_pct"),
        "ema": indicators.get("ema"),
        "ma": indicators.get("ma"),
        "rsi14": indicators.get("rsi14"),
        "macd": indicators.get("macd"),
        "bollinger": indicators.get("bollinger"),
        "atr14": indicators.get("atr14"),
        "atr_pct": indicators.get("atr_pct"),
        "volume": indicators.get("volume"),
        "support_resistance": indicators.get("support_resistance"),
        "volatility_annualized_pct": indicators.get("volatility_annualized_pct"),
        "sharpe_ratio": indicators.get("sharpe_ratio"),
        "max_drawdown_pct": indicators.get("max_drawdown_pct"),
        "signals": indicators.get("signals"),
    }


# ---------------------------------------------------------------------------
# Agents spécialisés Flash
# ---------------------------------------------------------------------------
class TechnicalFlowAgent:
    """Analyse la structure de prix, le momentum, le volume et la volatilité."""

    key = "technical"
    event_name = "agent_tech"
    label = "technical_flow"

    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None):
        self.client = client
        self.model = model or _fast_model()

    def deterministic(self, symbol: str, market_type: str, quant_payload: dict) -> dict:
        indicators = quant_payload.get("indicators") or {}
        trend = str(indicators.get("trend") or "neutral").lower()
        bias = _normalize_bias(trend)
        score = _clamp(quant_payload.get("technical_score"), 0, 100, 50)
        confidence = 58.0
        if int(quant_payload.get("candles_count") or 0) >= 150:
            confidence += 8
        if indicators.get("atr_pct") is not None:
            confidence += 4

        key_points = _string_list(indicators.get("signals"), 5)
        momentum = indicators.get("momentum_pct") or {}
        if momentum.get("5d") is not None:
            key_points.append(f"Momentum 5j : {float(momentum['5d']):+.2f}%")
        if indicators.get("trend_score") is not None:
            key_points.append(f"Score directionnel : {indicators['trend_score']:+d}")

        risks: list[str] = []
        rsi = indicators.get("rsi14")
        if rsi is not None and rsi >= 70:
            risks.append(f"RSI en surachat ({float(rsi):.1f}) : risque de correction.")
        if rsi is not None and rsi <= 30:
            risks.append(f"RSI en survente ({float(rsi):.1f}) : rebond fragile possible.")
        atr_pct = indicators.get("atr_pct")
        if atr_pct is not None and atr_pct >= 5:
            risks.append(f"Volatilité élevée (ATR {float(atr_pct):.2f}% du prix).")
        drawdown = indicators.get("max_drawdown_pct")
        if drawdown is not None and drawdown <= -25:
            risks.append(f"Drawdown historique important ({float(drawdown):.1f}%).")
        if not risks:
            risks.append("Aucun risque technique majeur détecté sur les données disponibles.")

        summary = (
            f"Tendance {trend} sur {symbol} ({market_type}), "
            f"score technique {score:.0f}/100, RSI {rsi if rsi is not None else 'n/a'}, "
            f"volume {((indicators.get('volume') or {}).get('ratio')) or 'n/a'}x."
        )
        return {
            "agent": self.label,
            "bias": bias,
            "score": round(float(score), 2),
            "confidence": round(min(confidence, 85.0), 2),
            "key_points": key_points[:5],
            "risks": risks[:5],
            "summary": summary,
            "support": (indicators.get("support_resistance") or {}).get("nearest_support"),
            "resistance": (indicators.get("support_resistance") or {}).get("nearest_resistance"),
            "source": "deterministic",
        }

    async def run(
        self,
        symbol: str,
        market_type: str,
        quant_payload: dict,
        mode: str = DEFAULT_ANALYSIS_MODE,
    ) -> dict:
        fallback = self.deterministic(symbol, market_type, quant_payload)
        if self.client is None:
            return fallback

        system_prompt = (
            "Tu es TechnicalFlowAgent, analyste technique institutionnel. Analyse uniquement "
            "les données fournies (prix, EMA, RSI, MACD, Bollinger, ATR, volume, niveaux). "
            "Réponds en JSON strict, sans markdown ni texte autour. "
            + _mode_directive(mode)
        )
        user_payload = {
            "symbol": symbol,
            "market_type": market_type,
            "timeframe": quant_payload.get("timeframe"),
            "price": quant_payload.get("price"),
            "technical_score": quant_payload.get("technical_score"),
            "indicators": _compact_indicators(quant_payload),
            "schema": {
                "bias": "bullish|bearish|neutral",
                "score": "0-100 (bullishness)",
                "confidence": "0-100",
                "key_points": ["point clé 1", "point clé 2", "point clé 3"],
                "risks": ["risque 1", "risque 2"],
                "summary": "synthèse en 2 phrases maximum",
            },
            "contraintes": (
                "Maximum 120 mots. Ne pas inventer de données. Le score doit refléter "
                "l'alignement structure/momentum/volume."
            ),
        }
        payload = await call_llm_json(
            self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=False, default=str),
            max_tokens=FAST_MAX_TOKENS,
            temperature=0.15,
            timeout=AGENT_TIMEOUT_SECONDS,
        )
        return _normalize_agent_output(payload, fallback, self.label)


class MacroFundamentalAgent:
    """Analyse valorisation, secteur, catalyseurs macro et corrélation."""

    key = "macro"
    event_name = "agent_macro"
    label = "macro_fundamental"

    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None):
        self.client = client
        self.model = model or _fast_model()

    def deterministic(self, symbol: str, market_type: str, quant_payload: dict) -> dict:
        indicators = quant_payload.get("indicators") or {}
        fundamentals = quant_payload.get("fundamentals") or {}
        momentum = indicators.get("momentum_pct") or {}
        score = 50.0
        key_points: list[str] = []
        risks: list[str] = []

        if market_type == "stock" and fundamentals:
            pe = _as_float(fundamentals.get("pe_ratio"), 0.0)
            target = _as_float(fundamentals.get("target_mean_price"), 0.0)
            price = _as_float(quant_payload.get("price"), 0.0)
            high_52 = _as_float(fundamentals.get("fifty_two_week_high"), 0.0)
            low_52 = _as_float(fundamentals.get("fifty_two_week_low"), 0.0)

            if pe > 0:
                if pe < 20:
                    score += 9
                    key_points.append(f"Valorisation modérée (P/E {pe:.1f}).")
                elif pe > 60:
                    score -= 9
                    risks.append(f"Valorisation tendue (P/E {pe:.1f}).")
                else:
                    key_points.append(f"P/E de {pe:.1f}.")
            if fundamentals.get("sector"):
                key_points.append(f"Secteur : {fundamentals['sector']}.")
            if target and price:
                upside = (target / price - 1) * 100
                if upside >= 8:
                    score += 7
                    key_points.append(f"Objectif analystes {upside:+.1f}% au-dessus du prix.")
                elif upside <= -8:
                    score -= 7
                    risks.append(f"Objectif analystes {upside:+.1f}% sous le prix.")
            if price and high_52 and low_52 and high_52 > low_52:
                position = (price - low_52) / (high_52 - low_52) * 100
                key_points.append(f"Position dans le range 52 semaines : {position:.0f}%.")
                if position >= 92:
                    risks.append("Prix proche du plus haut 52 semaines : risque de résistance.")
                elif position <= 12:
                    key_points.append("Prix proche du plus bas 52 semaines : potentiel de rebond.")
        else:
            key_points.append("Actif crypto : pas de valorisation fondamentale classique.")
            if momentum.get("20d") is not None:
                key_points.append(f"Momentum 20j : {float(momentum['20d']):+.2f}%.")
            atr_pct = indicators.get("atr_pct")
            if atr_pct is not None and atr_pct >= 5:
                risks.append(f"Volatilité crypto élevée (ATR {float(atr_pct):.2f}%).")
            risks.append("Sensibilité aux décisions de taux et à la liquidité globale.")

        five_day = momentum.get("5d")
        if five_day is not None:
            score += max(-10.0, min(10.0, float(five_day) * 0.6))
        twenty_day = momentum.get("20d")
        if twenty_day is not None:
            score += max(-8.0, min(8.0, float(twenty_day) * 0.25))

        if not key_points:
            key_points.append("Données fondamentales partielles : analyse basée sur la tendance.")
        if not risks:
            risks.append("Catalyseur macro non identifié dans les données disponibles.")

        score = _clamp(score, 0, 100, 50)
        bias = "bullish" if score >= 56 else "bearish" if score <= 44 else "neutral"
        confidence = 52.0 + (10.0 if fundamentals else 0.0) + (6.0 if momentum.get("20d") is not None else 0.0)

        return {
            "agent": self.label,
            "bias": bias,
            "score": round(score, 2),
            "confidence": round(min(confidence, 82.0), 2),
            "key_points": key_points[:5],
            "risks": risks[:5],
            "summary": (
                f"Vue fondamentale/macro {bias} pour {symbol} ({market_type}) : "
                f"score {score:.0f}/100."
            ),
            "sector": (fundamentals or {}).get("sector"),
            "source": "deterministic",
        }

    async def run(
        self,
        symbol: str,
        market_type: str,
        quant_payload: dict,
        mode: str = DEFAULT_ANALYSIS_MODE,
    ) -> dict:
        fallback = self.deterministic(symbol, market_type, quant_payload)
        if self.client is None:
            return fallback

        system_prompt = (
            "Tu es MacroFundamentalAgent, analyste macro et fondamental. Utilise uniquement "
            "les données fournies (valorisation, secteur, objectifs analystes, momentum, "
            "actualités). Réponds en JSON strict, sans markdown. "
            + _mode_directive(mode)
        )
        user_payload = {
            "symbol": symbol,
            "market_type": market_type,
            "price": quant_payload.get("price"),
            "fundamentals": quant_payload.get("fundamentals"),
            "news_headlines": (quant_payload.get("news_headlines") or [])[:6],
            "indicators": {
                "trend": (quant_payload.get("indicators") or {}).get("trend"),
                "momentum_pct": (quant_payload.get("indicators") or {}).get("momentum_pct"),
                "volatility_annualized_pct": (quant_payload.get("indicators") or {}).get(
                    "volatility_annualized_pct"
                ),
            },
            "schema": {
                "bias": "bullish|bearish|neutral",
                "score": "0-100 (attractivité fondamentale/macro)",
                "confidence": "0-100",
                "key_points": ["point clé 1", "point clé 2", "point clé 3"],
                "risks": ["risque 1", "risque 2"],
                "valuation_view": "courte phrase",
                "summary": "synthèse en 2 phrases maximum",
            },
            "contraintes": (
                "Maximum 120 mots. Ne pas inventer de ratios ni d'actualités. Si les "
                "données manquent, le préciser."
            ),
        }
        payload = await call_llm_json(
            self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=False, default=str),
            max_tokens=FAST_MAX_TOKENS,
            temperature=0.15,
            timeout=AGENT_TIMEOUT_SECONDS,
        )
        return _normalize_agent_output(payload, fallback, self.label)


class SentimentNewsAgent:
    """Analyse le sentiment des actualités récentes et les risques imminents."""

    key = "sentiment"
    event_name = "agent_sentiment"
    label = "sentiment_news"

    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None):
        self.client = client
        self.model = model or _fast_model()

    @staticmethod
    def _keyword_score(news_items: list[dict]) -> tuple[float, list[str], list[str]]:
        positive = (
            "surge",
            "rally",
            "beat",
            "record",
            "upgrade",
            "growth",
            "bullish",
            "adoption",
            "profit",
            "buy",
            "hausse",
            "record",
            "dépasse",
            "optimiste",
            "gains",
            "partnership",
            "approval",
            "inflow",
            "breakout",
        )
        negative = (
            "drop",
            "fall",
            "miss",
            "downgrade",
            "bearish",
            "loss",
            "hack",
            "lawsuit",
            "warning",
            "crash",
            "chute",
            "baisse",
            "perte",
            "risque",
            "probe",
            "bankruptcy",
            "outflow",
            "exploit",
            "sanction",
        )
        net = 0
        positive_points: list[str] = []
        negative_points: list[str] = []
        for item in news_items[:12]:
            title = str(item.get("title") or "").lower()
            if not title:
                continue
            pos_hits = sum(1 for token in positive if token in title)
            neg_hits = sum(1 for token in negative if token in title)
            net += pos_hits - neg_hits
            if pos_hits > neg_hits:
                positive_points.append(str(item.get("title"))[:160])
            elif neg_hits > pos_hits:
                negative_points.append(str(item.get("title"))[:160])
        return net, positive_points, negative_points

    def deterministic(self, symbol: str, market_type: str, news_items: list[dict]) -> dict:
        news_items = news_items or []
        net, positive_points, negative_points = self._keyword_score(news_items)
        score = _clamp(50 + net * 7, 0, 100, 50)
        bias = "bullish" if score >= 57 else "bearish" if score <= 43 else "neutral"

        key_points = positive_points[:4]
        risks = negative_points[:4]
        if not key_points and news_items:
            key_points = [str(item.get("title"))[:160] for item in news_items[:3] if item.get("title")]
        if not key_points:
            key_points = ["Aucune actualité récente exploitable."]
        if not risks:
            risks = ["Risque de headline surprise non couvert par les flux disponibles."]

        confidence = 30.0 + min(len(news_items), 8) * 5.5
        return {
            "agent": self.label,
            "bias": bias,
            "score": round(float(score), 2),
            "confidence": round(min(confidence, 80.0), 2),
            "key_points": key_points[:5],
            "risks": risks[:5],
            "headline_count": len(news_items),
            "summary": (
                f"Sentiment {bias} pour {symbol} ({market_type}) sur {len(news_items)} "
                f"actualité(s), score {score:.0f}/100."
            ),
            "source": "deterministic",
        }

    async def run(
        self,
        symbol: str,
        market_type: str,
        news_items: list[dict],
        mode: str = DEFAULT_ANALYSIS_MODE,
    ) -> dict:
        fallback = self.deterministic(symbol, market_type, news_items)
        if self.client is None:
            return fallback

        compact_news = [
            {
                "title": str(item.get("title") or "")[:220],
                "source": item.get("source"),
                "published_at": item.get("published_at"),
                "summary": str(item.get("summary") or "")[:220],
            }
            for item in (news_items or [])[:10]
        ]
        system_prompt = (
            "Tu es SentimentNewsAgent, analyste de sentiment de marché. Analyse uniquement "
            "les titres et résumés fournis. Distingue le bruit du risque réel. Réponds en "
            "JSON strict, sans markdown. "
            + _mode_directive(mode)
        )
        user_payload = {
            "symbol": symbol,
            "market_type": market_type,
            "news": compact_news,
            "schema": {
                "bias": "bullish|bearish|neutral",
                "sentiment": "bullish|bearish|neutral",
                "score": "0-100 (sentiment haussier)",
                "confidence": "0-100",
                "key_points": ["point clé 1", "point clé 2", "point clé 3"],
                "risks": ["risque imminent 1", "risque imminent 2"],
                "summary": "synthèse en 2 phrases maximum",
            },
            "contraintes": (
                "Maximum 120 mots. Si aucune actualité pertinente, retourner neutral avec "
                "une confiance basse. Ne pas inventer de news."
            ),
        }
        payload = await call_llm_json(
            self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=False, default=str),
            max_tokens=FAST_MAX_TOKENS,
            temperature=0.2,
            timeout=AGENT_TIMEOUT_SECONDS,
        )
        return _normalize_agent_output(payload, fallback, self.label)


# ---------------------------------------------------------------------------
# Plan de trade et stratégie finale
# ---------------------------------------------------------------------------
def _build_trade_plan(
    price: float,
    atr: float,
    risk_profile: str,
    verdict: str,
    confidence: float,
) -> dict:
    profile = _normalize_risk_profile(risk_profile)
    params = PROFILE_PARAMS[profile]
    entry = float(price)
    atr_value = float(atr) if atr and atr > 0 else entry * 0.02
    multiplier = params["atr_multiplier"]

    direction = "short" if verdict == "SELL" else "long"
    if direction == "long":
        stop_loss = entry - multiplier * atr_value
        take_profit_1 = entry + params["tp1_rr"] * multiplier * atr_value
        take_profit_2 = entry + params["tp2_rr"] * multiplier * atr_value
    else:
        stop_loss = entry + multiplier * atr_value
        take_profit_1 = entry - params["tp1_rr"] * multiplier * atr_value
        take_profit_2 = entry - params["tp2_rr"] * multiplier * atr_value

    risk_per_unit = abs(entry - stop_loss)
    reward = abs(take_profit_1 - entry)
    risk_reward = reward / risk_per_unit if risk_per_unit > 0 else 0.0

    base_size = params["base_position_pct"]
    position_size = base_size * (0.75 + 0.5 * (_clamp(confidence, 0, 100, 50) / 100))
    if verdict in {"HOLD", "WATCH"}:
        position_size *= 0.5
    position_size = max(0.5, min(position_size, 15.0))

    return {
        "entry_price": _round_price(entry),
        "stop_loss": _round_price(stop_loss),
        "take_profit_1": _round_price(take_profit_1),
        "take_profit_2": _round_price(take_profit_2),
        "position_size_pct": round(position_size, 2),
        "risk_reward_ratio": round(risk_reward, 2),
        "direction": direction,
        "atr": _round_price(atr_value),
        "atr_multiplier": multiplier,
        "risk_per_unit": _round_price(risk_per_unit),
    }


def _normalize_trade_plan(
    raw_plan: Any,
    fallback_plan: dict,
    price: float,
    atr: float,
    risk_profile: str,
    verdict: str,
    confidence: float,
) -> dict:
    plan = dict(fallback_plan)
    if not isinstance(raw_plan, dict):
        return plan

    entry = _as_float(raw_plan.get("entry_price"), 0.0)
    if entry <= 0 or abs(entry / max(price, 1e-9) - 1.0) > 0.2:
        entry = float(fallback_plan["entry_price"])

    regenerated = _build_trade_plan(entry, atr, risk_profile, verdict, confidence)
    stop_loss = _as_float(raw_plan.get("stop_loss"), 0.0)
    take_profit_1 = _as_float(raw_plan.get("take_profit_1"), 0.0)
    take_profit_2 = _as_float(raw_plan.get("take_profit_2"), 0.0)
    position_size = _as_float(raw_plan.get("position_size_pct"), 0.0)
    risk_reward = _as_float(raw_plan.get("risk_reward_ratio"), 0.0)
    direction = regenerated["direction"]

    if stop_loss <= 0 or (direction == "long" and stop_loss >= entry) or (
        direction == "short" and stop_loss <= entry
    ):
        stop_loss = float(regenerated["stop_loss"])
    if take_profit_1 <= 0 or (direction == "long" and take_profit_1 <= entry) or (
        direction == "short" and take_profit_1 >= entry
    ):
        take_profit_1 = float(regenerated["take_profit_1"])
    if take_profit_2 <= 0:
        take_profit_2 = float(regenerated["take_profit_2"])
    if direction == "long" and take_profit_2 < take_profit_1:
        take_profit_2 = float(regenerated["take_profit_2"])
    if direction == "short" and take_profit_2 > take_profit_1:
        take_profit_2 = float(regenerated["take_profit_2"])
    if position_size <= 0:
        position_size = float(regenerated["position_size_pct"])
    position_size = max(0.5, min(position_size, 15.0))

    risk_per_unit = abs(entry - stop_loss)
    computed_rr = abs(take_profit_1 - entry) / risk_per_unit if risk_per_unit > 0 else 0.0
    if risk_reward <= 0 or abs(risk_reward - computed_rr) > 2.5:
        risk_reward = computed_rr

    plan.update(
        {
            "entry_price": _round_price(entry),
            "stop_loss": _round_price(stop_loss),
            "take_profit_1": _round_price(take_profit_1),
            "take_profit_2": _round_price(take_profit_2),
            "position_size_pct": round(position_size, 2),
            "risk_reward_ratio": round(risk_reward, 2),
            "direction": direction,
            "atr": regenerated["atr"],
            "atr_multiplier": regenerated["atr_multiplier"],
            "risk_per_unit": _round_price(risk_per_unit),
        }
    )
    return plan


class ChiefStrategyAgent:
    """Débat Bull vs Bear et verdict final du Risk Manager."""

    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None):
        self.client = client
        self.model = model or _reasoning_model()

    def deterministic(
        self,
        symbol: str,
        market_type: str,
        risk_profile: str,
        quant_payload: dict,
        agents: dict,
        memory_context: dict,
        mode: str = DEFAULT_ANALYSIS_MODE,
    ) -> dict:
        technical = agents.get("technical") or {}
        macro = agents.get("macro") or {}
        sentiment = agents.get("sentiment") or {}
        indicators = quant_payload.get("indicators") or {}

        tech_score = _clamp(technical.get("score"), 0, 100, 50)
        macro_score = _clamp(macro.get("score"), 0, 100, 50)
        sent_score = _clamp(sentiment.get("score"), 0, 100, 50)
        weighted = tech_score * 0.45 + macro_score * 0.30 + sent_score * 0.25

        biases = [
            _normalize_bias(technical.get("bias")),
            _normalize_bias(macro.get("bias")),
            _normalize_bias(sentiment.get("bias")),
        ]
        bullish_count = biases.count("bullish")
        bearish_count = biases.count("bearish")
        agreement = max(bullish_count, bearish_count)

        if weighted >= 60 and biases[0] == "bullish" and biases[2] != "bearish":
            verdict = "BUY"
        elif weighted <= 40 and biases[0] == "bearish" and biases[2] != "bullish":
            verdict = "SELL"
        elif 47 <= weighted <= 53 or max(bullish_count, bearish_count) <= 1:
            verdict = "WATCH"
        else:
            verdict = "HOLD"

        gates = quant_payload.get("buy_gates") or {}
        gate_blocked = bool(verdict == "BUY" and gates and not gates.get("allowed", True))
        if gate_blocked:
            verdict = "WATCH"

        confidence = 42.0 + abs(weighted - 50) * 1.15 + agreement * 6.0
        confidence += (
            _clamp(technical.get("confidence"), 0, 100, 50)
            + _clamp(macro.get("confidence"), 0, 100, 50)
            + _clamp(sentiment.get("confidence"), 0, 100, 50)
        ) / 30.0

        performance = memory_context.get("performance") or {}
        total_history = int(performance.get("total_trades") or 0)
        if total_history >= 3:
            win_rate = _as_float(performance.get("win_rate_pct"), 50.0)
            if win_rate >= 55:
                confidence += 4
            elif win_rate < 40:
                confidence -= 6
        if verdict == "BUY" and memory_context.get("pitfalls"):
            confidence -= 1.5
        confidence = _clamp(confidence, 15, 96, 50)

        bull_case: list[str] = []
        bear_case: list[str] = []
        for agent in (technical, macro, sentiment):
            bias = _normalize_bias(agent.get("bias"))
            points = _string_list(agent.get("key_points"), 4)
            risks = _string_list(agent.get("risks"), 4)
            if bias == "bullish":
                bull_case.extend(points)
                bear_case.extend(risks)
            elif bias == "bearish":
                bear_case.extend(points)
                bear_case.extend(risks)
            else:
                bull_case.extend(points[:1])
                bear_case.extend(risks[:2])

        momentum = indicators.get("momentum_pct") or {}
        if momentum.get("5d") is not None and float(momentum["5d"]) > 0:
            bull_case.append(f"Momentum 5 jours positif ({float(momentum['5d']):+.2f}%).")
        if momentum.get("5d") is not None and float(momentum["5d"]) < 0:
            bear_case.append(f"Momentum 5 jours négatif ({float(momentum['5d']):+.2f}%).")
        if not bull_case:
            bull_case.append("Aucun catalyseur haussier majeur identifié.")
        if not bear_case:
            bear_case.append("Aucun risque baissier majeur identifié.")
        bull_case = _string_list(bull_case, 5)
        bear_case = _string_list(bear_case, 5)

        atr = _as_float(quant_payload.get("atr14"), _as_float(quant_payload.get("price")) * 0.02)
        plan = _build_trade_plan(
            price=_as_float(quant_payload.get("price")),
            atr=atr,
            risk_profile=risk_profile,
            verdict=verdict,
            confidence=confidence,
        )

        rules = memory_context.get("rules") or []
        memory_note = (
            f" Mémoire appliquée : {rules[0][:180]}" if rules else " Aucune mémoire consolidée."
        )
        rationale = (
            f"Verdict {verdict} pour {symbol} ({market_type}) avec une confiance de "
            f"{confidence:.0f}%. Score pondéré {weighted:.1f}/100 "
            f"(tech {tech_score:.0f}, macro {macro_score:.0f}, sentiment {sent_score:.0f})."
            f"{memory_note}"
        )

        return {
            "symbol": symbol,
            "market_type": market_type,
            "timeframe": quant_payload.get("timeframe"),
            "risk_profile": _normalize_risk_profile(risk_profile),
            "analysis_mode": _normalize_mode(mode),
            "buy_gates": gates or None,
            "buy_gates_blocked": gate_blocked,
            "bull_case": bull_case,
            "bear_case": bear_case,
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "trade_plan": plan,
            "rationale": rationale,
            "agent_consensus": {
                "technical_score": round(tech_score, 2),
                "macro_score": round(macro_score, 2),
                "sentiment_score": round(sent_score, 2),
                "weighted_score": round(weighted, 2),
                "bullish_votes": bullish_count,
                "bearish_votes": bearish_count,
            },
            "memory_applied": bool(memory_context.get("has_memory")),
            "memory_summary": memory_context.get("summary"),
            "memory_rules": memory_context.get("rules") or [],
            "memory_pitfalls": memory_context.get("pitfalls") or [],
            "source": "deterministic",
            "generated_at": datetime.now(UTC).isoformat(),
        }

    async def run(
        self,
        symbol: str,
        market_type: str,
        risk_profile: str,
        quant_payload: dict,
        agents: dict,
        memory_context: dict,
        mode: str = DEFAULT_ANALYSIS_MODE,
    ) -> dict:
        fallback = self.deterministic(
            symbol,
            market_type,
            risk_profile,
            quant_payload,
            agents,
            memory_context,
            mode=mode,
        )
        if self.client is None:
            return fallback

        system_prompt = (
            "Tu es ChiefStrategyAgent, stratège et Risk Manager. Organise un débat "
            "Bull vs Bear à partir des trois rapports d'agents, du profil de risque et "
            "de la mémoire de trading. Tranche avec un verdict BUY/SELL/HOLD/WATCH et "
            "construis un plan de trade cohérent. Réponds en JSON strict, sans markdown. "
            + _mode_directive(mode)
        )
        user_payload = {
            "symbol": symbol,
            "market_type": market_type,
            "timeframe": quant_payload.get("timeframe"),
            "risk_profile": _normalize_risk_profile(risk_profile),
            "analysis_mode": _normalize_mode(mode),
            "buy_gates": quant_payload.get("buy_gates"),
            "price": quant_payload.get("price"),
            "atr14": quant_payload.get("atr14"),
            "technical_score": quant_payload.get("technical_score"),
            "key_levels": quant_payload.get("levels"),
            "agents": {
                "technical": {
                    "bias": (agents.get("technical") or {}).get("bias"),
                    "score": (agents.get("technical") or {}).get("score"),
                    "confidence": (agents.get("technical") or {}).get("confidence"),
                    "key_points": (agents.get("technical") or {}).get("key_points"),
                    "risks": (agents.get("technical") or {}).get("risks"),
                    "summary": (agents.get("technical") or {}).get("summary"),
                },
                "macro": {
                    "bias": (agents.get("macro") or {}).get("bias"),
                    "score": (agents.get("macro") or {}).get("score"),
                    "confidence": (agents.get("macro") or {}).get("confidence"),
                    "key_points": (agents.get("macro") or {}).get("key_points"),
                    "risks": (agents.get("macro") or {}).get("risks"),
                    "summary": (agents.get("macro") or {}).get("summary"),
                },
                "sentiment": {
                    "bias": (agents.get("sentiment") or {}).get("bias"),
                    "score": (agents.get("sentiment") or {}).get("score"),
                    "confidence": (agents.get("sentiment") or {}).get("confidence"),
                    "key_points": (agents.get("sentiment") or {}).get("key_points"),
                    "risks": (agents.get("sentiment") or {}).get("risks"),
                    "summary": (agents.get("sentiment") or {}).get("summary"),
                },
            },
            "memory": memory_context.get("prompt_context"),
            "plan_reference": fallback["trade_plan"],
            "risk_rules": {
                "conservative": "Risque max 1% du capital, stop 2x ATR, privilégier BUY/HOLD.",
                "moderate": "Risque max 2% du capital, stop 1.6x ATR, R:R minimum 2.",
                "aggressive": "Risque max 3% du capital, stop 1.3x ATR, R:R minimum 2.4.",
            },
            "schema": {
                "bull_case": ["argument haussier 1", "argument haussier 2"],
                "bear_case": ["argument baissier 1", "argument baissier 2"],
                "verdict": "BUY|SELL|HOLD|WATCH",
                "confidence": "0-100",
                "trade_plan": {
                    "entry_price": "nombre",
                    "stop_loss": "nombre basé sur l'ATR",
                    "take_profit_1": "nombre",
                    "take_profit_2": "nombre",
                    "position_size_pct": "nombre adapté au profil",
                    "risk_reward_ratio": "nombre",
                },
                "rationale": "synthèse du débat en 3 phrases maximum",
            },
            "contraintes": (
                "Maximum 260 mots. Utilise plan_reference comme base et ajuste-le si les "
                "arguments le justifient. Le stop doit être cohérent avec l'ATR et le sens "
                "du trade. Ne pas inventer de niveaux sans lien avec les données."
            ),
        }
        payload = await call_llm_json(
            self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=False, default=str),
            max_tokens=CHIEF_MAX_TOKENS,
            temperature=0.25,
            timeout=CHIEF_TIMEOUT_SECONDS,
        )
        if not isinstance(payload, dict):
            return fallback

        verdict = _normalize_verdict(payload.get("verdict"), fallback["verdict"])
        gates = quant_payload.get("buy_gates") or {}
        gate_blocked = bool(verdict == "BUY" and gates and not gates.get("allowed", True))
        if gate_blocked:
            verdict = "WATCH"
        confidence = _clamp(payload.get("confidence"), 0, 100, fallback["confidence"])
        atr = _as_float(
            quant_payload.get("atr14"),
            _as_float(quant_payload.get("price")) * 0.02,
        )
        raw_plan = payload.get("trade_plan")
        fallback_for_verdict = _build_trade_plan(
            price=_as_float(quant_payload.get("price")),
            atr=atr,
            risk_profile=risk_profile,
            verdict=verdict,
            confidence=confidence,
        )
        plan = _normalize_trade_plan(
            raw_plan,
            fallback_for_verdict,
            price=_as_float(quant_payload.get("price")),
            atr=atr,
            risk_profile=risk_profile,
            verdict=verdict,
            confidence=confidence,
        )

        bull_case = _string_list(payload.get("bull_case"), 6) or fallback["bull_case"]
        bear_case = _string_list(payload.get("bear_case"), 6) or fallback["bear_case"]
        rationale = str(payload.get("rationale") or payload.get("summary") or "").strip()

        result = dict(fallback)
        result.update(
            {
                "bull_case": bull_case,
                "bear_case": bear_case,
                "verdict": verdict,
                "confidence": round(confidence, 2),
                "trade_plan": plan,
                "analysis_mode": _normalize_mode(mode),
                "buy_gates": gates or None,
                "buy_gates_blocked": gate_blocked,
                "rationale": rationale[:1200] if rationale else fallback["rationale"],
                "source": "llm",
                "generated_at": datetime.now(UTC).isoformat(),
            }
        )
        return result


# ---------------------------------------------------------------------------
# Construction des événements
# ---------------------------------------------------------------------------
def _options_summary(options: dict | None) -> dict | None:
    if not isinstance(options, dict):
        return None
    calls = options.get("calls") or []
    puts = options.get("puts") or []
    spot = _as_float(options.get("spot"), 0.0)

    def _atm(contracts: list[dict]) -> dict | None:
        valid = [
            contract
            for contract in contracts
            if _as_float(contract.get("strike"), 0.0) > 0
        ]
        if not valid:
            return None
        if spot > 0:
            valid.sort(key=lambda item: abs(_as_float(item.get("strike")) - spot))
        contract = valid[0]
        return {
            "strike": _as_float(contract.get("strike")),
            "last_price": _as_float(contract.get("last_price")),
            "implied_volatility_pct": round(
                _as_float(contract.get("implied_volatility")) * 100.0, 2
            ),
            "open_interest": _as_float(contract.get("open_interest")),
        }

    return {
        "expiration": options.get("expiration"),
        "spot": spot or None,
        "calls_count": len(calls),
        "puts_count": len(puts),
        "atm_call": _atm(calls),
        "atm_put": _atm(puts),
    }


def _build_quant_payload(
    symbol: str,
    market_type: str,
    timeframe: str,
    risk_profile: str,
    snapshot: dict,
    memory_context: dict,
) -> dict:
    ohlcv = snapshot.get("ohlcv")
    if ohlcv is None or getattr(ohlcv, "empty", True):
        raise RuntimeError(f"Aucune donnée OHLCV exploitable pour {symbol}.")

    technical = quant.compute_technicals(ohlcv, include_candles=True, candle_limit=200)
    if "error" in technical:
        raise RuntimeError(technical["error"])

    surge_score, surge_signals = quant._compute_surge_score(technical)
    indicators = {key: value for key, value in technical.items() if key != "candles"}
    news = snapshot.get("news") or []
    headlines = [
        str(item.get("title"))
        for item in news[:10]
        if isinstance(item, dict) and item.get("title")
    ]

    quant_payload = {
        "symbol": symbol,
        "market_type": market_type,
        "timeframe": timeframe,
        "risk_profile": risk_profile,
        "price": technical.get("price"),
        "last_close": snapshot.get("last_close"),
        "candles": technical.get("candles") or [],
        "candles_count": technical.get("candles_used"),
        "indicators": indicators,
        "trend": technical.get("trend"),
        "trend_score": technical.get("trend_score"),
        "rsi14": technical.get("rsi14"),
        "atr14": technical.get("atr14"),
        "atr_pct": technical.get("atr_pct"),
        "technical_score": surge_score,
        "technical_grade": quant._grade_score(surge_score),
        "signals": technical.get("signals") or [],
        "surge_signals": surge_signals,
        "levels": technical.get("support_resistance") or {},
        "price_snapshot": snapshot.get("price"),
        "fundamentals": snapshot.get("fundamentals"),
        "options": _options_summary(snapshot.get("options")),
        "news_headlines": headlines,
        "news_count": len(news),
        "memory_summary": memory_context.get("summary"),
        "memory_rules": memory_context.get("rules") or [],
        "memory_pitfalls": memory_context.get("pitfalls") or [],
        "fetched_at": snapshot.get("fetched_at"),
        "generated_at": datetime.now(UTC).isoformat(),
        "cached": False,
    }
    return quant_payload


async def _fetch_core_data(symbol: str, market_type: str, timeframe: str) -> dict:
    """Récupère uniquement OHLCV + prix pour émettre l'événement quant au plus vite."""
    if market_type == "crypto":
        price_coro = fetcher.fetch_crypto_price(symbol)
    else:
        price_coro = fetcher.fetch_stock_quote(symbol)

    ohlcv_result, price_result = await asyncio.gather(
        fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=300, market_type=market_type),
        price_coro,
        return_exceptions=True,
    )
    if isinstance(ohlcv_result, Exception):
        raise ohlcv_result
    price = None if isinstance(price_result, Exception) else price_result
    last_close = _as_float((price or {}).get("price"), 0.0)
    if last_close <= 0 and getattr(ohlcv_result, "empty", True) is False:
        last_close = _as_float(ohlcv_result["Close"].iloc[-1], 0.0)
    return {
        "ohlcv": ohlcv_result,
        "price": price,
        "last_close": last_close or None,
        "fetched_at": datetime.now(UTC).isoformat(),
    }


async def _fetch_context_extras(
    symbol: str,
    market_type: str,
    include_options: bool = True,
) -> dict:
    """Récupère news/fondamentaux/options sans retarder l'événement quant."""
    keys = ["news"]
    coros = [fetcher.fetch_news(symbol, market_type, limit=10)]
    if market_type == "stock":
        keys.append("fundamentals")
        coros.append(fetcher.fetch_stock_fundamentals(symbol))
        if include_options:
            keys.append("options")
            coros.append(fetcher.fetch_option_chain(symbol))

    results = await asyncio.gather(*coros, return_exceptions=True)
    extras: dict[str, Any] = {}
    for key, result in zip(keys, results):
        extras[key] = None if isinstance(result, Exception) else result
    return extras


async def _stream_cached(cached: dict) -> AsyncGenerator[str, None]:
    agents = cached.get("agents") or {}
    age = cached.get("age_minutes")

    def _tag(payload: Any) -> dict:
        if not isinstance(payload, dict):
            return {"cached": True, "cache_age_minutes": age}
        return {**payload, "cached": True, "cache_age_minutes": age}

    events = [
        ("quant", cached.get("quant") or {}),
        ("agent_tech", agents.get("technical") or _agent_error_payload("technical_flow", "absent du cache")),
        ("agent_macro", agents.get("macro") or _agent_error_payload("macro_fundamental", "absent du cache")),
        (
            "agent_sentiment",
            agents.get("sentiment") or _agent_error_payload("sentiment_news", "absent du cache"),
        ),
        ("synthesis", cached.get("synthesis") or {}),
    ]
    for event_name, payload in events:
        yield sse_event(event_name, _tag(payload))


async def _queue_agent(
    queue: asyncio.Queue,
    key: str,
    event_name: str,
    coroutine: Any,
) -> None:
    try:
        result = await coroutine
    except Exception as exc:
        result = _agent_error_payload(key, str(exc))
    await queue.put((key, event_name, result))


# ---------------------------------------------------------------------------
# Générateur SSE principal
# ---------------------------------------------------------------------------
async def stream_market_analysis(
    symbol: str,
    timeframe: str = "1d",
    risk_profile: str = "moderate",
    mode: str = DEFAULT_ANALYSIS_MODE,
    force_refresh: bool = False,
) -> AsyncGenerator[str, None]:
    """Produit le flux SSE complet d'analyse d'un symbole.

    `mode` : `auto` (équilibré), `research` (catalyseurs/actualités) ou
    `analytical` (rigueur quantitative et scénarios d'invalidation).
    """
    started = time.perf_counter()
    market_type = fetcher.detect_market_type(symbol)
    normalized_symbol = fetcher.normalize_symbol(symbol, market_type)
    normalized_timeframe = _normalize_timeframe(timeframe)
    normalized_risk = _normalize_risk_profile(risk_profile)
    normalized_mode = _normalize_mode(mode)
    ttl_minutes = int(os.getenv("CACHE_TTL_MINUTES", "30"))

    if not force_refresh:
        cached = db.get_cached_analysis(
            normalized_symbol,
            market_type,
            normalized_timeframe,
            normalized_risk,
            ttl_minutes=ttl_minutes,
        )
        if cached:
            async for event in _stream_cached(cached):
                yield event
            return

    try:
        memory_context = get_memory_context(
            normalized_symbol,
            market_type,
            normalized_risk,
        )
        core_data = await _fetch_core_data(normalized_symbol, market_type, normalized_timeframe)
        quant_payload = _build_quant_payload(
            normalized_symbol,
            market_type,
            normalized_timeframe,
            normalized_risk,
            core_data,
            memory_context,
        )
        quant_payload["analysis_mode"] = normalized_mode
        buy_gates = evaluate_buy_gates(
            normalized_symbol,
            market_type,
            normalized_risk,
            quant_payload,
        )
        quant_payload["buy_gates"] = buy_gates
        quant_payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        yield sse_event("quant", quant_payload)
    except Exception as exc:
        yield sse_event(
            "error",
            {
                "symbol": normalized_symbol,
                "market_type": market_type,
                "timeframe": normalized_timeframe,
                "stage": "quant",
                "message": str(exc),
                "generated_at": datetime.now(UTC).isoformat(),
            },
        )
        return

    extras = await _fetch_context_extras(
        normalized_symbol,
        market_type,
        include_options=(market_type == "stock"),
    )
    news_items = extras.get("news") or []
    quant_payload["news_headlines"] = [
        str(item.get("title"))
        for item in news_items[:10]
        if isinstance(item, dict) and item.get("title")
    ]
    quant_payload["news_count"] = len(news_items)
    quant_payload["fundamentals"] = extras.get("fundamentals")
    quant_payload["options"] = _options_summary(extras.get("options"))

    client = build_llm_client()
    technical_agent = TechnicalFlowAgent(client)
    macro_agent = MacroFundamentalAgent(client)
    sentiment_agent = SentimentNewsAgent(client)

    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(
            _queue_agent(
                queue,
                technical_agent.key,
                technical_agent.event_name,
                technical_agent.run(normalized_symbol, market_type, quant_payload, mode=normalized_mode),
            )
        ),
        asyncio.create_task(
            _queue_agent(
                queue,
                macro_agent.key,
                macro_agent.event_name,
                macro_agent.run(normalized_symbol, market_type, quant_payload, mode=normalized_mode),
            )
        ),
        asyncio.create_task(
            _queue_agent(
                queue,
                sentiment_agent.key,
                sentiment_agent.event_name,
                sentiment_agent.run(normalized_symbol, market_type, news_items, mode=normalized_mode),
            )
        ),
    ]
    async def _drain_agents() -> None:
        await asyncio.gather(*tasks, return_exceptions=True)

    gather_task = asyncio.create_task(_drain_agents())
    collected: dict[str, dict] = {}

    try:
        for _ in range(len(tasks)):
            key, event_name, result = await queue.get()
            collected[key] = result
            yield sse_event(event_name, result)
        await gather_task
    finally:
        if not gather_task.done():
            gather_task.cancel()
            await asyncio.gather(gather_task, return_exceptions=True)

    chief = ChiefStrategyAgent(client)
    try:
        synthesis_payload = await chief.run(
            normalized_symbol,
            market_type,
            normalized_risk,
            quant_payload,
            collected,
            memory_context,
            mode=normalized_mode,
        )
    except Exception as exc:
        synthesis_payload = chief.deterministic(
            normalized_symbol,
            market_type,
            normalized_risk,
            quant_payload,
            collected,
            memory_context,
            mode=normalized_mode,
        )
        synthesis_payload["error"] = str(exc)

    final_gates = evaluate_buy_gates(
        normalized_symbol,
        market_type,
        normalized_risk,
        quant_payload,
        confidence=synthesis_payload.get("confidence"),
    )
    synthesis_payload["buy_gates"] = final_gates
    if synthesis_payload.get("verdict") == "BUY" and not final_gates.get("allowed", True):
        synthesis_payload["verdict"] = "WATCH"
        synthesis_payload["buy_gates_blocked"] = True
        blockers = ", ".join(
            gate["name"] for gate in final_gates.get("blockers", [])
        )
        existing = str(synthesis_payload.get("rationale") or "").strip()
        synthesis_payload["rationale"] = (
            (existing + " " if existing else "")
            + f"Verdict BUY dégradé en WATCH : gates non satisfaites ({blockers})."
        )

    try:
        db.save_analysis_cache(
            symbol=normalized_symbol,
            market_type=market_type,
            timeframe=normalized_timeframe,
            risk_profile=normalized_risk,
            quant=quant_payload,
            agents=collected,
            synthesis=synthesis_payload,
        )
    except Exception as exc:
        synthesis_payload["cache_write_error"] = str(exc)

    synthesis_payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    yield sse_event("synthesis", synthesis_payload)
