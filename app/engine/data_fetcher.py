"""Récupération de données de marché pour RwaMarket.

Crypto : Binance REST public (aucune clé) avec repli CoinGecko.
Actions US : yfinance, avec Alpaca en source prioritaire si les clés sont
configurées, puis repli automatique sur Yahoo Finance.

Toutes les fonctions réseau exposées sont asynchrones (httpx) et les appels
yfinance, bloquants, sont exécutés dans un thread pour ne jamais bloquer la
boucle d'événements FastAPI.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

BINANCE_BASE = "https://api.binance.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"
FINNHUB_BASE = "https://finnhub.io/api/v1"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

REQUEST_TIMEOUT = 15.0
DEFAULT_HEADERS = {
    "User-Agent": "RwaMarket/1.0 (+https://rwamarket.rwazap.com)",
    "Accept": "application/json, text/plain, */*",
}

CRYPTO_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD", "DAI", "BTC", "ETH", "BNB")
KNOWN_CRYPTO = {
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "DOGE",
    "AVAX",
    "LINK",
    "DOT",
    "LTC",
    "TRX",
    "TON",
    "SHIB",
    "SUI",
    "NEAR",
    "APT",
    "ARB",
    "OP",
    "ATOM",
    "FIL",
    "INJ",
    "PEPE",
    "WIF",
    "SEI",
    "RNDR",
    "RENDER",
    "MATIC",
    "BCH",
    "ETC",
    "XLM",
    "HBAR",
    "ICP",
    "VET",
    "FET",
    "TIA",
    "JUP",
    "PYTH",
    "STRK",
    "WLD",
    "ORDI",
    "BONK",
    "FLOKI",
}

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "DOT": "polkadot",
    "LTC": "litecoin",
    "TRX": "tron",
    "TON": "the-open-network",
    "SHIB": "shiba-inu",
    "SUI": "sui",
    "NEAR": "near",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
    "ATOM": "cosmos",
    "FIL": "filecoin",
    "INJ": "injective-protocol",
    "PEPE": "pepe",
    "WIF": "dogwifcoin",
    "SEI": "sei-network",
    "RNDR": "render-token",
    "RENDER": "render-token",
    "MATIC": "matic-network",
    "BCH": "bitcoin-cash",
    "ETC": "ethereum-classic",
    "XLM": "stellar",
    "HBAR": "hedera-hashgraph",
    "ICP": "internet-computer",
    "VET": "vechain",
    "FET": "fetch-ai",
    "TIA": "celestia",
    "JUP": "jupiter-exchange-solana",
    "PYTH": "pyth-network",
    "STRK": "starknet",
    "WLD": "worldcoin-wld",
    "ORDI": "ordinals",
    "BONK": "bonk",
    "FLOKI": "floki",
}

BINANCE_INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}

TIMEFRAME_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
    "1M": 43200,
}

YFINANCE_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "4h": "60m",
    "1d": "1d",
    "1w": "1wk",
    "1M": "1mo",
}

YFINANCE_PERIODS = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1h": "730d",
    "4h": "730d",
    "1d": "2y",
    "1wk": "5y",
    "1mo": "10y",
}

ALPACA_TIMEFRAMES = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "4h": "1Hour",
    "1d": "1Day",
    "1w": "1Week",
    "1M": "1Month",
}


class DataFetchError(RuntimeError):
    """Erreur levée quand aucune source de données ne répond correctement."""


# ---------------------------------------------------------------------------
# Détection et normalisation des symboles
# ---------------------------------------------------------------------------
def detect_market_type(symbol: str) -> str:
    """Retourne 'crypto' ou 'stock' à partir de la forme du symbole."""
    raw = (symbol or "").strip().upper()
    if not raw:
        raise ValueError("Symbole vide.")

    if "/" in raw:
        base, quote = raw.split("/", 1)
        if quote in CRYPTO_QUOTES or base in KNOWN_CRYPTO:
            return "crypto"

    compact = raw.replace("-", "").replace("_", "")
    if compact.endswith(CRYPTO_QUOTES):
        return "crypto"

    base_part = re.split(r"[-_/]", raw)[0]
    if base_part in KNOWN_CRYPTO:
        return "crypto"

    return "stock"


def normalize_symbol(symbol: str, market_type: str | None = None) -> str:
    """Normalise un symbole (BTC-USD -> BTCUSDT, btc/usdt -> BTCUSDT)."""
    raw = (symbol or "").strip().upper()
    market = market_type or detect_market_type(raw)

    if market == "crypto":
        compact = raw.replace("/", "").replace("_", "").replace("-", "")
        for quote in sorted(CRYPTO_QUOTES, key=len, reverse=True):
            if compact.endswith(quote) and len(compact) > len(quote):
                base = compact[: -len(quote)]
                if quote == "USD":
                    return f"{base}USDT"
                return compact
        if compact in KNOWN_CRYPTO:
            return f"{compact}USDT"
        return compact or raw

    return raw


def _crypto_base_asset(symbol: str) -> str:
    normalized = normalize_symbol(symbol, "crypto")
    for quote in sorted(CRYPTO_QUOTES, key=len, reverse=True):
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return normalized[: -len(quote)]
    return normalized


def _normalize_timeframe(timeframe: str | None) -> str:
    tf = (timeframe or "1d").strip()
    if tf in BINANCE_INTERVALS:
        return tf
    lowered = tf.lower()
    for candidate in BINANCE_INTERVALS:
        if candidate.lower() == lowered:
            return candidate
    return "1d"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ramène un DataFrame OHLCV au schéma standard RwaMarket."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    result = df.copy()
    result.columns = [str(col).strip().title() for col in result.columns]
    if "Volume" not in result.columns:
        result["Volume"] = 0.0
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in result.columns:
            raise DataFetchError(f"Colonne OHLCV manquante : {col}")
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    result.index = pd.to_datetime(result.index, utc=True)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


# ---------------------------------------------------------------------------
# Helpers HTTP / exécution synchrone
# ---------------------------------------------------------------------------
async def _get_json(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = REQUEST_TIMEOUT,
) -> Any:
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)
    async with httpx.AsyncClient(timeout=timeout, headers=merged_headers, follow_redirects=True) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def _get_text(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = REQUEST_TIMEOUT,
) -> str:
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)
    async with httpx.AsyncClient(timeout=timeout, headers=merged_headers, follow_redirects=True) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.text


def _run_coroutine_sync(coro: Any) -> Any:
    """Exécute une coroutine depuis un contexte synchrone, même si une boucle tourne."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


async def _safe_await(coro: Any, default: Any = None) -> Any:
    try:
        return await coro
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Crypto : Binance puis CoinGecko
# ---------------------------------------------------------------------------
async def fetch_crypto_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 300,
) -> pd.DataFrame:
    """OHLCV crypto via Binance, avec repli automatique sur CoinGecko."""
    binance_symbol = normalize_symbol(symbol, "crypto")
    interval = BINANCE_INTERVALS.get(_normalize_timeframe(timeframe), "1d")
    safe_limit = max(1, min(int(limit), 1000))

    try:
        rows = await _get_json(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": binance_symbol, "interval": interval, "limit": safe_limit},
        )
        if isinstance(rows, list) and rows:
            df = pd.DataFrame(
                [
                    {
                        "Open": float(row[1]),
                        "High": float(row[2]),
                        "Low": float(row[3]),
                        "Close": float(row[4]),
                        "Volume": float(row[5]),
                        "timestamp": pd.to_datetime(row[0], unit="ms", utc=True),
                    }
                    for row in rows
                ]
            ).set_index("timestamp")
            return _normalize_ohlcv(df)
    except Exception:
        pass

    return await fetch_crypto_ohlcv_coingecko(symbol, timeframe, limit)


async def fetch_crypto_ohlcv_coingecko(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 300,
) -> pd.DataFrame:
    """Repli CoinGecko : OHLC + volumes fusionnés, puis rééchantillonnage."""
    base = _crypto_base_asset(symbol)
    coin_id = COINGECKO_IDS.get(base)
    if not coin_id:
        raise DataFetchError(f"Actif crypto inconnu pour CoinGecko : {base}")

    tf = _normalize_timeframe(timeframe)
    minutes = TIMEFRAME_MINUTES.get(tf, 1440)
    days = max(1, int(math.ceil((limit * minutes) / 1440)) + 1)

    ohlc_raw, volumes_raw = await asyncio.gather(
        _get_json(
            f"{COINGECKO_BASE}/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": days},
        ),
        _get_json(
            f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days},
        ),
        return_exceptions=True,
    )

    if isinstance(ohlc_raw, Exception) or not ohlc_raw:
        raise DataFetchError(f"CoinGecko n'a retourné aucune donnée pour {symbol}.")

    ohlc = pd.DataFrame(ohlc_raw, columns=["timestamp", "Open", "High", "Low", "Close"])
    ohlc["timestamp"] = pd.to_datetime(ohlc["timestamp"], unit="ms", utc=True)
    ohlc = ohlc.set_index("timestamp").sort_index()

    volumes = pd.Series(dtype=float)
    if isinstance(volumes_raw, dict) and volumes_raw.get("total_volumes"):
        volume_df = pd.DataFrame(volumes_raw["total_volumes"], columns=["timestamp", "Volume"])
        volume_df["timestamp"] = pd.to_datetime(volume_df["timestamp"], unit="ms", utc=True)
        volumes = volume_df.set_index("timestamp")["Volume"].sort_index()

    if not volumes.empty:
        volume_frame = volumes.rename("Volume").to_frame()
        merged = pd.merge_asof(
            ohlc.reset_index().sort_values("timestamp"),
            volume_frame.reset_index().sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        ).set_index("timestamp")
    else:
        merged = ohlc.copy()
        merged["Volume"] = 0.0

    resample_rule = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1D",
        "3d": "3D",
        "1w": "1W",
        "1M": "1MS",
    }.get(tf, "1D")

    if resample_rule not in {"1D", "3D", "1W", "1MS"} or len(merged) > limit:
        aggregated = merged.resample(resample_rule).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        )
        aggregated = aggregated.dropna(subset=["Close"])
        if not aggregated.empty:
            merged = aggregated

    return _normalize_ohlcv(merged.tail(limit))


async def fetch_crypto_price(symbol: str) -> dict:
    """Prix temps réel et statistiques 24h via Binance, repli CoinGecko."""
    binance_symbol = normalize_symbol(symbol, "crypto")
    try:
        data = await _get_json(
            f"{BINANCE_BASE}/api/v3/ticker/24hr",
            params={"symbol": binance_symbol},
        )
        if isinstance(data, dict) and data.get("lastPrice"):
            return {
                "symbol": binance_symbol,
                "market_type": "crypto",
                "price": float(data["lastPrice"]),
                "change_pct_24h": float(data.get("priceChangePercent", 0.0)),
                "change_24h": float(data.get("priceChange", 0.0)),
                "high_24h": float(data.get("highPrice", 0.0)),
                "low_24h": float(data.get("lowPrice", 0.0)),
                "volume_24h": float(data.get("volume", 0.0)),
                "quote_volume_24h": float(data.get("quoteVolume", 0.0)),
                "source": "binance",
                "fetched_at": datetime.now(UTC).isoformat(),
            }
    except Exception:
        pass

    base = _crypto_base_asset(symbol)
    coin_id = COINGECKO_IDS.get(base)
    if not coin_id:
        raise DataFetchError(f"Prix indisponible pour {symbol} (Binance et CoinGecko).")

    data = await _get_json(
        f"{COINGECKO_BASE}/simple/price",
        params={
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_vol": "true",
        },
    )
    entry = data.get(coin_id, {})
    if not entry or "usd" not in entry:
        raise DataFetchError(f"Prix indisponible pour {symbol}.")

    return {
        "symbol": normalize_symbol(symbol, "crypto"),
        "market_type": "crypto",
        "price": float(entry["usd"]),
        "change_pct_24h": float(entry.get("usd_24h_change", 0.0)),
        "change_24h": None,
        "high_24h": None,
        "low_24h": None,
        "volume_24h": None,
        "quote_volume_24h": float(entry.get("usd_24h_vol", 0.0)),
        "source": "coingecko",
        "fetched_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Actions US : Alpaca (optionnel) puis yfinance
# ---------------------------------------------------------------------------
def _alpaca_headers() -> dict | None:
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not secret:
        return None
    return {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}


async def fetch_stock_ohlcv_alpaca(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 300,
) -> pd.DataFrame:
    """OHLCV via Alpaca Market Data (IEX gratuit) si les clés sont présentes."""
    headers = _alpaca_headers()
    if headers is None:
        raise DataFetchError("Clés Alpaca absentes.")

    tf = _normalize_timeframe(timeframe)
    alpaca_tf = ALPACA_TIMEFRAMES.get(tf, "1Day")
    params = {
        "timeframe": alpaca_tf,
        "limit": max(1, min(int(limit), 10000)),
        "adjustment": "all",
        "feed": "iex",
        "sort": "asc",
    }
    data = await _get_json(
        f"{ALPACA_DATA_BASE}/stocks/{urllib.parse.quote(symbol.upper())}/bars",
        params=params,
        headers=headers,
    )
    bars = (data or {}).get("bars") or []
    if not bars:
        raise DataFetchError(f"Alpaca n'a retourné aucune bougie pour {symbol}.")

    df = pd.DataFrame(
        [
            {
                "Open": bar["o"],
                "High": bar["h"],
                "Low": bar["l"],
                "Close": bar["c"],
                "Volume": bar["v"],
                "timestamp": pd.to_datetime(bar["t"], utc=True),
            }
            for bar in bars
        ]
    ).set_index("timestamp")
    return _normalize_ohlcv(df)


async def fetch_stock_ohlcv_yfinance(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 300,
) -> pd.DataFrame:
    """OHLCV via yfinance (thread) avec repli Alpaca en cas d'échec."""
    tf = _normalize_timeframe(timeframe)
    interval = YFINANCE_INTERVALS.get(tf, "1d")
    period = YFINANCE_PERIODS.get(interval, "2y")

    def _download() -> pd.DataFrame:
        import yfinance as yf

        ticker = yf.Ticker(symbol.upper())
        return ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)

    try:
        df = await asyncio.to_thread(_download)
        if df is not None and not df.empty:
            normalized = _normalize_ohlcv(df)
            if tf == "4h" and not normalized.empty:
                aggregated = normalized.resample("4h").agg(
                    {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                ).dropna(subset=["Close"])
                if not aggregated.empty:
                    normalized = aggregated
            return normalized.tail(limit)
    except Exception:
        pass

    try:
        return await fetch_stock_ohlcv_alpaca(symbol, timeframe, limit)
    except Exception as exc:
        raise DataFetchError(f"Impossible de récupérer les bougies de {symbol} : {exc}") from exc


async def fetch_stock_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 300,
) -> pd.DataFrame:
    """OHLCV actions : Alpaca en priorité si configuré, yfinance sinon."""
    if _alpaca_headers() is not None:
        try:
            df = await fetch_stock_ohlcv_alpaca(symbol, timeframe, limit)
            if not df.empty:
                return df.tail(limit)
        except Exception:
            pass
    return await fetch_stock_ohlcv_yfinance(symbol, timeframe, limit)


async def fetch_stock_quote(symbol: str) -> dict:
    """Prix temps réel (ou dernière clôture) d'une action US via yfinance."""
    def _quote() -> dict:
        import yfinance as yf

        ticker = yf.Ticker(symbol.upper())
        price = None
        previous_close = None
        try:
            fast_info = ticker.fast_info
            price = _to_float(getattr(fast_info, "last_price", None))
            previous_close = _to_float(getattr(fast_info, "previous_close", None))
        except Exception:
            pass

        if price is None:
            history = ticker.history(period="5d", interval="1d", auto_adjust=True)
            if history is not None and not history.empty:
                price = _to_float(history["Close"].iloc[-1])
                if len(history) > 1:
                    previous_close = _to_float(history["Close"].iloc[-2])

        if price is None:
            raise DataFetchError(f"Prix indisponible pour {symbol}.")

        change = (price - previous_close) if previous_close else None
        change_pct = (change / previous_close * 100) if previous_close else None
        return {
            "symbol": symbol.upper(),
            "market_type": "stock",
            "price": price,
            "change_24h": change,
            "change_pct_24h": change_pct,
            "previous_close": previous_close,
            "high_24h": None,
            "low_24h": None,
            "volume_24h": None,
            "quote_volume_24h": None,
            "source": "yfinance",
            "fetched_at": datetime.now(UTC).isoformat(),
        }

    return await asyncio.to_thread(_quote)


async def fetch_stock_fundamentals(symbol: str) -> dict:
    """Fondamentaux essentiels : P/E, Market Cap, secteur, 52w high/low."""
    def _fundamentals() -> dict:
        import yfinance as yf

        ticker = yf.Ticker(symbol.upper())
        info: dict = {}
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        return {
            "symbol": symbol.upper(),
            "market_type": "stock",
            "name": info.get("longName") or info.get("shortName") or symbol.upper(),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange"),
            "market_cap": _to_float(info.get("marketCap")),
            "pe_ratio": _to_float(info.get("trailingPE")),
            "forward_pe": _to_float(info.get("forwardPE")),
            "peg_ratio": _to_float(info.get("trailingPegRatio")),
            "price_to_book": _to_float(info.get("priceToBook")),
            "dividend_yield": _to_float(info.get("dividendYield")),
            "beta": _to_float(info.get("beta")),
            "eps": _to_float(info.get("trailingEps")),
            "revenue": _to_float(info.get("totalRevenue")),
            "profit_margin": _to_float(info.get("profitMargins")),
            "fifty_two_week_high": _to_float(info.get("fiftyTwoWeekHigh")),
            "fifty_two_week_low": _to_float(info.get("fiftyTwoWeekLow")),
            "fifty_day_average": _to_float(info.get("fiftyDayAverage")),
            "two_hundred_day_average": _to_float(info.get("twoHundredDayAverage")),
            "recommendation": info.get("recommendationKey"),
            "target_mean_price": _to_float(info.get("targetMeanPrice")),
            "fetched_at": datetime.now(UTC).isoformat(),
        }

    return await asyncio.to_thread(_fundamentals)


def _safe_number(value: Any) -> float | None:
    """Convertit une valeur pandas/numpy en float natif, sinon None."""
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


async def fetch_piotroski_score(symbol: str) -> dict:
    """Calcule le Piotroski F-Score (0-9) d'une action US via yfinance.

    Les 9 critères couvrent la rentabilité, le levier/liquidité et l'efficacité
    opérationnelle. Aucun appel LLM : pur calcul déterministe.
    """
    def _compute() -> dict:
        import pandas as pd
        import yfinance as yf

        result: dict[str, Any] = {
            "symbol": symbol.upper(),
            "score": 0,
            "max_score": 9,
            "criteria": {},
            "fetched_at": datetime.now(UTC).isoformat(),
        }

        try:
            ticker = yf.Ticker(symbol.upper())
            quarterly_financials = ticker.quarterly_financials
            quarterly_cashflow = ticker.quarterly_cashflow
            annual_financials = ticker.financials
            annual_balance = ticker.balance_sheet
        except Exception as exc:
            result["error"] = f"Données financières indisponibles : {exc}"
            return result

        def is_empty(frame: Any) -> bool:
            return frame is None or not isinstance(frame, pd.DataFrame) or frame.empty

        if is_empty(quarterly_financials) or is_empty(quarterly_cashflow):
            result["error"] = "Données trimestrielles insuffisantes."
            return result
        if is_empty(annual_financials) or is_empty(annual_balance):
            result["error"] = "Données annuelles insuffisantes."
            return result

        def pick(frame: pd.DataFrame, label: str, column: int = 0) -> float | None:
            try:
                if label not in frame.index or frame.shape[1] <= column:
                    return None
                return _safe_number(frame.loc[label].iloc[column])
            except Exception:
                return None

        recent_ni = None
        for column in range(min(4, quarterly_financials.shape[1])):
            candidate = pick(quarterly_financials, "Net Income", column)
            if candidate is not None:
                recent_ni = candidate if recent_ni is None else recent_ni + candidate
        recent_ocf = None
        for column in range(min(4, quarterly_cashflow.shape[1])):
            candidate = pick(quarterly_cashflow, "Operating Cash Flow", column)
            if candidate is not None:
                recent_ocf = candidate if recent_ocf is None else recent_ocf + candidate

        def annual(label: str, column: int, frame: pd.DataFrame) -> float | None:
            return pick(frame, label, column)

        recent_assets = annual("Total Assets", 0, annual_balance)
        prev_assets = annual("Total Assets", 1, annual_balance)
        recent_current_assets = annual("Current Assets", 0, annual_balance)
        prev_current_assets = annual("Current Assets", 1, annual_balance)
        recent_current_liab = annual("Current Liabilities", 0, annual_balance)
        prev_current_liab = annual("Current Liabilities", 1, annual_balance)
        recent_lt_debt = annual("Long Term Debt", 0, annual_balance)
        prev_lt_debt = annual("Long Term Debt", 1, annual_balance)
        recent_shares = annual("Share Issued", 0, annual_balance)
        prev_shares = annual("Share Issued", 1, annual_balance)

        recent_revenue = annual("Total Revenue", 0, annual_financials)
        prev_revenue = annual("Total Revenue", 1, annual_financials)
        recent_gross = annual("Gross Profit", 0, annual_financials)
        prev_gross = annual("Gross Profit", 1, annual_financials)
        recent_ni_annual = annual("Net Income", 0, annual_financials)
        prev_ni_annual = annual("Net Income", 1, annual_financials)

        criteria: dict[str, dict] = {}
        score = 0

        def add(name: str, passed: bool | None, detail: str) -> None:
            nonlocal score
            counted = bool(passed)
            if counted:
                score += 1
            criteria[name] = {"passed": counted if passed is not None else None, "detail": detail}

        roa_recent = (recent_ni / recent_assets) if recent_ni is not None and recent_assets else None
        roa_prev = (
            (prev_ni_annual / prev_assets)
            if prev_ni_annual is not None and prev_assets
            else None
        )

        add(
            "roa_positive",
            None if roa_recent is None else roa_recent > 0,
            f"ROA={roa_recent:.4f}" if roa_recent is not None else "ROA indisponible",
        )
        add(
            "cfo_positive",
            None if recent_ocf is None else recent_ocf > 0,
            f"CFO={recent_ocf:.0f}" if recent_ocf is not None else "CFO indisponible",
        )
        add(
            "roa_improving",
            None if roa_recent is None or roa_prev is None else roa_recent > roa_prev,
            f"ROA {roa_prev} -> {roa_recent}" if roa_recent is not None and roa_prev is not None else "Indisponible",
        )
        add(
            "cfo_exceeds_net_income",
            None if recent_ocf is None or recent_ni is None else recent_ocf > recent_ni,
            f"CFO={recent_ocf} vs NI={recent_ni}" if recent_ocf is not None and recent_ni is not None else "Indisponible",
        )

        def leverage(debt: float | None, assets: float | None) -> float | None:
            return (debt / assets) if debt is not None and assets else None

        recent_lev = leverage(recent_lt_debt, recent_assets)
        prev_lev = leverage(prev_lt_debt, prev_assets)
        add(
            "leverage_decreasing",
            None if recent_lev is None or prev_lev is None else recent_lev < prev_lev,
            f"Levier {prev_lev} -> {recent_lev}" if recent_lev is not None and prev_lev is not None else "Indisponible",
        )

        def current_ratio(assets: float | None, liab: float | None) -> float | None:
            return (assets / liab) if assets is not None and liab not in (None, 0) else None

        recent_cr = current_ratio(recent_current_assets, recent_current_liab)
        prev_cr = current_ratio(prev_current_assets, prev_current_liab)
        add(
            "current_ratio_improving",
            None if recent_cr is None or prev_cr is None else recent_cr > prev_cr,
            f"Current ratio {prev_cr} -> {recent_cr}" if recent_cr is not None and prev_cr is not None else "Indisponible",
        )
        add(
            "no_dilution",
            None if recent_shares is None or prev_shares is None else recent_shares <= prev_shares,
            f"Actions {prev_shares} -> {recent_shares}" if recent_shares is not None and prev_shares is not None else "Indisponible",
        )

        recent_margin = (
            (recent_gross / recent_revenue)
            if recent_gross is not None and recent_revenue
            else None
        )
        prev_margin = (
            (prev_gross / prev_revenue) if prev_gross is not None and prev_revenue else None
        )
        add(
            "gross_margin_improving",
            None if recent_margin is None or prev_margin is None else recent_margin > prev_margin,
            f"Marge {prev_margin} -> {recent_margin}" if recent_margin is not None and prev_margin is not None else "Indisponible",
        )

        recent_turnover = (
            (recent_revenue / recent_assets)
            if recent_revenue is not None and recent_assets
            else None
        )
        prev_turnover = (
            (prev_revenue / prev_assets) if prev_revenue is not None and prev_assets else None
        )
        add(
            "asset_turnover_improving",
            None if recent_turnover is None or prev_turnover is None else recent_turnover > prev_turnover,
            f"Rotation {prev_turnover} -> {recent_turnover}" if recent_turnover is not None and prev_turnover is not None else "Indisponible",
        )

        result["score"] = score
        result["criteria"] = criteria
        return result

    return await asyncio.to_thread(_compute)


async def fetch_earnings_calendar(symbol: str) -> dict:
    """Retourne la prochaine date de publication des résultats d'une action US.

    Trois sources yfinance en cascade : `calendar`, `earnings_dates`, puis
    `info.earningsTimestamp`. Le timing (BMO/AMC) et l'EPS estimé sont inclus.
    """
    def _compute() -> dict:
        import pandas as pd
        import yfinance as yf

        result: dict[str, Any] = {
            "symbol": symbol.upper(),
            "earnings_date": None,
            "timing": None,
            "eps_estimate": None,
            "fetched_at": datetime.now(UTC).isoformat(),
        }

        try:
            ticker = yf.Ticker(symbol.upper())
            today = datetime.now(UTC).date()

            calendar = getattr(ticker, "calendar", None)
            if isinstance(calendar, dict):
                raw_dates = calendar.get("Earnings Date") or []
                if not isinstance(raw_dates, (list, tuple)):
                    raw_dates = [raw_dates]
                upcoming = []
                for raw in raw_dates:
                    try:
                        parsed = datetime.fromisoformat(str(raw)[:10]).date()
                    except ValueError:
                        continue
                    if parsed >= today:
                        upcoming.append(parsed)
                if upcoming:
                    result["earnings_date"] = str(min(upcoming))
                    eps = calendar.get("Earnings Average")
                    result["eps_estimate"] = _safe_number(eps)
                    return result

            try:
                earnings_df = ticker.earnings_dates
                if earnings_df is not None and not earnings_df.empty:
                    future = [
                        index.date() if hasattr(index, "date") else index
                        for index in earnings_df.index
                        if (index.date() if hasattr(index, "date") else index) >= today
                    ]
                    if future:
                        next_date = min(future)
                        result["earnings_date"] = str(next_date)
                        try:
                            row = earnings_df.loc[earnings_df.index[-1]]
                            result["eps_estimate"] = _safe_number(row.get("EPS Estimate"))
                        except Exception:
                            result["eps_estimate"] = None
                        if hasattr(next_date, "hour"):
                            hour = next_date.hour
                            result["timing"] = "BMO" if hour < 12 else ("AMC" if hour >= 16 else None)
                        return result
            except Exception:
                pass

            try:
                info = ticker.info or {}
                timestamp = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
                if timestamp:
                    earnings_date = datetime.fromtimestamp(int(timestamp), UTC).date()
                    if earnings_date >= today:
                        result["earnings_date"] = str(earnings_date)
                        result["timing"] = "AMC"
                        result["eps_estimate"] = _safe_number(info.get("epsForward"))
            except Exception:
                pass
        except Exception as exc:
            result["error"] = f"Calendrier earnings indisponible : {exc}"

        return result

    return await asyncio.to_thread(_compute)


async def fetch_option_chain(
    symbol: str,
    expiration: str | None = None,
    max_contracts: int = 40,
) -> dict:
    """Chaîne d'options de la prochaine échéance disponible via yfinance."""
    def _chain() -> dict:
        import yfinance as yf

        ticker = yf.Ticker(symbol.upper())
        try:
            expirations = list(ticker.options or [])
        except Exception as exc:
            raise DataFetchError(f"Chaîne d'options indisponible pour {symbol} : {exc}") from exc

        if not expirations:
            raise DataFetchError(f"Aucune échéance d'options pour {symbol}.")

        chosen = expiration
        if chosen not in expirations:
            today = datetime.now(UTC).date()
            future = [
                expiry
                for expiry in expirations
                if datetime.strptime(expiry, "%Y-%m-%d").date() >= today
            ]
            chosen = future[0] if future else expirations[-1]

        chain = ticker.option_chain(chosen)

        spot = None
        try:
            fast_info = ticker.fast_info
            spot = _to_float(getattr(fast_info, "last_price", None))
        except Exception:
            spot = None

        def _parse(frame: pd.DataFrame) -> list[dict]:
            if frame is None or frame.empty:
                return []
            working = frame.copy()
            working["strike"] = pd.to_numeric(working["strike"], errors="coerce")
            working = working.dropna(subset=["strike"])
            if spot:
                around_spot = working[
                    (working["strike"] >= spot * 0.6) & (working["strike"] <= spot * 1.4)
                ]
                if not around_spot.empty:
                    working = around_spot
                working = working.iloc[
                    (working["strike"] - spot).abs().argsort()[:max_contracts]
                ].sort_values("strike")
            else:
                working = working.head(max_contracts)

            contracts = []
            for _, row in working.iterrows():
                in_money = _to_float(row.get("inTheMoney"))
                contracts.append(
                    {
                        "contract_symbol": row.get("contractSymbol"),
                        "strike": _to_float(row.get("strike")),
                        "last_price": _to_float(row.get("lastPrice")),
                        "bid": _to_float(row.get("bid")),
                        "ask": _to_float(row.get("ask")),
                        "change": _to_float(row.get("change")),
                        "percent_change": _to_float(row.get("percentChange")),
                        "volume": _to_float(row.get("volume")),
                        "open_interest": _to_float(row.get("openInterest")),
                        "implied_volatility": _to_float(row.get("impliedVolatility")),
                        "in_the_money": bool(in_money) if in_money is not None else None,
                    }
                )
            return contracts

        return {
            "symbol": symbol.upper(),
            "market_type": "stock",
            "expiration": chosen,
            "expirations": expirations,
            "spot": spot,
            "calls": _parse(chain.calls),
            "puts": _parse(chain.puts),
            "fetched_at": datetime.now(UTC).isoformat(),
        }

    return await asyncio.to_thread(_chain)


# ---------------------------------------------------------------------------
# Actualités
# ---------------------------------------------------------------------------
def _parse_yfinance_news(raw_items: list[dict], symbol: str, limit: int) -> list[dict]:
    articles: list[dict] = []
    for item in raw_items or []:
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = content.get("title") or item.get("title")
        if not title:
            continue
        provider = content.get("provider") or {}
        canonical = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        if not isinstance(canonical, dict):
            canonical = {}
        articles.append(
            {
                "title": title.strip(),
                "summary": (content.get("summary") or content.get("description") or "").strip(),
                "url": canonical.get("url") or item.get("link"),
                "source": provider.get("displayName") or item.get("publisher") or "Yahoo Finance",
                "published_at": content.get("pubDate") or item.get("providerPublishTime"),
                "provider": "yfinance",
                "symbol": symbol.upper(),
            }
        )
        if len(articles) >= limit:
            break
    return articles


async def fetch_news_yfinance(symbol: str, limit: int = 8) -> list[dict]:
    def _news() -> list[dict]:
        import yfinance as yf

        ticker = yf.Ticker(symbol.upper())
        try:
            return _parse_yfinance_news(ticker.news or [], symbol, limit)
        except Exception:
            return []

    return await asyncio.to_thread(_news)


async def fetch_news_google(symbol: str, limit: int = 8) -> list[dict]:
    query = f"{symbol.upper()} stock" if detect_market_type(symbol) == "stock" else f"{symbol.upper()} crypto"
    try:
        text = await _get_text(
            GOOGLE_NEWS_RSS,
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        )
        root = ET.fromstring(text)
    except Exception:
        return []

    articles: list[dict] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        articles.append(
            {
                "title": title,
                "summary": (item.findtext("description") or "").strip(),
                "url": item.findtext("link"),
                "source": item.findtext("source") or "Google News",
                "published_at": item.findtext("pubDate"),
                "provider": "google_news",
                "symbol": symbol.upper(),
            }
        )
        if len(articles) >= limit:
            break
    return articles


async def fetch_news_finnhub(symbol: str, limit: int = 8) -> list[dict]:
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key or detect_market_type(symbol) != "stock":
        return []

    today = datetime.now(UTC).date()
    start = today - timedelta(days=7)
    try:
        data = await _get_json(
            f"{FINNHUB_BASE}/company-news",
            params={
                "symbol": symbol.upper(),
                "from": start.isoformat(),
                "to": today.isoformat(),
                "token": api_key,
            },
        )
    except Exception:
        return []

    articles: list[dict] = []
    for item in (data or [])[:limit]:
        title = (item.get("headline") or "").strip()
        if not title:
            continue
        published = item.get("datetime")
        articles.append(
            {
                "title": title,
                "summary": (item.get("summary") or "").strip(),
                "url": item.get("url"),
                "source": item.get("source") or "Finnhub",
                "published_at": (
                    datetime.fromtimestamp(published, tz=UTC).isoformat()
                    if isinstance(published, (int, float))
                    else published
                ),
                "provider": "finnhub",
                "symbol": symbol.upper(),
            }
        )
    return articles


async def fetch_news(symbol: str, market_type: str | None = None, limit: int = 10) -> list[dict]:
    """Agrège yfinance + Google News RSS + Finnhub, sans doublons."""
    market = market_type or detect_market_type(symbol)
    results = await asyncio.gather(
        fetch_news_yfinance(symbol, limit),
        fetch_news_google(symbol, limit),
        fetch_news_finnhub(symbol, limit) if market == "stock" else asyncio.sleep(0, result=[]),
        return_exceptions=True,
    )

    merged: list[dict] = []
    seen: set[str] = set()
    for group in results:
        if isinstance(group, Exception) or not group:
            continue
        for article in group:
            key = re.sub(r"\s+", " ", (article.get("title") or "").lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(article)
            if len(merged) >= limit:
                return merged
    return merged


# ---------------------------------------------------------------------------
# Point d'entrée unifié
# ---------------------------------------------------------------------------
async def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 300,
    market_type: str | None = None,
) -> pd.DataFrame:
    """Dispatcher OHLCV selon le type de marché détecté."""
    market = market_type or detect_market_type(symbol)
    if market == "crypto":
        return await fetch_crypto_ohlcv(symbol, timeframe, limit)
    return await fetch_stock_ohlcv(symbol, timeframe, limit)


def fetch_ohlcv_sync(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 300,
    market_type: str | None = None,
) -> pd.DataFrame:
    """Version synchrone de fetch_ohlcv (utilisée par le screener)."""
    return _run_coroutine_sync(fetch_ohlcv(symbol, timeframe, limit, market_type))


async def fetch_market_snapshot(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 300,
    include_news: bool = True,
    include_options: bool = True,
) -> dict:
    """Agrège OHLCV, prix, fondamentaux/options et actualités en un seul appel."""
    market = detect_market_type(symbol)
    normalized = normalize_symbol(symbol, market)

    tasks: dict[str, Any] = {
        "ohlcv": fetch_ohlcv(normalized, timeframe, limit, market),
        "price": (
            fetch_crypto_price(normalized) if market == "crypto" else fetch_stock_quote(normalized)
        ),
    }
    if include_news:
        tasks["news"] = fetch_news(normalized, market)
    if market == "stock":
        tasks["fundamentals"] = fetch_stock_fundamentals(normalized)
        if include_options:
            tasks["options"] = fetch_option_chain(normalized)

    keys = list(tasks)
    values = await asyncio.gather(*tasks.values(), return_exceptions=True)

    snapshot: dict[str, Any] = {
        "symbol": normalized,
        "requested_symbol": symbol.upper(),
        "market_type": market,
        "timeframe": _normalize_timeframe(timeframe),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    for key, value in zip(keys, values):
        if isinstance(value, Exception):
            snapshot[key] = None
            snapshot.setdefault("errors", {})[key] = str(value)
        else:
            snapshot[key] = value

    ohlcv = snapshot.get("ohlcv")
    if isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:
        snapshot["last_close"] = float(ohlcv["Close"].iloc[-1])
        snapshot["candles_count"] = len(ohlcv)
    else:
        snapshot["last_close"] = None
        snapshot["candles_count"] = 0

    return snapshot
