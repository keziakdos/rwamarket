"""Boîte à outils quantitative 100 % déterministe de RwaMarket.

Aucun appel LLM : tous les indicateurs sont calculés localement avec Pandas,
Numpy et SciPy. Inspiré de `staskh/trading_skills` (indicateurs, greeks) et du
screener de `prism-insight` (filtre mathématique en amont des agents).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import norm

from app.engine.data_fetcher import (
    DataFetchError,
    detect_market_type,
    fetch_ohlcv_sync,
    normalize_symbol,
)

TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_OPTION_VOLATILITY = 0.30


# ---------------------------------------------------------------------------
# Helpers numériques
# ---------------------------------------------------------------------------
def _finite(value: Any, digits: int | None = 6) -> float | None:
    """Convertit une valeur numérique en float natif JSON-safe."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def _series_last(series: pd.Series | None, digits: int | None = 6) -> float | None:
    if series is None or len(series) == 0:
        return None
    return _finite(series.iloc[-1], digits)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    return rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"line": line, "signal": signal_line, "hist": line - signal_line})


def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle.replace(0.0, np.nan) * 100.0
    percent_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {"middle": middle, "upper": upper, "lower": lower, "bandwidth": bandwidth, "percent_b": percent_b}
    )


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> dict[str, pd.Series]:
    """Calcule l'ADX de Wilder avec les DI+ et DI-.

    Portage déterministe de `trading_skills.technicals.compute_raw_indicators`
    (ADX 14) sans dépendance à pandas-ta.
    """
    high, low, close = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_dm_smooth / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm_smooth / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_series = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return {"adx": adx_series, "plus_di": plus_di, "minus_di": minus_di}


def _value_at_risk(close: pd.Series, confidence: float = 0.95) -> float | None:
    """VaR historique 1 jour en % de la valeur du portefeuille (perte positive)."""
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return None
    lower_tail = np.quantile(returns.to_numpy(), 1.0 - confidence)
    return _finite(-float(lower_tail) * 100.0, 4)


def _expected_shortfall(close: pd.Series, confidence: float = 0.95) -> float | None:
    """Expected Shortfall (CVaR) historique 1 jour en %, perte positive."""
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return None
    threshold = np.quantile(returns.to_numpy(), 1.0 - confidence)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return None
    return _finite(-float(tail.mean()) * 100.0, 4)


def _detect_crossover(fast: pd.Series, slow: pd.Series) -> dict | None:
    """Détecte le dernier croisement fast/slow et son ancienneté en bougies."""
    diff = (fast - slow).dropna()
    if len(diff) < 2:
        return None
    values = diff.to_numpy()
    for i in range(len(values) - 1, 0, -1):
        current, previous = values[i], values[i - 1]
        if current > 0 >= previous:
            return {"direction": "up", "days_ago": len(values) - 1 - i}
        if current < 0 <= previous:
            return {"direction": "down", "days_ago": len(values) - 1 - i}
    return None


def _cluster_levels(levels: Iterable[float], tolerance_pct: float = 0.75) -> list[dict]:
    """Regroupe des prix proches en niveaux clés avec leur nombre de touches."""
    cleaned = sorted(float(level) for level in levels if level and math.isfinite(float(level)) and level > 0)
    if not cleaned:
        return []

    tolerance = tolerance_pct / 100.0
    clusters: list[list[float]] = [[cleaned[0]]]
    for level in cleaned[1:]:
        reference = float(np.mean(clusters[-1]))
        if abs(level - reference) / reference <= tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    return [
        {"price": _finite(np.mean(cluster), 6), "touches": len(cluster)}
        for cluster in clusters
        if cluster
    ]


def _support_resistance(df: pd.DataFrame, current_price: float, lookback: int = 180) -> dict:
    window = df.tail(lookback)
    if window.empty:
        return {
            "supports": [],
            "resistances": [],
            "nearest_support": None,
            "nearest_resistance": None,
        }

    order = max(2, min(5, len(window) // 25))
    highs = window["High"].to_numpy(dtype=float)
    lows = window["Low"].to_numpy(dtype=float)

    try:
        pivot_high_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        pivot_low_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    except Exception:
        pivot_high_idx = np.array([int(np.argmax(highs))])
        pivot_low_idx = np.array([int(np.argmin(lows))])

    resistance_clusters = _cluster_levels(highs[pivot_high_idx].tolist())
    support_clusters = _cluster_levels(lows[pivot_low_idx].tolist())

    supports = sorted(
        [level for level in support_clusters if level["price"] < current_price * 1.001],
        key=lambda item: item["price"],
        reverse=True,
    )[:4]
    resistances = sorted(
        [level for level in resistance_clusters if level["price"] > current_price * 0.999],
        key=lambda item: item["price"],
    )[:4]

    if not supports:
        fallback = _finite(window["Low"].min(), 6)
        if fallback is not None:
            supports = [{"price": fallback, "touches": 1}]
    if not resistances:
        fallback = _finite(window["High"].max(), 6)
        if fallback is not None:
            resistances = [{"price": fallback, "touches": 1}]

    return {
        "supports": supports,
        "resistances": resistances,
        "nearest_support": supports[0]["price"] if supports else None,
        "nearest_resistance": resistances[0]["price"] if resistances else None,
    }


def _max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return 0.0
    equity = close.astype(float)
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0.0, np.nan)
    return float(drawdown.min() * 100.0) if drawdown.notna().any() else 0.0


def _serialize_candles(df: pd.DataFrame, limit: int) -> list[dict]:
    candles: list[dict] = []
    for timestamp, row in df.tail(limit).iterrows():
        candles.append(
            {
                "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                "open": _finite(row["Open"], 8),
                "high": _finite(row["High"], 8),
                "low": _finite(row["Low"], 8),
                "close": _finite(row["Close"], 8),
                "volume": _finite(row["Volume"], 4),
            }
        )
    return candles


# ---------------------------------------------------------------------------
# Analyse technique complète
# ---------------------------------------------------------------------------
def compute_technicals(
    df: pd.DataFrame,
    include_candles: bool = True,
    candle_limit: int = 120,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Calcule l'ensemble des indicateurs déterministes sur un DataFrame OHLCV."""
    if df is None or df.empty:
        return {"error": "Aucune donnée OHLCV disponible."}

    required = {"High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        return {"error": f"Colonnes manquantes : {', '.join(sorted(missing))}"}

    data = df.copy()
    if "Volume" not in data.columns:
        data["Volume"] = 0.0
    data = data.dropna(subset=["High", "Low", "Close"])
    if len(data) < 30:
        return {"error": "Historique insuffisant (30 bougies minimum)."}

    close = data["Close"].astype(float)
    volume = data["Volume"].astype(float)
    current_price = float(close.iloc[-1])

    ema9_series = _ema(close, 9)
    ema20_series = _ema(close, 20)
    sma50_series = _sma(close, 50)
    sma200_series = _sma(close, 200)

    rsi_series = _rsi(close, 14)
    macd_frame = _macd(close)
    bollinger_frame = _bollinger(close)
    atr_series = _atr(data, 14)
    adx_frame = _adx(data, 14)

    ema9 = _series_last(ema9_series)
    ema20 = _series_last(ema20_series)
    sma50 = _series_last(sma50_series)
    sma200 = _series_last(sma200_series)
    rsi14 = _series_last(rsi_series, 2)

    macd_line = _series_last(macd_frame["line"], 6)
    macd_signal = _series_last(macd_frame["signal"], 6)
    macd_hist = _series_last(macd_frame["hist"], 6)
    prev_macd_hist = _series_last(macd_frame["hist"].shift(1), 6)

    bb_middle = _series_last(bollinger_frame["middle"])
    bb_upper = _series_last(bollinger_frame["upper"])
    bb_lower = _series_last(bollinger_frame["lower"])
    bb_bandwidth = _series_last(bollinger_frame["bandwidth"], 2)
    bb_percent_b = _series_last(bollinger_frame["percent_b"], 4)

    atr14 = _series_last(atr_series)
    atr_pct = (atr14 / current_price * 100.0) if atr14 and current_price else None

    adx14 = _series_last(adx_frame["adx"], 2)
    plus_di = _series_last(adx_frame["plus_di"], 2)
    minus_di = _series_last(adx_frame["minus_di"], 2)

    volume_last = _finite(volume.iloc[-1], 4)
    if len(volume) > 20:
        volume_avg20 = _finite(volume.iloc[-21:-1].mean(), 4)
    else:
        volume_avg20 = _finite(volume.mean(), 4)
    volume_ratio = (
        _finite(volume_last / volume_avg20, 4)
        if volume_last is not None and volume_avg20 not in (None, 0)
        else None
    )

    log_returns = np.log(close / close.shift(1)).dropna()
    volatility_annualized = (
        _finite(float(log_returns.tail(periods_per_year).std(ddof=1)) * math.sqrt(periods_per_year) * 100.0, 4)
        if len(log_returns) > 2
        else None
    )
    sharpe_ratio = None
    if len(log_returns) > 2:
        std = float(log_returns.tail(periods_per_year).std(ddof=1))
        if std > 0:
            sharpe_ratio = _finite(float(log_returns.tail(periods_per_year).mean()) / std * math.sqrt(periods_per_year), 4)

    var_95 = _value_at_risk(close, 0.95)
    var_99 = _value_at_risk(close, 0.99)
    cvar_95 = _expected_shortfall(close, 0.95)

    momentum = {}
    for label, periods in (("1d", 1), ("5d", 5), ("20d", 20), ("60d", 60)):
        if len(close) > periods:
            base = float(close.iloc[-1 - periods])
            momentum[label] = _finite((current_price / base - 1.0) * 100.0, 4) if base else None
        else:
            momentum[label] = None

    levels = _support_resistance(data, current_price)

    ema_crossover = _detect_crossover(ema9_series, ema20_series)
    macd_crossover = _detect_crossover(macd_frame["line"], macd_frame["signal"])

    signals: list[str] = []
    if ema9 is not None and ema20 is not None:
        if ema9 > ema20:
            signals.append("EMA9 au-dessus de l'EMA20 (tendance courte haussière)")
        else:
            signals.append("EMA9 sous l'EMA20 (tendance courte baissière)")
    if sma50 is not None and sma200 is not None:
        if sma50 > sma200:
            signals.append("SMA50 > SMA200 (structure haussière de fond)")
        else:
            signals.append("SMA50 < SMA200 (structure baissière de fond)")
    if rsi14 is not None:
        if rsi14 >= 70:
            signals.append(f"RSI14 en surachat ({rsi14:.1f})")
        elif rsi14 <= 30:
            signals.append(f"RSI14 en survente ({rsi14:.1f})")
    if macd_hist is not None and prev_macd_hist is not None:
        if macd_hist > prev_macd_hist:
            signals.append("Histogramme MACD en expansion")
        else:
            signals.append("Histogramme MACD en contraction")
    if volume_ratio is not None and volume_ratio >= 1.8:
        signals.append(f"Volume anormal ({volume_ratio:.2f}x la moyenne 20j)")
    if adx14 is not None and adx14 > 25 and plus_di is not None and minus_di is not None:
        direction = "haussière" if plus_di > minus_di else "baissière"
        signals.append(f"Tendance {direction} confirmée (ADX {adx14:.1f}, DI+ {plus_di:.1f} / DI- {minus_di:.1f})")
    if current_price >= (bb_upper or float("inf")):
        signals.append("Prix au-dessus de la bande de Bollinger supérieure")
    elif current_price <= (bb_lower or float("-inf")):
        signals.append("Prix sous la bande de Bollinger inférieure")

    bullish_points = 0
    bearish_points = 0
    if ema9 is not None and ema20 is not None:
        if ema9 > ema20:
            bullish_points += 1
        else:
            bearish_points += 1
    if sma50 is not None:
        if current_price > sma50:
            bullish_points += 1
        else:
            bearish_points += 1
    if sma200 is not None:
        if current_price > sma200:
            bullish_points += 1
        else:
            bearish_points += 1
    if macd_line is not None and macd_signal is not None:
        if macd_line > macd_signal:
            bullish_points += 1
        else:
            bearish_points += 1
    if rsi14 is not None:
        if rsi14 >= 50:
            bullish_points += 1
        else:
            bearish_points += 1

    if bullish_points - bearish_points >= 3:
        trend = "bullish"
    elif bearish_points - bullish_points >= 3:
        trend = "bearish"
    else:
        trend = "neutral"

    result: dict[str, Any] = {
        "price": _finite(current_price),
        "candles_used": len(data),
        "last_timestamp": (
            data.index[-1].isoformat() if hasattr(data.index[-1], "isoformat") else str(data.index[-1])
        ),
        "trend": trend,
        "trend_score": bullish_points - bearish_points,
        "momentum_pct": momentum,
        "ema": {
            "ema9": ema9,
            "ema20": ema20,
            "ema9_vs_ema20_pct": (
                _finite((ema9 / ema20 - 1.0) * 100.0, 4) if ema9 and ema20 else None
            ),
            "crossover": ema_crossover,
        },
        "ma": {
            "sma50": sma50,
            "sma200": sma200,
            "price_vs_sma50_pct": (
                _finite((current_price / sma50 - 1.0) * 100.0, 4) if sma50 else None
            ),
            "price_vs_sma200_pct": (
                _finite((current_price / sma200 - 1.0) * 100.0, 4) if sma200 else None
            ),
        },
        "rsi14": rsi14,
        "macd": {
            "line": macd_line,
            "signal": macd_signal,
            "hist": macd_hist,
            "hist_rising": (
                macd_hist > prev_macd_hist
                if macd_hist is not None and prev_macd_hist is not None
                else None
            ),
            "crossover": macd_crossover,
        },
        "bollinger": {
            "middle": bb_middle,
            "upper": bb_upper,
            "lower": bb_lower,
            "bandwidth_pct": bb_bandwidth,
            "percent_b": bb_percent_b,
        },
        "atr14": atr14,
        "atr_pct": _finite(atr_pct, 4),
        "adx": {
            "adx14": adx14,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "trend_strength": (
                "strong" if adx14 is not None and adx14 > 25 else ("weak" if adx14 is not None else None)
            ),
        },
        "volume": {
            "last": volume_last,
            "avg20": volume_avg20,
            "ratio": volume_ratio,
        },
        "support_resistance": levels,
        "volatility_annualized_pct": volatility_annualized,
        "sharpe_ratio": sharpe_ratio,
        "var": {
            "var_95_1d_pct": var_95,
            "var_99_1d_pct": var_99,
            "cvar_95_1d_pct": cvar_95,
        },
        "max_drawdown_pct": _finite(_max_drawdown(close), 4),
        "signals": signals,
    }

    if include_candles:
        result["candles"] = _serialize_candles(data, candle_limit)

    return result


def compute_correlation_matrix(
    symbols: Iterable[str],
    timeframe: str = "1d",
    limit: int = 90,
    market_type: str | None = None,
) -> dict:
    """Calcule la matrice de corrélation des rendements entre plusieurs actifs.

    Extrait de `trading_skills.correlation` : aligne les clôtures, convertit en
    rendements et renvoie une matrice imbriquée JSON-safe. Aucun appel LLM.
    """
    wanted = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if len(wanted) < 2:
        return {"error": "Au moins 2 symboles sont nécessaires pour la corrélation."}

    closes: dict[str, pd.Series] = {}
    for symbol in wanted:
        try:
            resolved_market = market_type or detect_market_type(symbol)
            normalized = normalize_symbol(symbol, resolved_market)
            frame = fetch_ohlcv_sync(
                normalized,
                timeframe=timeframe,
                limit=limit,
                market_type=resolved_market,
            )
            if frame is None or frame.empty or "Close" not in frame.columns:
                continue
            series = frame["Close"].astype(float)
            series.name = symbol
            closes[symbol] = series
        except Exception:
            continue

    if len(closes) < 2:
        return {"error": "Données insuffisantes : moins de 2 symboles valides."}

    price_frame = pd.DataFrame(closes).dropna(how="all")
    returns_frame = price_frame.pct_change().dropna(how="all")
    returns_frame = returns_frame.dropna(axis=1, how="all")
    if returns_frame.shape[1] < 2 or len(returns_frame) < 2:
        return {"error": "Historique commun insuffisant pour calculer la corrélation."}

    correlation = returns_frame.corr()
    matrix: dict[str, dict[str, float | None]] = {}
    for row_symbol in correlation.index:
        matrix[row_symbol] = {
            column_symbol: _finite(correlation.loc[row_symbol, column_symbol], 4)
            for column_symbol in correlation.columns
        }

    return {
        "symbols": list(correlation.columns),
        "timeframe": timeframe,
        "observations": int(len(returns_frame)),
        "correlation_matrix": matrix,
    }


def compute_indicators(
    df: pd.DataFrame,
    indicators: list[str] | None = None,
    include_candles: bool = True,
    candle_limit: int = 120,
) -> dict:
    """Alias compatible avec l'API historique (les indicateurs sont toujours complets)."""
    return compute_technicals(df, include_candles=include_candles, candle_limit=candle_limit)


# ---------------------------------------------------------------------------
# Black-Scholes et Grecques
# ---------------------------------------------------------------------------
def _bs_d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry
    ) / (volatility * sqrt_t)
    return d1, d1 - volatility * sqrt_t


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float:
    """Prix théorique Black-Scholes d'un call ou d'un put européen."""
    option_type = option_type.lower()
    if time_to_expiry <= 0 or volatility <= 0:
        intrinsic = spot - strike if option_type == "call" else strike - spot
        return max(0.0, intrinsic)

    d1, d2 = _bs_d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    discount_q = math.exp(-dividend_yield * time_to_expiry)
    discount_r = math.exp(-rate * time_to_expiry)
    if option_type == "call":
        return float(spot * discount_q * norm.cdf(d1) - strike * discount_r * norm.cdf(d2))
    return float(strike * discount_r * norm.cdf(-d2) - spot * discount_q * norm.cdf(-d1))


def _bs_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> dict:
    if time_to_expiry <= 0:
        return {"error": "Option expirée."}
    if volatility <= 0 or spot <= 0 or strike <= 0:
        return {"error": "Paramètres invalides (volatilité/spot/strike)."}

    option_type = option_type.lower()
    d1, d2 = _bs_d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    sqrt_t = math.sqrt(time_to_expiry)
    discount_q = math.exp(-dividend_yield * time_to_expiry)
    discount_r = math.exp(-rate * time_to_expiry)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    if option_type == "call":
        delta = discount_q * cdf_d1
        theta = (
            -spot * discount_q * pdf_d1 * volatility / (2 * sqrt_t)
            - rate * strike * discount_r * cdf_d2
            + dividend_yield * spot * discount_q * cdf_d1
        ) / 365.0
        rho = strike * time_to_expiry * discount_r * cdf_d2 / 100.0
        price = spot * discount_q * cdf_d1 - strike * discount_r * cdf_d2
    else:
        delta = discount_q * (cdf_d1 - 1.0)
        theta = (
            -spot * discount_q * pdf_d1 * volatility / (2 * sqrt_t)
            + rate * strike * discount_r * norm.cdf(-d2)
            - dividend_yield * spot * discount_q * norm.cdf(-d1)
        ) / 365.0
        rho = -strike * time_to_expiry * discount_r * norm.cdf(-d2) / 100.0
        price = strike * discount_r * norm.cdf(-d2) - spot * discount_q * norm.cdf(-d1)

    gamma = discount_q * pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * discount_q * pdf_d1 * sqrt_t / 100.0

    return {
        "price": round(float(price), 4),
        "delta": round(float(delta), 4),
        "gamma": round(float(gamma), 6),
        "theta": round(float(theta), 4),
        "vega": round(float(vega), 4),
        "rho": round(float(rho), 4),
    }


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str,
    dividend_yield: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float | None:
    """Volatilité implicite par Newton-Raphson avec repli en bissection."""
    if market_price <= 0 or time_to_expiry <= 0 or spot <= 0 or strike <= 0:
        return None

    volatility = 0.30
    for _ in range(max_iterations):
        greeks = _bs_greeks(spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield)
        if "error" in greeks:
            return None
        price = greeks["price"]
        vega = greeks["vega"] * 100.0
        diff = price - market_price
        if abs(diff) < tolerance:
            return volatility
        if vega < 1e-8:
            break
        volatility -= diff / vega
        if not math.isfinite(volatility) or volatility <= 0:
            break
        volatility = min(volatility, 5.0)

    low, high = 1e-4, 5.0
    for _ in range(200):
        mid = (low + high) / 2.0
        price = black_scholes_price(spot, strike, time_to_expiry, rate, mid, option_type, dividend_yield)
        if abs(price - market_price) < tolerance:
            return mid
        if price > market_price:
            high = mid
        else:
            low = mid
    return None


def calculate_greeks(
    spot: float,
    strike: float,
    option_type: str,
    expiry: str | None = None,
    dte: int | None = None,
    as_of_date: str | None = None,
    market_price: float | None = None,
    rate: float = DEFAULT_RISK_FREE_RATE,
    volatility: float | None = None,
    dividend_yield: float = 0.0,
) -> dict:
    """Calcule les grecques d'une option (Delta, Gamma, Theta, Vega, Rho)."""
    option_type = (option_type or "").lower()
    if option_type not in {"call", "put"}:
        return {"error": "option_type doit être 'call' ou 'put'."}

    if dte is not None:
        days_to_expiry = int(dte)
        expiry_label = f"{days_to_expiry} DTE"
    elif expiry is not None:
        try:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            reference = (
                datetime.strptime(as_of_date, "%Y-%m-%d").date()
                if as_of_date
                else datetime.now(UTC).date()
            )
        except ValueError:
            return {"error": "Format de date invalide (YYYY-MM-DD attendu)."}
        days_to_expiry = (expiry_date - reference).days
        expiry_label = expiry
    else:
        return {"error": "Fournir 'expiry' ou 'dte'."}

    if days_to_expiry <= 0:
        return {"error": "Option expirée ou expirant aujourd'hui."}
    if spot <= 0 or strike <= 0:
        return {"error": "Spot et strike doivent être positifs."}

    time_to_expiry = days_to_expiry / 365.0
    iv = None
    if market_price is not None and market_price > 0:
        iv = implied_volatility(
            market_price, spot, strike, time_to_expiry, rate, option_type, dividend_yield
        )
        if iv is None:
            iv = volatility or DEFAULT_OPTION_VOLATILITY
    elif volatility is not None:
        iv = float(volatility)
    else:
        iv = DEFAULT_OPTION_VOLATILITY

    if iv > 3.0:
        iv = iv / 100.0

    greeks = _bs_greeks(spot, strike, time_to_expiry, rate, iv, option_type, dividend_yield)
    if "error" in greeks:
        return greeks

    breakeven = strike + greeks["price"] if option_type == "call" else strike - greeks["price"]
    intrinsic = max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)

    return {
        "spot": round(float(spot), 4),
        "strike": round(float(strike), 4),
        "expiry": expiry_label,
        "days_to_expiry": days_to_expiry,
        "option_type": option_type,
        "market_price": round(float(market_price), 4) if market_price else None,
        "theoretical_price": greeks["price"],
        "intrinsic_value": round(intrinsic, 4),
        "time_value": round(max(greeks["price"] - intrinsic, 0.0), 4),
        "iv_pct": round(iv * 100.0, 2),
        "risk_free_rate": rate,
        "dividend_yield": dividend_yield,
        "breakeven": round(breakeven, 4),
        "greeks": {
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "theta": greeks["theta"],
            "vega": greeks["vega"],
            "rho": greeks["rho"],
        },
    }


def preview_option_setup(
    spot_price: float,
    option_chain: dict,
    outlook: str = "bullish",
    risk_profile: str = "moderate",
    target_price: float | None = None,
    dte: int | None = None,
    rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict:
    """Sélectionne le contrat d'option le plus adapté au scénario fourni.

    Prêt pour le futur module d'options US : accepte la sortie de
    `data_fetcher.fetch_option_chain` et renvoie un plan complet (coût,
    break-even, perte max, profit cible, R:R et grecques).
    """
    if not option_chain or not isinstance(option_chain, dict):
        return {"error": "Chaîne d'options invalide."}

    outlook = outlook.lower()
    if outlook not in {"bullish", "bearish"}:
        return {"error": "outlook doit être 'bullish' ou 'bearish'."}

    risk_profile = risk_profile.lower()
    target_delta = {
        "conservative": 0.65,
        "moderate": 0.50,
        "aggressive": 0.32,
    }.get(risk_profile, 0.50)

    option_type = "call" if outlook == "bullish" else "put"
    contracts = option_chain.get("calls" if option_type == "call" else "puts") or []
    if not contracts:
        return {"error": f"Aucun contrat {option_type} disponible."}

    if dte is None:
        expiration = option_chain.get("expiration")
        if expiration:
            try:
                dte = max(
                    (datetime.strptime(expiration, "%Y-%m-%d").date() - datetime.now(UTC).date()).days,
                    1,
                )
            except ValueError:
                dte = 30
        else:
            dte = 30
    if dte <= 0:
        dte = 1

    spot = float(option_chain.get("spot") or spot_price)
    if spot <= 0:
        return {"error": "Prix du sous-jacent invalide."}

    candidates: list[dict] = []
    for contract in contracts:
        strike = _finite(contract.get("strike"))
        last_price = _finite(contract.get("last_price"))
        bid = _finite(contract.get("bid"))
        ask = _finite(contract.get("ask"))
        if strike is None or strike <= 0:
            continue

        premium = ask if ask and ask > 0 else last_price
        if premium is None or premium <= 0:
            continue
        if not (spot * 0.75 <= strike <= spot * 1.25):
            continue

        iv = _finite(contract.get("implied_volatility")) or DEFAULT_OPTION_VOLATILITY
        if iv > 3.0:
            iv = iv / 100.0

        greeks = calculate_greeks(
            spot=spot,
            strike=strike,
            option_type=option_type,
            dte=dte,
            rate=rate,
            volatility=iv,
        )
        if "error" in greeks:
            continue

        delta = abs(greeks["greeks"]["delta"])
        spread_pct = (
            ((ask - bid) / ((ask + bid) / 2.0) * 100.0)
            if ask and bid and ask > bid and (ask + bid) > 0
            else 0.0
        )
        score = abs(delta - target_delta) * 100.0 + spread_pct * 0.5

        candidates.append(
            {
                "contract": contract,
                "strike": strike,
                "premium": premium,
                "iv": iv,
                "greeks": greeks,
                "delta": delta,
                "spread_pct": round(spread_pct, 2),
                "score": round(score, 4),
            }
        )

    if not candidates:
        return {"error": "Aucun contrat exploitable pour ce scénario."}

    best = min(candidates, key=lambda item: item["score"])
    premium = best["premium"]
    cost_usd = round(premium * 100.0, 2)
    breakeven = best["greeks"]["breakeven"]

    if target_price is None:
        expected_move = max(0.05, best["iv"] * math.sqrt(dte / 365.0) * 0.8)
        target_price = spot * (1.0 + expected_move) if outlook == "bullish" else spot * (1.0 - expected_move)

    if outlook == "bullish":
        intrinsic_at_target = max(0.0, target_price - best["strike"])
    else:
        intrinsic_at_target = max(0.0, best["strike"] - target_price)
    target_profit = round(max(intrinsic_at_target - premium, 0.0) * 100.0, 2)
    max_loss = cost_usd
    risk_reward = round(target_profit / max_loss, 2) if max_loss > 0 else 0.0

    return {
        "symbol": option_chain.get("symbol"),
        "expiration": option_chain.get("expiration"),
        "outlook": outlook,
        "risk_profile": risk_profile,
        "spot": round(spot, 4),
        "target_price": round(target_price, 4),
        "option_type": option_type,
        "strike": best["strike"],
        "premium": round(premium, 4),
        "contract_symbol": best["contract"].get("contract_symbol"),
        "iv_pct": round(best["iv"] * 100.0, 2),
        "delta": best["greeks"]["greeks"]["delta"],
        "gamma": best["greeks"]["greeks"]["gamma"],
        "theta": best["greeks"]["greeks"]["theta"],
        "vega": best["greeks"]["greeks"]["vega"],
        "cost_usd": cost_usd,
        "breakeven": breakeven,
        "max_loss_usd": max_loss,
        "target_profit_usd": target_profit,
        "risk_reward": risk_reward,
        "spread_pct": best["spread_pct"],
        "rationale": (
            f"{option_type.upper()} strike {best['strike']} expirant dans {dte}j, "
            f"delta {best['greeks']['greeks']['delta']:.2f} proche de la cible "
            f"{target_delta:.2f} pour un profil {risk_profile}."
        ),
    }


# ---------------------------------------------------------------------------
# Éligibilité PMCC (extrait de trading_skills.scanner_pmcc)
# ---------------------------------------------------------------------------
def _contract_mid(contract: dict | None) -> float | None:
    if not contract:
        return None
    for key in ("mid", "mid_price", "mark"):
        value = _finite(contract.get(key))
        if value:
            return value
    bid = _finite(contract.get("bid"))
    ask = _finite(contract.get("ask"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return _finite((bid + ask) / 2.0)
    return _finite(contract.get("lastPrice") or contract.get("last"))


def _contract_spread_pct(contract: dict | None) -> float | None:
    if not contract:
        return None
    bid = _finite(contract.get("bid"))
    ask = _finite(contract.get("ask"))
    if bid is None or ask is None or ask <= 0 or bid <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return _finite((ask - bid) / mid * 100.0, 2) if mid else None


def score_pmcc_eligibility(
    underlying_price: float,
    leaps_contract: dict | None,
    short_contract: dict | None,
    annual_yield_pct: float | None = None,
    earnings_days: int | None = None,
    trend_score: float | None = None,
    has_weeklies: bool | None = None,
) -> dict:
    """Note l'éligibilité d'un sous-jacent à un Poor Man's Covered Call sur 14.

    Portage déterministe du scoring `scanner_pmcc` : précision du delta,
    liquidité, spread, rendement annualisé, tendance et proximité des résultats.
    Le résultat est directement consommable par l'UI ou le Risk Manager.
    """
    price = _finite(underlying_price)
    breakdown: dict[str, float] = {}
    notes: list[str] = []
    score = 0.0

    if not price or price <= 0:
        return {
            "pmcc_score": 0.0,
            "max_possible_score": 14.0,
            "eligible": False,
            "score_breakdown": {},
            "notes": ["Prix sous-jacent invalide."],
        }

    leaps_delta = abs(_finite((leaps_contract or {}).get("delta")) or 0.0)
    short_delta = abs(_finite((short_contract or {}).get("delta")) or 0.0)

    if leaps_delta:
        if 0.70 <= leaps_delta <= 0.85:
            breakdown["leaps_delta"] = 2.0
            notes.append(f"Delta LEAPS optimal ({leaps_delta:.2f}).")
        elif 0.60 <= leaps_delta <= 0.90:
            breakdown["leaps_delta"] = 1.0
        else:
            notes.append(f"Delta LEAPS hors plage ({leaps_delta:.2f}).")

    if short_delta:
        if 0.20 <= short_delta <= 0.35:
            breakdown["short_delta"] = 2.0
            notes.append(f"Delta short call optimal ({short_delta:.2f}).")
        elif 0.15 <= short_delta <= 0.45:
            breakdown["short_delta"] = 1.0
        else:
            notes.append(f"Delta short call hors plage ({short_delta:.2f}).")

    for label, contract in (("leaps", leaps_contract), ("short", short_contract)):
        if not contract:
            continue
        open_interest = _finite(contract.get("openInterest") or contract.get("open_interest")) or 0.0
        volume = _finite(contract.get("volume")) or 0.0
        liquidity_points = 0.0
        if open_interest >= 500:
            liquidity_points += 1.0
        elif open_interest >= 100:
            liquidity_points += 0.5
        if volume >= 50:
            liquidity_points += 1.0
        elif volume >= 10:
            liquidity_points += 0.5
        if liquidity_points:
            breakdown[f"{label}_liquidity"] = liquidity_points
            if liquidity_points >= 2.0:
                notes.append(f"Liquidité {label} solide (OI {open_interest:.0f}, vol {volume:.0f}).")

        spread = _contract_spread_pct(contract)
        if spread is not None:
            if spread <= 5.0:
                breakdown[f"{label}_spread"] = 1.0
            elif spread <= 10.0:
                breakdown[f"{label}_spread"] = 0.5
            else:
                notes.append(f"Spread {label} large ({spread:.1f}%).")

    if annual_yield_pct is not None:
        if annual_yield_pct >= 12.0:
            breakdown["annual_yield"] = 2.0
            notes.append(f"Rendement annualisé attractif ({annual_yield_pct:.1f}%).")
        elif annual_yield_pct >= 8.0:
            breakdown["annual_yield"] = 1.5
        elif annual_yield_pct >= 5.0:
            breakdown["annual_yield"] = 1.0

    if trend_score is not None:
        if trend_score >= 1.0:
            breakdown["trend"] = 1.5
            notes.append("Tendance haussière confirmée.")
        elif trend_score > 0:
            breakdown["trend"] = 0.75

    if earnings_days is not None:
        if earnings_days > 45:
            breakdown["earnings"] = 1.0
        elif earnings_days < 0:
            breakdown["earnings"] = 0.0
        elif earnings_days <= 21:
            breakdown["earnings"] = -2.0
            notes.append("Résultats trop proches : risque de gap/IV crush.")
        else:
            breakdown["earnings"] = -1.0

    if has_weeklies:
        breakdown["weeklies"] = 1.0

    score = round(max(0.0, min(sum(breakdown.values()), 14.0)), 2)
    return {
        "pmcc_score": score,
        "max_possible_score": 14.0,
        "eligible": score >= 8.0,
        "score_breakdown": breakdown,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Screener déterministe (prism-insight)
# ---------------------------------------------------------------------------
def _compute_surge_score(technicals: dict) -> tuple[float, list[str]]:
    signals: list[str] = []
    score = 0.0

    volume = technicals.get("volume") or {}
    ratio = volume.get("ratio")
    if ratio is not None:
        if ratio >= 3.0:
            score += 35.0
            signals.append(f"Volume x{ratio:.2f}")
        elif ratio >= 2.0:
            score += 28.0
            signals.append(f"Volume x{ratio:.2f}")
        elif ratio >= 1.5:
            score += 22.0
            signals.append(f"Volume x{ratio:.2f}")
        elif ratio >= 1.2:
            score += 15.0
        elif ratio >= 1.0:
            score += 8.0

    price = technicals.get("price")
    ema = technicals.get("ema") or {}
    ma = technicals.get("ma") or {}
    ema9 = ema.get("ema9")
    ema20 = ema.get("ema20")
    sma50 = ma.get("sma50")

    if price and ema9 and ema20 and sma50:
        if price > ema9 > ema20 > sma50:
            score += 30.0
            signals.append("Alignement EMA9>EMA20>SMA50")
        elif price > ema9 > ema20:
            score += 24.0
            signals.append("Alignement EMA9>EMA20")
        elif price > ema20 > sma50:
            score += 18.0
            signals.append("Prix au-dessus des moyennes")
        elif price > ema9:
            score += 10.0
    elif price and ema9 and price > ema9:
        score += 8.0

    rsi = technicals.get("rsi14")
    if rsi is not None:
        if 55 <= rsi <= 65:
            score += 25.0
            signals.append(f"RSI sain ({rsi:.1f})")
        elif 65 < rsi <= 70:
            score += 22.0
        elif 50 <= rsi < 55:
            score += 18.0
        elif 70 < rsi <= 75:
            score += 15.0
            signals.append(f"RSI élevé ({rsi:.1f})")
        elif 45 <= rsi < 50:
            score += 10.0
        elif 75 < rsi <= 80:
            score += 8.0
        elif rsi < 30:
            score += 4.0
            signals.append(f"RSI survente ({rsi:.1f})")

    momentum = technicals.get("momentum_pct") or {}
    five_day = momentum.get("5d")
    if five_day is not None:
        score += max(-10.0, min(10.0, five_day * 1.5))

    levels = technicals.get("support_resistance") or {}
    resistance = levels.get("nearest_resistance")
    if price and resistance and ratio and ratio >= 1.3:
        distance_pct = (resistance - price) / price * 100.0
        if 0 <= distance_pct <= 1.5:
            score += 10.0
            signals.append("Cassure de résistance imminente")

    return round(max(0.0, min(score, 100.0)), 2), signals


def _prism_composite(technicals: dict) -> float:
    """Score composite 0-100 façon PRISM : momentum 60 % + volume relatif 40 %.

    Reprend `normalize_and_score` de prism-insight (ratio_weight=0.6,
    abs_weight=0.4) en normalisant chaque composante sur 0-1.
    """
    volume = technicals.get("volume") or {}
    momentum = technicals.get("momentum_pct") or {}
    ratio = volume.get("ratio")
    five_day = momentum.get("5d")

    volume_component = 0.0
    if ratio is not None:
        volume_component = max(0.0, min((ratio - 1.0) / 2.0, 1.0))

    momentum_component = 0.0
    if five_day is not None:
        momentum_component = max(0.0, min((five_day + 5.0) / 15.0, 1.0))

    return round((momentum_component * 0.6 + volume_component * 0.4) * 100.0, 2)


def _grade_score(score: float) -> str:
    if score >= 80:
        return "A+"
    if score >= 70:
        return "A"
    if score >= 60:
        return "B"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "E"


def _screen_single_symbol(symbol: str, timeframe: str, limit: int) -> dict:
    market_type = detect_market_type(symbol)
    normalized = normalize_symbol(symbol, market_type)
    df = fetch_ohlcv_sync(normalized, timeframe=timeframe, limit=limit, market_type=market_type)
    technicals = compute_technicals(df, include_candles=False)
    if "error" in technicals:
        raise DataFetchError(technicals["error"])

    surge_score, signals = _compute_surge_score(technicals)
    levels = technicals.get("support_resistance") or {}
    volume = technicals.get("volume") or {}
    momentum = technicals.get("momentum_pct") or {}

    return {
        "symbol": normalized,
        "requested_symbol": symbol.upper(),
        "market_type": market_type,
        "price": technicals.get("price"),
        "surge_score": surge_score,
        "prism_composite": _prism_composite(technicals),
        "grade": _grade_score(surge_score),
        "trend": technicals.get("trend"),
        "rsi14": technicals.get("rsi14"),
        "volume_ratio": volume.get("ratio"),
        "momentum_5d_pct": momentum.get("5d"),
        "atr_pct": technicals.get("atr_pct"),
        "volatility_annualized_pct": technicals.get("volatility_annualized_pct"),
        "nearest_support": levels.get("nearest_support"),
        "nearest_resistance": levels.get("nearest_resistance"),
        "signals": signals,
    }


def run_prism_bullish_screener(
    symbols_list: Iterable[str],
    timeframe: str = "1d",
    limit: int = 180,
    workers: int = 8,
    top_n: int | None = None,
) -> list[dict]:
    """Scanne une liste de tickers et classe les meilleures opportunités.

    Fusionne le `scanner-bullish` de trading_skills (EMA, RSI, MACD, momentum)
    avec les poids momentum/volume de prism-insight. Le `surge_score` (0-100)
    combine volume anormal, alignement EMA, RSI, momentum court terme et
    proximité d'une résistance. Aucun appel LLM.
    """
    symbols = [str(symbol).strip().upper() for symbol in symbols_list if str(symbol).strip()]
    if not symbols:
        return []

    results: list[dict] = []
    max_workers = max(1, min(int(workers), len(symbols), 16))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_screen_single_symbol, symbol, timeframe, limit): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result.get("price") is not None:
                    results.append(result)
            except Exception as exc:
                print(f"[screener] {symbol} ignoré : {exc}", flush=True)

    results.sort(key=lambda item: item["surge_score"], reverse=True)
    if top_n is not None:
        return results[: max(int(top_n), 1)]
    return results


def run_deterministic_screener(
    symbols_list: Iterable[str],
    timeframe: str = "1d",
    limit: int = 180,
    workers: int = 8,
    top_n: int | None = None,
) -> list[dict]:
    """Alias rétro-compatible de `run_prism_bullish_screener`."""
    return run_prism_bullish_screener(
        symbols_list,
        timeframe=timeframe,
        limit=limit,
        workers=workers,
        top_n=top_n,
    )
