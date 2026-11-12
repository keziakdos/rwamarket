"""Serveur web FastAPI de RwaMarket.

Contient l'authentification, le dashboard Jinja2, les endpoints SSE / screener /
Paper Trading / options et le planificateur APScheduler.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel

from app import database as db
from app.agents import memory as trading_memory
from app.agents.orchestrator import sse_event, stream_market_analysis
from app.engine import data_fetcher as fetcher
from app.engine import notifier, paper_trader, quant_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rwamarket")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip() or "rwamarket"
SECRET_KEY = os.getenv("SECRET_KEY", "").strip() or "rwamarket-session-secret"
SESSION_COOKIE = "rwa_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7
BRUSSELS_TZ = ZoneInfo("Europe/Brussels")
serializer = URLSafeTimedSerializer(SECRET_KEY, salt="rwa-session-v1")


def _watchlist(env_name: str, default: str) -> list[str]:
    raw = os.getenv(env_name, default)
    symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return symbols or [item.strip().upper() for item in default.split(",") if item.strip()]


def get_crypto_watchlist() -> list[str]:
    return _watchlist("WATCHLIST_CRYPTO", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")


def get_stock_watchlist() -> list[str]:
    return _watchlist("WATCHLIST_STOCKS", "NVDA,AAPL,MSFT,TSLA,SPY")


def get_combined_watchlist() -> list[str]:
    symbols: list[str] = []
    for symbol in get_crypto_watchlist() + get_stock_watchlist():
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------
def _header_password(request: Request) -> str:
    header = request.headers.get("x-app-password", "").strip()
    if header:
        return header
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def is_authenticated(request: Request) -> bool:
    header = _header_password(request)
    if header and hmac.compare_digest(header, APP_PASSWORD):
        return True

    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return False
    try:
        value = serializer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return False
    return value == "authenticated"


def require_auth(request: Request) -> None:
    if not is_authenticated(request):
        raise HTTPException(
            status_code=401,
            detail="Authentification requise.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _session_cookie_value() -> str:
    return serializer.dumps("authenticated")


# ---------------------------------------------------------------------------
# Tâches planifiées
# ---------------------------------------------------------------------------
async def _run_screener() -> list[dict]:
    symbols = get_combined_watchlist()
    return await asyncio.to_thread(
        quant_tools.run_deterministic_screener,
        symbols,
        "1d",
        180,
        8,
        30,
    )


def _cached_synthesis(symbol: str) -> dict:
    market = fetcher.detect_market_type(symbol)
    normalized = fetcher.normalize_symbol(symbol, market)
    ttl = int(os.getenv("CACHE_TTL_MINUTES", "30"))
    cached = db.get_cached_analysis(normalized, market, "1d", "moderate", ttl_minutes=ttl)
    if cached and isinstance(cached.get("synthesis"), dict):
        return {
            "symbol": normalized,
            "market_type": market,
            "verdict": cached["synthesis"].get("verdict"),
            "confidence": cached["synthesis"].get("confidence"),
            "trade_plan": cached["synthesis"].get("trade_plan"),
        }
    return {"symbol": normalized, "market_type": market, "error": "cache indisponible"}


async def _warm_analysis(symbol: str) -> dict:
    try:
        async for _event in stream_market_analysis(
            symbol,
            timeframe="1d",
            risk_profile="moderate",
            force_refresh=False,
        ):
            pass
        return _cached_synthesis(symbol)
    except Exception as exc:
        logger.warning("Pré-analyse %s échouée: %s", symbol, exc)
        return {"symbol": symbol, "error": str(exc)}


async def _scheduled_position_check() -> None:
    try:
        result = await paper_trader.check_and_update_positions()
        logger.info(
            "SL/TP check: %s position(s) vérifiée(s), %s clôturée(s), %s erreur(s)",
            result.get("checked"),
            len(result.get("closed") or []),
            len(result.get("errors") or []),
        )
    except Exception:
        logger.exception("Échec de la vérification périodique des positions")


async def _scheduled_brief(label: str = "Morning Brief") -> None:
    try:
        await paper_trader.check_and_update_positions()
        results = await _run_screener()
        top = results[:3]
        analyses: list[dict] = []
        if top:
            gathered = await asyncio.gather(
                *[_warm_analysis(item["symbol"]) for item in top],
                return_exceptions=True,
            )
            analyses = [item for item in gathered if isinstance(item, dict)]
        portfolio = await paper_trader.get_portfolio_state(include_history=False)
        sent = await notifier.send_morning_brief(
            screener_results=results,
            analyses=analyses,
            portfolio=portfolio,
            session_label=label,
        )
        logger.info(
            "Brief planifié '%s': %s résultat(s), Telegram=%s",
            label,
            len(results),
            "envoyé" if sent else "silencieux/indisponible",
        )
    except Exception:
        logger.exception("Échec du brief planifié %s", label)


# ---------------------------------------------------------------------------
# Application et scheduler
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    scheduler = AsyncIOScheduler(timezone=BRUSSELS_TZ)
    scheduler.add_job(
        _scheduled_position_check,
        "interval",
        minutes=5,
        id="paper_position_check",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_brief,
        CronTrigger(hour=8, minute=30, timezone=BRUSSELS_TZ),
        args=["Morning Brief"],
        id="morning_brief_0830",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_brief,
        CronTrigger(hour=15, minute=30, timezone=BRUSSELS_TZ),
        args=["Brief 15:30"],
        id="morning_brief_1530",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    scheduler.start()
    app.state.scheduler = scheduler
    logger.info("Scheduler démarré (Europe/Brussels)")
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler arrêté")


app = FastAPI(
    title="RwaMarket",
    description="Analyse multi-agents et Paper Trading",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


# ---------------------------------------------------------------------------
# Modèles API
# ---------------------------------------------------------------------------
class OrderRequest(BaseModel):
    symbol: str
    side: str | None = None
    quantity: float | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_profile: str = "moderate"
    reasoning_summary: str | None = None
    market_type: str | None = None
    position_size_pct: float | None = None
    trade_plan: dict | None = None
    verdict: str | None = None


class CloseRequest(BaseModel):
    exit_price: float | None = None
    exit_reason: str = "manual"
    lesson_learned: str | None = None


# ---------------------------------------------------------------------------
# Contexte des templates
# ---------------------------------------------------------------------------
def _template_config() -> dict:
    return {
        "defaultSymbol": get_crypto_watchlist()[0],
        "defaultRisk": os.getenv("DEFAULT_RISK_PROFILE", "moderate").strip().lower(),
        "watchlist": get_combined_watchlist(),
        "crypto": get_crypto_watchlist(),
        "stocks": get_stock_watchlist(),
    }


def _template_context() -> dict:
    return {
        "watchlist": get_combined_watchlist(),
        "default_symbol": get_crypto_watchlist()[0],
        "default_risk": os.getenv("DEFAULT_RISK_PROFILE", "moderate").strip().lower(),
        "config": _template_config(),
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None},
    )


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, password: str = Form(...)):
    if hmac.compare_digest(password, APP_PASSWORD):
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=_session_cookie_value(),
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=False,
            path="/",
        )
        return response
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": "Mot de passe incorrect."},
        status_code=401,
    )


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=_template_context(),
    )


# ---------------------------------------------------------------------------
# Endpoints API
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    scheduler = getattr(app.state, "scheduler", None)
    return {
        "status": "ok",
        "time": datetime.now(UTC).isoformat(),
        "scheduler_running": bool(scheduler and scheduler.running),
        "telegram_enabled": notifier.telegram_enabled(),
    }


@app.get("/api/analyze/{symbol}/stream")
async def analyze_stream(
    symbol: str,
    request: Request,
    timeframe: str = "1d",
    risk_profile: str = "moderate",
    mode: str = "auto",
    force_refresh: bool = False,
    _auth: None = Depends(require_auth),
):
    async def event_generator():
        try:
            async for event in stream_market_analysis(
                symbol=symbol,
                timeframe=timeframe,
                risk_profile=risk_profile,
                mode=mode,
                force_refresh=force_refresh,
            ):
                if await request.is_disconnected():
                    break
                yield event
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Erreur SSE pour %s", symbol)
            yield sse_event(
                "error",
                {
                    "symbol": symbol,
                    "message": str(exc),
                    "generated_at": datetime.now(UTC).isoformat(),
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/screener")
async def screener(_auth: None = Depends(require_auth)):
    try:
        results = await _run_screener()
    except Exception as exc:
        logger.exception("Échec du screener")
        raise HTTPException(status_code=502, detail=f"Screener indisponible : {exc}") from exc
    return {
        "success": True,
        "count": len(results),
        "watchlist": get_combined_watchlist(),
        "crypto": get_crypto_watchlist(),
        "stocks": get_stock_watchlist(),
        "results": results,
        "generated_at": datetime.now(UTC).isoformat(),
    }


@app.get("/api/portfolio")
async def get_portfolio(_auth: None = Depends(require_auth)):
    try:
        return await paper_trader.get_portfolio_state()
    except Exception as exc:
        logger.exception("Échec de la récupération du portefeuille")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/memory")
async def get_trading_memory(
    symbol: str | None = None,
    market_type: str | None = None,
    risk_profile: str = "moderate",
    _auth: None = Depends(require_auth),
):
    """Retourne la mémoire PRISM consolidée (par symbole ou globale)."""
    if symbol:
        try:
            market = market_type or fetcher.detect_market_type(symbol)
            context = trading_memory.get_memory_context(symbol, market, risk_profile)
            return {"success": True, "symbol": symbol.upper(), "context": context}
        except Exception as exc:
            logger.exception("Échec de la mémoire pour %s", symbol)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        items = db.get_all_trading_memory(limit=200)
    except Exception as exc:
        logger.exception("Échec de la récupération de la mémoire globale")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "success": True,
        "count": len(items),
        "items": items,
        "generated_at": datetime.now(UTC).isoformat(),
    }


@app.post("/api/portfolio/order")
async def create_order(payload: OrderRequest, _auth: None = Depends(require_auth)):
    try:
        if payload.trade_plan:
            result = await paper_trader.open_position_from_plan(
                symbol=payload.symbol,
                trade_plan=payload.trade_plan,
                verdict=payload.verdict,
                risk_profile=payload.risk_profile,
                reasoning_summary=payload.reasoning_summary,
                market_type=payload.market_type,
                quantity=payload.quantity,
            )
        else:
            result = await paper_trader.open_position(
                symbol=payload.symbol,
                side=payload.side,
                quantity=payload.quantity,
                entry_price=payload.entry_price,
                stop_loss=payload.stop_loss,
                take_profit=payload.take_profit,
                risk_profile=payload.risk_profile,
                reasoning_summary=payload.reasoning_summary,
                market_type=payload.market_type,
                position_size_pct=payload.position_size_pct,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Échec de l'ouverture paper")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.post("/api/portfolio/close/{position_id}")
async def close_position(
    position_id: int,
    payload: CloseRequest | None = None,
    _auth: None = Depends(require_auth),
):
    body = payload or CloseRequest()
    try:
        return await paper_trader.close_position_manual(
            position_id=position_id,
            exit_price=body.exit_price,
            exit_reason=body.exit_reason,
            lesson_learned=body.lesson_learned,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Échec de la clôture paper")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/portfolio/check")
async def check_positions(_auth: None = Depends(require_auth)):
    try:
        return await paper_trader.check_and_update_positions()
    except Exception as exc:
        logger.exception("Échec du contrôle SL/TP manuel")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/options/preview/{symbol}")
async def options_preview(
    symbol: str,
    outlook: str = "bullish",
    risk_profile: str = "moderate",
    expiration: str | None = None,
    target_price: float | None = None,
    strike: float | None = None,
    dte: int | None = None,
    volatility: float | None = None,
    option_type: str | None = None,
    market_price: float | None = None,
    _auth: None = Depends(require_auth),
):
    market = fetcher.detect_market_type(symbol)
    if market != "stock":
        raise HTTPException(
            status_code=400,
            detail="Les options US sont disponibles uniquement pour les actions.",
        )

    normalized = fetcher.normalize_symbol(symbol, market)
    try:
        chain = await fetcher.fetch_option_chain(normalized, expiration=expiration)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Chaîne d'options indisponible : {exc}") from exc

    spot = chain.get("spot")
    if not spot:
        spot, _ = await paper_trader.get_current_price(normalized, "stock")
    if not spot:
        raise HTTPException(status_code=404, detail="Prix du sous-jacent indisponible.")

    setup = quant_tools.preview_option_setup(
        spot_price=float(spot),
        option_chain=chain,
        outlook=outlook,
        risk_profile=risk_profile,
        target_price=target_price,
    )

    custom_greeks = None
    if strike is not None and strike > 0:
        resolved_type = (option_type or ("call" if outlook == "bullish" else "put")).strip().lower()
        if resolved_type not in {"call", "put"}:
            resolved_type = "call"
        custom_greeks = quant_tools.calculate_greeks(
            spot=float(spot),
            strike=float(strike),
            option_type=resolved_type,
            dte=int(dte) if dte and dte > 0 else 30,
            market_price=market_price,
            volatility=volatility,
            rate=0.05,
        )

    return {
        "success": "error" not in setup,
        "symbol": normalized,
        "spot": spot,
        "expiration": chain.get("expiration"),
        "expirations": (chain.get("expirations") or [])[:8],
        "calls_count": len(chain.get("calls") or []),
        "puts_count": len(chain.get("puts") or []),
        "setup": setup,
        "custom_greeks": custom_greeks,
        "generated_at": datetime.now(UTC).isoformat(),
    }
