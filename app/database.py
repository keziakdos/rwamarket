"""Gestionnaire de base de données SQLite pour RwaMarket.

Toutes les écritures passent par des requêtes SQL paramétrées. La base est
ouverte en mode WAL afin de supporter des lectures concurrentes pendant que les
agents écrivent leurs résultats. Les connexions sont courtes et systématiquement
refermées via un context manager.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "rwamarket.db"
DB_PATH = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH))).expanduser()
DEFAULT_CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "30"))
DEFAULT_INITIAL_BALANCE = float(os.getenv("INITIAL_PAPER_BALANCE", "10000.0"))
# Délais de carence anti-rachat impulsif (extraits de PRISM-Insight).
# Après une vente normale/gagnante : 0h par défaut (la continuation est saine).
# Après une perte ou une sortie sur stop/tendance : 24h par défaut.
DEFAULT_REENTRY_COOLDOWN_HOURS = float(os.getenv("REENTRY_COOLDOWN_HOURS", "0"))
DEFAULT_REENTRY_COOLDOWN_LOSS_HOURS = float(os.getenv("REENTRY_COOLDOWN_LOSS_HOURS", "24"))
RISK_EXIT_KINDS = {"stop", "trend_exit"}


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------
def utcnow_iso() -> str:
    """Retourne l'heure UTC courante au format ISO 8601 (secondes)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _json_dumps(value: Any) -> str:
    """Sérialise une valeur Python en JSON, en tolérant les valeurs vides."""
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Any, default: Any = None) -> Any:
    """Désérialise un champ JSON stocké en base."""
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


@contextmanager
def get_connection() -> Iterable[sqlite3.Connection]:
    """Ouvre une connexion SQLite configurée (WAL, timeout, row factory)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=30000;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Initialisation du schéma
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Crée les tables, index et le compte Paper Trading initial si nécessaire."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS analysis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                risk_profile TEXT NOT NULL,
                quant_json TEXT NOT NULL,
                agents_json TEXT,
                synthesis_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_analysis_cache_key
                ON analysis_cache (symbol, market_type, timeframe, risk_profile);
            CREATE INDEX IF NOT EXISTS idx_analysis_cache_created
                ON analysis_cache (created_at);

            CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                balance REAL NOT NULL,
                initial_balance REAL NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('long', 'short')),
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                risk_profile TEXT NOT NULL DEFAULT 'moderate',
                reasoning_summary TEXT,
                opened_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
                reserved_cash REAL NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_paper_positions_status
                ON paper_positions (status);
            CREATE INDEX IF NOT EXISTS idx_paper_positions_symbol
                ON paper_positions (symbol, status);

            CREATE TABLE IF NOT EXISTS paper_trades_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('long', 'short')),
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                pnl_usd REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                exit_reason TEXT,
                reasoning_summary TEXT,
                lesson_learned TEXT,
                opened_at TEXT,
                closed_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_trades_history_symbol
                ON paper_trades_history (symbol, closed_at);
            CREATE INDEX IF NOT EXISTS idx_trades_history_closed
                ON paper_trades_history (closed_at);

            CREATE TABLE IF NOT EXISTS trading_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                compressed_rules TEXT NOT NULL,
                win_rate REAL NOT NULL DEFAULT 0,
                total_trades INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_trading_memory_key
                ON trading_memory (symbol, market_type);

            CREATE TABLE IF NOT EXISTS reentry_cooldowns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                last_sell_at TEXT NOT NULL,
                exit_reason TEXT,
                exit_kind TEXT,
                pnl_pct REAL NOT NULL DEFAULT 0,
                after_loss INTEGER NOT NULL DEFAULT 0,
                risk_exit INTEGER NOT NULL DEFAULT 0,
                cooldown_hours REAL NOT NULL DEFAULT 0,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_reentry_cooldowns_key
                ON reentry_cooldowns (symbol, market_type);
            CREATE INDEX IF NOT EXISTS idx_reentry_cooldowns_expires
                ON reentry_cooldowns (expires_at);
            """
        )

        existing = conn.execute("SELECT id FROM paper_account ORDER BY id LIMIT 1").fetchone()
        if existing is None:
            now = utcnow_iso()
            conn.execute(
                "INSERT INTO paper_account (balance, initial_balance, updated_at) VALUES (?, ?, ?)",
                (DEFAULT_INITIAL_BALANCE, DEFAULT_INITIAL_BALANCE, now),
            )


# ---------------------------------------------------------------------------
# Cache d'analyses
# ---------------------------------------------------------------------------
def save_analysis_cache(
    symbol: str,
    market_type: str,
    timeframe: str,
    risk_profile: str,
    quant: Any,
    agents: Any = None,
    synthesis: Any = None,
) -> dict:
    """Insère ou remplace l'analyse d'un symbole pour une clé donnée."""
    key = (symbol.upper(), market_type.lower(), timeframe.lower(), risk_profile.lower())
    now = utcnow_iso()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO analysis_cache
                (symbol, market_type, timeframe, risk_profile,
                 quant_json, agents_json, synthesis_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, market_type, timeframe, risk_profile)
            DO UPDATE SET
                quant_json = excluded.quant_json,
                agents_json = excluded.agents_json,
                synthesis_json = excluded.synthesis_json,
                created_at = excluded.created_at
            """,
            (
                key[0],
                key[1],
                key[2],
                key[3],
                _json_dumps(quant),
                _json_dumps(agents),
                _json_dumps(synthesis),
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM analysis_cache WHERE symbol = ? AND market_type = ?"
            " AND timeframe = ? AND risk_profile = ?",
            key,
        ).fetchone()
    return _cache_row_to_dict(row) or {}


def _cache_row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    data = dict(row)
    return {
        "id": data["id"],
        "symbol": data["symbol"],
        "market_type": data["market_type"],
        "timeframe": data["timeframe"],
        "risk_profile": data["risk_profile"],
        "quant": _json_loads(data["quant_json"], {}),
        "agents": _json_loads(data["agents_json"], {}),
        "synthesis": _json_loads(data["synthesis_json"], {}),
        "created_at": data["created_at"],
    }


def get_cached_analysis(
    symbol: str,
    market_type: str,
    timeframe: str,
    risk_profile: str,
    ttl_minutes: int | None = None,
) -> dict | None:
    """Retourne l'analyse en cache si elle est encore valide, sinon None."""
    ttl = DEFAULT_CACHE_TTL_MINUTES if ttl_minutes is None else int(ttl_minutes)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_cache WHERE symbol = ? AND market_type = ?"
            " AND timeframe = ? AND risk_profile = ?",
            (symbol.upper(), market_type.lower(), timeframe.lower(), risk_profile.lower()),
        ).fetchone()

    cached = _cache_row_to_dict(row)
    if cached is None:
        return None

    created = _parse_dt(cached["created_at"])
    if created is None:
        return None

    age_minutes = (datetime.now(UTC) - created).total_seconds() / 60.0
    if ttl >= 0 and age_minutes > ttl:
        return None

    cached["age_minutes"] = round(max(age_minutes, 0.0), 2)
    cached["ttl_minutes"] = ttl
    return cached


def invalidate_cache(
    symbol: str | None = None,
    market_type: str | None = None,
    timeframe: str | None = None,
    risk_profile: str | None = None,
) -> int:
    """Supprime les entrées de cache correspondant aux filtres fournis."""
    clauses: list[str] = []
    params: list[Any] = []
    if symbol is not None:
        clauses.append("symbol = ?")
        params.append(symbol.upper())
    if market_type is not None:
        clauses.append("market_type = ?")
        params.append(market_type.lower())
    if timeframe is not None:
        clauses.append("timeframe = ?")
        params.append(timeframe.lower())
    if risk_profile is not None:
        clauses.append("risk_profile = ?")
        params.append(risk_profile.lower())

    query = "DELETE FROM analysis_cache"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    with get_connection() as conn:
        cursor = conn.execute(query, params)
        return cursor.rowcount


def purge_expired_cache(ttl_minutes: int | None = None) -> int:
    """Supprime les analyses plus anciennes que le TTL configuré."""
    ttl = DEFAULT_CACHE_TTL_MINUTES if ttl_minutes is None else int(ttl_minutes)
    cutoff = datetime.now(UTC).timestamp() - (ttl * 60)
    with get_connection() as conn:
        rows = conn.execute("SELECT id, created_at FROM analysis_cache").fetchall()
        expired_ids = [
            row["id"]
            for row in rows
            if (created := _parse_dt(row["created_at"])) is not None
            and created.timestamp() < cutoff
        ]
        if not expired_ids:
            return 0
        placeholders = ",".join("?" for _ in expired_ids)
        conn.execute(f"DELETE FROM analysis_cache WHERE id IN ({placeholders})", expired_ids)
        return len(expired_ids)


# ---------------------------------------------------------------------------
# Compte Paper Trading
# ---------------------------------------------------------------------------
def get_paper_account() -> dict:
    """Retourne le compte Paper Trading (créé à la volée si absent)."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM paper_account ORDER BY id LIMIT 1").fetchone()
        if row is None:
            now = utcnow_iso()
            conn.execute(
                "INSERT INTO paper_account (balance, initial_balance, updated_at) VALUES (?, ?, ?)",
                (DEFAULT_INITIAL_BALANCE, DEFAULT_INITIAL_BALANCE, now),
            )
            row = conn.execute("SELECT * FROM paper_account ORDER BY id LIMIT 1").fetchone()
    account = dict(row)
    initial = float(account["initial_balance"]) or DEFAULT_INITIAL_BALANCE
    balance = float(account["balance"])
    account["pnl_usd"] = round(balance - initial, 4)
    account["pnl_pct"] = round(((balance / initial) - 1) * 100, 4) if initial else 0.0
    return account


def update_paper_balance(balance: float) -> dict:
    """Force le solde du compte Paper Trading à une valeur donnée."""
    if balance < 0:
        raise ValueError("Le solde ne peut pas être négatif.")
    account = get_paper_account()
    with get_connection() as conn:
        conn.execute(
            "UPDATE paper_account SET balance = ?, updated_at = ? WHERE id = ?",
            (float(balance), utcnow_iso(), account["id"]),
        )
    return get_paper_account()


def reset_paper_account(balance: float | None = None, clear_positions: bool = True) -> dict:
    """Réinitialise le compte simulé et, optionnellement, les positions ouvertes."""
    new_balance = float(balance) if balance is not None else DEFAULT_INITIAL_BALANCE
    if new_balance < 0:
        raise ValueError("Le solde initial ne peut pas être négatif.")

    account = get_paper_account()
    with get_connection() as conn:
        if clear_positions:
            conn.execute("DELETE FROM paper_positions WHERE status = 'open'")
        conn.execute(
            "UPDATE paper_account SET balance = ?, initial_balance = ?, updated_at = ? WHERE id = ?",
            (new_balance, new_balance, utcnow_iso(), account["id"]),
        )
    return get_paper_account()


def get_portfolio_snapshot() -> dict:
    """Retourne le compte, les positions ouvertes et le nombre de trades clôturés."""
    account = get_paper_account()
    positions = get_open_positions()
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS total FROM paper_trades_history").fetchone()
    return {
        "account": account,
        "open_positions": positions,
        "open_positions_count": len(positions),
        "closed_trades_count": int(row["total"]) if row else 0,
    }


# ---------------------------------------------------------------------------
# Positions Paper Trading
# ---------------------------------------------------------------------------
def get_open_positions() -> list[dict]:
    """Liste toutes les positions actuellement ouvertes."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_positions WHERE status = 'open' ORDER BY opened_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_position(position_id: int) -> dict | None:
    """Retourne une position par son identifiant."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM paper_positions WHERE id = ?", (int(position_id),)).fetchone()
    return _row_to_dict(row)


def create_position(
    symbol: str,
    market_type: str,
    side: str,
    entry_price: float,
    quantity: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    risk_profile: str = "moderate",
    reasoning_summary: str | None = None,
    enforce_balance: bool = True,
) -> dict:
    """Ouvre une position simulée et réserve le notionnel sur le solde cash."""
    side = side.lower()
    if side not in {"long", "short"}:
        raise ValueError("Le side doit être 'long' ou 'short'.")
    if entry_price <= 0 or quantity <= 0:
        raise ValueError("Le prix d'entrée et la quantité doivent être positifs.")

    cost = round(float(entry_price) * float(quantity), 8)
    account = get_paper_account()
    if enforce_balance and cost > float(account["balance"]) + 1e-9:
        raise ValueError(
            f"Solde insuffisant : {cost:.2f} requis, {float(account['balance']):.2f} disponible."
        )

    now = utcnow_iso()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO paper_positions
                (symbol, market_type, side, entry_price, quantity, stop_loss, take_profit,
                 risk_profile, reasoning_summary, opened_at, status, reserved_cash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                symbol.upper(),
                market_type.lower(),
                side,
                float(entry_price),
                float(quantity),
                float(stop_loss) if stop_loss is not None else None,
                float(take_profit) if take_profit is not None else None,
                risk_profile.lower(),
                reasoning_summary,
                now,
                cost,
            ),
        )
        position_id = cursor.lastrowid
        conn.execute(
            "UPDATE paper_account SET balance = balance - ?, updated_at = ? WHERE id = ?",
            (cost, now, account["id"]),
        )
    return get_position(position_id) or {}


def update_position_stops(
    position_id: int,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> dict | None:
    """Met à jour le Stop-Loss et/ou le Take-Profit d'une position ouverte."""
    position = get_position(position_id)
    if position is None:
        return None
    if position["status"] != "open":
        raise ValueError("Impossible de modifier une position clôturée.")

    new_sl = float(stop_loss) if stop_loss is not None else position["stop_loss"]
    new_tp = float(take_profit) if take_profit is not None else position["take_profit"]
    with get_connection() as conn:
        conn.execute(
            "UPDATE paper_positions SET stop_loss = ?, take_profit = ? WHERE id = ?",
            (new_sl, new_tp, int(position_id)),
        )
    return get_position(position_id)


def close_position(
    position_id: int,
    exit_price: float,
    exit_reason: str = "manual",
    lesson_learned: str | None = None,
) -> dict:
    """Clôture une position, enregistre l'historique et met à jour la mémoire."""
    if exit_price <= 0:
        raise ValueError("Le prix de sortie doit être positif.")

    position = get_position(position_id)
    if position is None:
        raise ValueError("Position introuvable.")
    if position["status"] != "open":
        raise ValueError("Position déjà clôturée.")

    entry = float(position["entry_price"])
    quantity = float(position["quantity"])
    side = position["side"]
    if side == "long":
        pnl_usd = (float(exit_price) - entry) * quantity
    else:
        pnl_usd = (entry - float(exit_price)) * quantity
    notional = entry * quantity
    pnl_pct = (pnl_usd / notional) * 100 if notional else 0.0
    pnl_usd = round(pnl_usd, 8)
    pnl_pct = round(pnl_pct, 4)
    reserved = float(position.get("reserved_cash") or notional)
    now = utcnow_iso()

    account = get_paper_account()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO paper_trades_history
                (symbol, market_type, side, entry_price, exit_price, quantity,
                 pnl_usd, pnl_pct, exit_reason, reasoning_summary, lesson_learned,
                 opened_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position["symbol"],
                position["market_type"],
                side,
                entry,
                float(exit_price),
                quantity,
                pnl_usd,
                pnl_pct,
                exit_reason,
                position.get("reasoning_summary"),
                lesson_learned,
                position.get("opened_at"),
                now,
            ),
        )
        trade_id = cursor.lastrowid
        conn.execute(
            "UPDATE paper_positions SET status = 'closed' WHERE id = ?",
            (int(position_id),),
        )
        conn.execute(
            "UPDATE paper_account SET balance = balance + ?, updated_at = ? WHERE id = ?",
            (reserved + pnl_usd, now, account["id"]),
        )
        row = conn.execute(
            "SELECT * FROM paper_trades_history WHERE id = ?", (trade_id,)
        ).fetchone()

    trade = dict(row)
    refresh_trading_memory_from_history(
        position["symbol"],
        position["market_type"],
        new_lesson=lesson_learned,
    )
    try:
        record_reentry_cooldown(
            position["symbol"],
            position["market_type"],
            exit_reason=exit_reason,
            pnl_pct=pnl_pct,
            closed_at=now,
        )
    except Exception:
        # Le cooldown est un garde-fou : une erreur ici ne doit jamais empêcher
        # l'enregistrement comptable du trade.
        pass
    return trade


def close_position_by_symbol(
    symbol: str,
    exit_price: float,
    exit_reason: str = "manual",
    lesson_learned: str | None = None,
) -> list[dict]:
    """Clôture toutes les positions ouvertes d'un symbole donné."""
    closed: list[dict] = []
    for position in get_open_positions():
        if position["symbol"].upper() == symbol.upper():
            closed.append(
                close_position(
                    position["id"],
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    lesson_learned=lesson_learned,
                )
            )
    return closed


# ---------------------------------------------------------------------------
# Historique des trades et statistiques
# ---------------------------------------------------------------------------
def get_trades_history(
    symbol: str | None = None,
    market_type: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Retourne l'historique des trades clôturés, du plus récent au plus ancien."""
    clauses: list[str] = []
    params: list[Any] = []
    if symbol is not None:
        clauses.append("symbol = ?")
        params.append(symbol.upper())
    if market_type is not None:
        clauses.append("market_type = ?")
        params.append(market_type.lower())

    query = "SELECT * FROM paper_trades_history"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY closed_at DESC, id DESC LIMIT ?"
    params.append(max(int(limit), 1))

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_trade(trade_id: int) -> dict | None:
    """Retourne un trade clôturé par son identifiant."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM paper_trades_history WHERE id = ?", (int(trade_id),)
        ).fetchone()
    return _row_to_dict(row)


def get_performance_stats(symbol: str | None = None, market_type: str | None = None) -> dict:
    """Calcule Win Rate, PnL, Profit Factor, Drawdown et Sharpe des trades clôturés."""
    trades = get_trades_history(symbol=symbol, market_type=market_type, limit=100000)
    account = get_paper_account()
    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "total_pnl_usd": 0.0,
            "avg_pnl_usd": 0.0,
            "avg_pnl_pct": 0.0,
            "best_trade_usd": 0.0,
            "worst_trade_usd": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "account": account,
        }

    ordered = sorted(trades, key=lambda t: t["closed_at"] or "")
    pnls = [float(t["pnl_usd"]) for t in ordered]
    pct_returns = [float(t["pnl_pct"]) / 100 for t in ordered]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    equity = float(account["initial_balance"])
    peak = equity
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

    sharpe = 0.0
    if len(pct_returns) >= 2:
        import statistics

        mean_ret = statistics.fmean(pct_returns)
        std_ret = statistics.pstdev(pct_returns)
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * (252 ** 0.5)

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round((len(wins) / len(trades)) * 100, 2),
        "total_pnl_usd": round(sum(pnls), 4),
        "avg_pnl_usd": round(sum(pnls) / len(pnls), 4),
        "avg_pnl_pct": round(sum(float(t["pnl_pct"]) for t in ordered) / len(ordered), 4),
        "best_trade_usd": round(max(pnls), 4),
        "worst_trade_usd": round(min(pnls), 4),
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "sharpe_ratio": round(sharpe, 4),
        "account": account,
    }


# ---------------------------------------------------------------------------
# Mémoire de trading auto-apprenante (inspirée de PRISM-Insight)
# ---------------------------------------------------------------------------
def get_trading_memory(symbol: str, market_type: str) -> dict | None:
    """Retourne la mémoire compressée d'un couple (symbole, marché)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM trading_memory WHERE symbol = ? AND market_type = ?",
            (symbol.upper(), market_type.lower()),
        ).fetchone()
    return _row_to_dict(row)


def get_all_trading_memory(limit: int = 200) -> list[dict]:
    """Liste les mémoires de trading les plus récemment mises à jour."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trading_memory ORDER BY updated_at DESC LIMIT ?",
            (max(int(limit), 1),),
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_trading_memory(
    symbol: str,
    market_type: str,
    compressed_rules: str,
    win_rate: float | None = None,
    total_trades: int | None = None,
) -> dict:
    """Crée ou met à jour la mémoire de trading d'un symbole."""
    now = utcnow_iso()
    existing = get_trading_memory(symbol, market_type)
    if existing is None:
        win_rate = float(win_rate or 0.0)
        total_trades = int(total_trades or 0)
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO trading_memory
                    (symbol, market_type, compressed_rules, win_rate, total_trades, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (symbol.upper(), market_type.lower(), compressed_rules, win_rate, total_trades, now),
            )
    else:
        win_rate = float(existing["win_rate"] if win_rate is None else win_rate)
        total_trades = int(existing["total_trades"] if total_trades is None else total_trades)
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE trading_memory
                SET compressed_rules = ?, win_rate = ?, total_trades = ?, updated_at = ?
                WHERE symbol = ? AND market_type = ?
                """,
                (
                    compressed_rules,
                    win_rate,
                    total_trades,
                    now,
                    symbol.upper(),
                    market_type.lower(),
                ),
            )
    return get_trading_memory(symbol, market_type) or {}


def refresh_trading_memory_from_history(
    symbol: str,
    market_type: str,
    new_lesson: str | None = None,
    max_lessons: int = 20,
    max_chars: int = 6000,
) -> dict:
    """Recalcule Win Rate / total trades et compresse les leçons récentes."""
    trades = get_trades_history(symbol=symbol, market_type=market_type, limit=500)
    total_trades = len(trades)
    wins = sum(1 for t in trades if float(t["pnl_usd"]) > 0)
    win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0.0

    existing = get_trading_memory(symbol, market_type)
    previous_rules = (existing or {}).get("compressed_rules") or ""
    lessons: list[str] = [part.strip() for part in previous_rules.split("\n") if part.strip()]

    if new_lesson:
        lessons.append(f"[{utcnow_iso()[:10]}] {new_lesson.strip()}")
    else:
        for trade in trades[:max_lessons]:
            if trade.get("lesson_learned"):
                entry = f"[{(trade['closed_at'] or '')[:10]}] {trade['lesson_learned'].strip()}"
                if entry not in lessons:
                    lessons.append(entry)

    lessons = lessons[-max_lessons:]
    compressed = "\n".join(lessons)
    if len(compressed) > max_chars:
        compressed = compressed[-max_chars:]

    if not compressed:
        outcome = "gagnant" if wins else "perdant"
        compressed = (
            f"Bilan {symbol.upper()} ({market_type}) : {total_trades} trade(s), "
            f"{win_rate:.1f}% de réussite. Conserver une gestion stricte du risque."
            if total_trades
            else f"Aucune leçon disponible pour {symbol.upper()} ({outcome})."
        )

    return upsert_trading_memory(
        symbol=symbol,
        market_type=market_type,
        compressed_rules=compressed,
        win_rate=win_rate,
        total_trades=total_trades,
    )


def delete_trading_memory(symbol: str, market_type: str) -> bool:
    """Supprime la mémoire d'un symbole. Retourne True si une ligne a été effacée."""
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM trading_memory WHERE symbol = ? AND market_type = ?",
            (symbol.upper(), market_type.lower()),
        )
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Cooldown anti-rachat impulsif (extrait de PRISM-Insight)
# ---------------------------------------------------------------------------
def classify_exit_kind(exit_reason: str | None, explicit: str | None = None) -> str:
    """Normalise un motif de sortie en `stop`, `trend_exit`, `target` ou `ai`.

    Les sorties liées au risque (stop ou tendance) sont traitées comme des
    sorties « churn-risk » même lorsqu'elles se clôturent à un profit marginal.
    """
    if explicit:
        return explicit
    reason = (exit_reason or "").lower()
    if (
        "tier1.5" in reason
        or "ma50" in reason
        or "ma_50" in reason
        or "50-day" in reason
        or "trend" in reason
        or "tendance" in reason
    ):
        return "trend_exit"
    if (
        "stop-loss" in reason
        or "stop_loss" in reason
        or "stoploss" in reason
        or "tier1" in reason
        or "abs7" in reason
        or "-7%" in reason
        or "hard stop" in reason
        or "stop" in reason
        or "stop loss" in reason
        or "stop_loss" in reason
    ):
        return "stop"
    if "target" in reason or "take-profit" in reason or "take_profit" in reason or "objectif" in reason:
        return "target"
    return "ai"


def record_reentry_cooldown(
    symbol: str,
    market_type: str,
    exit_reason: str = "manual",
    exit_kind: str | None = None,
    pnl_pct: float = 0.0,
    closed_at: str | None = None,
) -> dict:
    """Enregistre (ou rafraîchit) le cooldown de rachat d'un symbole.

    La fenêtre dépend de la nature de la sortie :
    - perte réalisée ou sortie sur stop/tendance : `REENTRY_COOLDOWN_LOSS_HOURS`
    - vente normale / gagnante : `REENTRY_COOLDOWN_HOURS` (0h par défaut)
    """
    kind = classify_exit_kind(exit_reason, exit_kind)
    risk_exit = kind in RISK_EXIT_KINDS
    try:
        pnl_value = float(pnl_pct)
    except (TypeError, ValueError):
        pnl_value = 0.0
    after_loss = pnl_value < 0
    hours = (
        DEFAULT_REENTRY_COOLDOWN_LOSS_HOURS
        if (after_loss or risk_exit)
        else DEFAULT_REENTRY_COOLDOWN_HOURS
    )

    sold_at = _parse_dt(closed_at) or datetime.now(UTC)
    sold_at = sold_at.replace(microsecond=0)
    expires_at = sold_at + timedelta(hours=hours)
    now = utcnow_iso()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO reentry_cooldowns
                (symbol, market_type, last_sell_at, exit_reason, exit_kind,
                 pnl_pct, after_loss, risk_exit, cooldown_hours, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, market_type) DO UPDATE SET
                last_sell_at = excluded.last_sell_at,
                exit_reason = excluded.exit_reason,
                exit_kind = excluded.exit_kind,
                pnl_pct = excluded.pnl_pct,
                after_loss = excluded.after_loss,
                risk_exit = excluded.risk_exit,
                cooldown_hours = excluded.cooldown_hours,
                expires_at = excluded.expires_at,
                created_at = excluded.created_at
            """,
            (
                symbol.upper(),
                market_type.lower(),
                sold_at.isoformat(),
                exit_reason,
                kind,
                pnl_value,
                1 if after_loss else 0,
                1 if risk_exit else 0,
                hours,
                expires_at.isoformat(),
                now,
            ),
        )

    row = get_reentry_cooldown(symbol, market_type)
    return row or {}


def get_reentry_cooldown(symbol: str, market_type: str) -> dict | None:
    """Retourne le cooldown actif ou expiré le plus récent d'un symbole."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM reentry_cooldowns
            WHERE symbol = ? AND market_type = ?
            """,
            (symbol.upper(), market_type.lower()),
        ).fetchone()
    result = _row_to_dict(row)
    if result is None:
        return None
    result["after_loss"] = bool(result.get("after_loss"))
    result["risk_exit"] = bool(result.get("risk_exit"))
    return result


def is_reentry_blocked(
    symbol: str,
    market_type: str,
    now: datetime | None = None,
) -> dict | None:
    """Retourne un verdict de blocage si le symbole est encore en cooldown.

    Sinon retourne `None` (achat autorisé). Comportement « fail-open » : toute
    erreur ou cooldown absent laisse passer l'entrée.
    """
    try:
        cooldown = get_reentry_cooldown(symbol, market_type)
    except Exception:
        return None
    if not cooldown:
        return None

    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    expires = _parse_dt(cooldown.get("expires_at"))
    sold_at = _parse_dt(cooldown.get("last_sell_at"))
    if expires is None or sold_at is None:
        return None
    if reference >= expires:
        return None

    gap_hours = (reference - sold_at).total_seconds() / 3600.0
    return {
        "action": "WOULD_BLOCK",
        "symbol": cooldown["symbol"],
        "market_type": cooldown["market_type"],
        "last_sell": cooldown["last_sell_at"],
        "last_ret": cooldown.get("pnl_pct"),
        "gap_hours": round(max(gap_hours, 0.0), 2),
        "window_hours": cooldown.get("cooldown_hours"),
        "after_loss": cooldown.get("after_loss", False),
        "exit_kind": cooldown.get("exit_kind"),
        "risk_exit": cooldown.get("risk_exit", False),
    }


def purge_expired_reentry_cooldowns(now: datetime | None = None) -> int:
    """Supprime les cooldowns expirés et retourne le nombre de lignes effacées."""
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM reentry_cooldowns WHERE expires_at <= ?",
            (reference.replace(microsecond=0).isoformat(),),
        )
        return cursor.rowcount


def clear_reentry_cooldown(symbol: str, market_type: str) -> bool:
    """Supprime manuellement le cooldown d'un symbole."""
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM reentry_cooldowns WHERE symbol = ? AND market_type = ?",
            (symbol.upper(), market_type.lower()),
        )
        return cursor.rowcount > 0


# Initialisation au premier import : la base est prête dès le démarrage de l'app.
init_db()
