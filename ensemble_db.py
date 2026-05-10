"""
Database routing: PostgreSQL when DATABASE_URL is a postgres URL, else SQLite.
SQL helpers adapt placeholders and time functions per dialect.
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import psycopg2
    from psycopg2 import errors as pg_errors
except ImportError:  # pragma: no cover
    psycopg2 = None
    pg_errors = None

_BASE_DIR = Path(__file__).resolve().parent

_RAW_DATABASE_URL = (os.environ.get("DATABASE_URL") or "").strip()


def _parse_sqlite_file_path(url: str) -> str:
    body = url.split(":", 1)[1]
    if body.startswith("///"):
        path_part = body[3:]
    elif body.startswith("//"):
        path_part = body[2:].lstrip("/")
    else:
        path_part = body
    if path_part.startswith("/") and len(path_part) > 1:
        p = Path(path_part)
    else:
        p = _BASE_DIR / path_part
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(p.resolve())


def _resolve_storage() -> tuple[bool, str]:
    """
    Returns (use_postgres, dsn_or_sqlite_path).
    If DATABASE_URL missing → SQLite file default.
    If sqlite: → file path.
    If postgres / postgresql → DSN string (normalized for psycopg2).
    """
    if not _RAW_DATABASE_URL:
        return False, str((_BASE_DIR / "conversations.db").resolve())
    low = _RAW_DATABASE_URL.lower()
    if low.startswith("postgresql://") or low.startswith("postgres://"):
        dsn = _RAW_DATABASE_URL
        if dsn.startswith("postgres://"):
            dsn = "postgresql://" + dsn[len("postgres://") :]
        return True, dsn
    if low.startswith("sqlite:"):
        return False, _parse_sqlite_file_path(_RAW_DATABASE_URL)
    return False, str((_BASE_DIR / "conversations.db").resolve())


USE_POSTGRES, _DSN_OR_SQLITE_PATH = _resolve_storage()
POSTGRES_DSN: Optional[str] = _DSN_OR_SQLITE_PATH if USE_POSTGRES else None
SQLITE_DB_PATH: str = _DSN_OR_SQLITE_PATH if not USE_POSTGRES else str((_BASE_DIR / "conversations.db").resolve())

# Back-compat: code that logs a "path" uses this label on Postgres
DB_LABEL = POSTGRES_DSN.split("@")[-1] if USE_POSTGRES and POSTGRES_DSN else SQLITE_DB_PATH


def connect_db(timeout: Optional[float] = None) -> Any:
    if USE_POSTGRES:
        if psycopg2 is None:
            raise RuntimeError("psycopg2-binary is required when DATABASE_URL is PostgreSQL")
        return psycopg2.connect(POSTGRES_DSN)
    kw: dict[str, Any] = {}
    if timeout is not None:
        kw["timeout"] = timeout
    return sqlite3.connect(SQLITE_DB_PATH, **kw)


def adapt(sql: str) -> str:
    if not USE_POSTGRES:
        return sql
    return sql.replace("?", "%s")


def is_unique_violation(exc: BaseException) -> bool:
    if isinstance(exc, sqlite3.IntegrityError):
        return True
    if pg_errors is not None and isinstance(exc, pg_errors.UniqueViolation):
        return True
    return False


def now_expr_insert() -> str:
    return "NOW()" if USE_POSTGRES else "datetime('now')"


def sql_cast_date(column: str) -> str:
    """Expression usable in WHERE for “calendar day of timestamp/text column”."""
    if USE_POSTGRES:
        return f"CAST({column} AS DATE)"
    return f"date({column})"


def sql_today_predicate(column: str = "created_at") -> str:
    if USE_POSTGRES:
        return f"{sql_cast_date(column)} = CURRENT_DATE"
    return f"date({column}) = date('now')"


def sql_created_after_interval_days(column: str, days: int) -> str:
    if USE_POSTGRES:
        return f"CAST({column} AS TIMESTAMP) > NOW() - INTERVAL '{int(days)} days'"
    return f"{column} > datetime('now', '-{int(days)} day')"


def sql_order_by_datetime_desc(column: str) -> str:
    if USE_POSTGRES:
        return f"CAST({column} AS TIMESTAMP) DESC"
    return f"datetime({column}) DESC"


def sql_date_bucket_expr(column: str = "created_at") -> str:
    """Expression for grouping or comparing by calendar date (PostgreSQL ↔ SQLite)."""
    if USE_POSTGRES:
        return f"CAST({column} AS DATE)"
    return f"date({column})"


def _table_columns_sqlite(conn: sqlite3.Connection, table: str) -> set[str]:
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in c.fetchall()}


def _table_columns_postgres(conn: Any, table: str) -> set[str]:
    c = conn.cursor()
    c.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    )
    return {row[0] for row in c.fetchall()}


def _init_db_sqlite(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            updated_at TEXT,
            title TEXT
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            model TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            session_id TEXT PRIMARY KEY,
            user_name TEXT,
            user_role TEXT,
            projects TEXT,
            preferences TEXT,
            memory_context TEXT,
            uploaded_text TEXT,
            updated_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS ensemble_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question TEXT,
            round1_gpt TEXT,
            round1_gemini TEXT,
            round1_claude TEXT,
            round2_gpt TEXT,
            round2_gemini TEXT,
            round2_claude TEXT,
            final_synthesis TEXT,
            created_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS learning_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            model TEXT,
            feedback_value INTEGER,
            timestamp TEXT
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS trial_usage (
            user_identifier TEXT PRIMARY KEY,
            trial_count INTEGER DEFAULT 0,
            is_pro INTEGER DEFAULT 0
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS rate_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            hit_ts REAL NOT NULL
        )
    """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_session_hit ON rate_limits (session_id, hit_ts)")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL COLLATE NOCASE UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            tier TEXT NOT NULL DEFAULT 'free',
            stripe_customer_id TEXT
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS user_ensemble_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            used_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_ensemble_usage_uid_time ON user_ensemble_usage (user_id, used_at)"
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS model_responses (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            model TEXT NOT NULL,
            content TEXT NOT NULL,
            is_ben_synthesis INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS ben_learning_events (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            source_response_id TEXT,
            event_type TEXT NOT NULL,
            payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_responses_tid_mid ON model_responses (thread_id, message_id)"
    )
    conn.commit()


def _init_db_postgres(conn: Any) -> None:
    c = conn.cursor()
    stmts = [
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            updated_at TEXT,
            title TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            session_id TEXT REFERENCES sessions(session_id),
            model TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS profiles (
            session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
            user_name TEXT,
            user_role TEXT,
            projects TEXT,
            preferences TEXT,
            memory_context TEXT,
            uploaded_text TEXT,
            updated_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ensemble_results (
            id SERIAL PRIMARY KEY,
            session_id TEXT REFERENCES sessions(session_id),
            question TEXT,
            round1_gpt TEXT,
            round1_gemini TEXT,
            round1_claude TEXT,
            round2_gpt TEXT,
            round2_gemini TEXT,
            round2_claude TEXT,
            final_synthesis TEXT,
            created_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS learning_feedback (
            id SERIAL PRIMARY KEY,
            category TEXT,
            model TEXT,
            feedback_value INTEGER,
            timestamp TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS trial_usage (
            user_identifier TEXT PRIMARY KEY,
            trial_count INTEGER DEFAULT 0,
            is_pro INTEGER DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS rate_limits (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            hit_ts DOUBLE PRECISION NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_rate_limits_session_hit ON rate_limits (session_id, hit_ts)",
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            tier VARCHAR(32) NOT NULL DEFAULT 'free',
            stripe_customer_id TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_ensemble_usage (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            used_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_user_ensemble_usage_uid_time ON user_ensemble_usage (user_id, used_at)",
        """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS model_responses (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            model TEXT NOT NULL,
            content TEXT NOT NULL,
            is_ben_synthesis BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ben_learning_events (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            source_response_id TEXT,
            event_type TEXT NOT NULL,
            payload TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_model_responses_tid_mid ON model_responses (thread_id, message_id)",
    ]
    for s in stmts:
        c.execute(s)
    conn.commit()


def init_db_tables() -> None:
    conn = connect_db()
    try:
        if USE_POSTGRES:
            _init_db_postgres(conn)
        else:
            _init_db_sqlite(conn)
    finally:
        conn.close()


def migrate_schema() -> None:
    conn = connect_db()
    try:
        if USE_POSTGRES:
            _migrate_postgres(conn)
        else:
            _migrate_sqlite(conn)
    finally:
        conn.close()


def _migrate_sqlite(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    cols = _table_columns_sqlite(conn, "profiles")
    if "uploaded_text" not in cols:
        try:
            c.execute("ALTER TABLE profiles ADD COLUMN uploaded_text TEXT")
        except sqlite3.OperationalError:
            pass
    if "active_tools" not in cols:
        try:
            c.execute("ALTER TABLE profiles ADD COLUMN active_tools TEXT")
        except sqlite3.OperationalError:
            pass

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            consensus_pct REAL NOT NULL DEFAULT 0,
            cost_usd REAL DEFAULT 0,
            baseline_cost_usd REAL DEFAULT 0,
            savings_usd REAL DEFAULT 0,
            mode TEXT,
            openai_usd REAL DEFAULT 0,
            gemini_usd REAL DEFAULT 0,
            anthropic_usd REAL DEFAULT 0
        )
    """
    )
    tcols = _table_columns_sqlite(conn, "telemetry_runs")
    for col_sql in (
        "ALTER TABLE telemetry_runs ADD COLUMN ensemble_wall_ms REAL",
        "ALTER TABLE telemetry_runs ADD COLUMN r1_parallel_wall_ms REAL",
        "ALTER TABLE telemetry_runs ADD COLUMN r1_gpt_ms REAL",
        "ALTER TABLE telemetry_runs ADD COLUMN r1_gemini_ms REAL",
        "ALTER TABLE telemetry_runs ADD COLUMN r1_claude_ms REAL",
        "ALTER TABLE telemetry_runs ADD COLUMN parallel_efficiency_pct REAL",
        "ALTER TABLE telemetry_runs ADD COLUMN routing_tier TEXT",
        "ALTER TABLE telemetry_runs ADD COLUMN streaming_active INTEGER DEFAULT 1",
        "ALTER TABLE telemetry_runs ADD COLUMN fast_first_active INTEGER DEFAULT 1",
        "ALTER TABLE telemetry_runs ADD COLUMN first_token_ms REAL",
        "ALTER TABLE telemetry_runs ADD COLUMN question_len INTEGER",
    ):
        col_name = col_sql.split("ADD COLUMN ")[1].split(" ")[0]
        if col_name not in tcols:
            try:
                c.execute(col_sql)
            except sqlite3.OperationalError:
                pass
            tcols.add(col_name)

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS rate_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            hit_ts REAL NOT NULL
        )
    """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_session_hit ON rate_limits (session_id, hit_ts)")

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL COLLATE NOCASE UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            tier TEXT NOT NULL DEFAULT 'free',
            stripe_customer_id TEXT
        )
    """
    )

    ucols = _table_columns_sqlite(conn, "users")
    if "tier" not in ucols:
        try:
            c.execute("ALTER TABLE users ADD COLUMN tier TEXT NOT NULL DEFAULT 'free'")
        except sqlite3.OperationalError:
            pass
        ucols.add("tier")
    if "stripe_customer_id" not in ucols:
        try:
            c.execute("ALTER TABLE users ADD COLUMN stripe_customer_id TEXT")
        except sqlite3.OperationalError:
            pass

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS user_ensemble_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            used_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_ensemble_usage_uid_time ON user_ensemble_usage (user_id, used_at)"
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS self_heals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            telemetry_run_id INTEGER NOT NULL UNIQUE,
            consensus_pct REAL,
            rationale TEXT,
            instruction_addendum TEXT
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS model_responses (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            model TEXT NOT NULL,
            content TEXT NOT NULL,
            is_ben_synthesis INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS ben_learning_events (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            source_response_id TEXT,
            event_type TEXT NOT NULL,
            payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_responses_tid_mid ON model_responses (thread_id, message_id)"
    )
    conn.commit()


def _migrate_postgres(conn: Any) -> None:
    c = conn.cursor()
    cols = _table_columns_postgres(conn, "profiles")
    if "uploaded_text" not in cols:
        c.execute("ALTER TABLE profiles ADD COLUMN uploaded_text TEXT")
    if "active_tools" not in cols:
        c.execute("ALTER TABLE profiles ADD COLUMN active_tools TEXT")

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_runs (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            consensus_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
            cost_usd DOUBLE PRECISION DEFAULT 0,
            baseline_cost_usd DOUBLE PRECISION DEFAULT 0,
            savings_usd DOUBLE PRECISION DEFAULT 0,
            mode TEXT,
            openai_usd DOUBLE PRECISION DEFAULT 0,
            gemini_usd DOUBLE PRECISION DEFAULT 0,
            anthropic_usd DOUBLE PRECISION DEFAULT 0
        )
        """
    )
    tcols = _table_columns_postgres(conn, "telemetry_runs")
    for col_name, col_type in (
        ("ensemble_wall_ms", "DOUBLE PRECISION"),
        ("r1_parallel_wall_ms", "DOUBLE PRECISION"),
        ("r1_gpt_ms", "DOUBLE PRECISION"),
        ("r1_gemini_ms", "DOUBLE PRECISION"),
        ("r1_claude_ms", "DOUBLE PRECISION"),
        ("parallel_efficiency_pct", "DOUBLE PRECISION"),
        ("routing_tier", "TEXT"),
        ("streaming_active", "INTEGER DEFAULT 1"),
        ("fast_first_active", "INTEGER DEFAULT 1"),
        ("first_token_ms", "DOUBLE PRECISION"),
        ("question_len", "INTEGER"),
    ):
        if col_name not in tcols:
            c.execute(f"ALTER TABLE telemetry_runs ADD COLUMN {col_name} {col_type}")

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS self_heals (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            telemetry_run_id INTEGER NOT NULL UNIQUE,
            consensus_pct DOUBLE PRECISION,
            rationale TEXT,
            instruction_addendum TEXT
        )
        """
    )

    ucols = _table_columns_postgres(conn, "users")
    if "tier" not in ucols:
        c.execute("ALTER TABLE users ADD COLUMN tier VARCHAR(32) NOT NULL DEFAULT 'free'")
        ucols.add("tier")
    if "stripe_customer_id" not in ucols:
        c.execute("ALTER TABLE users ADD COLUMN stripe_customer_id TEXT")
        ucols.add("stripe_customer_id")

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS user_ensemble_usage (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            used_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_ensemble_usage_uid_time ON user_ensemble_usage (user_id, used_at)"
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS model_responses (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            model TEXT NOT NULL,
            content TEXT NOT NULL,
            is_ben_synthesis BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS ben_learning_events (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            source_response_id TEXT,
            event_type TEXT NOT NULL,
            payload TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_responses_tid_mid ON model_responses (thread_id, message_id)"
    )

    conn.commit()


# --- BEN v3: thread model responses (append-only raw answers + BEN synthesis metadata) ---


def create_thread(title: str | None = None) -> str:
    """Create a thread row; returns new thread id."""
    tid = str(uuid.uuid4())
    conn = connect_db()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute(
            adapt(
                "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, NOW(), NOW())"
            ),
            (tid, title),
        )
    else:
        c.execute(
            adapt(
                "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, datetime('now'), datetime('now'))"
            ),
            (tid, title),
        )
    conn.commit()
    conn.close()
    return tid


def ensure_thread_row(thread_id: str, title: str | None = None) -> None:
    """Insert a threads row if missing (session may predate the threads table)."""
    conn = connect_db()
    c = conn.cursor()
    c.execute(adapt("SELECT 1 FROM threads WHERE id = ?"), (thread_id,))
    if c.fetchone():
        conn.close()
        return
    if USE_POSTGRES:
        c.execute(
            adapt(
                "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, NOW(), NOW())"
            ),
            (thread_id, title),
        )
    else:
        c.execute(
            adapt(
                "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, datetime('now'), datetime('now'))"
            ),
            (thread_id, title),
        )
    conn.commit()
    conn.close()


def touch_thread(thread_id: str) -> None:
    conn = connect_db()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute(adapt("UPDATE threads SET updated_at = NOW() WHERE id = ?"), (thread_id,))
    else:
        c.execute(
            adapt("UPDATE threads SET updated_at = datetime('now') WHERE id = ?"),
            (thread_id,),
        )
    conn.commit()
    conn.close()


def save_model_response(
    thread_id: str,
    message_id: str,
    model: str,
    content: str,
    *,
    is_ben_synthesis: bool = False,
) -> str:
    """
    Raw model responses are append-only.
    Never UPDATE or DELETE existing model outputs.
    """
    rid = str(uuid.uuid4())
    conn = connect_db()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute(
            adapt(
                "INSERT INTO model_responses (id, thread_id, message_id, model, content, is_ben_synthesis) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            ),
            (rid, thread_id, message_id, model, content, is_ben_synthesis),
        )
    else:
        c.execute(
            adapt(
                "INSERT INTO model_responses (id, thread_id, message_id, model, content, is_ben_synthesis) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            ),
            (rid, thread_id, message_id, model, content, 1 if is_ben_synthesis else 0),
        )
    conn.commit()
    conn.close()
    return rid


def load_model_responses(thread_id: str, message_id: str) -> list[dict[str, Any]]:
    conn = connect_db()
    c = conn.cursor()
    c.execute(
        adapt(
            "SELECT id, model, content, is_ben_synthesis, created_at FROM model_responses "
            "WHERE thread_id = ? AND message_id = ? ORDER BY created_at ASC"
        ),
        (thread_id, message_id),
    )
    rows = c.fetchall()
    conn.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "model": r[1],
                "content": r[2],
                "is_ben_synthesis": bool(r[3]),
                "created_at": r[4],
            }
        )
    return out


def load_last_user_message(thread_id: str) -> str:
    conn = connect_db()
    c = conn.cursor()
    c.execute(
        adapt(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'user' "
            "ORDER BY id DESC LIMIT 1"
        ),
        (thread_id,),
    )
    row = c.fetchone()
    conn.close()
    return (row[0] or "") if row else ""


def load_last_user_message_id(thread_id: str) -> str:
    conn = connect_db()
    c = conn.cursor()
    c.execute(
        adapt(
            "SELECT id FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1"
        ),
        (thread_id,),
    )
    row = c.fetchone()
    conn.close()
    return str(row[0]) if row else ""


def save_learning_event(
    thread_id: str,
    event_type: str,
    source_response_id: Optional[str] = None,
    payload: Optional[str] = None,
) -> str:
    eid = str(uuid.uuid4())
    conn = connect_db()
    c = conn.cursor()
    c.execute(
        adapt(
            "INSERT INTO ben_learning_events (id, thread_id, source_response_id, event_type, payload) "
            "VALUES (?, ?, ?, ?, ?)"
        ),
        (eid, thread_id, source_response_id, event_type, payload),
    )
    conn.commit()
    conn.close()
    return eid


def insert_message_return_id(session_id: str, model: str, role: str, content: str) -> str:
    conn = connect_db()
    c = conn.cursor()
    ts = datetime.now().isoformat()
    if USE_POSTGRES:
        c.execute(
            adapt(
                "INSERT INTO messages (session_id, model, role, content, timestamp) "
                "VALUES (?, ?, ?, ?, ?) RETURNING id"
            ),
            (session_id, model, role, content, ts),
        )
        row = c.fetchone()
        mid = str(row[0]) if row else ""
    else:
        c.execute(
            adapt(
                "INSERT INTO messages (session_id, model, role, content, timestamp) VALUES (?, ?, ?, ?, ?)"
            ),
            (session_id, model, role, content, ts),
        )
        mid = str(c.lastrowid)
    conn.commit()
    conn.close()
    return mid
