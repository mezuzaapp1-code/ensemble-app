import os
import uuid
import json
import asyncio
import re
import subprocess
import tempfile
import time
import io
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional
from pathlib import Path

from dotenv import load_dotenv

from ensemble_db import (
    USE_POSTGRES,
    adapt as sqlq,
    connect_db,
    is_unique_violation,
    init_db_tables,
    migrate_schema as migrate_db_schema,
    now_expr_insert,
    SQLITE_DB_PATH,
)
import bcrypt
import jwt
from fastapi import Depends, FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx
from google import genai
try:
    from openai import AsyncOpenAI
    OPENAI_IMPORT_ERROR = None
except Exception as _openai_import_exc:
    AsyncOpenAI = None
    OPENAI_IMPORT_ERROR = _openai_import_exc
from anthropic import AsyncAnthropic
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import uvicorn

_BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = _BASE_DIR / ".env"
DOTENV_LOADED = load_dotenv(dotenv_path=DOTENV_PATH, override=False)


# Local SQLite file path when not on Postgres (for logging / founder display)
DB_PATH = SQLITE_DB_PATH
INDEX_HTML = _BASE_DIR / "index.html"
UPGRADE_HTML = _BASE_DIR / "upgrade.html"

# Canonical IDs for probes, streaming, and fallbacks (no legacy haiku / 1.5-flash).
MODEL_REGISTRY = {
    "openai_default": "gpt-4o-mini",
    "gemini_default": "gemini-2.5-flash",
    "gemini_fast": "gemini-2.5-flash",
    "gemini_fallback_chain": (),
    "claude_primary": "claude-sonnet-4-5",
    "claude_fallback_chain": (
        "claude-sonnet-4-5",
    ),
}

CLAUDE_MODEL = MODEL_REGISTRY["claude_primary"]
GEMINI_MODEL = MODEL_REGISTRY["gemini_default"]
GEMINI_FAST_MODEL = MODEL_REGISTRY["gemini_fast"]
OPENAI_DEFAULT_MODEL = MODEL_REGISTRY["openai_default"]
CLAUDE_FALLBACK_MODELS = [m for m in MODEL_REGISTRY["claude_fallback_chain"] if m != CLAUDE_MODEL]
GEMINI_FALLBACK_MODELS = list(MODEL_REGISTRY["gemini_fallback_chain"])

FREE_USER_LIMIT = max(0, int(os.getenv("FREE_USER_LIMIT", "5") or "5"))

_TIER_ROUTING_CTX: ContextVar[Optional["TierRouting"]] = ContextVar("_tier_routing_ctx", default=None)


@dataclass(frozen=True)
class TierRouting:
    tier: str
    openai_main: str
    openai_fast: str
    gemini_main: str
    claude_main: Optional[str]
    claude_fallbacks: tuple[str, ...]
    ben_use_claude: bool


def routing_for_db_tier(db_tier: Optional[str]) -> TierRouting:
    t = (db_tier or "free").strip().lower()
    if t == "pro":
        return TierRouting(
            tier="pro",
            openai_main="gpt-4o",
            openai_fast="gpt-4o",
            gemini_main="gemini-1.5-pro",
            claude_main="claude-3-5-sonnet-20241022",
            claude_fallbacks=("claude-3-5-sonnet-20241022",),
            ben_use_claude=True,
        )
    return TierRouting(
        tier="free",
        openai_main="gpt-4o-mini",
        openai_fast="gpt-4o-mini",
        gemini_main="gemini-1.5-flash",
        claude_main=None,
        claude_fallbacks=(),
        ben_use_claude=False,
    )


def current_tier_routing() -> Optional[TierRouting]:
    return _TIER_ROUTING_CTX.get()


def effective_openai_default(model_key_hint: Optional[str] = None) -> str:
    tr = current_tier_routing()
    if tr:
        if model_key_hint == "gpt-fast":
            return tr.openai_fast
        return tr.openai_main
    return OPENAI_DEFAULT_MODEL


def effective_gemini_default() -> str:
    tr = current_tier_routing()
    return tr.gemini_main if tr else GEMINI_FAST_MODEL


def effective_claude_primary() -> Optional[str]:
    tr = current_tier_routing()
    if tr:
        return tr.claude_main
    return CLAUDE_MODEL


def effective_claude_fallbacks_for_call() -> tuple[str, ...]:
    tr = current_tier_routing()
    if tr:
        return tr.claude_fallbacks
    return tuple(CLAUDE_FALLBACK_MODELS)


def _gemini_candidate_models(primary: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in (primary, *GEMINI_FALLBACK_MODELS):
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _create_openai_client():
    if AsyncOpenAI is None:
        print(f"[startup] OpenAI SDK import failed; OpenAI disabled: {OPENAI_IMPORT_ERROR}")
        return None
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return None
    try:
        return AsyncOpenAI(api_key=key)
    except Exception as e:
        print(f"[startup] OpenAI client initialization failed; OpenAI disabled: {e}")
        return None


def _create_gemini_client():
    key = (os.getenv("GEMINI_KEY") or "").strip()
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


def _create_anthropic_client():
    key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return None
    try:
        return AsyncAnthropic(api_key=key)
    except Exception:
        return None


gemini_client = _create_gemini_client()
openai_client = _create_openai_client()
claude_client = _create_anthropic_client()

app = FastAPI()

STRIPE_CHECKOUT_URL = os.getenv("STRIPE_CHECKOUT_URL", "https://checkout.stripe.com/pay/placeholder")

# Developer bypass (trial limits, etc.) — off in production; local: ENSEMBLE_DEV_MODE=1
DEV_MODE = os.getenv("ENSEMBLE_DEV_MODE", "").strip().lower() in ("1", "true", "yes", "on")

# ========================
# DATABASE INITIALIZATION (ensemble_db: SQLite or PostgreSQL from DATABASE_URL)
# ========================

JWT_SESSION_DAYS = max(1, int(os.getenv("JWT_SESSION_DAYS", "1")))
JWT_REMEMBER_ME_DAYS = max(1, int(os.getenv("JWT_REMEMBER_ME_DAYS", "7")))
auth_scheme = HTTPBearer(auto_error=False)


def _jwt_secret() -> str:
    s = (os.getenv("JWT_SECRET") or "").strip()
    if not s:
        print(
            "[auth] JWT_SECRET is not set — using insecure development default "
            "(set JWT_SECRET for production)."
        )
        # HMAC SHA-256 JWT requires key length recommendation ≥ 32 octets (RFC 7518).
        s = "ensemble-development-jwt-signing-secret-min-length-thirty-two"
    return s


def normalize_account_email(raw: str) -> str:
    em = (raw or "").strip().lower()
    if len(em) < 3:
        raise HTTPException(status_code=422, detail="Invalid email")
    if "@" not in em:
        raise HTTPException(status_code=422, detail="Invalid email")
    _, domain = em.rsplit("@", 1)
    if "." not in domain:
        raise HTTPException(status_code=422, detail="Invalid email")
    return em


def create_access_token(user_id: int, *, remember_me: bool = False) -> str:
    now = datetime.now(timezone.utc)
    days = JWT_REMEMBER_ME_DAYS if remember_me else JWT_SESSION_DAYS
    exp = now + timedelta(days=days)
    token = jwt.encode(
        {"sub": str(user_id), "iat": now, "exp": exp, "rm": bool(remember_me)},
        _jwt_secret(),
        algorithm="HS256",
    )
    return token if isinstance(token, str) else token.decode("utf-8")


def require_login(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(auth_scheme),
    ],
) -> int:
    """Require ``Authorization: Bearer <jwt>``. Cookies are not used; missing/invalid token → 401."""
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(status_code=401, detail="Please login")
    token = credentials.credentials.strip()
    if not token:
        raise HTTPException(status_code=401, detail="Please login")
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
        uid = int(payload.get("sub"))
    except (jwt.PyJWTError, TypeError, ValueError) as exc:
        if os.getenv("ENSEMBLE_DEBUG_AUTH", "").strip().lower() in ("1", "true", "yes", "on"):
            print(f"[auth] JWT decode failed ({type(exc).__name__}): {exc}")
        raise HTTPException(status_code=401, detail="Please login")

    conn = connect_db()
    c = conn.cursor()
    xe(c, "SELECT id FROM users WHERE id = ?", (uid,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Please login")
    return uid


ENSEMBLE_RATE_LIMIT_MSG = (
    "You've reached your limit. \n"
    "   Upgrade to Pro for unlimited access."
)


def enforce_ensemble_rate_limit(session_id: str, is_pro: bool) -> None:
    """
    Per-session_id limits (SQLite sliding window via hit_ts timestamps).
    Free: 10 requests/min, 100/day. Pro: 60/min, unlimited per day.
    Raises HTTPException 429 when exceeded; inserts one row when allowed.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return
    conn = connect_db(timeout=10.0)
    try:
        if not USE_POSTGRES:
            conn.execute("PRAGMA busy_timeout = 6000")
        now = time.time()
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute("BEGIN")
        else:
            conn.execute("BEGIN IMMEDIATE")
        cutoff_prune = now - (48 * 3600)
        xe(cur, "DELETE FROM rate_limits WHERE hit_ts < ?", (cutoff_prune,))
        minute_ago = now - 60.0
        day_ago = now - (24 * 3600)
        per_min_cap = 60 if is_pro else 10

        xe(
            cur,
            "SELECT COUNT(*) FROM rate_limits WHERE session_id = ? AND hit_ts > ?",
            (sid, minute_ago),
        )
        n_min = int(cur.fetchone()[0])
        if n_min >= per_min_cap:
            conn.rollback()
            raise HTTPException(status_code=429, detail=ENSEMBLE_RATE_LIMIT_MSG)

        if not is_pro:
            xe(
                cur,
                "SELECT COUNT(*) FROM rate_limits WHERE session_id = ? AND hit_ts > ?",
                (sid, day_ago),
            )
            if int(cur.fetchone()[0]) >= 100:
                conn.rollback()
                raise HTTPException(status_code=429, detail=ENSEMBLE_RATE_LIMIT_MSG)

        xe(cur, "INSERT INTO rate_limits (session_id, hit_ts) VALUES (?, ?)", (sid, now))
        conn.commit()
    except HTTPException:
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def consensus_agreement_pct(consensus_data) -> float:
    """Share of consensus rows marked high-confidence (proxy for model agreement)."""
    if not consensus_data:
        return 0.0
    n = 0
    high = 0
    for row in consensus_data:
        if not isinstance(row, dict):
            continue
        n += 1
        st = str(row.get("status", "")).upper()
        if "HIGH" in st:
            high += 1
    return round(100.0 * high / n, 1) if n else 0.0


DEFAULT_ACTIVE_TOOLS_JSON = json.dumps(["gpt", "gemini", "claude"])


def normalize_active_tools_list(raw: list[str] | None) -> list[str]:
    order_idx = {"gpt": 0, "gemini": 1, "claude": 2}
    if not raw:
        return json.loads(DEFAULT_ACTIVE_TOOLS_JSON)
    xs: list[str] = []
    for k in raw:
        kk = str(k).strip().lower()
        if kk in order_idx:
            xs.append(kk)
    out = sorted(set(xs), key=lambda x: order_idx[x])
    if not out:
        return ["gpt"]
    return out


def routing_tier_label(token_saver_mode: str, web_search: bool) -> str:
    if token_saver_mode == "ECONOMY":
        return "Economy"
    if web_search:
        return "Premium"
    return "Standard"


SYSTEM_INSTRUCTIONS_FILE = _BASE_DIR / "system_instructions.txt"

BEN_SUPREME_JUDGE_SYSTEM_BASE = """
Role: You are BEN, the Supreme Judge of an Ensemble Intelligence system. 
Mission: Synthesize multiple AI sources into a single, verified response.

STRICT LANGUAGE: Respond in English only. Every heading, bullet, label, quoted term, and body paragraph must be in English.

JUDGMENT FRAMEWORK:
1. Consensus: Prioritize facts agreed by all 3 models (High Confidence).
2. Hierarchy: User Files > Cross-Model Consensus > Individual Insights.
3. Conflict Management: Flag contradictions as "CONFLICT ALERT". Never guess.

OUTPUT STRUCTURE:
## TL;DR
[One sentence concise answer]
## Unified Answer
[Merged response. Bold = consensus points]
## Trust Map
- ✅ High-confidence consensus across available models: [point]
- ⚠️ Only one model said: [point]
- ❌ CONFLICT: Model X says Y, Model Z says W
## Delta Insights
[Unique points each model contributed]
## Next Action
[Strictly under 10 words - Actionable button text]
"""


def _ben_auto_learned_suffix() -> str:
    p = SYSTEM_INSTRUCTIONS_FILE
    if not p.is_file():
        return ""
    try:
        raw = p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not raw:
        return ""
    tail = raw[-8000:] if len(raw) > 8000 else raw
    return (
        "\n\n--- LEARNED INSTRUCTIONS (auto-updated when consensus is low) ---\n"
        + tail
        + "\n"
    )


def ben_supreme_judge_system_prompt() -> str:
    return BEN_SUPREME_JUDGE_SYSTEM_BASE.strip() + _ben_auto_learned_suffix()


def _truncate_audit_text(label: str, text: str, max_len: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return f"{label}:\n{s}" if s else f"{label}:\n[empty]"
    return f"{label}:\n{s[:max_len]} …[truncated]"


def _parse_analyzer_json(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _append_system_instructions_autolearn(blurb: str) -> None:
    iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    chunk = f"\n\n### [{iso}] Consensus auto-learn (agreement below 50%)\n{blurb.strip()}\n"
    SYSTEM_INSTRUCTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYSTEM_INSTRUCTIONS_FILE, "a", encoding="utf-8") as fh:
        fh.write(chunk)


def _telemetry_self_heal_exists(telemetry_run_id: int) -> bool:
    try:
        conn = connect_db()
        c = conn.cursor()
        xe(c, "SELECT 1 FROM self_heals WHERE telemetry_run_id = ?", (telemetry_run_id,))
        row = c.fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


def _insert_self_heal(
    telemetry_run_id: int,
    consensus_pct: float,
    rationale: str,
    instruction_addendum: str,
) -> None:
    conn = connect_db()
    cu = conn.cursor()
    row = (telemetry_run_id, consensus_pct, rationale[:1200], instruction_addendum[:2000])
    if USE_POSTGRES:
        xe(
            cu,
            """
            INSERT INTO self_heals
            (telemetry_run_id, consensus_pct, rationale, instruction_addendum)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (telemetry_run_id) DO NOTHING
            """,
            row,
        )
    else:
        xe(
            cu,
            """
            INSERT OR IGNORE INTO self_heals
            (telemetry_run_id, consensus_pct, rationale, instruction_addendum)
            VALUES (?, ?, ?, ?)
            """,
            row,
        )
    conn.commit()
    conn.close()


def _analyzer_model_key() -> str:
    k = (os.getenv("ENSEMBLE_ANALYZER_MODEL") or "gpt-fast").strip()
    return k if k else "gpt-fast"


async def _run_consensus_analyzer_autofix(
    *,
    telemetry_run_id: int,
    consensus_pct: float,
    session_id: str,
    question: str,
    round1: dict,
    round2: dict,
    consensus_data: list,
) -> None:
    if consensus_pct >= 50.0 or telemetry_run_id <= 0:
        return
    if _telemetry_self_heal_exists(telemetry_run_id):
        return
    r1_preview = "\n".join(
        [
            _truncate_audit_text("gpt / MODEL A Round1", str(round1.get("gpt") or ""), 2200),
            _truncate_audit_text("gemini / MODEL B Round1", str(round1.get("gemini") or ""), 2200),
            _truncate_audit_text("claude / MODEL C Round1", str(round1.get("claude") or ""), 2200),
        ]
    )
    r2_preview = "\n".join(
        [
            _truncate_audit_text("gpt Round2 critique", str(round2.get("gpt") or ""), 1800),
            _truncate_audit_text("gemini Round2 critique", str(round2.get("gemini") or ""), 1800),
            _truncate_audit_text("claude Round2 critique", str(round2.get("claude") or ""), 1800),
        ]
    )
    cq = _truncate_audit_text("User question", question, 2000)
    try:
        consensus_blob = json.dumps(consensus_data, ensure_ascii=False)[:6500]
    except TypeError:
        consensus_blob = "[]"
    prompt = f"""You are the Ensemble Analyzer (dedicated reviewer). Model agreement (consensus rate) was {consensus_pct:.1f}% (< 50%).

Produce ONE actionable line of guidance the Supreme Judge should follow next time a similar disagreement appears.
Do not quote secrets. Be specific about how to reconcile or rank conflicting claims.

{cq}

{r1_preview}

{r2_preview}

Consensus rows (may be abbreviated JSON):
{consensus_blob}

Reply with ONLY valid JSON:
{{"instruction_addendum": "<=500 chars single line rule for BEN>", "rationale_one_line": "<=200 chars plain English>"}}
"""
    mk = _analyzer_model_key()
    try:
        raw = await ask_model_timed(mk, prompt, session_id or None, timeout_sec=90.0)
    except Exception as ex:
        print(f"[autofix] analyzer model call failed: {ex}")
        return
    if not raw or "Error:" in raw or _is_budget_error(raw):
        print(f"[autofix] analyzer unusable response: {_ascii_preview(str(raw))}")
        return
    parsed = _parse_analyzer_json(raw)
    if not isinstance(parsed, dict):
        print("[autofix] analyzer JSON parse failed")
        return
    instr = str(parsed.get("instruction_addendum") or "").strip().replace("\n", " ")
    rationale = str(parsed.get("rationale_one_line") or "").strip().replace("\n", " ")
    if len(instr) < 12:
        print("[autofix] analyzer returned empty instruction")
        return
    instr = instr[:500]
    rationale = rationale[:200] if rationale else "Auto-learn applied from conflicting model outputs."
    _append_system_instructions_autolearn(instr)
    _insert_self_heal(telemetry_run_id, consensus_pct, rationale, instr)
    print(f"[autofix] telemetry#{telemetry_run_id} learned instruction ({len(instr)} chars)")


def _schedule_consensus_analyzer_if_needed(
    pct: float,
    telemetry_run_id: int,
    *,
    session_id: str,
    question: str,
    round1: dict,
    round2: dict,
    consensus_data,
) -> None:
    if pct >= 50.0 or telemetry_run_id <= 0:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    cd = consensus_data if isinstance(consensus_data, list) else []
    asyncio.create_task(
        _run_consensus_analyzer_autofix(
            telemetry_run_id=telemetry_run_id,
            consensus_pct=float(pct),
            session_id=session_id or "",
            question=str(question or ""),
            round1=round1 if isinstance(round1, dict) else {},
            round2=round2 if isinstance(round2, dict) else {},
            consensus_data=cd,
        )
    )


def parallel_eff_ratio(lat_ms: dict, parallel_wall_ms: float) -> float:
    vals = [float(v) for v in lat_ms.values() if isinstance(v, (int, float)) and float(v) > 0]
    if not vals or parallel_wall_ms <= 0:
        return 0.0
    s = sum(vals)
    return round(min(100.0, (s / parallel_wall_ms) * (100.0 / len(vals))), 2)


def get_profile_active_tool_set(session_id: str) -> set[str]:
    p = get_profile(session_id)
    raw = DEFAULT_ACTIVE_TOOLS_JSON
    if p and p.get("active_tools"):
        raw = str(p["active_tools"])
    try:
        lst = json.loads(raw)
    except Exception:
        lst = json.loads(DEFAULT_ACTIVE_TOOLS_JSON)
    return set(normalize_active_tools_list(lst if isinstance(lst, list) else []))


def record_telemetry_run(
    consensus_data,
    session_cost: dict,
    mode: str,
    routing_tier: str,
    *,
    perf: Optional[dict] = None,
) -> Optional[tuple[float, int]]:
    """Persist ensemble run telemetry. Returns ``(consensus_pct, telemetry_row_id)`` on success."""
    perf = perf or {}
    try:
        pct = consensus_agreement_pct(consensus_data)
        pmc = session_cost.get("per_model_cost_usd") or {}
        openai_usd = float(pmc.get("gpt") or 0)
        gemini_usd = float(pmc.get("gemini") or 0)
        anthropic_usd = float((pmc.get("claude") or 0) + (pmc.get("ben") or 0))
        cost_usd = float(session_cost.get("estimated_cost_usd") or 0)
        baseline_usd = float(session_cost.get("baseline_full_cost_usd") or 0)
        savings_usd = float(session_cost.get("usd_saved_vs_full") or 0)

        ensemble_wall_ms = perf.get("ensemble_wall_ms")
        r1_wall_ms = perf.get("r1_parallel_wall_ms")
        r1g = perf.get("r1_gpt_ms")
        r1gem = perf.get("r1_gemini_ms")
        r1cl = perf.get("r1_claude_ms")
        parallel_pct = perf.get("parallel_efficiency_pct")
        stream_act = int(1 if perf.get("streaming_active", True) else 0)
        fast_first = int(1 if perf.get("fast_first_active", True) else 0)
        ftok = perf.get("first_token_ms")
        qlen = perf.get("question_len")

        conn = connect_db()
        cu = conn.cursor()
        ins = (
            """
            INSERT INTO telemetry_runs (
                created_at, consensus_pct, cost_usd, baseline_cost_usd, savings_usd, mode,
                openai_usd, gemini_usd, anthropic_usd,
                ensemble_wall_ms, r1_parallel_wall_ms, r1_gpt_ms, r1_gemini_ms, r1_claude_ms,
                parallel_efficiency_pct, routing_tier, streaming_active, fast_first_active,
                first_token_ms, question_len
            ) VALUES (
            """
            + now_expr_insert()
            + """, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        )
        if USE_POSTGRES:
            ins += " RETURNING id"
        params = (
            pct,
            cost_usd,
            baseline_usd,
            savings_usd,
            mode,
            openai_usd,
            gemini_usd,
            anthropic_usd,
            ensemble_wall_ms,
            r1_wall_ms,
            r1g,
            r1gem,
            r1cl,
            parallel_pct,
            routing_tier,
            stream_act,
            fast_first,
            ftok,
            qlen,
        )
        cu.execute(sqlq(ins), params)
        if USE_POSTGRES:
            rid = int((cu.fetchone() or (0,))[0])
        else:
            rid = int(cu.lastrowid or 0)
        conn.commit()
        conn.close()
        return (float(pct), rid)
    except Exception as ex:
        print(f"[telemetry] record failed: {ex}")
        return None


init_db_tables()
migrate_db_schema()

# Sentinel: omit param to preserve DB value when calling save_profile
_PROFILE_KEEP = object()

# ========================
# DATA MODELS
# ========================

class Message(BaseModel):
    model: str
    content: str

class AskRequest(BaseModel):
    session_id: str
    model: str
    message: str

class RunRequest(BaseModel):
    session_id: str
    question: str
    web_search: Optional[bool] = False

class TestAIRequest(BaseModel):
    prompt: str = "Reply with one short sentence: backend connectivity test passed."


class SimilarAIToolsRequest(BaseModel):
    query: str


class CodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"

class NewSessionRequest(BaseModel):
    title: str = "New Conversation"
    user_name: Optional[str] = None
    user_role: Optional[str] = None
    projects: Optional[str] = None
    preferences: Optional[str] = None
    memory_context: Optional[str] = None

class ProfileRequest(BaseModel):
    user_name: Optional[str] = None
    user_role: Optional[str] = None
    projects: Optional[str] = None
    preferences: Optional[str] = None
    memory_context: Optional[str] = None


class ActiveToolsRequest(BaseModel):
    """Subset of ensemble analyst keys wired to GPT / Gemini / Claude."""
    active_tools: list[str]


class AuthRegisterBody(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=8)
    remember_me: bool = False


class AuthLoginBody(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str
    remember_me: bool = False


class FeedbackRequest(BaseModel):
    session_id: str
    model: str
    feedback_value: int
    category: str


def xe(cur: Any, sql: str, params: tuple | list = ()) -> Any:
    """Run SQL using ``?`` placeholders; adapted to ``%s`` on PostgreSQL."""
    return cur.execute(sqlq(sql), params)


def fetch_user_account(uid: int) -> tuple[str, Optional[str]]:
    conn = connect_db()
    c = conn.cursor()
    xe(c, "SELECT tier, stripe_customer_id FROM users WHERE id = ?", (uid,))
    row = c.fetchone()
    conn.close()
    if not row:
        return "free", None
    tier = (row[0] or "free").strip().lower()
    raw_sid = row[1]
    sid = str(raw_sid).strip() if raw_sid else None
    return tier, sid


def count_user_ensemble_usage_24h(uid: int) -> int:
    conn = connect_db()
    c = conn.cursor()
    if USE_POSTGRES:
        xe(
            c,
            "SELECT COUNT(*) FROM user_ensemble_usage WHERE user_id = ? "
            "AND used_at > NOW() - INTERVAL '24 hours'",
            (uid,),
        )
    else:
        xe(
            c,
            "SELECT COUNT(*) FROM user_ensemble_usage WHERE user_id = ? "
            "AND datetime(used_at) > datetime('now', '-24 hours')",
            (uid,),
        )
    n = int(c.fetchone()[0])
    conn.close()
    return n


def record_user_ensemble_message(uid: int) -> None:
    conn = connect_db()
    c = conn.cursor()
    try:
        xe(c, "INSERT INTO user_ensemble_usage (user_id) VALUES (?)", (uid,))
        conn.commit()
    finally:
        conn.close()


"""Shared DB row for ensemble usage when JWT is not required (FK target for ``user_ensemble_usage``)."""
_ENSEMBLE_GUEST_EMAIL = "__ensemble_guest__@system.local"


def get_or_create_guest_ensemble_user_id() -> int:
    conn = connect_db()
    c = conn.cursor()
    xe(c, "SELECT id FROM users WHERE email = ?", (_ENSEMBLE_GUEST_EMAIL,))
    row = c.fetchone()
    if row:
        uid = int(row[0])
        conn.close()
        return uid
    pw = bcrypt.hashpw(b"__ensemble_guest_not_for_login__", bcrypt.gensalt()).decode("ascii")
    try:
        ins = "INSERT INTO users (email, password_hash, tier) VALUES (?, ?, 'free')"
        if USE_POSTGRES:
            ins += " RETURNING id"
        xe(c, ins, (_ENSEMBLE_GUEST_EMAIL, pw))
        conn.commit()
        if USE_POSTGRES:
            uid = int(c.fetchone()[0])
        else:
            uid = int(c.lastrowid)
        conn.close()
        return uid
    except Exception:
        conn.rollback()
        conn.close()
        conn = connect_db()
        c = conn.cursor()
        xe(c, "SELECT id FROM users WHERE email = ?", (_ENSEMBLE_GUEST_EMAIL,))
        row = c.fetchone()
        conn.close()
        if row:
            return int(row[0])
        raise


# ========================
# DATABASE HELPERS
# ========================

def get_conversation_history(session_id):
    """Retrieve conversation history from database"""
    conn = connect_db()
    c = conn.cursor()
    xe(c, """
        SELECT role, content FROM messages 
        WHERE session_id = ? 
        ORDER BY timestamp ASC
    """, (session_id,))
    messages = c.fetchall()
    conn.close()
    return messages

def save_profile(
    session_id,
    user_name=None,
    user_role=None,
    projects=None,
    preferences=None,
    memory_context=None,
    uploaded_text=_PROFILE_KEEP,
    active_tools=_PROFILE_KEEP,
):
    """Save or update a user's profile. Use sentinel _PROFILE_KEEP to leave blobs/tools unchanged."""
    conn = connect_db()
    c = conn.cursor()
    xe(
        c,
        """
        SELECT user_name, user_role, projects, preferences, memory_context, uploaded_text,
               COALESCE(active_tools, ?)
        FROM profiles WHERE session_id = ?
        """,
        (DEFAULT_ACTIVE_TOOLS_JSON, session_id),
    )
    existing = c.fetchone()

    if existing:
        (
            current_name,
            current_role,
            current_projects,
            current_preferences,
            current_context,
            current_uploaded,
            current_tools,
        ) = existing
        user_name = user_name if user_name is not None else current_name
        user_role = user_role if user_role is not None else current_role
        projects = projects if projects is not None else current_projects
        preferences = preferences if preferences is not None else current_preferences
        memory_context = memory_context if memory_context is not None else current_context
        if uploaded_text is _PROFILE_KEEP:
            uploaded_use = current_uploaded
        else:
            uploaded_use = uploaded_text
        if active_tools is _PROFILE_KEEP:
            tools_use = current_tools
        else:
            lst = normalize_active_tools_list(active_tools if isinstance(active_tools, list) else [])
            tools_use = json.dumps(lst)

        xe(c, """
            UPDATE profiles
            SET user_name = ?, user_role = ?, projects = ?, preferences = ?, memory_context = ?,
                uploaded_text = ?, active_tools = ?, updated_at = ?
            WHERE session_id = ?
        """, (
            user_name,
            user_role,
            projects,
            preferences,
            memory_context,
            uploaded_use,
            tools_use,
            datetime.now().isoformat(),
            session_id,
        ))
    else:
        up_ins = None if uploaded_text is _PROFILE_KEEP else uploaded_text
        if active_tools is _PROFILE_KEEP:
            tools_ins = DEFAULT_ACTIVE_TOOLS_JSON
        else:
            tools_ins = json.dumps(normalize_active_tools_list(active_tools if isinstance(active_tools, list) else []))

        xe(c, """
            INSERT INTO profiles (
                session_id, user_name, user_role, projects, preferences, memory_context,
                uploaded_text, active_tools, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            user_name,
            user_role,
            projects,
            preferences,
            memory_context,
            up_ins,
            tools_ins,
            datetime.now().isoformat(),
        ))

    conn.commit()
    conn.close()


def get_profile(session_id):
    """Retrieve stored user profile/memory from database."""
    conn = connect_db()
    c = conn.cursor()
    xe(
        c,
        """
        SELECT user_name, user_role, projects, preferences, memory_context, uploaded_text,
               COALESCE(active_tools, ?)
        FROM profiles WHERE session_id = ?
        """,
        (DEFAULT_ACTIVE_TOOLS_JSON, session_id),
    )
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "user_name": row[0],
        "user_role": row[1],
        "projects": row[2],
        "preferences": row[3],
        "memory_context": row[4],
        "uploaded_text": row[5],
        "active_tools": row[6],
    }


def profile_for_client(profile):
    """Strip large blobs from profile before JSON responses."""
    if not profile:
        return None
    d = dict(profile)
    txt = (d.pop("uploaded_text", None) or "").strip()
    d["has_uploaded_document"] = bool(txt)
    if txt:
        d["uploaded_char_count"] = len(txt)
    raw_tools = d.get("active_tools") or DEFAULT_ACTIVE_TOOLS_JSON
    try:
        parsed = json.loads(raw_tools)
    except Exception:
        parsed = json.loads(DEFAULT_ACTIVE_TOOLS_JSON)
    d["active_tools"] = normalize_active_tools_list(parsed if isinstance(parsed, list) else [])
    return d


def get_uploaded_prompt_injection(session_id):
    """Text block prefixed to ensemble prompts when a document was uploaded."""
    profile = get_profile(session_id)
    if not profile:
        return ""
    raw = profile.get("uploaded_text") or ""
    blob = raw.strip()
    if not blob:
        return ""
    return (
        "\n=== UPLOADED DOCUMENT (user attached file; read and weigh this heavily) ===\n"
        f"{blob}\n"
        "=== END UPLOADED DOCUMENT ===\n"
    )


async def persist_session_upload(session_id: str, file: UploadFile) -> dict:
    """Read file, extract text (PyPDF2 for PDF), merge into profile uploaded_text + memory note."""
    conn = connect_db()
    c = conn.cursor()
    xe(c, "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
    if not c.fetchone():
        conn.close()
        return {"success": False, "error": "Session not found"}
    conn.close()

    fname = file.filename or "upload.bin"
    content = await file.read()

    extracted = ""
    try:
        if fname.lower().endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    extracted += t + "\n"
        else:
            extracted = content.decode("utf-8")
    except Exception as e:
        return {"success": False, "error": f"Could not extract text: {e}"}

    profile = get_profile(session_id)
    current_context = (profile.get("memory_context") or "") if profile else ""
    note = f"\n--- Attached document indexed for analysis ({fname}, {len(extracted)} chars) ---\n"
    new_context = current_context + note

    save_profile(
        session_id,
        user_name=profile.get("user_name") if profile else None,
        user_role=profile.get("user_role") if profile else None,
        projects=profile.get("projects") if profile else None,
        preferences=profile.get("preferences") if profile else None,
        memory_context=new_context,
        uploaded_text=extracted,
    )

    return {
        "success": True,
        "filename": fname,
        "extracted_length": len(extracted),
        "has_uploaded_document": True,
    }


def build_profile_context(session_id):
    """Build a reusable memory/context message for a session"""
    profile = get_profile(session_id)
    if not profile:
        return []

    parts = []
    if profile.get("user_name"):
        parts.append(f"Name: {profile['user_name']}")
    if profile.get("user_role"):
        parts.append(f"Role: {profile['user_role']}")
    if profile.get("projects"):
        parts.append(f"Current projects: {profile['projects']}")
    if profile.get("preferences"):
        parts.append(f"Preferences: {profile['preferences']}")
    if profile.get("memory_context"):
        parts.append(f"Context from past conversations: {profile['memory_context']}")

    if not parts:
        return []

    content = (
        "Please remember the user profile and context for this conversation.\n"
        + "\n".join(parts)
    )
    return [{"role": "user", "content": content}]


def save_message(session_id, model, role, content):
    """Save a message to database"""
    conn = connect_db()
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    xe(c, """
        INSERT INTO messages (session_id, model, role, content, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, model, role, content, timestamp))
    conn.commit()
    conn.close()

def create_session(session_id, title):
    """Create a new conversation session"""
    conn = connect_db()
    c = conn.cursor()
    now = datetime.now().isoformat()
    xe(c, """
        INSERT INTO sessions (session_id, created_at, updated_at, title)
        VALUES (?, ?, ?, ?)
    """, (session_id, now, now, title))
    conn.commit()
    conn.close()

def update_session_timestamp(session_id):
    """Update session's last update time"""
    conn = connect_db()
    c = conn.cursor()
    xe(c, """
        UPDATE sessions SET updated_at = ? WHERE session_id = ?
    """, (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()

# ========================
# AI MODEL FUNCTIONS
# ========================

async def ask_gpt(prompt, session_id=None, model=None):
    """Call GPT with optional conversation context."""
    if model is None:
        model = effective_openai_default()
    if openai_client is None:
        return "GPT Error: OpenAI client unavailable. Set OPENAI_API_KEY or check initialization."
    messages = [{"role": "user", "content": prompt}]
    
    if session_id:
        profile_context = build_profile_context(session_id)
        history = get_conversation_history(session_id)
        messages = profile_context + [
            {"role": role if role == "user" else "assistant", "content": content}
            for role, content in history
            if "Error" not in content
        ] + messages
    
    response = await openai_client.chat.completions.create(
        model=model,
        messages=messages
    )
    msg = response.choices[0].message
    return msg.content or ""

async def ask_gemini(prompt, session_id=None, model=None):
    """Call Gemini with optional conversation context."""
    if model is None:
        model = effective_gemini_default()
    if gemini_client is None:
        return "GEMINI Error: Gemini client unavailable. Set GEMINI_KEY or check initialization."
    if session_id:
        profile_text = ""
        profile = get_profile(session_id)
        if profile:
            profile_text = f"User Profile: Name={profile.get('user_name')}, Role={profile.get('user_role')}, Projects={profile.get('projects')}\n"

        history_text = ""
        history = get_conversation_history(session_id)
        for role, content in history:
            history_text += f"{role.upper()}: {content}\n"

        contents = f"{profile_text}\n{history_text}\nUSER: {prompt}"
    else:
        contents = prompt

    candidates = _gemini_candidate_models(model)
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=candidate,
                contents=contents,
            )
            return response.text
        except Exception as e:
            last_error = e
            err = str(e).lower()
            _print_provider_error(f"gemini:{candidate}", str(e))
            if "404" in err or "not_found" in err or "not available" in err:
                if candidate != candidates[-1]:
                    print(f"[gemini] model {candidate} unavailable, trying fallback")
                continue
            raise
    raise last_error if last_error else RuntimeError("Gemini call failed with unknown error")

async def ask_claude(prompt, session_id=None, system_prompt=None, model=None):
    """Call Claude with optional conversation context and system prompt"""
    eff_model = model if model is not None else effective_claude_primary()
    if eff_model is None:
        return "CLAUDE Error: Claude is available on BEN Pro only."
    if claude_client is None:
        return "CLAUDE Error: Anthropic client unavailable. Set ANTHROPIC_API_KEY or check initialization."
    messages = [{"role": "user", "content": prompt}]
    
    if session_id:
        profile_context = build_profile_context(session_id)
        history = get_conversation_history(session_id)
        messages = profile_context + [
            {"role": role if role == "user" else "assistant", "content": content}
            for role, content in history
            if "Error" not in content
        ] + messages
    
    model_order = [eff_model] + [m for m in effective_claude_fallbacks_for_call() if m != eff_model]
    last_error = None
    for candidate in model_order:
        kwargs = {
            "model": candidate,
            "max_tokens": 1000,
            "messages": messages
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        try:
            message = await claude_client.messages.create(**kwargs)
            return message.content[0].text
        except Exception as e:
            last_error = e
            err = str(e).lower()
            _print_provider_error(f"anthropic:{candidate}", str(e))
            if (
                ("not_found_error" in err or "404" in err or _is_budget_error(err))
                and candidate != model_order[-1]
            ):
                print(f"[anthropic] model {candidate} unavailable/budget-limited, trying fallback")
                continue
            raise
    raise last_error if last_error else RuntimeError("Claude call failed with unknown error")

async def ask_model(model_key, prompt, session_id=None):
    """Call the specified model with conversation context"""
    try:
        if model_key == "gpt":
            return await ask_gpt(prompt, session_id)
        elif model_key == "gemini":
            return await ask_gemini(prompt, session_id)
        elif model_key == "claude":
            return await ask_claude(prompt, session_id)
        elif model_key == "gpt-fast":
            return await ask_gpt(prompt, session_id, model=effective_openai_default("gpt-fast"))
        elif model_key == "gemini-fast":
            return await ask_gemini(prompt, session_id, model=effective_gemini_default())
        else:
            return "Unknown model"
    except Exception as e:
        err_text = str(e)
        _print_provider_error(model_key, err_text)
        if model_key == "claude" and _is_budget_error(err_text):
            try:
                fallback = await ask_gpt(prompt, session_id, model=OPENAI_DEFAULT_MODEL)
                return f"[Budget Fallback: GPT-4o-mini]\n{fallback}"
            except Exception as e2:
                _print_provider_error("gpt-fallback", str(e2))
        return f"{model_key.upper()} Error: {err_text}"


STREAM_R1_TIMEOUT_SEC = 12.0
STREAM_R2_TIMEOUT_SEC = 12.0


async def ask_model_timed(model_key, prompt, session_id=None, timeout_sec=STREAM_R2_TIMEOUT_SEC):
    """Non-streaming call with timeout; timed-out models return placeholder text."""
    try:
        return await asyncio.wait_for(ask_model(model_key, prompt, session_id), timeout=timeout_sec)
    except asyncio.TimeoutError:
        return f"[{model_key} timed out after {int(timeout_sec)}s — skipped]"


def _loud_startup_error(message: str):
    banner = "\n" + ("!" * 100)
    print(banner)
    print("!!! STARTUP API CONNECTIVITY ERROR !!!")
    print(message)
    print(banner)


def _ascii_preview(text: str, max_len: int = 120) -> str:
    """Windows consoles often use legacy code pages; strip non-ASCII for startup logs."""
    s = (text or "")[:max_len].replace("\n", " ")
    return s.encode("ascii", errors="replace").decode("ascii")

def _is_budget_error(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in [
        "rate limit",
        "rate_limit",
        "429",
        "insufficient funds",
        "insufficient_funds",
        "insufficient_quota",
        "credit",
        "quota",
        "billing",
    ])


def _print_provider_error(provider: str, err_text: str):
    line = f"[provider-error] {provider}: {err_text}"
    if _is_budget_error(err_text):
        print(f"\x1b[31m{line}\x1b[0m")
    else:
        print(line)
    try:
        from founder_status import log_status_error

        log_status_error(f"{provider}: {err_text}", source=str(provider))
    except Exception:
        pass

def _is_failed_model_output(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    markers = [
        " error:",
        " timed out ",
        "timed out after",
        "[retrying]",
        "[maintenance]",
        "authentication",
        "unauthorized",
        "invalid api key",
        "not_found_error",
        "404",
        "[skipped",
        "stream truncated",
    ]
    return any(m in f" {t} " for m in markers)

def _normalize_model_result(value, model_key: str) -> str:
    if isinstance(value, Exception):
        err = str(value).lower()
        if "timed out" in err:
            msg = f"[Retrying] {model_key.upper()} timed out. Temporary delay."
        elif ("not_found_error" in err or "404" in err or "401" in err or "authentication" in err or "unauthorized" in err):
            msg = f"[Maintenance] {model_key.upper()} unavailable right now."
        else:
            msg = f"{model_key.upper()} Error: {value}"
        print(f"[survivor] {msg}")
        return msg
    text = str(value or "")
    lower = text.lower()
    if "timed out" in lower:
        return f"[Retrying] {model_key.upper()} timed out. Temporary delay."
    if ("not_found_error" in lower or "404" in lower or "401" in lower or "authentication" in lower or "unauthorized" in lower):
        return f"[Maintenance] {model_key.upper()} unavailable right now."
    return text


BEN_EXPERTS_UNAVAILABLE_MSG = (
    "I'm currently having trouble reaching my experts. Please check your API keys."
)


def _ensemble_normalize(value, model_key: str) -> str:
    """
    Ensemble storage / BEN input: failed Gemini or Claude outputs become "" so BEN can treat
    them as absent. GPT failures stay as normalized text for single-analyst fallback.
    """
    if isinstance(value, Exception) and model_key in ("gemini", "claude"):
        print(f"[ensemble] {model_key} task failed; storing empty string")
        return ""
    raw = _normalize_model_result(value, model_key)
    if model_key in ("gemini", "claude") and _is_failed_model_output(raw):
        return ""
    return raw


def _estimate_tokens(text: str) -> int:
    # Lightweight approximation for cost telemetry
    return max(1, len((text or "").strip()) // 4) if (text or "").strip() else 0

def get_token_saver_mode(prompt: str) -> str:
    """
    Heuristic mode selector:
    - ECONOMY for short/simple asks
    - FULL for complex asks
    """
    text = (prompt or "").strip()
    if not text:
        return "ECONOMY"
    words = len(text.split())
    has_complex_signals = any(k in text.lower() for k in [
        "compare",
        "architecture",
        "scalability",
        "security",
        "tradeoff",
        "step-by-step",
        "detailed",
        "multi",
        "benchmark",
    ])
    punctuation_load = sum(text.count(ch) for ch in [":", ";", "?", "(", ")", ",", "\n"])
    if words <= 18 and punctuation_load <= 3 and not has_complex_signals:
        return "ECONOMY"
    return "FULL"


def is_product_idea_question(prompt: str) -> bool:
    """Heuristic: user is pitching or building a new product / venture idea."""
    t = (prompt or "").lower()
    if len(t.split()) < 4:
        return False
    cues = [
        "startup",
        "mvp",
        "saas",
        "build an app",
        "build a",
        "launch a",
        "new app",
        "new product",
        "product idea",
        "side project",
        "raise funding",
        "pitch",
        "venture",
        "monetize",
        "go-to-market",
        "gtm",
        "feature set",
        "roadmap",
        "competitor",
        "differentiate",
        "niche",
        "platform for",
        "tool for",
        "ai app",
        "ai tool",
    ]
    return any(c in t for c in cues)


async def tavily_search_similar_tools(query: str, max_results: int = 12) -> tuple[list[dict], str]:
    """
    Tavily web search for comparable AI tools/products.
    Requires TAVILY_API_KEY in environment.
    """
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return [], "TAVILY_API_KEY not set; search skipped."
    payload = {
        "api_key": key,
        "query": f"AI tools or products similar to this idea: {query}",
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_images": False,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post("https://api.tavily.com/search", json=payload)
            r.raise_for_status()
            body = r.json()
    except Exception as e:
        return [], f"Tavily search failed: {e}"
    results = body.get("results") or []
    out = []
    for row in results[:max_results]:
        out.append({
            "title": (row.get("title") or "").strip(),
            "url": (row.get("url") or "").strip(),
            "content": (row.get("content") or row.get("snippet") or "").strip()[:800],
        })
    return out, ""


def _names_from_search_hits(hits: list[dict]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for h in hits:
        title = (h.get("title") or "").strip()
        if not title:
            continue
        name = title.split("|")[0].split(" - ")[0].strip()[:80]
        low = name.lower()
        if low in seen:
            continue
        seen.add(low)
        names.append(name)
    return names


async def _llm_list_additional_competitors(
    question: str, existing: list[str], snippet_blob: str
) -> list[str]:
    """When search is thin, ask cheap models for more named competitors (product-idea guard)."""
    if openai_client is None and gemini_client is None:
        return []
    names_line = ", ".join(existing[:8]) if existing else "(none yet)"
    prompt = f"""Project idea (one line):
{question}

Existing names from web snippets: {names_line}

Snippets (truncated):
{snippet_blob[:3500]}

List EXACTLY 3 additional distinct AI products or tools that compete in the same space.
Rules: one product name per line, no numbering, no bullets, no explanations, English only.
If you must guess, label the line with "(example category)" but still give a plausible product name."""

    extra: list[str] = []
    seen = {x.lower() for x in existing}
    for model_key in ("gpt-fast", "gemini-fast"):
        if (openai_client is None and model_key == "gpt-fast") or (
            gemini_client is None and model_key == "gemini-fast"
        ):
            continue
        try:
            text = await ask_model_timed(model_key, prompt, None, timeout_sec=18.0)
        except Exception:
            continue
        for line in (text or "").splitlines():
            s = line.strip().lstrip("-*•0123456789.)").strip()
            if not s or len(s) < 2:
                continue
            if "error" in s.lower() and model_key.split("-")[0] in s.lower():
                continue
            low = s.lower()
            if low in seen:
                continue
            seen.add(low)
            extra.append(s[:120])
        if len(extra) >= 5:
            break
    return extra[:8]


async def _llm_competitors_comma_fallback(question: str) -> list[str]:
    """Last resort: three comma-separated product names for product-idea minimum."""
    prompt = (
        f"Startup / product idea (one line): {question}\n\n"
        "Reply with ONLY three real-world competing software or AI tool names, "
        "comma-separated, no descriptions, no punctuation besides commas."
    )
    try:
        raw = await ask_model_timed("gpt-fast", prompt, None, timeout_sec=12.0)
    except Exception:
        return []
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if len(p) < 2 or len(p) > 100:
            continue
        low = p.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(p)
        if len(out) >= 3:
            break
    return out


async def _infer_common_weakness(snippet_blob: str) -> str:
    if not snippet_blob.strip():
        return "pricing, latency, or limited customization (typical for the category)"
    prompt = f"""From these short blurbs about competing tools, respond with ONE short phrase (max 8 words)
describing the main shared weakness (e.g. "high cost and slow responses"). No quotes.

Blurbs:
{snippet_blob[:4000]}"""
    try:
        out = await ask_model_timed("gpt-fast", prompt, None, timeout_sec=12.0)
        s = (out or "").strip().split("\n")[0].strip()
        return s[:120] if s else "cost and speed tradeoffs"
    except Exception:
        return "cost and speed tradeoffs"


async def build_market_benchmark_context(
    question: str,
    token_saver_mode: str,
    session_id: Optional[str],
    web_search_enabled: bool,
) -> dict:
    """
    Tavily + optional LLM enrichment. For product ideas, ensure at least 3 named competitors before BEN.
    Returns dict: markdown (for UI card), ben_supplement (for BEN prompt), meta.
    """
    product_idea = is_product_idea_question(question)
    hits, search_note = await tavily_search_similar_tools(question, max_results=12)
    snippet_blob = "\n".join(f"{h['title']}: {h['content']}" for h in hits if h.get("content"))
    names = _names_from_search_hits(hits)

    if product_idea and len(names) < 3:
        extra = await _llm_list_additional_competitors(question, names, snippet_blob)
        for n in extra:
            low = n.lower()
            if low not in {x.lower() for x in names}:
                names.append(n)
            if len(names) >= 5:
                break
    if product_idea and len(names) < 3:
        for n in await _llm_competitors_comma_fallback(question):
            low = n.lower()
            if low not in {x.lower() for x in names}:
                names.append(n)
            if len(names) >= 8:
                break
    note_shortfall = ""
    if product_idea and len(names) < 3:
        note_shortfall = (
            "\n\n_Could not confirm three distinct named competitors automatically "
            "(search or model availability). BEN should still discuss typical substitutes "
            "in this category before recommending build decisions._"
        )
    # Hard floor: honest note if API dead; product ideas append shortfall guidance
    competitor_block = (
        "\n".join(f"- {n}" for n in names[:12])
        if names
        else "- (no indexed competitors yet — enable TAVILY_API_KEY for live market scan)"
    )
    if note_shortfall:
        competitor_block += note_shortfall
    weakness = await _infer_common_weakness(snippet_blob)
    n_similar = len(hits) if hits else max(len(names), 0)
    token_hint = ""
    if token_saver_mode == "ECONOMY":
        token_hint = (
            " **BEN can outperform on value** by routing simple turns through **Token Saver** eco-models "
            "(lower cost, faster cycles) while reserving full ensemble depth for complex prompts."
        )

    md = (
        f"### Market status\n\n"
        f"I found **{n_similar}** comparable results from open web search. "
        f"Their main shared weakness looks like **{weakness}**.{token_hint}\n\n"
        f"**Competitors / similar tools (minimum enforced for product ideas):**\n"
        f"{competitor_block}"
    )
    if search_note and not hits:
        md += f"\n\n_Search note:_ {search_note}"

    ben_supplement = (
        "COMPETITIVE LANDSCAPE (use in synthesis; cite only as market context):\n"
        f"- Comparable web results counted: {n_similar}\n"
        f"- Named competitors/tools:\n{competitor_block}\n"
        f"- Typical weakness pattern: {weakness}\n"
        f"- Token Saver mode active: {token_saver_mode == 'ECONOMY'}\n"
        f"- User triggered optional web enrichment in ensemble: {web_search_enabled}\n"
    )
    return {
        "markdown": md,
        "ben_supplement": ben_supplement,
        "competitor_names": names[:12],
        "hit_count": n_similar,
        "weakness_phrase": weakness,
        "product_idea": product_idea,
        "search_note": search_note,
    }


def _calculate_session_cost(round1: dict, round2: dict, final: str, mode: str = "FULL") -> dict:
    """
    Estimate session token usage and cost.
    Only successful model outputs contribute to token totals.
    """
    # Approx USD / 1K output tokens (rough telemetry, not billing source of truth)
    per_1k = {
        "gpt": 0.005,
        "gemini": 0.001,
        "claude": 0.015,
        "ben": 0.015,  # BEN uses Claude currently
    }

    model_tokens = {"gpt": 0, "gemini": 0, "claude": 0, "ben": 0}
    for model_key in ("gpt", "gemini", "claude"):
        r1 = str(round1.get(model_key, "") or "")
        r2 = str(round2.get(model_key, "") or "")
        if not _is_failed_model_output(r1):
            model_tokens[model_key] += _estimate_tokens(r1)
        if not _is_failed_model_output(r2):
            model_tokens[model_key] += _estimate_tokens(r2)
    if final and not _is_failed_model_output(final):
        model_tokens["ben"] = _estimate_tokens(final)

    model_cost_usd = {
        k: round((model_tokens[k] / 1000.0) * per_1k[k], 6)
        for k in model_tokens
    }
    total_tokens = sum(model_tokens.values())
    total_cost_usd = round(sum(model_cost_usd.values()), 6)
    # Estimate savings vs hypothetical FULL run.
    baseline_tokens = total_tokens
    baseline_cost = total_cost_usd
    if mode == "ECONOMY":
        # Approximate missing Gemini+Claude rounds as same output volume as GPT rounds.
        gpt_tokens = model_tokens["gpt"]
        baseline_tokens = total_tokens + (2 * gpt_tokens)
        baseline_cost = total_cost_usd + round(
            (gpt_tokens / 1000.0) * (per_1k["gemini"] + per_1k["claude"]),
            6,
        )
    saved_tokens = max(0, baseline_tokens - total_tokens)
    saved_usd = round(max(0.0, baseline_cost - total_cost_usd), 6)
    return {
        "mode": mode,
        "estimated_tokens": total_tokens,
        "estimated_cost_usd": total_cost_usd,
        "baseline_full_tokens": baseline_tokens,
        "baseline_full_cost_usd": round(baseline_cost, 6),
        "tokens_saved_vs_full": saved_tokens,
        "usd_saved_vs_full": saved_usd,
        "per_model_tokens": model_tokens,
        "per_model_cost_usd": model_cost_usd,
    }


def _build_ben_source_context(round1: dict, round2: dict):
    labels = {"gpt": "MODEL A", "gemini": "MODEL B", "claude": "MODEL C"}
    lines = []
    used = []
    for key in ("gpt", "gemini", "claude"):
        a = str(round1.get(key, "") or "")
        c = str(round2.get(key, "") or "")
        if _is_failed_model_output(a) or _is_failed_model_output(c):
            continue
        lines.append(f"{labels[key]}: Answer: {a} | Critique: {c}")
        used.append(key)
    # Fallback: use any non-empty round1 answer (critique may be empty).
    if not lines:
        for key in ("gpt", "gemini", "claude"):
            a = str(round1.get(key, "") or "")
            c = str(round2.get(key, "") or "")
            if a and not _is_failed_model_output(a):
                lines.append(f"{labels[key]}: Answer: {a} | Critique: {c}")
                used.append(key)
    return "\n".join(lines), used


def generate_ben_summary(
    round1: dict, round2: dict, benchmark_supplement: str = ""
) -> Optional[str]:
    """
    If exactly one analyst produced a valid answer, return that as the final reply (no compare step).
    If none did, return the experts-unavailable message.
    If two or more are valid, return None (caller runs full BEN / multi-model synthesis).
    """
    keys = ("gpt", "gemini", "claude")
    valid = [k for k in keys if not _is_failed_model_output(round1.get(k, ""))]
    if len(valid) == 0:
        return BEN_EXPERTS_UNAVAILABLE_MSG
    if len(valid) >= 2:
        return None
    only = valid[0]
    labels = {"gpt": "GPT", "gemini": "Gemini", "claude": "Claude"}
    body = (round1.get(only) or "").strip()
    crit = round2.get(only, "")
    if _is_failed_model_output(crit):
        crit = ""
    title = labels[only]
    out = (
        f"## TL;DR\nDelivering **{title}**'s analysis (other ensemble analysts were unavailable).\n\n"
        f"## Unified Answer\n\n{body}\n"
    )
    if crit.strip():
        out += f"\n## Gaps & critique\n\n{crit.strip()}\n"
    if benchmark_supplement.strip():
        out += f"\n### Market / competitors\n\n{benchmark_supplement.strip()}\n"
    return out


async def run_credit_probe(emit_logs: bool = True) -> dict:
    """Run provider readiness probe and return structured budget/readiness status."""
    if emit_logs:
        print(f"[startup] dotenv path: {DOTENV_PATH}")
        print(f"[startup] dotenv loaded: {DOTENV_LOADED}")

    checks = {
        "openai": {"env": "OPENAI_API_KEY", "ready": False, "status": "Missing Key", "detail": ""},
        "gemini": {"env": "GEMINI_KEY", "ready": False, "status": "Missing Key", "detail": ""},
        "anthropic": {"env": "ANTHROPIC_API_KEY", "ready": False, "status": "Missing Key", "detail": ""},
    }
    if emit_logs:
        print("[startup] Anthropic env var name in use: ANTHROPIC_API_KEY")
    for provider, meta in checks.items():
        has_key = bool((os.getenv(meta["env"]) or "").strip())
        if has_key:
            meta["status"] = "Key OK"
            if emit_logs:
                print(f"[startup] {meta['env']}: OK")
        else:
            meta["detail"] = f"Missing environment variable: {meta['env']}"
            if emit_logs:
                _loud_startup_error(meta["detail"])

    probe_specs = [
        ("openai", lambda: ask_gpt("say hi")),
        ("gemini", lambda: ask_gemini("say hi", model=GEMINI_FAST_MODEL)),
        ("anthropic", lambda: ask_claude("say hi", model=CLAUDE_MODEL)),
    ]
    if emit_logs:
        print("[startup] running model probes with prompt: 'say hi'")
    results = await asyncio.gather(*(fn() for _, fn in probe_specs), return_exceptions=True)

    for (provider, _), result in zip(probe_specs, results):
        meta = checks[provider]
        if isinstance(result, Exception):
            err = str(result)
            meta["detail"] = err
            if _is_budget_error(err):
                meta["status"] = "Low Funds"
            else:
                meta["status"] = "Error"
            _print_provider_error(provider, err)
            continue
        text = str(result)
        meta["detail"] = text
        if _is_budget_error(text):
            meta["status"] = "Low Funds"
            _print_provider_error(provider, text)
        elif "error" in text.lower() or "unknown model" in text.lower():
            meta["status"] = "Error"
            _print_provider_error(provider, text)
        else:
            meta["status"] = "Ready"
            meta["ready"] = True
            if emit_logs:
                preview = _ascii_preview(text)
                print(f"[startup] {provider} probe OK: {preview}")
    return checks


@app.on_event("startup")
async def startup_api_connectivity_check():
    """Validate .env loading and quickly probe model APIs."""
    jwt_raw = (os.getenv("JWT_SECRET") or "").strip()
    if jwt_raw:
        print(
            f"[startup] JWT_SECRET is set (length {len(jwt_raw)}); "
            "if you rotated this value, users must sign in again (401 until then)."
        )
    else:
        print("[startup] JWT_SECRET unset — using embedded dev default for JWT signing.")
    checks = await run_credit_probe(emit_logs=True)
    for provider, meta in checks.items():
        if not meta["ready"] and meta["status"] not in ("Low Funds",):
            _loud_startup_error(f"{provider} probe failed: {meta['detail']}")


async def stream_gpt_tokens(prompt, session_id=None, model=None):
    """Yield incremental text from OpenAI chat completions (streaming)."""
    if model is None:
        model = effective_openai_default()
    if openai_client is None:
        yield "GPT Error: OpenAI client unavailable. Set OPENAI_API_KEY or check initialization."
        return
    messages = [{"role": "user", "content": prompt}]
    if session_id:
        profile_context = build_profile_context(session_id)
        history = get_conversation_history(session_id)
        messages = profile_context + [
            {"role": role if role == "user" else "assistant", "content": content}
            for role, content in history
            if "Error" not in content
        ] + messages
    stream = await openai_client.chat.completions.create(
        model=model, messages=messages, stream=True
    )
    async for chunk in stream:
        piece = chunk.choices[0].delta.content or ""
        if piece:
            yield piece


async def stream_gemini_tokens(prompt, session_id=None, model=None):
    """Yield text from Gemini using stable SDK call (single-chunk)."""
    if model is None:
        model = effective_gemini_default()
    if gemini_client is None:
        yield "GEMINI Error: Gemini client unavailable. Set GEMINI_KEY or check initialization."
        return
    if session_id:
        profile_text = ""
        profile = get_profile(session_id)
        if profile:
            profile_text = (
                f"User Profile: Name={profile.get('user_name')}, Role={profile.get('user_role')}, "
                f"Projects={profile.get('projects')}\n"
            )
        history_text = ""
        history = get_conversation_history(session_id)
        for role, content in history:
            history_text += f"{role.upper()}: {content}\n"
        full_prompt = f"{profile_text}\n{history_text}\nUSER: {prompt}"
    else:
        full_prompt = prompt

    candidates = _gemini_candidate_models(model)
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=candidate,
                contents=full_prompt,
            )
            t = getattr(response, "text", None) or ""
            if t:
                yield t
            return
        except Exception as e:
            last_error = e
            err = str(e).lower()
            _print_provider_error(f"gemini-stream:{candidate}", str(e))
            if "404" in err or "not_found" in err or "not available" in err:
                if candidate != candidates[-1]:
                    print(f"[gemini] stream model {candidate} unavailable, trying fallback")
                continue
            raise
    if last_error:
        raise last_error


async def stream_claude_tokens(prompt, session_id=None, system_prompt=None, model=None):
    """Yield incremental text from Claude messages.stream."""
    eff_model = model if model is not None else effective_claude_primary()
    if eff_model is None:
        yield "CLAUDE Error: Claude is available on BEN Pro only."
        return
    if claude_client is None:
        yield "CLAUDE Error: Anthropic client unavailable. Set ANTHROPIC_API_KEY or check initialization."
        return
    msgs = [{"role": "user", "content": prompt}]
    if session_id:
        profile_context = build_profile_context(session_id)
        history = get_conversation_history(session_id)
        msgs = profile_context + [
            {"role": role if role == "user" else "assistant", "content": content}
            for role, content in history
            if "Error" not in content
        ] + msgs
    model_order = [eff_model] + [m for m in effective_claude_fallbacks_for_call() if m != eff_model]
    last_error = None
    for candidate in model_order:
        kwargs = {"model": candidate, "max_tokens": 1000, "messages": msgs}
        if system_prompt:
            kwargs["system"] = system_prompt
        try:
            async with claude_client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
                return
        except Exception as e:
            last_error = e
            err = str(e).lower()
            _print_provider_error(f"anthropic-stream:{candidate}", str(e))
            if (
                ("not_found_error" in err or "404" in err or _is_budget_error(err))
                and candidate != model_order[-1]
            ):
                print(f"[anthropic] stream model {candidate} unavailable/budget-limited, trying fallback")
                continue
            raise
    if last_error:
        raise last_error


async def stream_round1_model(model_key: str, prompt: str, session_id: str):
    """Dispatch Round 1 streaming by provider key."""
    om_main = effective_openai_default()
    om_fast = effective_openai_default("gpt-fast")
    gm = effective_gemini_default()
    if model_key == "gpt":
        async for t in stream_gpt_tokens(prompt, session_id, model=om_main):
            yield t
    elif model_key == "gpt-fast":
        async for t in stream_gpt_tokens(prompt, session_id, model=om_fast):
            yield t
    elif model_key == "gemini":
        async for t in stream_gemini_tokens(prompt, session_id, model=gm):
            yield t
    elif model_key == "claude":
        async for t in stream_claude_tokens(prompt, session_id):
            yield t
    else:
        yield ""


async def stream_ben_early_draft(question: str, round1_live: dict, session_id: Optional[str]):
    """
    Provisional Supreme Judge streaming as soon as at least one Round 1 model has finished.
    Uses whatever text is present in round1_live; labels missing slots as provisional.
    """
    ctx = f"""
STRICT INSTRUCTION: Respond in English only.

User question:
{question}

--- Partial ensemble (more analysts may still be running) ---
MODEL A (technical view): {round1_live.get("gpt", "") or "[not received yet]"}
MODEL B (implementation view): {round1_live.get("gemini", "") or "[not received yet]"}
MODEL C (strategic view): {round1_live.get("claude", "") or "[not received yet]"}

Produce a living synthesis: merge what is known, mark gaps as PROVISIONAL, use short headings.
"""
    ben_system = (
        "You are BEN, the Supreme Judge. This is a preliminary streaming synthesis; "
        "some model outputs may still be missing. Be concise. English only."
    )
    tr = current_tier_routing()
    if tr and not tr.ben_use_claude:
        if openai_client is None:
            yield "GPT Error: OpenAI client unavailable. Set OPENAI_API_KEY or check initialization."
            return
        combined = f"{ben_system}\n\n{ctx}"
        async for text in stream_gpt_tokens(combined, session_id, model=tr.openai_main):
            yield text
        return
    if claude_client is None:
        yield "CLAUDE Error: Anthropic client unavailable. Set ANTHROPIC_API_KEY or check initialization."
        return
    primary = effective_claude_primary() or CLAUDE_MODEL
    async with claude_client.messages.stream(
        model=primary,
        max_tokens=900,
        system=ben_system,
        messages=[{"role": "user", "content": ctx}],
    ) as stream:
        async for text in stream.text_stream:
            yield text

async def run_ben_supreme_judge(round1, round2, session_id=None, benchmark_supplement: str = ""):
    """BEN — Supreme Judge: Blind Evaluation and Synthesis"""
    preset = generate_ben_summary(round1, round2, benchmark_supplement)
    if preset is not None:
        return preset
    source_block, used_models = _build_ben_source_context(round1, round2)
    if not source_block:
        return BEN_EXPERTS_UNAVAILABLE_MSG
    bench = f"\n\n--- MARKET / COMPETITORS ---\n{benchmark_supplement.strip()}\n" if benchmark_supplement.strip() else ""
    # Anonymize models to ensure "Blind Evaluation"
    context_for_ben = f"""
STRICT INSTRUCTION: Respond in English only.

--- SOURCE DATA ---
{source_block}
{bench}
"""

    ben_system_prompt = ben_supreme_judge_system_prompt()
    learned_user = _ben_auto_learned_suffix()
    tr = current_tier_routing()
    if tr and not tr.ben_use_claude:
        if openai_client is None:
            return BEN_EXPERTS_UNAVAILABLE_MSG
        try:
            return await ask_gpt(
                ben_system_prompt
                + "\n\n"
                + context_for_ben
                + learned_user
                + "\n\nFollow the OUTPUT STRUCTURE from the system instructions above.",
                session_id=session_id,
                model=tr.openai_main,
            )
        except Exception as e:
            return f"## TL;DR\n{BEN_EXPERTS_UNAVAILABLE_MSG}\n## Unified Answer\n({e})"
    if claude_client is None:
        if openai_client is not None:
            try:
                return await ask_gpt(
                    context_for_ben
                    + learned_user
                    + "\n\nFollow the OUTPUT STRUCTURE from the system role.",
                    session_id=session_id,
                    model=OPENAI_DEFAULT_MODEL,
                )
            except Exception as e:
                return f"## TL;DR\n{BEN_EXPERTS_UNAVAILABLE_MSG}\n## Unified Answer\n({e})"
        return BEN_EXPERTS_UNAVAILABLE_MSG
    try:
        return await ask_claude(context_for_ben, session_id=session_id, system_prompt=ben_system_prompt)
    except Exception as e:
        if openai_client is not None:
            try:
                return await ask_gpt(
                    context_for_ben
                    + learned_user
                    + "\n\nFollow OUTPUT STRUCTURE: TL;DR, Unified Answer, Trust Map, Next Action.",
                    session_id=session_id,
                    model=OPENAI_DEFAULT_MODEL,
                )
            except Exception:
                pass
        return (
            f"## TL;DR\n{BEN_EXPERTS_UNAVAILABLE_MSG}\n## Unified Answer\n"
            f"Synthesis failed ({e}). Analysts available: {', '.join(used_models) or 'none'}."
        )

async def stream_ben_supreme_judge(round1, round2, session_id=None, benchmark_supplement: str = ""):
    """BEN — Supreme Judge: Streaming Blind Evaluation"""
    preset = generate_ben_summary(round1, round2, benchmark_supplement)
    if preset is not None:
        yield preset
        return
    source_block, used_models = _build_ben_source_context(round1, round2)
    if not source_block:
        yield BEN_EXPERTS_UNAVAILABLE_MSG
        return
    bench = f"\n\n--- MARKET / COMPETITORS ---\n{benchmark_supplement.strip()}\n" if benchmark_supplement.strip() else ""
    context_for_ben = f"""
STRICT INSTRUCTION: Respond in English only.

--- SOURCE DATA ---
{source_block}
{bench}
"""
    
    ben_system_prompt = ben_supreme_judge_system_prompt()

    learned_user = _ben_auto_learned_suffix()
    tr = current_tier_routing()
    if tr and not tr.ben_use_claude:
        if openai_client is None:
            yield BEN_EXPERTS_UNAVAILABLE_MSG
            return
        full_prompt = (
            ben_system_prompt
            + "\n\n"
            + context_for_ben
            + learned_user
            + "\n\nFollow the synthesis structure (TL;DR, headings, English)."
        )
        async for chunk in stream_gpt_tokens(full_prompt, session_id, model=tr.openai_main):
            yield chunk
        return
    if claude_client is None:
        if openai_client is not None:
            try:
                text = await ask_gpt(
                    context_for_ben
                    + learned_user
                    + "\n\nFollow the synthesis structure (TL;DR, headings, English).",
                    session_id=session_id,
                    model=OPENAI_DEFAULT_MODEL,
                )
                yield text or BEN_EXPERTS_UNAVAILABLE_MSG
            except Exception:
                yield BEN_EXPERTS_UNAVAILABLE_MSG
        else:
            yield BEN_EXPERTS_UNAVAILABLE_MSG
        return

    try:
        primary = effective_claude_primary() or CLAUDE_MODEL
        async with claude_client.messages.stream(
            model=primary,
            max_tokens=1024,
            system=ben_system_prompt,
            messages=[{"role": "user", "content": context_for_ben}]
        ) as stream:
            async for text in stream.text_stream:
                yield text
    except Exception as e:
        if openai_client is not None:
            try:
                text = await ask_gpt(
                    context_for_ben
                    + learned_user
                    + "\n\nSame structure: TL;DR, Unified Answer, Trust Map, Next Action.",
                    session_id=session_id,
                    model=OPENAI_DEFAULT_MODEL,
                )
                yield text or BEN_EXPERTS_UNAVAILABLE_MSG
                return
            except Exception:
                pass
        yield (
            f"## TL;DR\n{BEN_EXPERTS_UNAVAILABLE_MSG}\n## Unified Answer\n"
            f"Synthesis failed ({e}). Analysts: {', '.join(used_models) or 'none'}."
        )

# ========================
# SESSION MANAGEMENT ENDPOINTS
# ========================


@app.post("/auth/register")
def auth_register(body: AuthRegisterBody):
    email = normalize_account_email(body.email)
    pw = bcrypt.hashpw(body.password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")
    conn = connect_db()
    c = conn.cursor()
    try:
        ins = """
            INSERT INTO users (email, password_hash)
            VALUES (?, ?)
            """
        if USE_POSTGRES:
            ins += " RETURNING id"
        xe(c, ins, (email, pw))
        conn.commit()
        if USE_POSTGRES:
            user_id = int(c.fetchone()[0])
        else:
            user_id = int(c.lastrowid)
    except Exception as ex:
        conn.rollback()
        conn.close()
        if is_unique_violation(ex):
            raise HTTPException(status_code=400, detail="Email already registered") from ex
        raise
    xe(c, "SELECT tier FROM users WHERE id = ?", (user_id,))
    trow = c.fetchone()
    conn.close()
    tier_out = ((trow[0] or "free").strip().lower() if trow else "free")

    tok = create_access_token(int(user_id), remember_me=bool(body.remember_me))
    return {"success": True, "access_token": tok, "token_type": "Bearer", "tier": tier_out}


@app.post("/auth/refresh")
def auth_refresh(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(auth_scheme),
    ],
):
    """Issue a new JWT with the same remember-me duration as the current (still-valid) token."""
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(status_code=401, detail="Please login")
    token = credentials.credentials.strip()
    if not token:
        raise HTTPException(status_code=401, detail="Please login")
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired") from None
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Please login") from None
    try:
        uid = int(payload.get("sub"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Please login") from None
    remember_me = bool(payload.get("rm"))
    conn = connect_db()
    c = conn.cursor()
    xe(c, "SELECT id FROM users WHERE id = ?", (uid,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Please login")
    tok = create_access_token(uid, remember_me=remember_me)
    return {"access_token": tok, "token_type": "Bearer"}


@app.post("/auth/login")
def auth_login(body: AuthLoginBody):
    email = normalize_account_email(body.email)
    conn = connect_db()
    c = conn.cursor()
    xe(c, "SELECT id, password_hash, COALESCE(tier, 'free') FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()
    if row is None or not bcrypt.checkpw(
        body.password.encode("utf-8"),
        str(row[1]).encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Please login")

    tok = create_access_token(int(row[0]), remember_me=bool(body.remember_me))
    tier_out = (row[2] or "free").strip().lower()
    return {"access_token": tok, "token_type": "Bearer", "tier": tier_out}


@app.get("/")
async def serve_index():
    """Serve the main web UI."""
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/login")
async def serve_login():
    """Same SPA shell as `/`; client routes by pathname (`/login` vs `/`)."""
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/upgrade")
async def upgrade_landing():
    """Stripe / checkout landing (see upgrade.html)."""
    if UPGRADE_HTML.is_file():
        return FileResponse(UPGRADE_HTML, media_type="text/html")
    return HTMLResponse("<p>upgrade.html not found.</p>", status_code=404)


@app.get("/api/billing/checkout-url")
def billing_checkout_url():
    """Expose Stripe Checkout URL from env for the upgrade page CTA."""
    return {"url": STRIPE_CHECKOUT_URL}


@app.post("/session/new")
def create_new_session(req: NewSessionRequest):
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    create_session(session_id, req.title)
    save_profile(
        session_id,
        user_name=req.user_name,
        user_role=req.user_role,
        projects=req.projects,
        preferences=req.preferences,
        memory_context=req.memory_context
    )
    return {
        "success": True,
        "session_id": session_id,
        "title": req.title,
        "created_at": datetime.now().isoformat()
    }

@app.get("/session/{session_id}")
def get_session_history(session_id: str):
    """Retrieve full conversation history for a session"""
    conn = connect_db()
    c = conn.cursor()
    
    # Get session info
    xe(c, "SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    session = c.fetchone()
    
    if not session:
        conn.close()
        return {"success": False, "error": "Session not found"}
    
    # Get all messages
    xe(c, """
        SELECT model, role, content, timestamp FROM messages 
        WHERE session_id = ? 
        ORDER BY timestamp ASC
    """, (session_id,))
    messages = c.fetchall()
    prof = get_profile(session_id)
    # Get trial count
    user_id = prof.get("user_name", "anonymous") if prof else "anonymous"
    xe(c, "SELECT trial_count FROM trial_usage WHERE user_identifier = ?", (user_id,))
    t_row = c.fetchone()
    trial_count = t_row[0] if t_row else 0
    conn.close()
    
    return {
        "success": True,
        "session_id": session_id,
        "title": session[3],
        "created_at": session[1],
        "updated_at": session[2],
        "profile": profile_for_client(prof),
        "trial_count": trial_count,
        "messages": [
            {
                "model": msg[0],
                "role": msg[1],
                "content": msg[2],
                "timestamp": msg[3]
            }
            for msg in messages
        ]
    }

@app.post("/session/{session_id}/profile")
def update_session_profile(session_id: str, req: ProfileRequest):
    """Update or create profile/memory data for a session"""
    conn = connect_db()
    c = conn.cursor()
    xe(c, "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
    if not c.fetchone():
        conn.close()
        return {"success": False, "error": "Session not found"}
    conn.close()

    save_profile(
        session_id,
        user_name=req.user_name,
        user_role=req.user_role,
        projects=req.projects,
        preferences=req.preferences,
        memory_context=req.memory_context,
    )

    return {"success": True, "session_id": session_id, "profile": profile_for_client(get_profile(session_id))}


@app.patch("/session/{session_id}/active-tools")
def patch_session_active_tools(session_id: str, body: ActiveToolsRequest):
    """BEN Workspace: persist which GPT/Gemini/Claude lanes are routed for this session."""
    conn = connect_db()
    c = conn.cursor()
    xe(c, "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
    if not c.fetchone():
        conn.close()
        return {"success": False, "error": "Session not found"}
    conn.close()

    lst = normalize_active_tools_list(body.active_tools)
    save_profile(session_id, active_tools=lst)
    return {"success": True, "session_id": session_id, "profile": profile_for_client(get_profile(session_id))}

@app.get("/sessions")
def list_sessions():
    """List all conversation sessions"""
    conn = connect_db()
    c = conn.cursor()
    xe(c, "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC")
    sessions = c.fetchall()
    conn.close()
    
    return {
        "success": True,
        "sessions": [
            {
                "session_id": s[0],
                "title": s[1],
                "created_at": s[2],
                "updated_at": s[3]
            }
            for s in sessions
        ]
    }

# ========================
# CONVERSATION ENDPOINTS
# ========================

@app.post("/ask")
async def ask_single_model(req: AskRequest):
    """Ask a question to a single model with conversation context"""
    try:
        # Verify session exists
        conn = connect_db()
        c = conn.cursor()
        xe(c, "SELECT session_id FROM sessions WHERE session_id = ?", (req.session_id,))
        if not c.fetchone():
            conn.close()
            return {"success": False, "error": "Session not found"}
        conn.close()
        
        # Save user message
        save_message(req.session_id, req.model, "user", req.message)
        
        # Get response from model with conversation history
        response = await ask_model(req.model, req.message, req.session_id)
        
        # Save assistant response
        save_message(req.session_id, req.model, "assistant", response)
        
        # Update session timestamp
        update_session_timestamp(req.session_id)
        
        return {
            "success": True,
            "session_id": req.session_id,
            "model": req.model,
            "user_message": req.message,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ========================
# ENSEMBLE ENDPOINTS
# ========================

@app.post("/ensemble/feedback")
def submit_feedback(req: FeedbackRequest):
    """Store user feedback for the learning engine"""
    try:
        conn = connect_db()
        c = conn.cursor()
        xe(c, """
            INSERT INTO learning_feedback (category, model, feedback_value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (req.category, req.model, req.feedback_value, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/ensemble/export_pdf")
async def export_pdf(req: AskRequest):
    """Convert the synthesis into a clean PDF"""
    try:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "AI Ensemble - Synthesis Report")
        
        c.setFont("Helvetica", 11)
        text = req.message.replace("## ", "\n").replace("**", "")
        lines = simpleSplit(text, "Helvetica", 11, width - 100)
        
        y = height - 80
        for line in lines:
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)
            c.drawString(50, y, line)
            y -= 15
            
        c.save()
        buffer.seek(0)
        
        filename = f"synthesis_{uuid.uuid4().hex[:8]}.pdf"
        with open(filename, "wb") as f:
            f.write(buffer.read())
            
        return {"success": True, "filename": filename}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/ensemble/generate_code")
async def generate_code(req: AskRequest):
    """Tell Claude to write code based on the synthesis"""
    prompt = f"Based on this analysis, write the complete production-ready Python/JS code to implement the core logic:\n{req.message}"
    code = await ask_model("claude", prompt)
    return {"success": True, "code": code}

@app.post("/ensemble/research_examples")
async def research_examples(req: AskRequest):
    """Use Gemini to find real-world examples"""
    prompt = f"Find 3 real-world examples or case studies of companies/projects implementing the ideas discussed here:\n{req.message}"
    examples = await ask_model("gemini-fast", prompt)
    return {"success": True, "examples": examples}

@app.post("/ensemble/run")
async def run_ensemble(req: RunRequest):
    """Run the 3-round ensemble analysis with multi-turn capability"""
    _uid = get_or_create_guest_ensemble_user_id()
    tr_tok = None
    try:
        # Verify session exists + trial usage (single connection)
        conn = connect_db()
        c = conn.cursor()
        xe(c, "SELECT session_id FROM sessions WHERE session_id = ?", (req.session_id,))
        if not c.fetchone():
            conn.close()
            return {"success": False, "error": "Session not found"}

        db_tier, _ = fetch_user_account(_uid)
        routing = routing_for_db_tier(db_tier)
        if routing.tier != "pro" and not DEV_MODE:
            if count_user_ensemble_usage_24h(_uid) >= FREE_USER_LIMIT:
                conn.close()
                raise HTTPException(status_code=403, detail={"error": "LIMIT_REACHED"})
        record_user_ensemble_message(_uid)
        tr_tok = _TIER_ROUTING_CTX.set(routing)

        profile = get_profile(req.session_id)
        user_id = profile.get("user_name", "anonymous") if profile else "anonymous"

        xe(c, "SELECT trial_count, is_pro FROM trial_usage WHERE user_identifier = ?", (user_id,))
        row = c.fetchone()

        trial_count = 0
        is_pro = 0
        if row:
            trial_count, is_pro = row
        else:
            xe(c, "INSERT INTO trial_usage (user_identifier, trial_count) VALUES (?, 0)", (user_id,))
            conn.commit()

        # Legacy trial superseded by FREE_USER_LIMIT + tier for authenticated ensemble.

        enforce_ensemble_rate_limit(req.session_id, bool(is_pro))

        # Save user question
        save_message(req.session_id, "ensemble", "user", req.question)

        # Increment trial count
        xe(c, "UPDATE trial_usage SET trial_count = trial_count + 1 WHERE user_identifier = ?", (user_id,))
        conn.commit()
        conn.close()
        
        # Categorize question for Learning Engine
        category_prompt = f"""
Analyze the following question and classify it into exactly one of these categories: technical, strategy, code, creative.
Return ONLY the category name.
Question: {req.question}
"""
        category = (await ask_model("gpt", category_prompt)).strip().lower()
        if category not in ["technical", "strategy", "code", "creative"]:
            category = "technical"
            
        # Fetch historical stats
        conn = connect_db()
        c = conn.cursor()
        xe(c, """
            SELECT model, SUM(feedback_value) as score, COUNT(*) as total
            FROM learning_feedback
            WHERE category = ?
            GROUP BY model
        """, (category,))
        stats_rows = c.fetchall()
        conn.close()
        
        total_feedback = sum(row[2] for row in stats_rows)
        model_scores = {row[0]: row[1] for row in stats_rows}
        model_totals = {row[0]: row[2] for row in stats_rows}
        
        learning_stats_msg = "No historical data for this category yet. Rate responses to train the engine."
        weights_instruction = ""
        
        if total_feedback > 0:
            accuracies = {}
            for m in ["gpt", "gemini", "claude"]:
                m_score = model_scores.get(m, 0)
                m_total = model_totals.get(m, 0)
                if m_total > 0:
                    acc = max(0, (m_score + m_total) / (2 * m_total)) * 100
                else:
                    acc = 50.0
                accuracies[m] = acc
            
            best_model = max(accuracies, key=accuracies.get)
            best_accuracy = accuracies[best_model]
            learning_stats_msg = f"Based on {total_feedback} previous questions, {best_model.capitalize()} is {best_accuracy:.0f}% more accurate for {category} questions."
            weights_instruction = f"""
Historical data for this category ({category}) indicates the following model accuracy:
GPT: {accuracies['gpt']:.0f}%
Gemini: {accuracies['gemini']:.0f}%
Claude: {accuracies['claude']:.0f}%
Apply these learned weights automatically to prioritize the advice of the most accurate model.
"""

        # Get conversation history for context
        history_text = ""
        history = get_conversation_history(req.session_id)
        if history:
            for role, content in history[:-1]:  # Exclude the question we just saved
                history_text += f"{role.upper()}: {content}\n\n"

        uploaded_block = get_uploaded_prompt_injection(req.session_id)

        token_saver_mode = get_token_saver_mode(req.question)
        active_workspace_tools = get_profile_active_tool_set(req.session_id)
        tier_allowed = {"gpt", "gemini"} if routing.tier == "free" else {"gpt", "gemini", "claude"}
        active_workspace_tools = sorted(set(active_workspace_tools) & tier_allowed)
        if not active_workspace_tools:
            active_workspace_tools = ["gpt"]
        ben_tool_order = ["gpt", "gemini", "claude"]
        econ_lane_notice = "[Maintenance] Token saver mode: model skipped for cost efficiency."
        lane_off_notice = "(BEN Workspace: this analyst is turned off.)"

        if token_saver_mode == "ECONOMY":
            routed_models = []
            if "gpt" in active_workspace_tools:
                routed_models.append("gpt")
        else:
            routed_models = [m for m in ben_tool_order if m in active_workspace_tools]
        if not routed_models:
            routed_models = ["gpt"]

        skip_lane_r1_msgs: dict[str, str] = {}
        tier_free_lane = "(BEN Free tier: Claude is available on BEN Pro.)"
        for mm in ben_tool_order:
            if mm in routed_models:
                continue
            if mm == "claude" and routing.tier == "free":
                skip_lane_r1_msgs[mm] = tier_free_lane
            else:
                skip_lane_r1_msgs[mm] = (
                    econ_lane_notice if token_saver_mode == "ECONOMY" and mm != "gpt" else lane_off_notice
                )

        # =========================
        # ROUND 1
        # =========================

        gpt_r1_prompt = f"""
{uploaded_block}
You are GPT, the technical analyst.

Analyze this question from an engineering and architecture perspective.
Focus on system design, scalability, risks, security, edge cases, and tradeoffs.

{f"Conversation context:{history_text}" if history_text else ""}

Question:
{req.question}
"""

        gemini_r1_prompt = f"""
{uploaded_block}
You are Gemini, the implementation analyst.

Analyze this question from an implementation perspective.
Focus on APIs, code structure, backend execution, data flow, and practical build steps.

{f"Conversation context:{history_text}" if history_text else ""}

Question:
{req.question}
"""

        claude_r1_prompt = f"""
{uploaded_block}
You are Claude, the deep reasoning analyst.

Analyze this question deeply.
Focus on strategy, reasoning quality, hidden assumptions, business risks, human behavior, and long-term implications.

{f"Conversation context:{history_text}" if history_text else ""}

Question:
{req.question}
"""

        prompts_r1 = {"gpt": gpt_r1_prompt, "gemini": gemini_r1_prompt, "claude": claude_r1_prompt}
        r1_gather_tasks = []
        for mid in routed_models:
            if token_saver_mode == "ECONOMY" and mid == "gpt":
                r1_gather_tasks.append(ask_model("gpt-fast", prompts_r1[mid], req.session_id))
            else:
                r1_gather_tasks.append(ask_model(mid, prompts_r1[mid], req.session_id))

        r1_gather_out = await asyncio.gather(*r1_gather_tasks, return_exceptions=True) if r1_gather_tasks else []
        r1_live = dict(zip(routed_models, r1_gather_out))

        round1 = {}
        for mm in ben_tool_order:
            if mm in routed_models:
                round1[mm] = _normalize_model_result(r1_live[mm], mm)
            else:
                round1[mm] = skip_lane_r1_msgs[mm]

        # Save Round 1 responses (role field = analyst id for timeline UI)
        for model, response in round1.items():
            save_message(req.session_id, "ensemble-round1", model, response)

        # =========================
        # ROUND 1 CONSENSUS ANALYSIS
        # =========================
        consensus_prompt = f"""
Analyze the 3 responses provided to the original question.
Extract the key claims/points made across all responses.
For each claim, determine how many models mentioned or supported it:
- HIGH CONFIDENCE: all 3 models mentioned it
- MEDIUM: 2 models mentioned it
- UNVERIFIED: only 1 model mentioned it
- CONTRADICTION: models actively disagreed on this point

Return a JSON array of objects, with no markdown formatting or extra text.
Format:
[
  {{"claim": "The system should use microservices", "status": "HIGH CONFIDENCE"}},
  {{"claim": "Use MongoDB", "status": "UNVERIFIED"}},
  {{"claim": "Use GraphQL vs REST", "status": "CONTRADICTION"}}
]

Original Question: {req.question}

GPT: {round1.get("gpt", "")}
Gemini: {round1.get("gemini", "")}
Claude: {round1.get("claude", "")}
"""
        async def compute_consensus_data():
            """Runs on GPT only; Round 2 does not depend on this result."""
            raw = await ask_model("gpt", consensus_prompt)
            out = []
            try:
                json_str = raw.replace('```json', '').replace('```', '').strip()
                out = json.loads(json_str)
            except Exception as e:
                print("Failed to parse consensus JSON:", e)
            return out

        # =========================
        # ROUND 2 (parallel GPT/Gemini/Claude critics)
        # =========================
        # Overlap with consensus extraction: consensus only reads Round 1; critics only read Round 1.

        uploaded_block_r2 = get_uploaded_prompt_injection(req.session_id)
        shared_round1 = f"""
{uploaded_block_r2}
Original Question:
{req.question}

GPT Answer:
{round1.get("gpt", "")}

Gemini Answer:
{round1.get("gemini", "")}

Claude Answer:
{round1.get("claude", "")}
"""

        gpt_r2_prompt = f"""
You are GPT acting as a technical critic.

Critique the answers below.
Find technical gaps, weak assumptions, missing architecture, security risks, scalability risks, and engineering contradictions.

{shared_round1}
"""

        gemini_r2_prompt = f"""
You are Gemini acting as an implementation critic.

Critique the answers below.
Find missing build steps, vague APIs, unrealistic implementation claims, backend bottlenecks, and operational risks.

{shared_round1}
"""

        claude_r2_prompt = f"""
You are Claude acting as a deep reasoning critic.

Critique the answers below.
Find contradictions, shallow thinking, hidden assumptions, strategic weaknesses, incentive problems, and overlooked risks.

{shared_round1}
"""

        openai_output = str(round1.get("gpt", "") or "")
        print(f"DEBUG: OpenAI responded with: {openai_output[:100]}...")

        econ_r2_skip = "[Maintenance] Skipped in ECONOMY mode."

        async def r2_for_run(mid: str):
            tier_free_r2 = "(BEN Free tier: Claude is available on BEN Pro.)"
            if mid not in routed_models:
                if mid == "claude" and routing.tier == "free":
                    return tier_free_r2
                if token_saver_mode == "ECONOMY" and mid != "gpt":
                    return econ_r2_skip
                return lane_off_notice
            if mid == "gpt":
                return await ask_model_timed(
                    "gpt-fast", gpt_r2_prompt, req.session_id, STREAM_R2_TIMEOUT_SEC
                )
            if mid == "gemini":
                return await ask_model_timed(
                    "gemini-fast", gemini_r2_prompt, req.session_id, STREAM_R2_TIMEOUT_SEC
                )
            return await ask_model_timed("claude", claude_r2_prompt, req.session_id, STREAM_R2_TIMEOUT_SEC)

        round2_results, consensus_data = await asyncio.gather(
            asyncio.gather(*[r2_for_run(m) for m in ben_tool_order], return_exceptions=True),
            compute_consensus_data(),
        )
        round2 = {
            "gpt": _normalize_model_result(round2_results[0], "gpt"),
            "gemini": _normalize_model_result(round2_results[1], "gemini"),
            "claude": _normalize_model_result(round2_results[2], "claude"),
        }

        # Save Round 2 responses
        for model, response in round2.items():
            save_message(req.session_id, "ensemble-round2", model, response)

        # =========================
        # ROUND 3 (BEN — SUPREME JUDGE)
        # =========================

        bench_supplement = ""
        benchmark_markdown = None
        if is_product_idea_question(req.question) or req.web_search:
            bench = await build_market_benchmark_context(
                req.question,
                token_saver_mode,
                req.session_id,
                bool(req.web_search),
            )
            bench_supplement = bench.get("ben_supplement") or ""
            benchmark_markdown = bench.get("markdown")

        final = await run_ben_supreme_judge(
            round1, round2, req.session_id, benchmark_supplement=bench_supplement
        )

        # Save final synthesis
        save_message(req.session_id, "ensemble-final", "assistant", final)
        session_cost = _calculate_session_cost(round1, round2, final, token_saver_mode)
        cd = consensus_data if isinstance(consensus_data, list) else []
        tel = record_telemetry_run(
            cd,
            session_cost,
            token_saver_mode,
            routing_tier_label(token_saver_mode, bool(req.web_search)),
            perf={},
        )
        if tel:
            p_tel, tid = tel
            _schedule_consensus_analyzer_if_needed(
                p_tel,
                tid,
                session_id=req.session_id,
                question=str(req.question or ""),
                round1=round1,
                round2=round2,
                consensus_data=cd,
            )

        # Update session timestamp
        update_session_timestamp(req.session_id)

        return {
            "success": True,
            "session_id": req.session_id,
            "question": req.question,
            "category": category,
            "learning_stats_msg": learning_stats_msg,
            "consensus_data": consensus_data,
            "round1": round1,
            "round2": round2,
            "final": final,
            "benchmark_markdown": benchmark_markdown,
            "session_cost": session_cost,
            "trial_count": trial_count + 1,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if tr_tok is not None:
            _TIER_ROUTING_CTX.reset(tr_tok)

@app.post("/upload")
async def upload_global(session_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a document for a session (multipart: session_id + file). Stores text for ensemble prompts."""
    try:
        result = await persist_session_upload(session_id, file)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/session/{session_id}/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """Upload a file and extract text (PyPDF2 for PDF). Same storage as POST /upload."""
    try:
        return await persist_session_upload(session_id, file)
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/execute")
async def execute_code(req: CodeExecutionRequest):
    """Securely (?) execute code for the engineer tool"""
    if req.language != "python":
        return {"success": False, "error": "Only Python is supported"}
        
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
            tf.write(req.code)
            tf_path = tf.name
            
        result = subprocess.run(
            ["python", tf_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        os.unlink(tf_path)
        
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/test-ai")
async def test_ai(req: TestAIRequest):
    """Temporary connectivity test endpoint (non-streaming)."""
    try:
        result = await ask_model("gpt-fast", req.prompt)
        return {"success": True, "model": OPENAI_DEFAULT_MODEL, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/credit-check")
async def credit_check():
    """Run provider readiness/budget probe on demand."""
    checks = await run_credit_probe(emit_logs=True)
    return {"success": True, "checks": checks}


@app.post("/tools/similar-ai")
async def tools_similar_ai(req: SimilarAIToolsRequest):
    """Tavily-backed scan + LLM weakness line; use for product ideation or competitor discovery."""
    ctx = await build_market_benchmark_context(
        (req.query or "").strip(),
        "FULL",
        None,
        web_search_enabled=True,
    )
    return {
        "success": True,
        "markdown": ctx.get("markdown"),
        "ben_supplement": ctx.get("ben_supplement"),
        "competitor_names": ctx.get("competitor_names"),
        "hit_count": ctx.get("hit_count"),
        "weakness_phrase": ctx.get("weakness_phrase"),
        "product_idea": ctx.get("product_idea"),
        "search_note": ctx.get("search_note"),
    }


@app.post("/ensemble/stream")
async def run_ensemble_stream(req: RunRequest):
    """NDJSON stream: parallel Round 1 token streams, provisional BEN drafts, then consensus + R2 + final BEN."""
    uid = get_or_create_guest_ensemble_user_id()
    db_tier, _stripe_cust = fetch_user_account(uid)
    routing = routing_for_db_tier(db_tier)
    if routing.tier != "pro" and not DEV_MODE:
        if count_user_ensemble_usage_24h(uid) >= FREE_USER_LIMIT:
            raise HTTPException(status_code=403, detail={"error": "LIMIT_REACHED"})
    record_user_ensemble_message(uid)

    profile = get_profile(req.session_id)
    user_id = profile.get("user_name", "anonymous") if profile else "anonymous"
    conn_pre = connect_db()
    c_pre = conn_pre.cursor()
    xe(c_pre, "SELECT trial_count, is_pro FROM trial_usage WHERE user_identifier = ?", (user_id,))
    row_pre = c_pre.fetchone()
    trial_count_pre = 0
    is_pro_pre = 0
    if row_pre:
        trial_count_pre, is_pro_pre = int(row_pre[0]), int(row_pre[1] or 0)
    else:
        xe(c_pre, "INSERT INTO trial_usage (user_identifier, trial_count) VALUES (?, 0)", (user_id,))
        conn_pre.commit()
    conn_pre.close()

    # Legacy 3-message trial — superseded by tier + FREE_USER_LIMIT for logged-in users.

    enforce_ensemble_rate_limit(req.session_id, bool(is_pro_pre))

    async def event_generator():
        category = "technical"
        token_saver_mode = "FULL"
        round1 = {"gpt": "", "gemini": "", "claude": ""}
        round2 = {"gpt": "", "gemini": "", "claude": ""}
        consensus_data = []
        full_final = ""
        bench_supplement = ""
        def emit(payload: dict) -> str:
            # SSE framing: one JSON event per data line
            return f"data: {json.dumps(payload)}\n\n"
        tr_var = _TIER_ROUTING_CTX.set(routing)
        try:
            conn = connect_db()
            c = conn.cursor()
            xe(
                c,
                "UPDATE trial_usage SET trial_count = trial_count + 1 WHERE user_identifier = ?",
                (user_id,),
            )
            conn.commit()
            conn.close()

            save_message(req.session_id, "ensemble", "user", req.question)
            token_saver_mode = get_token_saver_mode(req.question)

            def _pipe(step: int, label: str):
                return emit({"type": "pipeline", "step": step, "label": label})

            if token_saver_mode == "ECONOMY":
                yield emit({
                    "type": "token_saver",
                    "mode": token_saver_mode,
                    "message": "Token Saver Active: Using high-speed eco-models",
                })

            yield _pipe(0, "Researching")

            category_prompt = f"Categorize: {req.question}. Return one word: technical, strategy, code, creative."
            category = (await ask_model_timed("gpt-fast", category_prompt, None, STREAM_R2_TIMEOUT_SEC)).strip().lower()

            ensemble_wall_loop_start = asyncio.get_running_loop().time()
            active_workspace_tools = get_profile_active_tool_set(req.session_id)
            tier_allowed = {"gpt", "gemini"} if routing.tier == "free" else {"gpt", "gemini", "claude"}
            active_workspace_tools = sorted(set(active_workspace_tools) & tier_allowed)
            if not active_workspace_tools:
                active_workspace_tools = ["gpt"]

            web_data = ""
            if req.web_search:
                search_prompt = (
                    f"Search for the latest information on: {req.question}. Summarize the top findings."
                )
                if "gemini" in active_workspace_tools:
                    web_data = await ask_model_timed("gemini-fast", search_prompt, None, STREAM_R2_TIMEOUT_SEC)
                elif "gpt" in active_workspace_tools:
                    web_data = await ask_model_timed("gpt-fast", search_prompt, None, STREAM_R2_TIMEOUT_SEC)
                else:
                    web_data = ""

            uploaded_block = get_uploaded_prompt_injection(req.session_id)
            q_doc = f"{req.question}\n{uploaded_block}" if uploaded_block.strip() else req.question
            q_with_web = f"{q_doc}\n\n[LATEST WEB SEARCH RESULTS]:\n{web_data}" if web_data else q_doc

            gpt_r1 = f"Analyst: GPT. Technical/Architectural view. Question: {q_with_web}"
            gem_r1 = f"Analyst: Gemini. Implementation view. Question: {q_with_web}"
            claude_r1 = f"Analyst: Claude. Strategic/Reasoning view. Question: {q_with_web}"

            ben_tool_order = ["gpt", "gemini", "claude"]
            econ_lane_notice = "[Maintenance] Token saver mode: model skipped for cost efficiency."
            lane_off_notice = "(BEN Workspace: this analyst is turned off.)"

            if token_saver_mode == "ECONOMY":
                routed_models = []
                if "gpt" in active_workspace_tools:
                    routed_models.append("gpt")
            else:
                routed_models = [m for m in ben_tool_order if m in active_workspace_tools]
            if not routed_models:
                routed_models = ["gpt"]

            skip_lane_banner: dict[str, str] = {}
            tier_free_lane = "(BEN Free tier: Claude is available on BEN Pro.)"
            for mm in ben_tool_order:
                if mm in routed_models:
                    continue
                if mm == "claude" and routing.tier == "free":
                    skip_lane_banner[mm] = tier_free_lane
                else:
                    skip_lane_banner[mm] = econ_lane_notice if token_saver_mode == "ECONOMY" and mm != "gpt" else lane_off_notice

            evt_q = asyncio.Queue()
            round1_live: dict[str, str] = {"gpt": "", "gemini": "", "claude": ""}
            r1_final: dict[str, str] = {}
            r1_done_flags = {"gpt": False, "gemini": False, "claude": False}
            SKIP_MSG = "[Retrying] Temporary delay. No complete response yet."
            leader_info = {"sent": False, "first_ms": None}
            r1_lat_ms = {"gpt": None, "gemini": None, "claude": None}

            loop_wm = asyncio.get_running_loop()
            parallel_wave_t0_holder: dict[str, float | None] = {"t": None}
            prompt_map_r1 = {"gpt": gpt_r1, "gemini": gem_r1, "claude": claude_r1}

            async def push_skipped_lane(mid: str, note: str):
                round1_live[mid] = note
                r1_final[mid] = note
                r1_done_flags[mid] = True
                save_message(req.session_id, "ensemble-round1", mid, note)
                await evt_q.put(("r1_chunk", mid, note))
                await evt_q.put(("r1_done", mid))

            async def pump_model(mid: str, prompt_line: str, stream_key: Optional[str] = None):
                t_mid0 = loop_wm.time()
                buf: list[str] = []
                deadline = loop_wm.time() + STREAM_R1_TIMEOUT_SEC
                use_key = stream_key or mid
                try:
                    async for piece in stream_round1_model(use_key, prompt_line, req.session_id):
                        if loop_wm.time() >= deadline:
                            if not buf:
                                round1_live[mid] = SKIP_MSG
                                await evt_q.put(("r1_chunk", mid, ""))
                            else:
                                suffix = "\n\n[Stream truncated — 12s limit]"
                                buf.append(suffix)
                                round1_live[mid] = "".join(buf)
                                await evt_q.put(("r1_chunk", mid, suffix))
                            break
                        if leader_info["sent"] is False and piece and piece.strip():
                            pv = parallel_wave_t0_holder.get("t")
                            if pv is not None:
                                dt_ms = (loop_wm.time() - pv) * 1000.0
                                if dt_ms <= 2100.0:
                                    leader_info["sent"] = True
                                    leader_info["first_ms"] = dt_ms
                                    await evt_q.put(("fast_leader", dt_ms, mid))
                        buf.append(piece)
                        round1_live[mid] = "".join(buf)
                        await evt_q.put(("r1_chunk", mid, piece))
                except Exception as ex:
                    err = _normalize_model_result(ex, mid)
                    buf.append(err)
                    round1_live[mid] = "".join(buf)
                    await evt_q.put(("r1_chunk", mid, err))
                text = "".join(buf) or SKIP_MSG
                round1_live[mid] = text
                r1_final[mid] = text
                r1_done_flags[mid] = True
                r1_lat_ms[mid] = (loop_wm.time() - t_mid0) * 1000.0
                save_message(req.session_id, "ensemble-round1", mid, text)
                await evt_q.put(("r1_done", mid))

            tools_active_payload = {mm: (mm in routed_models) for mm in ben_tool_order}
            yield emit(
                {
                    "type": "round1_start",
                    "models": ben_tool_order,
                    "tools_active": tools_active_payload,
                    "timeout_sec": STREAM_R1_TIMEOUT_SEC,
                }
            )
            parallel_wave_t0_holder["t"] = loop_wm.time()

            for _mid, lbl in skip_lane_banner.items():
                asyncio.create_task(push_skipped_lane(_mid, lbl))
            for mid in routed_models:
                stream_key_opt = None
                if token_saver_mode == "ECONOMY" and mid == "gpt":
                    stream_key_opt = "gpt-fast"
                asyncio.create_task(pump_model(mid, prompt_map_r1[mid], stream_key_opt))

            draft_task: Optional[asyncio.Task] = None
            draft_generation = 0

            async def cancel_draft():
                nonlocal draft_task
                if draft_task is None or draft_task.done():
                    return
                draft_task.cancel()
                try:
                    await draft_task
                except asyncio.CancelledError:
                    pass
                draft_task = None

            pending_r1_done = 3
            r1_parallel_wall_ms_snapshot = 0.0
            parallel_eff_pct_snapshot = 0.0

            async def get_next_evt():
                if pending_r1_done > 0:
                    return await evt_q.get()
                try:
                    return await asyncio.wait_for(evt_q.get(), timeout=0.04)
                except asyncio.TimeoutError:
                    return None

            while True:
                if pending_r1_done <= 0 and (draft_task is None or draft_task.done()):
                    stray = await get_next_evt()
                    if stray is None:
                        break
                    evt = stray
                else:
                    evt = await evt_q.get()

                kind, *rest = evt
                if kind == "fast_leader":
                    dt_leader, md_leader = rest
                    yield emit(
                        {
                            "type": "fast_first_stream",
                            "model": md_leader,
                            "ms_since_start": round(dt_leader, 2),
                            "within_target_sec": 2.0,
                        }
                    )
                    continue
                if kind == "r1_chunk":
                    mid, piece = rest
                    yield emit({"type": "round1_chunk", "model": mid, "content": piece})
                    continue
                if kind == "r1_done":
                    pending_r1_done -= 1
                    yield emit({"type": "round1_complete", "model": rest[0]})
                    await cancel_draft()
                    draft_generation += 1
                    gen_local = draft_generation
                    snap_now = {
                        k: (
                            r1_final[k]
                            if r1_done_flags[k]
                            else (round1_live.get(k, "") + " [still generating…]")
                        )
                        for k in ("gpt", "gemini", "claude")
                    }
                    yield emit({"type": "ben_reset", "draft": True, "generation": gen_local})

                    async def run_one_draft(snap_inner=snap_now, gg=gen_local):
                        try:
                            async for piece in stream_ben_early_draft(req.question, snap_inner, req.session_id):
                                await evt_q.put(("ben_chunk", gg, piece, True))
                        except asyncio.CancelledError:
                            return
                        except Exception as exc:
                            await evt_q.put(("ben_chunk", gg, f"\n[Draft error: {exc}]\n", True))

                    draft_task = asyncio.create_task(run_one_draft())
                    continue
                if kind == "ben_chunk":
                    gen_m, piece, is_draft = rest
                    if gen_m != draft_generation and is_draft:
                        continue
                    yield emit({"type": "ben_chunk", "content": piece, "draft": is_draft, "generation": gen_m})
                    continue

            pv_ts = parallel_wave_t0_holder.get("t")
            pv_base = pv_ts if pv_ts is not None else loop_wm.time()
            r1_parallel_wall_ms_snapshot = max(0.0, (loop_wm.time() - pv_base) * 1000.0)
            eff_lat_only = {
                kk: vv for kk, vv in r1_lat_ms.items() if isinstance(vv, (int, float)) and float(vv) > 0
            }
            parallel_eff_pct_snapshot = parallel_eff_ratio(eff_lat_only, r1_parallel_wall_ms_snapshot)

            if draft_task:
                try:
                    await draft_task
                except asyncio.CancelledError:
                    pass
                draft_task = None

            round1 = {
                "gpt": r1_final.get("gpt", SKIP_MSG),
                "gemini": r1_final.get("gemini", SKIP_MSG),
                "claude": r1_final.get("claude", SKIP_MSG),
            }

            yield _pipe(1, "Analyzing consensus")

            consensus_prompt = (
                "Analyze claims and status (HIGH CONFIDENCE, MEDIUM, UNVERIFIED, CONTRADICTION). "
                f"JSON format. GPT: {round1['gpt']}, Gemini: {round1['gemini']}, Claude: {round1['claude']}"
            )

            async def compute_consensus_data():
                raw = await ask_model_timed("gpt-fast", consensus_prompt, req.session_id, STREAM_R2_TIMEOUT_SEC)
                out: list = []
                try:
                    json_str = raw.replace("```json", "").replace("```", "").strip()
                    out = json.loads(json_str)
                except Exception:
                    pass
                return out

            shared = (
                f"{uploaded_block}Q: {req.question}\nGPT: {round1['gpt']}\n"
                f"Gemini: {round1['gemini']}\nClaude: {round1['claude']}"
            )
            openai_output = str(round1.get("gpt", "") or "")
            print(f"DEBUG: OpenAI responded with: {openai_output[:100]}...")

            econ_r2_skip = "[Maintenance] Skipped in ECONOMY mode."
            tier_free_r2 = "(BEN Free tier: Claude is available on BEN Pro.)"

            async def r2_for_model(mid: str):
                if mid not in routed_models:
                    if mid == "claude" and routing.tier == "free":
                        return tier_free_r2
                    if token_saver_mode == "ECONOMY" and mid != "gpt":
                        return econ_r2_skip
                    return lane_off_notice
                prompts_r2 = {
                    "gpt": f"Critic: GPT. Technical gaps. Data: {shared}",
                    "gemini": f"Critic: Gemini. Implementation gaps. Data: {shared}",
                    "claude": f"Critic: Claude. Reasoning gaps. Data: {shared}",
                }
                stream_keys = {"gpt": "gpt-fast", "gemini": "gemini-fast", "claude": "claude"}
                return await ask_model_timed(
                    stream_keys[mid], prompts_r2[mid], req.session_id, STREAM_R2_TIMEOUT_SEC
                )

            r2_pack, consensus_result = await asyncio.gather(
                asyncio.gather(*[r2_for_model(m) for m in ben_tool_order], return_exceptions=True),
                compute_consensus_data(),
                return_exceptions=True,
            )
            if isinstance(consensus_result, Exception):
                consensus_data = []
            else:
                consensus_data = consensus_result

            def _coerce_r2(v, name):
                if isinstance(v, Exception):
                    return _normalize_model_result(v, name)
                return _normalize_model_result(v, name)

            if isinstance(r2_pack, Exception):
                r2_rows = [_coerce_r2(r2_pack, m) for m in ben_tool_order]
            else:
                r2_rows = [_coerce_r2(r2_pack[i], ben_tool_order[i]) for i in range(3)]

            round2 = {
                "gpt": r2_rows[0],
                "gemini": r2_rows[1],
                "claude": r2_rows[2],
            }
            for m, r in round2.items():
                save_message(req.session_id, "ensemble-round2", m, r)

            if is_product_idea_question(req.question) or req.web_search:
                bench = await build_market_benchmark_context(
                    req.question,
                    token_saver_mode,
                    req.session_id,
                    bool(req.web_search),
                )
                bench_markdown = bench.get("markdown") or ""
                bench_supplement = bench.get("ben_supplement") or ""
                yield emit(
                    {
                        "type": "benchmark_card",
                        "markdown": bench_markdown,
                        "product_idea": bench.get("product_idea"),
                        "hit_count": bench.get("hit_count"),
                        "competitors": bench.get("competitor_names") or [],
                    }
                )

            await cancel_draft()
            draft_generation += 1
            fin_gen = draft_generation
            yield _pipe(2, "Finalizing answer")
            yield emit({"type": "ben_reset", "draft": False, "generation": fin_gen})

            full_final = ""
            async for chunk in stream_ben_supreme_judge(
                round1, round2, req.session_id, benchmark_supplement=bench_supplement
            ):
                full_final += chunk
                yield emit({"type": "ben_chunk", "content": chunk, "draft": False, "generation": fin_gen})

            save_message(req.session_id, "ensemble-final", "assistant", full_final)
            update_session_timestamp(req.session_id)
            session_cost = _calculate_session_cost(round1, round2, full_final, token_saver_mode)
            print(
                f"[token-saver] mode={token_saver_mode} saved_tokens={session_cost.get('tokens_saved_vs_full', 0)} "
                f"saved_usd={session_cost.get('usd_saved_vs_full', 0)}"
            )
            perf_snapshot = {
                "ensemble_wall_ms": round((loop_wm.time() - ensemble_wall_loop_start) * 1000.0, 2),
                "r1_parallel_wall_ms": round(r1_parallel_wall_ms_snapshot, 2),
                "r1_gpt_ms": r1_lat_ms.get("gpt"),
                "r1_gemini_ms": r1_lat_ms.get("gemini"),
                "r1_claude_ms": r1_lat_ms.get("claude"),
                "parallel_efficiency_pct": parallel_eff_pct_snapshot,
                "streaming_active": True,
                "fast_first_active": bool(leader_info.get("sent")),
                "first_token_ms": leader_info.get("first_ms"),
                "question_len": len(req.question or ""),
            }
            tel = record_telemetry_run(
                consensus_data,
                session_cost,
                token_saver_mode,
                routing_tier_label(token_saver_mode, bool(req.web_search)),
                perf=perf_snapshot,
            )
            if tel:
                p_tel, tid = tel
                _schedule_consensus_analyzer_if_needed(
                    p_tel,
                    tid,
                    session_id=req.session_id,
                    question=str(req.question or ""),
                    round1=round1,
                    round2=round2,
                    consensus_data=consensus_data,
                )

            yield emit(
                {
                    "type": "done",
                    "round1": round1,
                    "round2": round2,
                    "consensus_data": consensus_data,
                    "final": full_final,
                    "category": category,
                    "session_cost": session_cost,
                }
            )

        except Exception as e:
            try:
                from founder_status import log_status_error

                log_status_error(str(e), "ensemble-stream")
            except Exception:
                pass
            yield emit({"type": "error", "content": str(e)})
            session_cost = _calculate_session_cost(round1, round2, full_final, token_saver_mode)
            yield emit(
                {
                    "type": "done",
                    "round1": round1,
                    "round2": round2,
                    "consensus_data": consensus_data,
                    "final": full_final or f"Partial response due to stream error: {e}",
                    "category": category,
                    "session_cost": session_cost,
                }
            )
        finally:
            _TIER_ROUTING_CTX.reset(tr_var)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


try:
    from founder_status import append_changelog_if_dirty, register_founders_routes

    register_founders_routes(
        app,
        base_dir=_BASE_DIR,
        db_connect=connect_db,
        model_registry=MODEL_REGISTRY,
        run_credit_probe=run_credit_probe,
    )

    @app.on_event("startup")
    async def founder_changelog_hook():
        append_changelog_if_dirty(_BASE_DIR)

except ImportError as _fe:
    print(f"[founder] Control Suite disabled: {_fe}")


if __name__ == "__main__":
    _port = int(os.environ.get("PORT") or os.getenv("ENSEMBLE_PORT") or "8080")
    _reload_local = os.getenv("ENSEMBLE_RELOAD", "").strip().lower() in ("1", "true", "yes", "on")
    uvicorn.run("main:app", host="0.0.0.0", port=_port, reload=_reload_local)