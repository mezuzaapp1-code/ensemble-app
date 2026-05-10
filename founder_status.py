"""
Founder's Control Suite: /status dashboard, APIs, error ring, git auditor, backup/restore hooks.
"""
from __future__ import annotations

import html as html_escape
import os
import subprocess
import sys
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ensemble_db import (
    adapt as db_adapt,
    sql_created_after_interval_days,
    sql_date_bucket_expr,
    sql_order_by_datetime_desc,
    sql_today_predicate,
)

# --- Error ring (last 50 raw; dashboard shows last 3 translated) ---
_ERROR_RING: deque = deque(maxlen=50)


def log_status_error(raw: str, source: str = "app") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    _ERROR_RING.appendleft({"ts": ts, "source": source, "raw": raw[:2000]})


def translate_error_to_english(raw: str) -> str:
    t = (raw or "").lower()
    if "429" in raw or "rate" in t or "resource_exhausted" in t:
        return "Rate limit / too many requests"
    if "404" in raw or "not_found" in t or "not found" in t:
        return "Model name error or endpoint not found"
    if "401" in raw or "403" in raw or "unauthorized" in t or "permission" in t:
        return "Authentication failed or access denied"
    if "timeout" in t or "timed out" in t:
        return "Request timed out"
    if "budget" in t or "insufficient" in t or "billing" in t:
        return "Billing or quota issue"
    if "invalid api key" in t or "api key" in t and "invalid" in t:
        return "Invalid or missing API key"
    return "Error — see raw detail below"


def get_last_errors_translated(limit: int = 3) -> list[dict]:
    out = []
    for entry in list(_ERROR_RING)[:limit]:
        raw = entry.get("raw", "")
        out.append(
            {
                "time": entry.get("ts"),
                "source": entry.get("source"),
                "friendly": translate_error_to_english(raw),
                "raw": raw[:500],
            }
        )
    return out


# --- Git & changelog ---------------------------------------------------------

FILE_CHANGE_SUMMARY = {
    "main.py": "Brain/Logic updated",
    "index.html": "Appearance/UI updated",
    "requirements.txt": "System dependencies updated",
}


def summarize_changed_file(rel: str) -> str:
    rel = rel.replace("\\", "/")
    return FILE_CHANGE_SUMMARY.get(rel.split("/")[-1], f"{rel} changed")


def git_diff_names(base: Path) -> list[str]:
    if not (base / ".git").exists():
        return []
    try:
        r = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            return []
        return [x.strip() for x in r.stdout.splitlines() if x.strip()]
    except Exception as ex:
        log_status_error(str(ex), "git")
        return []


def append_changelog_if_dirty(base: Path) -> None:
    names = git_diff_names(base)
    if not names:
        return
    changelog = base / "CHANGELOG.md"
    summaries = [summarize_changed_file(n) for n in names[:25]]
    line = (
        f"- {datetime.now(timezone.utc).isoformat()} — "
        f"{'; '.join(dict.fromkeys(summaries))}\n"
    )
    prev = changelog.read_text(encoding="utf-8") if changelog.exists() else ""
    if line.strip() in prev:
        return
    header = "# Changelog\n\n" if not prev.strip() else ""
    if not prev.strip():
        changelog.write_text(header + "## Auto-detect (uncommitted vs last commit)\n" + line, encoding="utf-8")
    else:
        changelog.write_text(prev.rstrip() + "\n" + line, encoding="utf-8")


# --- Telemetry reads (table created in main.migrate_schema) -------------------

def _truth_consensus_pct(conn: Any) -> float:
    c = conn.cursor()
    win = sql_created_after_interval_days("created_at", 30)
    c.execute(db_adapt(f"SELECT AVG(consensus_pct) FROM telemetry_runs WHERE {win}"))
    row = c.fetchone()
    v = row[0]
    if v is None:
        return 0.0
    return round(float(v), 1)


def _cfo_from_db(conn: Any) -> dict:
    c = conn.cursor()
    bucket = sql_date_bucket_expr("created_at")
    w14 = sql_created_after_interval_days("created_at", 14)
    c.execute(
        db_adapt(
            f"""
        SELECT {bucket} AS d,
               SUM(openai_usd) AS o, SUM(gemini_usd) AS g, SUM(anthropic_usd) AS a,
               SUM(cost_usd) AS t, SUM(savings_usd) AS s
        FROM telemetry_runs
        WHERE {w14}
        GROUP BY {bucket}
        ORDER BY d DESC
        """
        )
    )
    daily_rows = [
        {
            "date": str(r[0]),
            "openai_usd": r[1] or 0,
            "gemini_usd": r[2] or 0,
            "anthropic_usd": r[3] or 0,
            "total_usd": r[4] or 0,
            "savings_usd": r[5] or 0,
        }
        for r in c.fetchall()
    ]

    c.execute(
        """
        SELECT SUM(savings_usd), SUM(baseline_cost_usd), SUM(cost_usd)
        FROM telemetry_runs
        """
    )
    agg = c.fetchone()
    total_savings = float(agg[0] or 0)
    total_baseline = float(agg[1] or 0)
    total_spent = float(agg[2] or 0)

    bucket7 = sql_date_bucket_expr("created_at")
    w7 = sql_created_after_interval_days("created_at", 7)
    c.execute(
        db_adapt(
            f"""
        SELECT AVG(day_total) FROM (
            SELECT {bucket7} AS d, SUM(cost_usd) AS day_total
            FROM telemetry_runs
            WHERE {w7}
            GROUP BY {bucket7}
        ) sub
        """
        )
    )
    avg_daily = c.fetchone()[0]
    avg_daily = float(avg_daily) if avg_daily is not None else 0.0

    budget = float(os.getenv("FOUNDER_API_BUDGET_USD", "0") or 0)
    runway_days: Optional[float] = None
    if avg_daily > 0 and budget > 0:
        runway_days = round(budget / avg_daily, 1)

    runway_level = "ok"
    if runway_days is not None:
        if runway_days < 7:
            runway_level = "red"
        elif runway_days < 14:
            runway_level = "yellow"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_spend = next((d["total_usd"] for d in daily_rows if d["date"] == today), 0.0)

    return {
        "daily_series": daily_rows,
        "today_total_usd": today_spend,
        "avg_daily_burn_usd": round(avg_daily, 6),
        "budget_usd": budget,
        "runway_days": runway_days,
        "runway_level": runway_level,
        "token_saver_savings_usd": round(total_savings, 6),
        "would_have_spent_usd": round(total_baseline, 6),
        "actually_spent_usd": round(total_spent, 6),
    }


def _performance_from_db(conn: Any) -> dict:
    """Today’s ensemble timing + routing mix from telemetry_runs."""
    empty = {
        "runs_today": 0,
        "avg_ms": {"gpt": None, "gemini": None, "claude": None},
        "bar_scale_ms": 1.0,
        "parallel_efficiency_pct_avg": None,
        "streaming_active_pct": None,
        "fast_first_pct": None,
        "ensemble_wall_slowest_ms": None,
        "ensemble_wall_fastest_ms": None,
        "routing_pct": {"Economy": None, "Standard": None, "Premium": None},
    }
    c = conn.cursor()
    today_w = sql_today_predicate("created_at")
    c.execute(db_adapt(f"SELECT COUNT(*) FROM telemetry_runs WHERE {today_w}"))
    n = int((c.fetchone() or [0])[0])
    if n <= 0:
        return empty
    c.execute(
        db_adapt(
            f"""
        SELECT
          AVG(CASE WHEN r1_gpt_ms IS NOT NULL AND r1_gpt_ms > 0 THEN r1_gpt_ms END),
          AVG(CASE WHEN r1_gemini_ms IS NOT NULL AND r1_gemini_ms > 0 THEN r1_gemini_ms END),
          AVG(CASE WHEN r1_claude_ms IS NOT NULL AND r1_claude_ms > 0 THEN r1_claude_ms END),
          AVG(parallel_efficiency_pct),
          SUM(CASE WHEN COALESCE(streaming_active, 1) = 1 THEN 1 ELSE 0 END),
          SUM(CASE WHEN COALESCE(fast_first_active, 1) = 1 THEN 1 ELSE 0 END)
        FROM telemetry_runs WHERE {today_w}
        """
        )
    )
    row = c.fetchone() or (None,) * 6
    avg_gpt, avg_ge, avg_cl = row[0], row[1], row[2]
    avg_par_raw = row[3]
    stream_hits = int(row[4] or 0)
    fast_hits = int(row[5] or 0)
    vals = []
    out_avg: dict[str, Optional[float]] = {"gpt": None, "gemini": None, "claude": None}
    if avg_gpt is not None:
        out_avg["gpt"] = round(float(avg_gpt), 2)
        vals.append(out_avg["gpt"])
    if avg_ge is not None:
        out_avg["gemini"] = round(float(avg_ge), 2)
        vals.append(out_avg["gemini"])
    if avg_cl is not None:
        out_avg["claude"] = round(float(avg_cl), 2)
        vals.append(out_avg["claude"])
    bar_scale_ms = float(max(vals) if vals else 1.0)
    avg_par_avg = round(float(avg_par_raw), 2) if avg_par_raw is not None else None
    streaming_pct = round(100.0 * stream_hits / n, 1) if n else None
    fast_pct = round(100.0 * fast_hits / n, 1) if n else None

    c.execute(
        db_adapt(
            f"""
        SELECT MAX(ensemble_wall_ms), MIN(ensemble_wall_ms)
        FROM telemetry_runs
        WHERE {today_w} AND ensemble_wall_ms IS NOT NULL
        """
        )
    )
    smin = c.fetchone()
    slowest_ms = round(float(smin[0]), 2) if smin and smin[0] is not None else None
    fastest_ms = round(float(smin[1]), 2) if smin and smin[1] is not None else None

    c.execute(
        db_adapt(
            f"""
        SELECT routing_tier, COUNT(*) FROM telemetry_runs
        WHERE {today_w} GROUP BY routing_tier
        """
        )
    )
    tier_rows = c.fetchall()
    buckets = {"Economy": 0, "Standard": 0, "Premium": 0, "_other": 0}
    for lbl, ct in tier_rows:
        key = str(lbl or "").strip()
        if key in buckets:
            buckets[key] += int(ct or 0)
        else:
            buckets["_other"] += int(ct or 0)
    total_lab = buckets["Economy"] + buckets["Standard"] + buckets["Premium"] + buckets["_other"]
    routing_pct: dict[str, Optional[float]] = {"Economy": None, "Standard": None, "Premium": None}
    if total_lab > 0:
        routing_pct["Economy"] = round(100.0 * buckets["Economy"] / total_lab, 1)
        routing_pct["Standard"] = round(100.0 * buckets["Standard"] / total_lab, 1)
        routing_pct["Premium"] = round(100.0 * (buckets["Premium"] + buckets["_other"]) / total_lab, 1)

    return {
        "runs_today": n,
        "avg_ms": out_avg,
        "bar_scale_ms": bar_scale_ms,
        "parallel_efficiency_pct_avg": avg_par_avg,
        "streaming_active_pct": streaming_pct,
        "fast_first_pct": fast_pct,
        "ensemble_wall_slowest_ms": slowest_ms,
        "ensemble_wall_fastest_ms": fastest_ms,
        "routing_pct": routing_pct,
        "streaming_status_label": (
            "Active"
            if stream_hits == n
            else ("Inactive" if stream_hits == 0 else "Partial")
        ),
        "fast_first_status_label": (
            "Active" if fast_hits == n else ("Inactive" if fast_hits == 0 else "Partial")
        ),
    }


def _truth_message(pct: float) -> str:
    if pct > 80:
        return "🍃 High Agreement: Consider Aggressive Token Saver"
    if pct < 50:
        return "🛡️ Low Agreement: Keep Full Ensemble Active"
    return "Moderate agreement — adjust Token Saver to your workload."


_JWT_PLACEHOLDER = "ensemble-development-jwt-signing-secret-min-length-thirty-two"


def _env_nonempty(name: str) -> bool:
    return bool((os.getenv(name) or "").strip())


def _railway_bearer_token() -> str:
    return (os.getenv("RAILWAY_TOKEN") or os.getenv("RAILWAY_API_KEY") or "").strip()


RAILWAY_GQL_URL = (os.getenv("RAILWAY_GRAPHQL_URL") or "https://backboard.railway.com/graphql/v2").strip()

_RAILWAY_MEASUREMENTS = [
    "CPU_USAGE",
    "CPU_LIMIT",
    "MEMORY_USAGE_GB",
    "MEMORY_LIMIT_GB",
    "DISK_USAGE_GB",
    "EPHEMERAL_DISK_USAGE_GB",
    "NETWORK_RX_GB",
    "NETWORK_TX_GB",
    "BACKUP_USAGE_GB",
]


async def _railway_gql(
    token: str, query: str, variables: Optional[dict] = None
) -> tuple[Optional[dict], Optional[str]]:
    variables = variables or {}
    payload = {"query": query.strip(), "variables": variables}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(RAILWAY_GQL_URL, json=payload, headers=headers)
            r.raise_for_status()
            body = r.json()
    except Exception as exc:
        return None, str(exc)[:800]
    errs = body.get("errors") or []
    err_msg = None
    if errs:
        err_msg = "; ".join(str(e.get("message", e)) for e in errs)[:900]
    return body.get("data"), err_msg


async def railway_cloud_infra_snapshot() -> dict:
    """
    Billing-ish signals from Railway Public GraphQL (estimatedUsage sums).
    Credits “left” are not always exposed reliably; optionally mirror dashboard via env hints.
    """
    out: dict[str, Any] = {
        "configured": False,
        "graphql_url": RAILWAY_GQL_URL.split("?", 1)[0],
        "error": None,
        "graphql_error": None,
        "workspace_id": None,
        "workspace_name": None,
        "plan_summary": None,
        "subscription_plan_limit": None,
        "projected_cycle_spend_usd": None,
        "estimated_daily_spend_usd": None,
        "estimated_monthly_burn_usd": None,
        "credits_remaining_usd": None,
        "credits_line": "",
        "measurement_rows": 0,
    }
    token = _railway_bearer_token()
    if not token:
        out["error"] = "Add RAILWAY_TOKEN or RAILWAY_API_KEY (Railway account / workspace token)."
        return out
    out["configured"] = True

    wid = (os.getenv("RAILWAY_WORKSPACE_ID") or "").strip()
    ws_name = None

    if not wid:
        q_ws = """
        query FounderWorkspaces {
          workspaces {
            edges {
              node {
                id
                name
              }
            }
          }
        }
        """
        data, err = await _railway_gql(token, q_ws, {})
        if err and not data:
            out["graphql_error"] = err
            out["error"] = "Could not list workspaces via GraphQL. Set RAILWAY_WORKSPACE_ID manually."
            return out
        if err:
            out["graphql_error"] = err
        edges = (((data or {}).get("workspaces") or {}).get("edges")) or []
        nodes = [e.get("node") or {} for e in edges if isinstance(e, dict)]
        if len(nodes) == 1:
            wid = str(nodes[0].get("id") or "").strip()
            ws_name = (nodes[0].get("name") or "").strip() or None
        elif nodes:
            want = (os.getenv("RAILWAY_WORKSPACE_NAME_MATCH") or "").strip().lower()
            pick = None
            if want:
                for n in nodes:
                    if str(n.get("name") or "").strip().lower() == want:
                        pick = n
                        break
            if pick is None:
                out["error"] = (
                    "Multiple workspaces on this token — set RAILWAY_WORKSPACE_ID "
                    "(Ctrl/Cmd+K in Railway dashboard → copy id), optionally RAILWAY_WORKSPACE_NAME_MATCH."
                )
                return out
            wid = str(pick.get("id") or "").strip()
            ws_name = (pick.get("name") or "").strip() or None

    if not wid:
        out["error"] = "Missing RAILWAY_WORKSPACE_ID and workspace list was empty."
        return out

    q_w = """
    query FounderWorkspaceMeta($wid: String!) {
      workspace(workspaceId: $wid) {
        id
        name
        plan
        subscriptionModel
        subscriptionPlanLimit
      }
    }
    """
    meta, err = await _railway_gql(token, q_w, {"wid": wid})
    if err and not meta:
        out["graphql_error"] = err
        out["error"] = "Could not load workspace meta (check token scope and workspace id)."
        return out
    if err:
        out["graphql_error"] = err
    wm = ((meta or {}).get("workspace")) or {}
    out["workspace_id"] = str(wm.get("id") or wid)
    out["workspace_name"] = (wm.get("name") or ws_name or "").strip() or None
    plan = wm.get("plan")
    subm = wm.get("subscriptionModel")
    parts = [str(x) for x in (plan, subm) if x]
    out["plan_summary"] = " · ".join(parts) if parts else None
    spl = wm.get("subscriptionPlanLimit")
    out["subscription_plan_limit"] = spl

    q_est = """
    query FounderEstimated($wid: String!, $meas: [MetricMeasurement!]!, $inc: Boolean!) {
      estimatedUsage(
        workspaceId: $wid
        measurements: $meas
        includeDeleted: $inc
      ) {
        estimatedValue
        measurement
        projectId
      }
    }
    """
    pid = (os.getenv("RAILWAY_PROJECT_ID") or "").strip() or None
    q_est_proj = """
    query FounderEstimatedProject($pid: String!, $meas: [MetricMeasurement!]!, $inc: Boolean!) {
      estimatedUsage(
        projectId: $pid
        measurements: $meas
        includeDeleted: $inc
      ) {
        estimatedValue
        measurement
        projectId
      }
    }
    """
    meas_try = list(_RAILWAY_MEASUREMENTS)
    rows: list[dict] = []
    last_est_err = None
    while meas_try:
        scope = {"meas": meas_try, "inc": False}
        if pid:
            data_e, er = await _railway_gql(token, q_est_proj, {"pid": pid, **scope})
        else:
            data_e, er = await _railway_gql(token, q_est, {"wid": wid, **scope})
        block = ((data_e or {}).get("estimatedUsage")) if data_e is not None else None
        if isinstance(block, list) and block:
            rows = [r for r in block if isinstance(r, dict)]
            last_est_err = er
            break
        last_est_err = er
        if len(meas_try) <= 1:
            rows = []
            break
        meas_try = meas_try[:-1]

    if last_est_err and not rows:
        out["graphql_error"] = (out.get("graphql_error") or "").strip()
        sep = "; " if out["graphql_error"] else ""
        out["graphql_error"] = (out["graphql_error"] or "") + sep + (last_est_err or "")
    out["measurement_rows"] = len(rows)
    total = 0.0
    for r in rows:
        try:
            total += float(r.get("estimatedValue") or 0.0)
        except (TypeError, ValueError):
            continue
    if rows:
        out["projected_cycle_spend_usd"] = round(total, 4)
        daily = round(total / 30.0, 4)
        out["estimated_daily_spend_usd"] = daily
        out["estimated_monthly_burn_usd"] = round(daily * 30.0, 4)

    hint = (os.getenv("RAILWAY_CREDITS_REMAINING_HINT_USD") or "").strip()
    cycle_allow = (os.getenv("RAILWAY_INCLUDED_USAGE_USD_PER_CYCLE") or "").strip()
    if hint:
        try:
            out["credits_remaining_usd"] = round(float(hint), 4)
            out["credits_line"] = "Mirrors Railway UI via RAILWAY_CREDITS_REMAINING_HINT_USD."
        except ValueError:
            out["credits_line"] = "RAILWAY_CREDITS_REMAINING_HINT_USD is set but not a valid number."
    elif cycle_allow and out["projected_cycle_spend_usd"] is not None:
        try:
            allow = float(cycle_allow)
            out["credits_remaining_usd"] = round(max(0.0, allow - float(out["projected_cycle_spend_usd"])), 4)
            out["credits_line"] = (
                "Rough remainder: RAILWAY_INCLUDED_USAGE_USD_PER_CYCLE minus projected cycle (not official)."
            )
        except ValueError:
            out["credits_line"] = "RAILWAY_INCLUDED_USAGE_USD_PER_CYCLE invalid."
    else:
        out["credits_line"] = (
            "Railway does not always expose “credits left” on this API. "
            "Set RAILWAY_CREDITS_REMAINING_HINT_USD to mirror the dashboard, "
            "or RAILWAY_INCLUDED_USAGE_USD_PER_CYCLE (e.g. 5) for a rough estimate."
        )

    if not rows and not out.get("graphql_error"):
        out["graphql_error"] = last_est_err
    return out


def railway_config_env_dashboard() -> dict:
    jwt_raw = (os.getenv("JWT_SECRET") or "").strip()
    jwt_secure = bool(jwt_raw) and jwt_raw != _JWT_PLACEHOLDER

    def mk(
        *,
        env_key_display: str,
        label: str,
        present: bool,
        group: str,
        optional: bool = False,
        note: str = "",
    ) -> dict:
        return {
            "env_key_display": env_key_display,
            "label": label,
            "present": present,
            "group": group,
            "optional": optional,
            "note": note,
        }

    port_ok = _env_nonempty("PORT") or _env_nonempty("ENSEMBLE_PORT")
    checks: list[dict] = [
        mk(
            env_key_display="OPENAI_API_KEY",
            label="OpenAI · GPT lanes",
            present=_env_nonempty("OPENAI_API_KEY"),
            group="Model providers",
        ),
        mk(
            env_key_display="GEMINI_KEY",
            label="Google · Gemini lanes",
            present=_env_nonempty("GEMINI_KEY"),
            group="Model providers",
        ),
        mk(
            env_key_display="ANTHROPIC_API_KEY",
            label="Anthropic · Claude · BEN",
            present=_env_nonempty("ANTHROPIC_API_KEY"),
            group="Model providers",
        ),
        mk(
            env_key_display="JWT_SECRET",
            label="JWT signing secret",
            present=jwt_secure,
            group="Auth / security",
            note="Production should set a unique strong secret (not the dev fallback).",
        ),
        mk(
            env_key_display="PORT (+ ENSEMBLE_PORT locally)",
            label="Listen port",
            present=port_ok,
            group="Runtime",
            note="Railway sets PORT; local dev may use ENSEMBLE_PORT.",
        ),
        mk(
            env_key_display="DATABASE_URL",
            label="SQLite / database URL",
            present=_env_nonempty("DATABASE_URL"),
            group="Data",
            optional=True,
            note="Optional — app defaults to SQLite next to main.py.",
        ),
        mk(
            env_key_display="TAVILY_API_KEY",
            label="Market / competitor web search",
            present=_env_nonempty("TAVILY_API_KEY"),
            group="Enhancements",
            optional=True,
        ),
        mk(
            env_key_display="FOUNDER_API_BUDGET_USD",
            label="Founder runway budget hint",
            present=_env_nonempty("FOUNDER_API_BUDGET_USD"),
            group="Enhancements",
            optional=True,
        ),
        mk(
            env_key_display="RAILWAY_TOKEN or RAILWAY_API_KEY",
            label="Railway GraphQL · Cloud Infrastructure Cost",
            present=bool(_railway_bearer_token()),
            group="Enhancements",
            optional=True,
            note="Optional — Bearer from railway.com/account/tokens; projected usage on /status.",
        ),
        mk(
            env_key_display="RAILWAY_WORKSPACE_ID",
            label="Railway workspace id",
            present=_env_nonempty("RAILWAY_WORKSPACE_ID"),
            group="Enhancements",
            optional=True,
            note="Optional — omit if this token sees exactly one workspace.",
        ),
        mk(
            env_key_display="ENSEMBLE_ANALYZER_MODEL",
            label="Consensus analyzer routing",
            present=_env_nonempty("ENSEMBLE_ANALYZER_MODEL"),
            group="Enhancements",
            optional=True,
            note="Optional — omit to use default analyzer lane (gpt-fast).",
        ),
    ]
    core = [c for c in checks if not c.get("optional")]
    core_missing = sum(1 for c in core if not c["present"])
    return {
        "checks": checks,
        "core_required_missing": core_missing,
        "jwt_using_placeholder": bool(jwt_raw) and jwt_raw == _JWT_PLACEHOLDER,
    }


def self_heals_today_count(conn: Any) -> int:
    c = conn.cursor()
    try:
        tw = sql_today_predicate("created_at")
        c.execute(db_adapt(f"SELECT COUNT(*) FROM self_heals WHERE {tw}"))
        return int((c.fetchone() or [0])[0])
    except Exception:
        return 0


def _model_strings_for_display(model_registry: dict) -> dict[str, list[str]]:
    def _uniq(xs: list) -> list[str]:
        return list(dict.fromkeys([x for x in xs if x]))

    return {
        "openai": _uniq([model_registry.get("openai_default")]),
        "gemini": _uniq(
            [
                model_registry.get("gemini_default"),
                model_registry.get("gemini_fast"),
                *list(model_registry.get("gemini_fallback_chain") or ()),
            ]
        ),
        "anthropic": _uniq(
            [
                model_registry.get("claude_primary"),
                *list(model_registry.get("claude_fallback_chain") or ()),
            ]
        ),
    }


BACKUPS_SUBDIR = "backups"
STABLE_PREFIX = "stable_v"


def _backups_dir(base: Path) -> Path:
    d = base / BACKUPS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _zip_ignore(path: Path, base: Path) -> bool:
    parts = path.relative_to(base).parts
    if "__pycache__" in parts or ".git" in parts:
        return True
    if "venv" in parts or ".venv" in parts or "node_modules" in parts:
        return True
    if path.name == ".env":
        return True
    if BACKUPS_SUBDIR in parts:
        return True
    if path.suffix.lower() == ".zip" and STABLE_PREFIX in path.name:
        return True
    return False


def create_stable_backup_zip(base: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = _backups_dir(base) / f"{STABLE_PREFIX}{ts}.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in base.rglob("*"):
            if p.is_dir():
                continue
            if _zip_ignore(p, base):
                continue
            arc = p.relative_to(base).as_posix()
            zf.write(p, arcname=arc)
    return dest


def latest_stable_zip(base: Path) -> Optional[Path]:
    d = base / BACKUPS_SUBDIR
    if not d.is_dir():
        return None
    zs = sorted(d.glob(f"{STABLE_PREFIX}*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
    return zs[0] if zs else None


def extract_stable_zip(base: Path, zip_path: Path) -> None:
    base = base.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if ".." in name or name.startswith("/"):
                continue
            dest_path = (base / name).resolve()
            try:
                dest_path.relative_to(base)
            except ValueError:
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest_path, "wb") as out:
                out.write(src.read())


def restart_server_after_restore(base: Path) -> None:
    port = (os.environ.get("PORT") or os.getenv("ENSEMBLE_PORT") or "8080").strip()
    exe = sys.executable
    args = [exe, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", port]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    subprocess.Popen(
        args,
        cwd=str(base),
        close_fds=True,
        creationflags=creationflags,
        start_new_session=sys.platform != "win32",
    )
    os._exit(0)


def register_founders_routes(
    app: Any,
    *,
    base_dir: Path,
    db_connect: Callable[[], Any],
    model_registry: dict,
    run_credit_probe: Callable[..., Awaitable[dict]],
) -> None:
    router = APIRouter(tags=["founder"])
    status_html = base_dir / "status.html"

    @router.get("/status", response_class=HTMLResponse)
    async def status_page():
        if not status_html.is_file():
            return HTMLResponse("<h1>status.html missing</h1>", status_code=500)
        return FileResponse(status_html, media_type="text/html")

    @router.get("/api/founder/dashboard")
    async def founder_dashboard():
        checks = await run_credit_probe(emit_logs=False)
        conn = db_connect()
        n_runs = 0
        heals_today = 0
        try:
            cur = conn.cursor()
            cur.execute(db_adapt("SELECT COUNT(*) FROM telemetry_runs"))
            n_runs = int(cur.fetchone()[0] or 0)
            pct = _truth_consensus_pct(conn) if n_runs else 0.0
            cfo = _cfo_from_db(conn)
            performance = _performance_from_db(conn)
            heals_today = self_heals_today_count(conn)
        finally:
            conn.close()

        railway_env = railway_config_env_dashboard()
        railway_cloud = await railway_cloud_infra_snapshot()

        if n_runs == 0:
            truth_tip = "No consensus data yet — run a few ensemble chats to fill the truth meter."
        else:
            truth_tip = _truth_message(pct)
        names = _model_strings_for_display(model_registry)
        auditor_files = git_diff_names(base_dir)
        auditor_summaries = [{"file": f, "summary": summarize_changed_file(f)} for f in auditor_files[:40]]

        health = {}
        for name, meta in checks.items():
            health[name] = {
                "ready": meta.get("ready", False),
                "status": meta.get("status"),
                "env": meta.get("env"),
                "detail_preview": (str(meta.get("detail", ""))[:160]),
            }

        exec_line = (
            f"BEN improved its logic {heals_today} time{'s' if heals_today != 1 else ''} today "
            "based on model disagreements."
        )

        free_lim = max(0, int(os.getenv("FREE_USER_LIMIT", "5") or "5"))
        monetization = {
            "free_user_limit_24h": free_lim,
            "free_models": "gpt-4o-mini · gemini-1.5-flash (Claude lane off)",
            "pro_models": "gpt-4o · gemini-1.5-pro · claude-3-5-sonnet-20241022",
            "routing_note": (
                "Smart routing: DB tier `free` uses budget lanes only; tier `pro` enables full ensemble "
                "(including Claude). Usage rows enforce the rolling 24h cap."
            ),
        }

        return JSONResponse(
            {
                "health": health,
                "model_strings": names,
                "truth_consensus_pct": pct,
                "truth_message": truth_tip,
                "cfo": cfo,
                "performance": performance,
                "auditor": {"uncommitted_files": auditor_summaries},
                "errors": get_last_errors_translated(3),
                "railway_env": railway_env,
                "railway_cloud": railway_cloud,
                "monetization": monetization,
                "self_heals_today": heals_today,
                "self_heal_exec_line": exec_line,
            }
        )

    @router.get("/review-auto-fix", response_class=HTMLResponse)
    async def review_auto_fix_page():
        conn = db_connect()
        rows: list[tuple[Any, ...]] = []
        try:
            cur = conn.cursor()
            ob = sql_order_by_datetime_desc("created_at")
            cur.execute(
                db_adapt(
                    f"""
                SELECT created_at, telemetry_run_id, consensus_pct, rationale, instruction_addendum
                FROM self_heals ORDER BY {ob} LIMIT 120
                """
                )
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        lis = ""
        if not rows:
            lis = "<li>No self-heals recorded yet. When ensemble consensus stays under 50% on a run, the Analyzer may append a learned line to guide BEN.</li>"
        else:
            for created_at, tr_id, cons_pct, rationale, instr in rows:
                ra = html_escape.escape(str(rationale or ""))
                ins = html_escape.escape(str(instr or ""))
                lis += (
                    f"<li><strong>{html_escape.escape(str(created_at))}</strong> · "
                    f"telemetry #{int(tr_id or 0)} · consensus was {cons_pct}%<br/>"
                    f"<em>Rationale:</em> {ra}<br/>"
                    f"<strong>Applied instruction:</strong><br/><code>{ins}</code></li>"
                )
        body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Review Auto-Fix</title>
<style>
body {{ font-family: system-ui,sans-serif; background:#0c0d10;color:#e8e9ec;line-height:1.5; padding:28px 20px 60px; max-width:760px;margin:0 auto; }}
h1 {{ font-size:1.25rem; margin:0 0 12px; }}
p {{ color:#8b909a;font-size:.95rem;margin:0 0 20px; }}
a {{ color:#6366f1; }}
ul {{ padding-left:1.15rem; }}
li {{ margin-bottom:16px; }}
code {{ display:block; white-space:pre-wrap; background:#15171c; border:1px solid #252830; padding:10px; border-radius:8px; font-size:.85rem; margin-top:6px; }}
</style></head><body>
<a href="/status">← Status dashboard</a>
<h1>Review Auto-Fix</h1>
<p>Each entry is an automatic instruction line appended to <code>system_instructions.txt</code> and folded into BEN’s system prompt so similar disagreements are handled more consistently next time.</p>
<ul>{lis}</ul>
</body></html>"""
        return HTMLResponse(body)

    @router.post("/api/founder/backup")
    async def api_backup():
        try:
            p = create_stable_backup_zip(base_dir)
            return {"success": True, "path": str(p.relative_to(base_dir))}
        except Exception as e:
            log_status_error(str(e), "backup")
            raise HTTPException(500, str(e))

    @router.post("/api/founder/restore")
    async def api_restore():
        z = latest_stable_zip(base_dir)
        if not z or not z.is_file():
            raise HTTPException(404, "No stable backup zip found in /backups")
        try:
            extract_stable_zip(base_dir, z)
            restart_server_after_restore(base_dir)
            return {"success": True, "restarted": True}
        except Exception as e:
            log_status_error(str(e), "restore")
            raise HTTPException(500, str(e))

    app.include_router(router)
