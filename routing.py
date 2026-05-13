"""Tier-based model routing, token saver mode, and BEN analyst lane selection.

Loaded by ``main`` after ``MODEL_REGISTRY`` is defined; call ``install_model_defaults`` once
so effective model IDs match the registry when no billing tier context is active.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

CHAT_MODEL_KEYS = frozenset({"gpt", "gpt-fast", "gemini", "gemini-fast", "claude", "ben"})

TIER_ROUTING_CTX: ContextVar[Optional["TierRouting"]] = ContextVar("tier_routing_ctx", default=None)

# Populated by ``install_model_defaults`` from ``main`` (mirrors MODEL_REGISTRY-derived values).
_DEFAULT_OPENAI = ""
_DEFAULT_GEMINI_FAST = ""
_DEFAULT_CLAUDE_PRIMARY = ""
_DEFAULT_CLAUDE_FALLBACKS: tuple[str, ...] = ()
_DEFAULT_GEMINI_FALLBACKS: tuple[str, ...] = ()


def install_model_defaults(
    *,
    openai_default: str,
    gemini_fast: str,
    claude_primary: str,
    claude_fallbacks: tuple[str, ...],
    gemini_fallbacks: tuple[str, ...],
) -> None:
    """Wire registry-derived model IDs used when ``TierRouting`` context is unset."""
    global _DEFAULT_OPENAI, _DEFAULT_GEMINI_FAST, _DEFAULT_CLAUDE_PRIMARY
    global _DEFAULT_CLAUDE_FALLBACKS, _DEFAULT_GEMINI_FALLBACKS
    _DEFAULT_OPENAI = openai_default
    _DEFAULT_GEMINI_FAST = gemini_fast
    _DEFAULT_CLAUDE_PRIMARY = claude_primary
    _DEFAULT_CLAUDE_FALLBACKS = claude_fallbacks
    _DEFAULT_GEMINI_FALLBACKS = gemini_fallbacks


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
    return TIER_ROUTING_CTX.get()


def effective_openai_default(model_key_hint: Optional[str] = None) -> str:
    tr = current_tier_routing()
    if tr:
        if model_key_hint == "gpt-fast":
            return tr.openai_fast
        return tr.openai_main
    return _DEFAULT_OPENAI


def effective_gemini_default() -> str:
    tr = current_tier_routing()
    return tr.gemini_main if tr else _DEFAULT_GEMINI_FAST


def effective_claude_primary() -> Optional[str]:
    tr = current_tier_routing()
    if tr:
        return tr.claude_main
    return _DEFAULT_CLAUDE_PRIMARY


def effective_claude_fallbacks_for_call() -> tuple[str, ...]:
    tr = current_tier_routing()
    if tr:
        return tr.claude_fallbacks
    return _DEFAULT_CLAUDE_FALLBACKS


async def _run_with_routing(routing: TierRouting, coro: Any):
    tok = TIER_ROUTING_CTX.set(routing)
    try:
        return await coro
    finally:
        TIER_ROUTING_CTX.reset(tok)


def _stored_label_for_model_key(model_key: str) -> str:
    mk = (model_key or "").strip().lower()
    if mk == "gpt":
        return effective_openai_default()
    if mk == "gpt-fast":
        return effective_openai_default("gpt-fast")
    if mk in ("gemini", "gemini-fast"):
        return effective_gemini_default()
    if mk == "claude":
        return effective_claude_primary() or _DEFAULT_CLAUDE_PRIMARY
    return mk


def _ben_placeholder_mixed_failure_success(texts: list[str]) -> bool:
    if len(texts) < 2:
        return False

    def _failed(t: str) -> bool:
        x = (t or "").lower()
        return (
            " error:" in x
            or x.startswith("gpt error")
            or x.startswith("claude error")
            or x.startswith("gemini error")
        )

    flags = [_failed(t) for t in texts]
    return any(flags) and not all(flags)


def _gemini_candidate_models(primary: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in (primary, *_DEFAULT_GEMINI_FALLBACKS):
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


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
    has_complex_signals = any(
        k in text.lower()
        for k in [
            "compare",
            "architecture",
            "scalability",
            "security",
            "tradeoff",
            "step-by-step",
            "detailed",
            "multi",
            "benchmark",
        ]
    )
    punctuation_load = sum(text.count(ch) for ch in [":", ";", "?", "(", ")", ",", "\n"])
    if words <= 18 and punctuation_load <= 3 and not has_complex_signals:
        return "ECONOMY"
    return "FULL"


def routing_tier_label(token_saver_mode: str, web_search: bool) -> str:
    if token_saver_mode == "ECONOMY":
        return "Economy"
    if web_search:
        return "Premium"
    return "Standard"


BEN_TOOL_ORDER = ["gpt", "gemini", "claude"]
ECON_LANE_NOTICE = "[Maintenance] Token saver mode: model skipped for cost efficiency."
LANE_OFF_NOTICE = "(BEN Workspace: this analyst is turned off.)"
TIER_FREE_LANE_NOTICE = "(BEN Free tier: Claude is available on BEN Pro.)"


def prepare_workspace_tools_for_ensemble(active_tools: set[str], tier: str) -> list[str]:
    """Intersect saved workspace tools with tier-allowed analysts; default GPT-only when empty."""
    tier_allowed = frozenset({"gpt", "gemini"}) if tier == "free" else frozenset({"gpt", "gemini", "claude"})
    out = sorted(set(active_tools) & tier_allowed)
    return out if out else ["gpt"]


def compute_ben_r1_lane_plan(
    *,
    token_saver_mode: str,
    active_workspace_tools: list[str],
    tier: str,
) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Decide which analysts run in round 1 and skip-reason copy for disabled lanes.

    Returns ``(ben_tool_order, routed_models, skip_lane_messages)`` — same structure as
    ``/ensemble/run`` and ``/ensemble/stream`` used inline before extraction.
    """
    ben_tool_order = list(BEN_TOOL_ORDER)
    if token_saver_mode == "ECONOMY":
        routed_models: list[str] = []
        if "gpt" in active_workspace_tools:
            routed_models.append("gpt")
    else:
        routed_models = [m for m in ben_tool_order if m in active_workspace_tools]
    if not routed_models:
        routed_models = ["gpt"]

    skip_lane: dict[str, str] = {}
    for mm in ben_tool_order:
        if mm in routed_models:
            continue
        if mm == "claude" and tier == "free":
            skip_lane[mm] = TIER_FREE_LANE_NOTICE
        else:
            skip_lane[mm] = ECON_LANE_NOTICE if token_saver_mode == "ECONOMY" and mm != "gpt" else LANE_OFF_NOTICE

    return ben_tool_order, routed_models, skip_lane
