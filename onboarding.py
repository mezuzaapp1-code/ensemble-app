"""Session profile, onboarding fields, language prompts, and uploaded-document helpers.

Separated from ``main`` so FastAPI routes and streaming stay thin; behavior matches the
previous implementations that lived in ``main.py``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from datetime import datetime
from typing import Any

import PyPDF2
from fastapi import UploadFile

from ensemble_db import adapt as sqlq, connect_db


def _xe(cur: Any, sql: str, params: tuple | list = ()) -> Any:
    return cur.execute(sqlq(sql), params)


DEFAULT_ACTIVE_TOOLS_JSON = json.dumps(["gpt", "gemini", "claude"])

# Sentinel: omit param to preserve DB value when calling save_profile
_PROFILE_KEEP = object()


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


def _encrypt_sensitive_field(plain: str) -> str:
    """Store-at-rest obfuscation for date of birth (symmetric XOR + base64; key from JWT_SECRET)."""
    s = (plain or "").strip()
    if not s:
        return ""
    secret = (os.getenv("JWT_SECRET") or "ensemble-dev-default-secret").encode()
    key = hashlib.sha256(secret + b"|ensemble-dob|").digest()
    b = s.encode("utf-8")
    xored = bytes(b[i] ^ key[i % len(key)] for i in range(len(b)))
    return "x1:" + base64.urlsafe_b64encode(xored).decode("ascii")


def build_language_instruction(profile: dict) -> str:
    """Short instruction injected into model prompts for response language."""
    lang = (profile.get("preferred_language") or "auto").strip().lower()
    if lang in ("", "auto"):
        return "Respond in the same language as the user's message."
    labels = {
        "en": "English",
        "english": "English",
        "he": "Hebrew",
        "ar": "Arabic",
        "ru": "Russian",
        "other": "the user's preferred language",
    }
    label = labels.get(lang, lang)
    return f"Always respond in {label}. Do not switch languages."


def profile_save_kwargs_from_request(req: Any) -> dict[str, Any]:
    """Map optional onboarding-related fields from a profile request body to ``save_profile`` kwargs."""
    kw: dict[str, Any] = {}
    if getattr(req, "date_of_birth", None) is not None:
        kw["date_of_birth_plain"] = req.date_of_birth
    if getattr(req, "preferred_language", None) is not None:
        kw["preferred_language"] = req.preferred_language
    if getattr(req, "ai_usage_category", None) is not None:
        kw["ai_usage_category"] = req.ai_usage_category
    if getattr(req, "onboarding_completed", None) is not None:
        kw["onboarding_completed"] = req.onboarding_completed
    return kw


def save_profile(
    session_id,
    user_name=None,
    user_role=None,
    projects=None,
    preferences=None,
    memory_context=None,
    uploaded_text=_PROFILE_KEEP,
    active_tools=_PROFILE_KEEP,
    preferred_language=_PROFILE_KEEP,
    ai_usage_category=_PROFILE_KEEP,
    onboarding_completed=_PROFILE_KEEP,
    date_of_birth_plain=_PROFILE_KEEP,
):
    """Save or update a user's profile. Use sentinel _PROFILE_KEEP to leave blobs/tools unchanged."""
    conn = connect_db()
    c = conn.cursor()
    _xe(
        c,
        """
        SELECT user_name, user_role, projects, preferences, memory_context, uploaded_text,
               COALESCE(active_tools, ?),
               COALESCE(preferred_language, 'auto'), ai_usage_category,
               COALESCE(onboarding_completed, 0), date_of_birth_encrypted
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
            current_pl,
            current_auc,
            current_oc,
            current_dob_enc,
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

        if preferred_language is _PROFILE_KEEP:
            pl_use = current_pl or "auto"
        else:
            pl_use = preferred_language or "auto"
        if ai_usage_category is _PROFILE_KEEP:
            auc_use = current_auc
        else:
            auc_use = ai_usage_category
        if onboarding_completed is _PROFILE_KEEP:
            oc_use = int(current_oc or 0)
        else:
            oc_use = 1 if onboarding_completed else 0
        if date_of_birth_plain is _PROFILE_KEEP:
            dob_use = current_dob_enc
        else:
            dob_use = _encrypt_sensitive_field(date_of_birth_plain) if (date_of_birth_plain or "").strip() else ""

        _xe(
            c,
            """
            UPDATE profiles
            SET user_name = ?, user_role = ?, projects = ?, preferences = ?, memory_context = ?,
                uploaded_text = ?, active_tools = ?, preferred_language = ?, ai_usage_category = ?,
                onboarding_completed = ?, date_of_birth_encrypted = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (
                user_name,
                user_role,
                projects,
                preferences,
                memory_context,
                uploaded_use,
                tools_use,
                pl_use,
                auc_use,
                oc_use,
                dob_use,
                datetime.now().isoformat(),
                session_id,
            ),
        )
    else:
        up_ins = None if uploaded_text is _PROFILE_KEEP else uploaded_text
        if active_tools is _PROFILE_KEEP:
            tools_ins = DEFAULT_ACTIVE_TOOLS_JSON
        else:
            tools_ins = json.dumps(normalize_active_tools_list(active_tools if isinstance(active_tools, list) else []))

        pl_ins = "auto"
        if preferred_language is not _PROFILE_KEEP:
            pl_ins = preferred_language or "auto"
        auc_ins = None if ai_usage_category is _PROFILE_KEEP else ai_usage_category
        oc_ins = 0 if onboarding_completed is _PROFILE_KEEP else (1 if onboarding_completed else 0)
        dob_ins = ""
        if date_of_birth_plain is not _PROFILE_KEEP:
            dob_ins = _encrypt_sensitive_field(date_of_birth_plain) if (date_of_birth_plain or "").strip() else ""

        _xe(
            c,
            """
            INSERT INTO profiles (
                session_id, user_name, user_role, projects, preferences, memory_context,
                uploaded_text, active_tools, preferred_language, ai_usage_category,
                onboarding_completed, date_of_birth_encrypted, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_name,
                user_role,
                projects,
                preferences,
                memory_context,
                up_ins,
                tools_ins,
                pl_ins,
                auc_ins,
                oc_ins,
                dob_ins,
                datetime.now().isoformat(),
            ),
        )

    conn.commit()
    conn.close()


def get_profile(session_id):
    """Retrieve stored user profile/memory from database."""
    conn = connect_db()
    c = conn.cursor()
    _xe(
        c,
        """
        SELECT user_name, user_role, projects, preferences, memory_context, uploaded_text,
               COALESCE(active_tools, ?),
               COALESCE(preferred_language, 'auto'), ai_usage_category,
               COALESCE(onboarding_completed, 0), date_of_birth_encrypted
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
        "preferred_language": row[7],
        "ai_usage_category": row[8],
        "onboarding_completed": bool(row[9]),
        "date_of_birth_encrypted": row[10],
    }


def profile_for_client(profile):
    """Strip large blobs from profile before JSON responses."""
    if not profile:
        return None
    d = dict(profile)
    txt = (d.pop("uploaded_text", None) or "").strip()
    d.pop("date_of_birth_encrypted", None)
    d["has_uploaded_document"] = bool(txt)
    if txt:
        d["uploaded_char_count"] = len(txt)
    raw_tools = d.get("active_tools") or DEFAULT_ACTIVE_TOOLS_JSON
    try:
        parsed = json.loads(raw_tools)
    except Exception:
        parsed = json.loads(DEFAULT_ACTIVE_TOOLS_JSON)
    d["active_tools"] = normalize_active_tools_list(parsed if isinstance(parsed, list) else [])
    if "onboarding_completed" not in d:
        d["onboarding_completed"] = False
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
    _xe(c, "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
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
    lang_line = build_language_instruction(profile)
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
        content = lang_line
        return [{"role": "user", "content": content}]

    content = (
        lang_line + "\n\nPlease remember the user profile and context for this conversation.\n"
        + "\n".join(parts)
    )
    return [{"role": "user", "content": content}]


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
