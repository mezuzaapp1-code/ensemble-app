from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from openai import OpenAI
from anthropic import Anthropic

import os
import uuid
import sqlite3
import json
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

load_dotenv()

app = FastAPI()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SESSIONS = {}
DB_PATH = os.getenv("DB_PATH", "conversations.db")

executor = ThreadPoolExecutor(max_workers=6)

# ========================
# DATABASE INITIALIZATION
# ========================

def init_db():
    """Initialize SQLite database for persistent conversation storage"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Sessions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            updated_at TEXT,
            title TEXT
        )
    """)
    
    # Messages table
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            model TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    
    # User profile / memory table
    c.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            session_id TEXT PRIMARY KEY,
            user_name TEXT,
            user_role TEXT,
            projects TEXT,
            preferences TEXT,
            memory_context TEXT,
            updated_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    
    # Ensemble results table
    c.execute("""
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
    """)
    
    # Learning Engine Feedback table
    c.execute("""
        CREATE TABLE IF NOT EXISTS learning_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            model TEXT,
            feedback_value INTEGER,
            timestamp TEXT
        )
    """)
    
    conn.commit()
    conn.close()

init_db()

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

class FeedbackRequest(BaseModel):
    session_id: str
    model: str
    feedback_value: int
    category: str

# ========================
# DATABASE HELPERS
# ========================

def get_conversation_history(session_id):
    """Retrieve conversation history from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT role, content FROM messages 
        WHERE session_id = ? 
        ORDER BY timestamp ASC
    """, (session_id,))
    messages = c.fetchall()
    conn.close()
    return messages

def save_profile(session_id, user_name=None, user_role=None, projects=None, preferences=None, memory_context=None):
    """Save or update a user's profile/memory data"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_name, user_role, projects, preferences, memory_context FROM profiles WHERE session_id = ?", (session_id,))
    existing = c.fetchone()

    if existing:
        current_name, current_role, current_projects, current_preferences, current_context = existing
        user_name = user_name if user_name is not None else current_name
        user_role = user_role if user_role is not None else current_role
        projects = projects if projects is not None else current_projects
        preferences = preferences if preferences is not None else current_preferences
        memory_context = memory_context if memory_context is not None else current_context

        c.execute("""
            UPDATE profiles
            SET user_name = ?, user_role = ?, projects = ?, preferences = ?, memory_context = ?, updated_at = ?
            WHERE session_id = ?
        """, (
            user_name,
            user_role,
            projects,
            preferences,
            memory_context,
            datetime.now().isoformat(),
            session_id
        ))
    else:
        c.execute("""
            INSERT INTO profiles (session_id, user_name, user_role, projects, preferences, memory_context, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            user_name,
            user_role,
            projects,
            preferences,
            memory_context,
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()


def get_profile(session_id):
    """Retrieve stored user profile/memory from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT user_name, user_role, projects, preferences, memory_context
        FROM profiles
        WHERE session_id = ?
    """, (session_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "user_name": row[0],
        "user_role": row[1],
        "projects": row[2],
        "preferences": row[3],
        "memory_context": row[4]
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("""
        INSERT INTO messages (session_id, model, role, content, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, model, role, content, timestamp))
    conn.commit()
    conn.close()

def create_session(session_id, title):
    """Create a new conversation session"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("""
        INSERT INTO sessions (session_id, created_at, updated_at, title)
        VALUES (?, ?, ?, ?)
    """, (session_id, now, now, title))
    conn.commit()
    conn.close()

def update_session_timestamp(session_id):
    """Update session's last update time"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE sessions SET updated_at = ? WHERE session_id = ?
    """, (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()

# ========================
# AI MODEL FUNCTIONS
# ========================

def ask_gpt(prompt, session_id=None):
    """Call GPT-4o with optional conversation context"""
    messages = [{"role": "user", "content": prompt}]
    
    if session_id:
        profile_context = build_profile_context(session_id)
        history = get_conversation_history(session_id)
        messages = profile_context + [
            {"role": role if role == "user" else "assistant", "content": content}
            for role, content in history
            if "Error" not in content
        ] + messages
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=800
    )
    return response.choices[0].message.content

def ask_gemini(prompt, session_id=None):
    """Call Gemini with optional conversation context"""
    if session_id:
        profile_context = build_profile_context(session_id)
        history = get_conversation_history(session_id)
        # Gemini API expects contents as list
        contents = []
        for msg in profile_context:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
        for role, content in history:
            if "Error" in content:
                continue
            contents.append({"role": role if role == "user" else "assistant", "parts": [{"text": content}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
    else:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    
    return response.text

def ask_claude(prompt, session_id=None):
    """Call Claude with optional conversation context"""
    messages = [{"role": "user", "content": prompt}]
    
    if session_id:
        profile_context = build_profile_context(session_id)
        history = get_conversation_history(session_id)
        messages = profile_context + [
            {"role": role if role == "user" else "assistant", "content": content}
            for role, content in history
            if "Error" not in content
        ] + messages
    
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=messages
    )
    return message.content[0].text

def ask_model(model, prompt, session_id=None):
    """Call the specified model with conversation context"""
    try:
        if model == "gpt":
            return ask_gpt(prompt, session_id)
        elif model == "gemini":
            return ask_gemini(prompt, session_id)
        elif model == "claude":
            return ask_claude(prompt, session_id)
        else:
            return "Unknown model"
    except Exception as e:
        return f"{model.upper()} Error: {str(e)}"

def run_parallel(tasks, session_id=None):
    """Run multiple AI model tasks in parallel"""
    futures = {}
    
    for key, model, prompt in tasks:
        futures[key] = executor.submit(ask_model, model, prompt, session_id)
    
    results = {}
    for key, future in futures.items():
        results[key] = future.result()
    
    return results

@app.get("/landing")
def landing():
    return FileResponse("landing.html")

@app.get("/")
def home():
    return FileResponse("index.html")

# ========================
# SESSION MANAGEMENT ENDPOINTS
# ========================

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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get session info
    c.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    session = c.fetchone()
    
    if not session:
        conn.close()
        return {"success": False, "error": "Session not found"}
    
    # Get all messages
    c.execute("""
        SELECT model, role, content, timestamp FROM messages 
        WHERE session_id = ? 
        ORDER BY timestamp ASC
    """, (session_id,))
    messages = c.fetchall()
    conn.close()
    profile = get_profile(session_id)
    
    return {
        "success": True,
        "session_id": session_id,
        "title": session[3],
        "created_at": session[1],
        "updated_at": session[2],
        "profile": profile,
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
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
        memory_context=req.memory_context
    )

    return {"success": True, "session_id": session_id, "profile": get_profile(session_id)}

@app.get("/sessions")
def list_sessions():
    """List all conversation sessions"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC")
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
def ask_single_model(req: AskRequest):
    """Ask a question to a single model with conversation context"""
    try:
        # Verify session exists
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT session_id FROM sessions WHERE session_id = ?", (req.session_id,))
        if not c.fetchone():
            conn.close()
            return {"success": False, "error": "Session not found"}
        conn.close()
        
        # Save user message
        save_message(req.session_id, req.model, "user", req.message)
        
        # Get response from model with conversation history
        response = ask_model(req.model, req.message, req.session_id)
        
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
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO learning_feedback (category, model, feedback_value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (req.category, req.model, req.feedback_value, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/ensemble/run")
def run_ensemble(req: RunRequest):
    """Run the 3-round ensemble analysis with multi-turn capability"""
    try:
        # Verify session exists
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT session_id FROM sessions WHERE session_id = ?", (req.session_id,))
        if not c.fetchone():
            conn.close()
            return {"success": False, "error": "Session not found"}
        
        # Save user question
        save_message(req.session_id, "ensemble", "user", req.question)
        
        conn.close()
        
        # Categorize question for Learning Engine
        category_prompt = f"""
Analyze the following question and classify it into exactly one of these categories: technical, strategy, code, creative.
Return ONLY the category name.
Question: {req.question}
"""
        category = ask_model("gpt", category_prompt).strip().lower()
        if category not in ["technical", "strategy", "code", "creative"]:
            category = "technical"
            
        # Fetch historical stats
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
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
        
        # =========================
        # ROUND 1
        # =========================
        
        gpt_r1_prompt = f"""
You are GPT, the technical analyst.

Analyze this question from an engineering and architecture perspective.
Focus on system design, scalability, risks, security, edge cases, and tradeoffs.

{f"Conversation context:{history_text}" if history_text else ""}

Question:
{req.question}
"""

        gemini_r1_prompt = f"""
You are Gemini, the implementation analyst.

Analyze this question from an implementation perspective.
Focus on APIs, code structure, backend execution, data flow, and practical build steps.

{f"Conversation context:{history_text}" if history_text else ""}

Question:
{req.question}
"""

        claude_r1_prompt = f"""
You are Claude, the deep reasoning analyst.

Analyze this question deeply.
Focus on strategy, reasoning quality, hidden assumptions, business risks, human behavior, and long-term implications.

{f"Conversation context:{history_text}" if history_text else ""}

Question:
{req.question}
"""

        round1 = run_parallel([
            ("gpt", "gpt", gpt_r1_prompt),
            ("gemini", "gemini", gemini_r1_prompt),
            ("claude", "claude", claude_r1_prompt)
        ], req.session_id)

        # Save Round 1 responses
        for model, response in round1.items():
            save_message(req.session_id, f"ensemble-round1", "assistant", response)

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
        consensus_raw = ask_model("gpt", consensus_prompt)
        consensus_data = []
        try:
            json_str = consensus_raw.replace('```json', '').replace('```', '').strip()
            consensus_data = json.loads(json_str)
        except Exception as e:
            print("Failed to parse consensus JSON:", e)

        # =========================
        # ROUND 2
        # =========================

        shared_round1 = f"""
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

        round2 = run_parallel([
            ("gpt", "gpt", gpt_r2_prompt),
            ("gemini", "gemini", gemini_r2_prompt),
            ("claude", "claude", claude_r2_prompt)
        ], req.session_id)

        # Save Round 2 responses
        for model, response in round2.items():
            save_message(req.session_id, f"ensemble-round2", "assistant", response)

        # =========================
        # ROUND 3 (SYNTHESIS)
        # =========================

        synthesis_prompt = f"""
You are the final synthesis engine.

You see the original question, all Round 1 answers, and all Round 2 critiques.

Your job:
1. Produce the final recommendation.
2. Identify consensus.
3. Identify disagreements.
4. Explain the best architecture.
5. List major risks.
6. Give scaling strategy.
7. Give business model.
8. Give next 3 practical actions.

{weights_instruction}

Original Question:
{req.question}

=== ROUND 1 ANSWERS ===

GPT:
{round1.get("gpt", "")}

Gemini:
{round1.get("gemini", "")}

Claude:
{round1.get("claude", "")}

=== ROUND 2 CRITIQUES ===

GPT Critique:
{round2.get("gpt", "")}

Gemini Critique:
{round2.get("gemini", "")}

Claude Critique:
{round2.get("claude", "")}
"""

        final = ask_model("gpt", synthesis_prompt, req.session_id)

        # Save final synthesis
        save_message(req.session_id, "ensemble-final", "assistant", final)
        
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
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }