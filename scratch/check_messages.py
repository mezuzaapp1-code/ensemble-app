import sqlite3
import os

db_path = os.getenv("DB_PATH", "conversations.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()
try:
    c.execute("SELECT session_id, model, role, content FROM messages ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    print("Last 20 messages:")
    for row in rows:
        print(row)
except Exception as e:
    print("Error:", e)
conn.close()
