import sqlite3
import os

db_path = os.getenv("DB_PATH", "conversations.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()
try:
    c.execute("SELECT * FROM profiles ORDER BY updated_at DESC LIMIT 5")
    rows = c.fetchall()
    print("Last 5 profiles:")
    for row in rows:
        print(row)
except Exception as e:
    print("Error:", e)
conn.close()
