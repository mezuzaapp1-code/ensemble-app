import sqlite3
import os

db_path = os.getenv("DB_PATH", "conversations.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()
try:
    c.execute("SELECT * FROM trial_usage")
    rows = c.fetchall()
    print("trial_usage table:")
    for row in rows:
        print(row)
except Exception as e:
    print("Error:", e)
conn.close()
