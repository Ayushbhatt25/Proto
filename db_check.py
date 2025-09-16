import sqlite3
import os
import main

main.init_db()
main.seed_teacher()
print("DB path:", os.path.abspath(main.DB_PATH))

conn = sqlite3.connect(main.DB_PATH)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print("Tables:", [r[0] for r in cur.fetchall()])

counts = {}
for t in ["students", "teachers", "sessions", "attendance_records"]:
    try:
        cur.execute("SELECT COUNT(*) FROM " + t)
        counts[t] = cur.fetchone()[0]
    except Exception as e:
        counts[t] = f"error: {e}"
print("Counts:", counts)

conn.close()
