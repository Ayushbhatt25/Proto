import os
import sqlite3
import main
from datetime import datetime

main.init_db()
main.seed_teacher()

conn = sqlite3.connect(main.DB_PATH)
cur = conn.cursor()

# Insert sample student
cur.execute("INSERT INTO students (name, image_path, present) VALUES (?, ?, ?)", ("Alice", os.path.join(main.IMAGE_FOLDER, "Alice_sample.jpg"), 0))
student_id = cur.lastrowid

# Create sample session
session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
cur.execute("INSERT OR REPLACE INTO sessions (session_id, date, class_name, teacher_name) VALUES (?, ?, ?, ?)", (session_id, datetime.now().date().isoformat(), "Class A", "Mr. Kumar"))

# Insert attendance record for student name
cur.execute(
    "INSERT INTO attendance_records (session_id, student_name, date, is_present, accuracy, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
    (session_id, "Alice", datetime.now().date().isoformat(), 1, 0.95, datetime.now().isoformat())
)

conn.commit()

# Show quick summaries
cur.execute("SELECT id, name, present FROM students")
print("Students:", cur.fetchall())
cur.execute("SELECT session_id, date, class_name, teacher_name FROM sessions ORDER BY date DESC")
print("Sessions:", cur.fetchall())
cur.execute("SELECT session_id, student_name, is_present, accuracy FROM attendance_records ORDER BY id DESC LIMIT 5")
print("Attendance records (latest 5):", cur.fetchall())

conn.close()
