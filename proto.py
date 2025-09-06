from flask import Flask, request, jsonify
import sqlite3
import os
import traceback
from flask_cors import CORS
from face_recognition_simple import SimpleFaceRecognitionService
import base64

app = Flask(__name__)
CORS(app)

# Automatically select persistent or temp data dir
PERSISTENT_DIR = "/mnt/data"
TMP_DIR = "/tmp/data"

# Root Route [changed by : Ayush]
# -->
@app.route("/", methods=["GET"])
def home():
    return jsonify({"success": True, "message": "Server is Running"})
# ---

def get_data_dir():
    # If persistent disk exists and is writable, use it
    if os.path.exists(PERSISTENT_DIR) and os.access(PERSISTENT_DIR, os.W_OK):
        return PERSISTENT_DIR
    # Otherwise, fall back to /tmp/data
    return TMP_DIR

DATA_DIR = get_data_dir()
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
IMAGE_FOLDER = os.path.join(DATA_DIR, "student_images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Initialize face recognition service
face_service = SimpleFaceRecognitionService()

# Database Setup 
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Students table
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT,
            face_encoding TEXT,
            present INTEGER DEFAULT 0
        )
    """)
    # Teachers table
    c.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

# Teacher Verification 
def verify_teacher(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM teachers WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

# Routes 

# Student Registration
@app.route("/student/register", methods=["POST"])
def register_student():
    try:
        name = request.form.get("name")
        image = request.files.get("image")

        if not name or not image:
            return jsonify({"success": False, "error": "Name and image required"}), 400

        # Save image to persistent folder
        image_path = os.path.join(IMAGE_FOLDER, f"{name}_{image.filename}")
        image.save(image_path)

        # Generate face encoding
        face_encoding = face_service.load_face_from_image_path(image_path)
        if face_encoding is None:
            return jsonify({"success": False, "error": "No face detected in the image. Please upload a clear photo with a visible face."}), 400

        # Convert face encoding to JSON string for storage
        face_encoding_json = face_service.get_face_encoding_as_json(face_encoding)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO students (name, image_path, face_encoding) VALUES (?, ?, ?)", 
                 (name, image_path, face_encoding_json))
        student_id = c.lastrowid
        conn.commit()
        conn.close()

        # Add to face recognition service
        face_service.add_known_face(face_encoding, name, student_id)

        return jsonify({
            "success": True, 
            "message": f"Student {name} registered successfully",
            "student_id": student_id
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Teacher Login
@app.route("/teacher/login", methods=["POST"])
def teacher_login():
    try:
        username = request.form.get("username")
        password = request.form.get("password")

        teacher_id = verify_teacher(username, password)
        if not teacher_id:
            return jsonify({"success": False, "error": "Invalid credentials"}), 401

        return jsonify({"success": True, "message": "Login successful", "teacher_id": teacher_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Mark Attendance
@app.route("/teacher/attendance", methods=["POST"])
def mark_attendance():
    try:
        data = request.json
        recognized_students = data.get("recognized_students", [])

        present_students = []
        unidentifiable_students = []

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        for student in recognized_students:
            sid = student.get("id")
            confidence = student.get("confidence", 1.0)

            if confidence >= 0.5:
                c.execute("UPDATE students SET present=1 WHERE id=?", (sid,))
                present_students.append(sid)
            else:
                unidentifiable_students.append(sid)

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "present_students": present_students,
            "unidentifiable_students": unidentifiable_students
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Manual Mark
@app.route("/teacher/manual_mark", methods=["POST"])
def manual_mark():
    try:
        student_id = request.form.get("student_id")
        status = request.form.get("status")  # "present" or "absent"

        if not student_id or status not in ["present", "absent"]:
            return jsonify({"success": False, "error": "Invalid input"}), 400

        present_value = 1 if status == "present" else 0

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE students SET present=? WHERE id=?", (present_value, student_id))
        conn.commit()
        conn.close()

        return jsonify({"success": True, "message": f"Student {student_id} marked as {status}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# List Students
@app.route("/students", methods=["GET"])
def list_students():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name, present FROM students")
        rows = c.fetchall()
        conn.close()

        students = [
            {"id": row[0], "name": row[1], "status": "present" if row[2] == 1 else "absent"}
            for row in rows
        ]
        return jsonify({"students": students})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Face Recognition Endpoints

# Recognize faces in uploaded image
@app.route("/face/recognize", methods=["POST"])
def recognize_faces():
    try:
        image = request.files.get("image")
        tolerance = float(request.form.get("tolerance", 0.6))

        if not image:
            return jsonify({"success": False, "error": "Image required"}), 400

        # Read image bytes
        image_bytes = image.read()
        
        # Recognize faces
        results = face_service.recognize_faces_in_image(image_bytes, tolerance)
        
        return jsonify({
            "success": True,
            "faces_detected": len(results),
            "recognitions": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Recognize faces from base64 image
@app.route("/face/recognize-base64", methods=["POST"])
def recognize_faces_base64():
    try:
        data = request.json
        base64_image = data.get("image")
        tolerance = float(data.get("tolerance", 0.6))

        if not base64_image:
            return jsonify({"success": False, "error": "Base64 image required"}), 400

        # Recognize faces
        results = face_service.recognize_faces_in_image(base64_image, tolerance)
        
        return jsonify({
            "success": True,
            "faces_detected": len(results),
            "recognitions": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Mark attendance using face recognition
@app.route("/attendance/face-mark", methods=["POST"])
def mark_attendance_face():
    try:
        image = request.files.get("image")
        tolerance = float(request.form.get("tolerance", 0.6))

        if not image:
            return jsonify({"success": False, "error": "Image required"}), 400

        # Read image bytes
        image_bytes = image.read()
        
        # Recognize faces
        results = face_service.recognize_faces_in_image(image_bytes, tolerance)
        
        # Filter results with confidence >= 60%
        recognized_students = []
        for result in results:
            if result['confidence'] >= 60 and result['id'] is not None:
                recognized_students.append({
                    'id': result['id'],
                    'name': result['name'],
                    'confidence': result['confidence']
                })

        # Mark attendance for recognized students
        present_students = []
        unidentifiable_students = []

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        for student in recognized_students:
            sid = student.get("id")
            confidence = student.get("confidence", 0)

            if confidence >= 60:
                c.execute("UPDATE students SET present=1 WHERE id=?", (sid,))
                present_students.append({
                    "id": sid,
                    "name": student.get("name"),
                    "confidence": confidence
                })
            else:
                unidentifiable_students.append(sid)

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "faces_detected": len(results),
            "recognized_students": recognized_students,
            "present_students": present_students,
            "unidentifiable_students": unidentifiable_students
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Load all students into face recognition service
@app.route("/face/load-students", methods=["POST"])
def load_students_into_face_service():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name, image_path, face_encoding FROM students")
        rows = c.fetchall()
        conn.close()

        # Clear existing faces
        face_service.clear_known_faces()

        loaded_count = 0
        for row in rows:
            student_id, name, image_path, face_encoding_json = row
            
            if face_encoding_json:
                # Load from stored encoding
                try:
                    face_encoding = face_service.load_face_encoding_from_json(face_encoding_json)
                    face_service.add_known_face(face_encoding, name, student_id)
                    loaded_count += 1
                except Exception as e:
                    print(f"Error loading face encoding for {name}: {str(e)}")
            elif image_path and os.path.exists(image_path):
                # Generate encoding from image
                face_encoding = face_service.load_face_from_image_path(image_path)
                if face_encoding is not None:
                    face_service.add_known_face(face_encoding, name, student_id)
                    loaded_count += 1
                    
                    # Update database with new encoding
                    face_encoding_json = face_service.get_face_encoding_as_json(face_encoding)
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("UPDATE students SET face_encoding=? WHERE id=?", (face_encoding_json, student_id))
                    conn.commit()
                    conn.close()

        return jsonify({
            "success": True,
            "message": f"Loaded {loaded_count} students into face recognition service",
            "loaded_count": loaded_count,
            "total_students": len(rows)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Get face recognition service status
@app.route("/face/status", methods=["GET"])
def face_service_status():
    try:
        return jsonify({
            "success": True,
            "known_faces_count": face_service.get_known_faces_count(),
            "known_face_names": face_service.known_face_names
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Reset attendance for all students
@app.route("/attendance/reset", methods=["POST"])
def reset_attendance():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE students SET present=0")
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Attendance reset for all students"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Seed Teacher
def seed_teacher():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO teachers (username, password) VALUES (?, ?)", ("teacher1", "1234"))
    conn.commit()
    conn.close()

# Global error handler: returns JSON instead of HTML for 500 errors
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Initialization (runs at import, not just __main__)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
init_db()
seed_teacher()

# Load existing students into face recognition service
def load_existing_students():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name, image_path, face_encoding FROM students")
        rows = c.fetchall()
        conn.close()

        loaded_count = 0
        for row in rows:
            student_id, name, image_path, face_encoding_json = row
            
            if face_encoding_json:
                # Load from stored encoding
                try:
                    face_encoding = face_service.load_face_encoding_from_json(face_encoding_json)
                    face_service.add_known_face(face_encoding, name, student_id)
                    loaded_count += 1
                except Exception as e:
                    print(f"Error loading face encoding for {name}: {str(e)}")
            elif image_path and os.path.exists(image_path):
                # Generate encoding from image
                face_encoding = face_service.load_face_from_image_path(image_path)
                if face_encoding is not None:
                    face_service.add_known_face(face_encoding, name, student_id)
                    loaded_count += 1
                    
                    # Update database with new encoding
                    face_encoding_json = face_service.get_face_encoding_as_json(face_encoding)
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("UPDATE students SET face_encoding=? WHERE id=?", (face_encoding_json, student_id))
                    conn.commit()
                    conn.close()

        print(f"Loaded {loaded_count} students into face recognition service")
    except Exception as e:
        print(f"Error loading existing students: {str(e)}")

# Load existing students
load_existing_students()

# Main 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
