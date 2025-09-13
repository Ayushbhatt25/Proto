from flask import Flask, request, jsonify
import sqlite3
import os
import traceback
from flask_cors import CORS
import face_recognition
import cv2
import numpy as np
from PIL import Image
import pickle
from typing import Dict, List
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

# Face Recognition Service Class
class FaceRecognitionService:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_file = "face_model.pkl"
        
    def train_model(self, students_folder: str = "students"):
        """
        Train the face recognition model using images from student folders
        Each student should have a folder with their name containing 2 images
        """
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(students_folder):
            raise Exception(f"Students folder '{students_folder}' not found")
        
        student_folders = [f for f in os.listdir(students_folder) if os.path.isdir(os.path.join(students_folder, f))]
        
        if not student_folders:
            raise Exception("No student folders found")
        
        for student_name in student_folders:
            student_path = os.path.join(students_folder, student_name)
            image_files = [f for f in os.listdir(student_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            if len(image_files) < 2:
                print(f"Warning: {student_name} has less than 2 images, skipping...")
                continue
            
            # Process up to 2 images per student
            for image_file in image_files[:2]:
                image_path = os.path.join(student_path, image_file)
                try:
                    # Load image
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        # Use the first face found
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(student_name)
                        print(f"Trained on {student_name} from {image_file}")
                    else:
                        print(f"No face found in {image_path}")
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
        
        if not self.known_face_encodings:
            raise Exception("No valid face encodings found during training")
        
        # Save the trained model
        model_data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model trained successfully with {len(self.known_face_encodings)} face encodings")
        return {"message": f"Model trained successfully with {len(self.known_face_encodings)} face encodings"}
    
    def load_model(self):
        """Load the trained model from file"""
        if not os.path.exists(self.model_file):
            raise Exception("No trained model found. Please train the model first.")
        
        with open(self.model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        self.known_face_encodings = model_data['encodings']
        self.known_face_names = model_data['names']
        print(f"Model loaded with {len(self.known_face_encodings)} face encodings")
    
    def recognize_faces(self, image_data: bytes) -> Dict[str, float]:
        """
        Recognize faces in the uploaded image and return accuracy scores
        """
        if not self.known_face_encodings:
            self.load_model()
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image. Please ensure it's a valid image file.")
            
            # Convert BGR to RGB (OpenCV uses BGR, face_recognition uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                return {"message": "No faces found in the image"}
            
            results = {}
            
            for face_encoding in face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                # Find the best match
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    # Calculate accuracy (1 - distance, where distance is between 0 and 1)
                    accuracy = max(0, 1 - face_distances[best_match_index])
                    person_name = self.known_face_names[best_match_index]
                    
                    # If multiple faces of the same person, keep the highest accuracy
                    if person_name in results:
                        results[person_name] = max(results[person_name], accuracy)
                    else:
                        results[person_name] = accuracy
            
            return results
            
        except Exception as e:
            print(f"Error in recognize_faces: {str(e)}")
            raise ValueError(f"Error processing image: {str(e)}")

# Initialize face recognition service
face_service = FaceRecognitionService()

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

        # Store student in database (simplified for new face recognition system)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO students (name, image_path) VALUES (?, ?)", 
                 (name, image_path))
        student_id = c.lastrowid
        conn.commit()
        conn.close()

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

# Train the face recognition model
@app.route("/train", methods=["POST"])
def train_model():
    """Train the face recognition model using student folders"""
    try:
        students_folder = request.form.get("students_folder", "students")
        result = face_service.train_model(students_folder)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Upload image for face recognition
@app.route("/upload", methods=["POST"])
def upload_image():
    """Upload an image for face recognition"""
    try:
        # Check if image file is present
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check content type and file extension
        if not file.content_type or not file.content_type.startswith('image/'):
            # Fallback: check file extension
            if not file.filename:
                return jsonify({"success": False, "error": "No filename provided"}), 400
            
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
            if f'.{file_ext}' not in valid_extensions:
                return jsonify({"success": False, "error": "File must be an image (jpg, jpeg, png, bmp, gif)"}), 400
        
        # Read the uploaded file
        image_data = file.read()
        
        if len(image_data) == 0:
            return jsonify({"success": False, "error": "Empty file uploaded"}), 400
        
        # Recognize faces in the image
        results = face_service.recognize_faces(image_data)
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "recognized_faces": results,
            "total_faces_found": len(results)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

# Get model status
@app.route("/model/status", methods=["GET"])
def model_status():
    """Check if the model is trained and ready"""
    try:
        face_service.load_model()
        return jsonify({
            "success": True,
            "status": "ready",
            "total_encodings": len(face_service.known_face_encodings),
            "known_people": list(set(face_service.known_face_names))
        })
    except:
        return jsonify({"success": True, "status": "not_trained"})

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

# Note: Face recognition model needs to be trained separately using /train endpoint

# Main 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
