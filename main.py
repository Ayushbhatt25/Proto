from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import cv2
import numpy as np
from PIL import Image
import os
import pickle
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime, date
from pydantic import BaseModel
import sqlite3
import traceback

app = FastAPI(title="Face Recognition API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store trained model data
known_face_encodings = []
known_face_names = []
model_file = "face_model.pkl"

# Data directories (persistent if available)
PERSISTENT_DIR = "/mnt/data"
TMP_DIR = "/tmp/data"

def get_data_dir():
    if os.path.exists(PERSISTENT_DIR) and os.access(PERSISTENT_DIR, os.W_OK):
        return PERSISTENT_DIR
    # Fallback for local/dev (including Windows)
    local_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(local_dir, exist_ok=True)
    return local_dir

DATA_DIR = get_data_dir()
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
IMAGE_FOLDER = os.path.join(DATA_DIR, "student_images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Pydantic models for API
class AttendanceRecord(BaseModel):
    student_name: str
    date: str
    is_present: bool
    accuracy: Optional[float] = None
    timestamp: str

class AttendanceSession(BaseModel):
    session_id: str
    date: str
    class_name: str
    teacher_name: str
    records: List[AttendanceRecord]

# Attendance data now stored in SQLite (see DB schema below)

class FaceRecognitionService:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
    def train_model(self, students_folder: str = "students"):
        """
        Train the face recognition model using images from student folders
        Each student should have a folder with their name containing 2 images
        """
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(students_folder):
            raise HTTPException(status_code=404, detail=f"Students folder '{students_folder}' not found")
        
        student_folders = [f for f in os.listdir(students_folder) if os.path.isdir(os.path.join(students_folder, f))]
        
        if not student_folders:
            raise HTTPException(status_code=404, detail="No student folders found")
        
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
            raise HTTPException(status_code=400, detail="No valid face encodings found during training")
        
        # Save the trained model
        model_data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model trained successfully with {len(self.known_face_encodings)} face encodings")
        return {"message": f"Model trained successfully with {len(self.known_face_encodings)} face encodings"}
    
    def load_model(self):
        """Load the trained model from file"""
        if not os.path.exists(model_file):
            raise Exception("No trained model found. Please train the model first.")
        
        with open(model_file, 'rb') as f:
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

# Initialize the face recognition service
face_service = FaceRecognitionService()

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.post("/train")
async def train_model(students_folder: str = "students"):
    """Train the face recognition model using student folders"""
    try:
        result = face_service.train_model(students_folder)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for face recognition"""
    # Check content type and file extension
    if not file.content_type or not file.content_type.startswith('image/'):
        # Fallback: check file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        if f'.{file_ext}' not in valid_extensions:
            raise HTTPException(status_code=400, detail="File must be an image (jpg, jpeg, png, bmp, gif)")
    
    try:
        # Read the uploaded file
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Recognize faces in the image
        results = face_service.recognize_faces(image_data)
        
        return {
            "filename": file.filename,
            "recognized_faces": results,
            "total_faces_found": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model/status")
async def model_status():
    """Check if the model is trained and ready"""
    try:
        face_service.load_model()
        return {
            "status": "ready",
            "total_encodings": len(face_service.known_face_encodings),
            "known_people": list(set(face_service.known_face_names))
        }
    except:
        return {"status": "not_trained"}

# =========================
# SQLite and Attendance APIs
# =========================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT,
            present INTEGER DEFAULT 0
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
        """
    )
    # Attendance sessions
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            class_name TEXT NOT NULL,
            teacher_name TEXT NOT NULL
        )
        """
    )
    # Attendance records per session
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            date TEXT NOT NULL,
            is_present INTEGER NOT NULL,
            accuracy REAL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    conn.close()

def verify_teacher(username: str, password: str) -> Optional[int]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM teachers WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def seed_teacher():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO teachers (username, password) VALUES (?, ?)", ("teacher1", "1234"))
    conn.commit()
    conn.close()

@app.post("/student/register")
async def register_student(name: str = Form(...), image: UploadFile = File(...)):
    try:
        # Save image to persistent folder
        filename = f"{name}_{image.filename}"
        image_path = os.path.join(IMAGE_FOLDER, filename)
        file_bytes = await image.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image uploaded")
        with open(image_path, "wb") as f:
            f.write(file_bytes)

        # Store student in database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO students (name, image_path) VALUES (?, ?)", (name, image_path))
        student_id = c.lastrowid
        conn.commit()
        conn.close()

        return {"success": True, "message": f"Student {name} registered successfully", "student_id": student_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})

@app.post("/teacher/login")
async def teacher_login(username: str = Form(...), password: str = Form(...)):
    try:
        teacher_id = verify_teacher(username, password)
        if not teacher_id:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"success": True, "message": "Login successful", "teacher_id": teacher_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})

class RecognizedStudent(BaseModel):
    id: int
    confidence: Optional[float] = 1.0

class AttendanceMarkRequest(BaseModel):
    recognized_students: List[RecognizedStudent]

@app.post("/teacher/attendance")
async def mark_attendance(payload: AttendanceMarkRequest):
    try:
        present_students: List[int] = []
        unidentifiable_students: List[int] = []

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        for student in payload.recognized_students:
            confidence_value = student.confidence if student.confidence is not None else 1.0
            if confidence_value >= 0.5:
                c.execute("UPDATE students SET present=1 WHERE id=?", (student.id,))
                present_students.append(student.id)
            else:
                unidentifiable_students.append(student.id)

        conn.commit()
        conn.close()

        return {
            "success": True,
            "present_students": present_students,
            "unidentifiable_students": unidentifiable_students,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})

@app.post("/teacher/manual_mark")
async def manual_mark(student_id: int = Form(...), status: str = Form(...)):
    try:
        if status not in ["present", "absent"]:
            raise HTTPException(status_code=400, detail="Invalid status")
        present_value = 1 if status == "present" else 0
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE students SET present=? WHERE id=?", (present_value, student_id))
        conn.commit()
        conn.close()
        return {"success": True, "message": f"Student {student_id} marked as {status}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})

@app.get("/students")
async def list_students():
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
        return {"students": students}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})

@app.post("/attendance/reset")
async def reset_attendance():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE students SET present=0")
        conn.commit()
        conn.close()
        return {"success": True, "message": "Attendance reset for all students"}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})

@app.post("/attendance/session")
async def create_attendance_session(session: AttendanceSession):
    """Create a new attendance session (stored in DB)"""
    try:
        if not session.session_id:
            session.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Ensure timestamps exist
        for record in session.records:
            if not record.timestamp:
                record.timestamp = datetime.now().isoformat()

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Insert session
        c.execute(
            "INSERT OR REPLACE INTO sessions (session_id, date, class_name, teacher_name) VALUES (?, ?, ?, ?)",
            (session.session_id, session.date, session.class_name, session.teacher_name),
        )

        # Insert records
        for record in session.records:
            c.execute(
                """
                INSERT INTO attendance_records (session_id, student_name, date, is_present, accuracy, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    record.student_name,
                    record.date,
                    1 if record.is_present else 0,
                    record.accuracy,
                    record.timestamp,
                ),
            )

        conn.commit()
        conn.close()

        return {"message": "Attendance session created successfully", "session_id": session.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating attendance session: {str(e)}")

@app.get("/attendance/sessions")
async def get_attendance_sessions():
    """Get all attendance sessions from DB with nested records"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT session_id, date, class_name, teacher_name FROM sessions ORDER BY date DESC, session_id DESC")
        sessions_rows = c.fetchall()
        sessions: List[Dict] = []
        for row in sessions_rows:
            sid, sdate, class_name, teacher_name = row
            c.execute(
                "SELECT student_name, date, is_present, accuracy, timestamp FROM attendance_records WHERE session_id=?",
                (sid,),
            )
            rec_rows = c.fetchall()
            records = [
                {
                    "student_name": r[0],
                    "date": r[1],
                    "is_present": bool(r[2]),
                    "accuracy": r[3],
                    "timestamp": r[4],
                }
                for r in rec_rows
            ]
            sessions.append(
                {
                    "session_id": sid,
                    "date": sdate,
                    "class_name": class_name,
                    "teacher_name": teacher_name,
                    "records": records,
                }
            )
        conn.close()
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving attendance sessions: {str(e)}")

@app.get("/attendance/sessions/{session_id}")
async def get_attendance_session(session_id: str):
    """Get a specific attendance session from DB"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT session_id, date, class_name, teacher_name FROM sessions WHERE session_id=?", (session_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Attendance session not found")
        sid, sdate, class_name, teacher_name = row
        c.execute(
            "SELECT student_name, date, is_present, accuracy, timestamp FROM attendance_records WHERE session_id=?",
            (sid,),
        )
        rec_rows = c.fetchall()
        conn.close()
        records = [
            {
                "student_name": r[0],
                "date": r[1],
                "is_present": bool(r[2]),
                "accuracy": r[3],
                "timestamp": r[4],
            }
            for r in rec_rows
        ]
        return {
            "session_id": sid,
            "date": sdate,
            "class_name": class_name,
            "teacher_name": teacher_name,
            "records": records,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving attendance session: {str(e)}")

@app.get("/attendance/student/{student_name}")
async def get_student_attendance(student_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get attendance records for a specific student from DB"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        params: List = [student_name.lower()]
        query = (
            "SELECT r.session_id, r.date, s.class_name, s.teacher_name, r.is_present, r.accuracy, r.timestamp "
            "FROM attendance_records r JOIN sessions s ON r.session_id = s.session_id "
            "WHERE LOWER(r.student_name) = ?"
        )
        if start_date:
            query += " AND r.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND r.date <= ?"
            params.append(end_date)
        query += " ORDER BY r.date DESC, r.timestamp DESC"
        c.execute(query, tuple(params))
        rows = c.fetchall()
        conn.close()
        records = [
            {
                "session_id": r[0],
                "date": r[1],
                "class_name": r[2],
                "teacher_name": r[3],
                "is_present": bool(r[4]),
                "accuracy": r[5],
                "timestamp": r[6],
            }
            for r in rows
        ]
        return {"student_name": student_name, "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving student attendance: {str(e)}")

@app.get("/attendance/stats")
async def get_attendance_stats():
    """Get overall attendance statistics from DB"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = c.fetchone()[0]
        c.execute("SELECT COUNT(*), SUM(is_present) FROM attendance_records")
        rec_row = c.fetchone()
        total_records = rec_row[0] or 0
        present_records = rec_row[1] or 0
        attendance_rate = (present_records / total_records * 100) if total_records > 0 else 0
        c.execute("SELECT COUNT(DISTINCT student_name) FROM attendance_records")
        unique_students = c.fetchone()[0] or 0
        conn.close()
        return {
            "total_sessions": total_sessions,
            "total_records": total_records,
            "present_records": present_records,
            "absent_records": total_records - present_records,
            "attendance_rate": round(attendance_rate, 2),
            "unique_students": unique_students,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating attendance stats: {str(e)}")

@app.delete("/attendance/sessions/{session_id}")
async def delete_attendance_session(session_id: str):
    """Delete an attendance session and its records from DB"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Ensure session exists
        c.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,))
        if not c.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Attendance session not found")
        # Delete records first, then session
        c.execute("DELETE FROM attendance_records WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()
        return {"message": "Attendance session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting attendance session: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = c.fetchone()[0]
        conn.close()
    except Exception:
        total_sessions = 0
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": len(face_service.known_face_encodings) > 0,
        "total_sessions": total_sessions,
    }

if __name__ == "__main__":
    # Ensure data directories and DB are ready at startup
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    init_db()
    seed_teacher()
    uvicorn.run(app, host="0.0.0.0", port=8000)
