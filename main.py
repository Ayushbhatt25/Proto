from fastapi import FastAPI, File, UploadFile, HTTPException
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
attendance_file = "attendance_data.pkl"

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

# In-memory storage for attendance data (in production, use a database)
attendance_sessions: List[AttendanceSession] = []

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

@app.post("/attendance/session")
async def create_attendance_session(session: AttendanceSession):
    """Create a new attendance session"""
    try:
        # Generate session ID if not provided
        if not session.session_id:
            session.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add timestamp to records
        for record in session.records:
            if not record.timestamp:
                record.timestamp = datetime.now().isoformat()
        
        attendance_sessions.append(session)
        
        # Save to file (in production, save to database)
        with open(attendance_file, 'wb') as f:
            pickle.dump(attendance_sessions, f)
        
        return {"message": "Attendance session created successfully", "session_id": session.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating attendance session: {str(e)}")

@app.get("/attendance/sessions")
async def get_attendance_sessions():
    """Get all attendance sessions"""
    try:
        return {"sessions": attendance_sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving attendance sessions: {str(e)}")

@app.get("/attendance/sessions/{session_id}")
async def get_attendance_session(session_id: str):
    """Get a specific attendance session"""
    try:
        session = next((s for s in attendance_sessions if s.session_id == session_id), None)
        if not session:
            raise HTTPException(status_code=404, detail="Attendance session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving attendance session: {str(e)}")

@app.get("/attendance/student/{student_name}")
async def get_student_attendance(student_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get attendance records for a specific student"""
    try:
        student_records = []
        
        for session in attendance_sessions:
            for record in session.records:
                if record.student_name.lower() == student_name.lower():
                    # Filter by date range if provided
                    if start_date and record.date < start_date:
                        continue
                    if end_date and record.date > end_date:
                        continue
                    
                    student_records.append({
                        "session_id": session.session_id,
                        "date": record.date,
                        "class_name": session.class_name,
                        "teacher_name": session.teacher_name,
                        "is_present": record.is_present,
                        "accuracy": record.accuracy,
                        "timestamp": record.timestamp
                    })
        
        return {"student_name": student_name, "records": student_records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving student attendance: {str(e)}")

@app.get("/attendance/stats")
async def get_attendance_stats():
    """Get overall attendance statistics"""
    try:
        if not attendance_sessions:
            return {"message": "No attendance data available"}
        
        total_sessions = len(attendance_sessions)
        total_records = sum(len(session.records) for session in attendance_sessions)
        present_records = sum(
            sum(1 for record in session.records if record.is_present) 
            for session in attendance_sessions
        )
        
        attendance_rate = (present_records / total_records * 100) if total_records > 0 else 0
        
        # Get unique students
        all_students = set()
        for session in attendance_sessions:
            for record in session.records:
                all_students.add(record.student_name)
        
        return {
            "total_sessions": total_sessions,
            "total_records": total_records,
            "present_records": present_records,
            "absent_records": total_records - present_records,
            "attendance_rate": round(attendance_rate, 2),
            "unique_students": len(all_students)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating attendance stats: {str(e)}")

@app.delete("/attendance/sessions/{session_id}")
async def delete_attendance_session(session_id: str):
    """Delete an attendance session"""
    try:
        global attendance_sessions
        session = next((s for s in attendance_sessions if s.session_id == session_id), None)
        if not session:
            raise HTTPException(status_code=404, detail="Attendance session not found")
        
        attendance_sessions = [s for s in attendance_sessions if s.session_id != session_id]
        
        # Save updated data
        with open(attendance_file, 'wb') as f:
            pickle.dump(attendance_sessions, f)
        
        return {"message": "Attendance session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting attendance session: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": len(face_service.known_face_encodings) > 0,
        "total_sessions": len(attendance_sessions)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
