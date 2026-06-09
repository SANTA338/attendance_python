import os
import sys
import tkinter as tk
from tkinter import messagebox

# Safety Net: Catch absolute startup crashes and show a Windows Alert Box
def show_error_window(title, message):
    root = tk.Tk()
    root.withdraw() # Hide main small window
    messagebox.showerror(title, message)
    root.destroy()
    sys.exit()

try:
    import cv2
    import numpy as np
    import pandas as pd
    import face_recognition
    import mysql.connector
    from datetime import datetime
except ImportError as e:
    show_error_window("Dependency Error", f"A required library is missing!\n\nError: {e}\n\nPlease run: pip install -r requirements.txt")

# =========================================================================
# CONFIGURATION
# =========================================================================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",          # Change to your MySQL username
    "password": "password",  # Change to your MySQL password
    "database": "attendance_system"
}
CSV_FILE_PATH = "attendance.csv"
DATASET_PATH = "dataset"

def connect_to_database():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            print("[INFO] Connected to MySQL successfully.")
            return conn
    except Exception as e:
        # Don't crash the system if MySQL is off, just warn the user
        print(f"[WARNING] Database offline. Running in CSV-only mode. Details: {e}")
        return None

def mark_attendance(student_name, db_connection):
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')
    
    # CSV Backup
    attendance_marked_csv = False
    if os.path.exists(CSV_FILE_PATH):
        try:
            df = pd.read_csv(CSV_FILE_PATH)
            duplicate_csv = df[(df['Name'] == student_name) & (df['Date'] == current_date)]
            if not duplicate_csv.empty:
                attendance_marked_csv = True
        except Exception:
            df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    else:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        
    if not attendance_marked_csv:
        new_record = pd.DataFrame([{'Name': student_name, 'Date': current_date, 'Time': current_time}])
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(CSV_FILE_PATH, index=False)
        print(f"[CSV LOG] Attendance recorded for {student_name}.")

    # MySQL Entry
    if db_connection:
        try:
            cursor = db_connection.cursor()
            check_query = "SELECT * FROM attendance WHERE student_name = %s AND date = %s"
            cursor.execute(check_query, (student_name, current_date))
            
            if cursor.fetchone() is None:
                insert_query = "INSERT INTO attendance (student_name, date, time) VALUES (%s, %s, %s)"
                cursor.execute(insert_query, (student_name, current_date, current_time))
                db_connection.commit()
                print(f"[MYSQL LOG] Success: Attendance stored for {student_name}.")
            cursor.close()
        except Exception as e:
            print(f"[MYSQL ERROR] Entry failed: {e}")

def load_and_encode_dataset():
    known_encodings = []
    known_names = []
    
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        return known_encodings, known_names

    print("[INFO] Processing dataset folders...")
    for student_name in os.listdir(DATASET_PATH):
        student_folder = os.path.join(DATASET_PATH, student_name)
        if os.path.isdir(student_folder):
            for img_name in os.listdir(student_folder):
                img_path = os.path.join(student_folder, img_name)
                try:
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(student_name)
                except Exception:
                    pass
    return known_encodings, known_names

def register_new_student(frame, db_connection):
    print("\n" + "="*40)
    student_name = input("[REGISTRATION] Enter Student Name: ").strip()
    if not student_name:
        print("[CANCELLED] Name entry blank.")
        return None, None

    student_dir = os.path.join(DATASET_PATH, student_name)
    os.makedirs(student_dir, exist_ok=True)
    
    img_path = os.path.join(student_dir, f"captured_{int(datetime.now().timestamp())}.jpg")
    cv2.imwrite(img_path, frame)
    print(f"[REGISTRATION] Image saved to {img_path}")
    
    if db_connection:
        try:
            cursor = db_connection.cursor()
            cursor.execute("SELECT * FROM students WHERE name = %s", (student_name,))
            if cursor.fetchone() is None:
                cursor.execute("INSERT INTO students (name) VALUES (%s)", (student_name,))
                db_connection.commit()
                print(f"[MYSQL] Profile registered.")
            cursor.close()
        except Exception as e:
            print(f"[MYSQL ERROR] Registration entry skipped: {e}")
            
    print("="*40 + "\n")
    return load_and_encode_dataset()

def main():
    print("[INFO] Starting system...")
    db_conn = connect_to_database()
    known_encodings, known_names = load_and_encode_dataset()
    
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        show_error_window("OpenCV Error", "Could not load Haar Cascade XML metadata file.")

    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        show_error_window("Camera Error", "Your system cannot open or access the webcam.\n\n1. Check if another app (Zoom/Teams/Browser) is using it.\n2. Verify system camera privacy settings.")

    print("\n[SUCCESS] System running. Camera stream active.")
    print("-> Press 'r' in webcam window to register.")
    print("-> Press 'q' in webcam window to quit.\n")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces_detected = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces_detected]
        
        if len(known_encodings) > 0:
            live_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for face_encoding, face_loc in zip(live_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                
                top, right, bottom, left = face_loc
                box_color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                
                if name != "Unknown":
                    mark_attendance(name, db_conn)
                    
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), box_color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No Profiles Loaded. Press 'r' to Register.", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow('Smart Attendance Tracker', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            updated_enc, updated_names = register_new_student(frame, db_conn)
            if updated_enc is not None:
                known_encodings, known_names = updated_enc, updated_names
        elif key == ord('q'):
            break
            
    video_capture.release()
    cv2.destroyAllWindows()
    if db_conn and db_conn.is_connected():
        db_conn.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        show_error_window("Runtime Crash", f"System crashed unexpected:\n\n{e}")