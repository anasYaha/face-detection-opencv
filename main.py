import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import numpy as np
import os
import pickle
from PIL import Image, ImageTk
import pathlib
from datetime import datetime


class FaceRecognitionSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        self.root.geometry("400x300")

        # Initialize face detection and recognition
        self.init_face_detection()

        # Data file for storing face data
        self.data_file = "face_data.pkl"
        self.faces_dir = "faces"

        # Create faces directory if it doesn't exist
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

        # Load existing data
        self.face_data = self.load_face_data()

        # Setup GUI
        self.setup_gui()

    def init_face_detection(self):
        """Initialize face detection classifier"""
        path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(str(path))

        # Initialize face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    def setup_gui(self):
        """Setup the main GUI"""
        # Title
        title_label = tk.Label(self.root, text="Face Recognition System",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=20)

        # Enroll button
        enroll_btn = tk.Button(self.root, text="Enroll New Face",
                               command=self.enroll_face,
                               bg="#4CAF50", fg="white",
                               font=("Arial", 12), width=20, height=2)
        enroll_btn.pack(pady=10)

        # Camera button
        camera_btn = tk.Button(self.root, text="Start Camera Recognition",
                               command=self.start_camera,
                               bg="#2196F3", fg="white",
                               font=("Arial", 12), width=20, height=2)
        camera_btn.pack(pady=10)

        # View enrolled faces button
        view_btn = tk.Button(self.root, text="View Enrolled Faces",
                             command=self.view_enrolled_faces,
                             bg="#FF9800", fg="white",
                             font=("Arial", 12), width=20, height=2)
        view_btn.pack(pady=10)

        # Status label
        self.status_label = tk.Label(self.root, text=f"Enrolled faces: {len(self.face_data)}",
                                     font=("Arial", 10))
        self.status_label.pack(pady=10)

    def load_face_data(self):
        """Load face data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'rb') as f:
                    return pickle.load(f)
            return {}
        except:
            return {}

    def save_face_data(self):
        """Save face data to file"""
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.face_data, f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")

    def enroll_face(self):
        """Enroll a new face"""
        # Get user information
        first_name = simpledialog.askstring("First Name", "Enter first name:")
        if not first_name:
            return

        last_name = simpledialog.askstring("Last Name", "Enter last name:")
        if not last_name:
            return

        age = simpledialog.askstring("Age", "Enter age:")
        if not age:
            return

        try:
            age = int(age)
        except ValueError:
            messagebox.showerror("Error", "Age must be a number")
            return

        # Capture face
        self.capture_face_for_enrollment(first_name, last_name, age)

    def capture_face_for_enrollment(self, first_name, last_name, age):
        """Capture face for enrollment"""
        camera = cv2.VideoCapture(0)
        captured = False

        messagebox.showinfo("Instructions",
                            "Position your face in the camera and press SPACE to capture.\nPress ESC to cancel.")

        while not captured:
            ret, frame = camera.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to capture", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Enrollment - Position your face and press SPACE', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space key
                if len(faces) > 0:
                    # Save the face
                    x, y, w, h = faces[0]  # Take the first detected face
                    face_img = gray[y:y + h, x:x + w]

                    # Generate unique ID
                    user_id = len(self.face_data) + 1

                    # Save face image
                    face_filename = f"{self.faces_dir}/face_{user_id}.jpg"
                    cv2.imwrite(face_filename, face_img)

                    # Save user data
                    self.face_data[user_id] = {
                        'first_name': first_name,
                        'last_name': last_name,
                        'age': age,
                        'face_file': face_filename,
                        'enrolled_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    self.save_face_data()
                    self.train_recognizer()

                    captured = True
                    messagebox.showinfo("Success",
                                        f"Face enrolled successfully for {first_name} {last_name}")
                else:
                    messagebox.showwarning("Warning", "No face detected. Please try again.")

            elif key == 27:  # ESC key
                break

        camera.release()
        cv2.destroyAllWindows()

        # Update status
        self.status_label.config(text=f"Enrolled faces: {len(self.face_data)}")

    def train_recognizer(self):
        """Train the face recognizer with enrolled faces"""
        if len(self.face_data) == 0:
            return

        faces = []
        labels = []

        for user_id, data in self.face_data.items():
            if os.path.exists(data['face_file']):
                face_img = cv2.imread(data['face_file'], cv2.IMREAD_GRAYSCALE)
                faces.append(face_img)
                labels.append(user_id)

        if len(faces) > 0:
            self.face_recognizer.train(faces, np.array(labels))

    def start_camera(self):
        """Start camera for face recognition"""
        if len(self.face_data) == 0:
            messagebox.showwarning("Warning", "No faces enrolled yet. Please enroll faces first.")
            return

        # Train recognizer before starting
        self.train_recognizer()

        camera = cv2.VideoCapture(0)

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

            if len(faces) == 0:
                # No face detected - gray border
                cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10),
                              (128, 128, 128), 3)
                cv2.putText(frame, "No face detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]

                # Recognize face
                user_id, confidence = self.face_recognizer.predict(face_img)

                # Lower confidence means better match (closer to 0)
                if confidence < 100:  # Threshold for recognition
                    # Face recognized - green border
                    cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10),
                                  (0, 255, 0), 3)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Display user info
                    user_info = self.face_data[user_id]
                    name = f"{user_info['first_name']} {user_info['last_name']}"
                    age = user_info['age']

                    cv2.putText(frame, f"Name: {name}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Age: {age}", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {round(100 - confidence, 1)}%",
                                (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    # Face not recognized - red border
                    cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10),
                                  (0, 0, 255), 3)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown Person", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, "?", (x + w // 2 - 10, y + h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.imshow('Face Recognition - Press Q to quit', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

    def view_enrolled_faces(self):
        """View all enrolled faces"""
        if len(self.face_data) == 0:
            messagebox.showinfo("Info", "No faces enrolled yet.")
            return

        # Create new window
        view_window = tk.Toplevel(self.root)
        view_window.title("Enrolled Faces")
        view_window.geometry("600x400")

        # Create scrollable frame
        canvas = tk.Canvas(view_window)
        scrollbar = ttk.Scrollbar(view_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add face data
        for user_id, data in self.face_data.items():
            frame = ttk.Frame(scrollable_frame, relief="ridge", borderwidth=2)
            frame.pack(fill="x", padx=10, pady=5)

            info_text = f"ID: {user_id} | Name: {data['first_name']} {data['last_name']} | Age: {data['age']} | Enrolled: {data['enrolled_date']}"
            label = ttk.Label(frame, text=info_text, padding=10)
            label.pack()

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def run(self):
        """Run the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = FaceRecognitionSystem()
    app.run()