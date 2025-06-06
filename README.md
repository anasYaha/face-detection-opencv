## 🎯 Face Detection & Recognition System using OpenCV & Tkinter
A desktop application built with Python, OpenCV, and Tkinter that allows you to enroll, detect, and recognize human faces in real-time using your webcam.

## 🧠 Features
📸 Face Enrollment Capture and store user faces with first name, last name, and age.

🎥 Real-Time Recognition Recognize faces through your webcam using trained LBPH recognizer.

📁 Face Data Management Stores images and metadata locally (/faces directory and face_data.pkl file).

👁️ View Enrolled Users GUI to list all registered users with their details.

🖼️ Simple GUI Intuitive and responsive interface using Tkinter.

## 🏗️ How It Works
Face Detection: Uses Haar cascades (haarcascade_frontalface_default.xml) from OpenCV to detect faces.

Face Recognition: Uses OpenCV’s LBPH (Local Binary Patterns Histograms) face recognizer.

Data Storage:

Face images stored as .jpg in /faces/. User data serialized in face_data.pkl.

## 📦 Requirements
Install dependencies with:

pip install opencv-python opencv-contrib-python numpy pillow

## 🖥️ Application Preview
Main Menu Enroll New Face

Start Camera Recognition

View Enrolled Faces

Enrollment Prompts for name and age

Captures face when pressing SPACE

Cancels with ESC

Recognition Green box: recognized

Red box: unknown

Confidence displayed in real time

## ❗ Notes
Make sure your webcam is working and accessible.

Close OpenCV windows with Q or ESC depending on context.

Facial recognition is sensitive to lighting and angle — enroll clearly framed photos.

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Contributions
Pull requests are welcome! If you have ideas for improvements or new features, feel free to fork the repo and submit a PR.
