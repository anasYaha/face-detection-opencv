# 🎯 Face Detection & Recognition System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-yellowgreen)

A complete face recognition system with enrollment, detection, and real-time recognition capabilities.

## 📦 Installation
pip install opencv-python opencv-contrib-python numpy pillow
## 🧠 Core Features
📸 Face enrollment with metadata

🎥 Real-time webcam recognition

📁 Automatic data storage (images + pickle)

👤 User management interface

## 🖥️ How It Works
Run main.py to launch the application

Choose between:

Enroll New User: Capture face samples

Start Recognition: Real-time detection

View Users: Browse enrolled persons

🏗️ System Architecture

### Key Bindings
Key	  ----> Action
SPACE ---->	Capture face sample
ESC  	---->	Cancel enrollment
Q	   	----> Exit recognition

## ⚙️ Technical Specifications
Face Detection: Haar Cascades

Recognition: LBPH (Local Binary Patterns Histograms)

Resolution: 640x480 (default)

FPS: ~30 (depending on hardware)

## 💡 Best Practices
Ensure good lighting conditions

Face the camera directly during enrollment

Capture multiple angles (5 samples recommended)

Keep background uncluttered

## 🚨 Troubleshooting
Camera not working?

Try different camera indexes (0, 1, 2)

Check webcam permissions

Poor recognition?

Re-enroll with better samples

Adjust recognition threshold

## 📜 License
MIT License - Free for personal and commercial use
