import cv2
import numpy as np
import pathlib
from typing import List, Tuple, Optional


class FaceDetectionService:

    def __init__(self):
        self.face_cascade = None
        self.face_recognizer = None
        self.recognition_threshold = 100  # FIX: Increased threshold for better accuracy
        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        try:
            # Initialize face cascade
            cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(str(cascade_path))

            if self.face_cascade.empty():
                print("Error: Could not load face cascade classifier")
                self.face_cascade = None
                return

            # Initialize face recognizer
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            print("Face detection and recognition components initialized successfully")

        except Exception as e:
            print(f"Error initializing face detection: {e}")
            self.face_cascade = None
            self.face_recognizer = None

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.face_cascade is None or frame is None:
            return []

        try:
            if frame.size == 0:
                return []

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # FIX: Improved detection parameters for better accuracy
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,  # More sensitive
                minNeighbors=3,  # Reduced for better detection
                minSize=(80, 80),  # Smaller minimum size
                maxSize=(300, 300)  # Added maximum size
            )
            return faces.tolist()
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []

    def extract_face_roi(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int],
                         target_size: Tuple[int, int] = (200, 200)) -> Optional[np.ndarray]:
        try:
            if frame is None or frame.size == 0:
                return None

            x, y, w, h = face_coords

            if x < 0 or y < 0 or w <= 0 or h <= 0:
                print("Invalid face coordinates")
                return None

            if x + w > frame.shape[1] or y + h > frame.shape[0]:
                print("Face coordinates exceed frame boundaries")
                return None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y + h, x:x + w]

            if face_roi.size == 0:
                print("Empty face ROI extracted")
                return None

            # FIX: Apply preprocessing for better recognition
            face_roi = cv2.equalizeHist(face_roi)  # Histogram equalization
            return cv2.resize(face_roi, target_size)
        except Exception as e:
            print(f"Error extracting face ROI: {e}")
            return None

    def train_recognizer(self, face_images: List[np.ndarray], labels: List[int]) -> bool:
        if not face_images or not labels or self.face_recognizer is None:
            print("Cannot train: missing images, labels, or recognizer not initialized")
            return False

        if len(face_images) != len(labels):
            print("Error: Number of face images and labels must match")
            return False

        try:
            # Filter out None images and apply preprocessing
            valid_images = []
            valid_labels = []

            for img, label in zip(face_images, labels):
                if img is not None and img.size > 0:
                    # FIX: Apply same preprocessing as in extraction
                    processed_img = cv2.equalizeHist(img)
                    valid_images.append(processed_img)
                    valid_labels.append(label)

            if len(valid_images) < 3:  # FIX: Minimum samples per person
                print(f"Not enough valid images to train ({len(valid_images)} found, need at least 3)")
                return False

            self.face_recognizer.train(valid_images, np.array(valid_labels))
            print(f"Recognizer trained successfully with {len(valid_images)} samples")
            return True
        except Exception as e:
            print(f"Error training recognizer: {e}")
            return False

    def recognize_face(self, face_roi: np.ndarray) -> Tuple[int, float]:
        if self.face_recognizer is None:
            return -1, 1000.0  # Very high confidence (bad)

        try:
            if face_roi is None or face_roi.size == 0:
                return -1, 1000.0

            # FIX: Apply same preprocessing as training
            processed_roi = cv2.equalizeHist(face_roi)
            user_id, confidence = self.face_recognizer.predict(processed_roi)

            # FIX: Add debugging info
            print(f"Recognition result: ID={user_id}, Confidence={confidence:.2f}")

            return int(user_id), float(confidence)
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return -1, 1000.0

    def is_face_recognized(self, confidence: float) -> bool:
        try:
            # FIX: Lower confidence values mean better matches
            # Adjust threshold based on your testing results
            is_recognized = confidence < self.recognition_threshold
            print(
                f"Recognition check: confidence={confidence:.2f}, threshold={self.recognition_threshold}, recognized={is_recognized}")
            return is_recognized
        except Exception as e:
            print(f"Error checking recognition confidence: {e}")
            return False

    def set_recognition_threshold(self, threshold: float) -> None:

        self.recognition_threshold = threshold
        print(f"Recognition threshold set to {threshold}")
