import cv2
import numpy as np
from typing import Optional, Callable, List


class CameraService:

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.camera = None

    def start_camera(self) -> bool:
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            return self.camera.isOpened()
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def stop_camera(self) -> None:
        if self.camera:
            self.camera.release()
            self.camera = None
        cv2.destroyAllWindows()

    def capture_frame(self) -> Optional[np.ndarray]:
        if not self.camera:
            return None

        ret, frame = self.camera.read()
        return frame if ret else None

    def capture_faces_for_enrollment(self, required_samples: int = 5) -> List[np.ndarray]:
        from services.face_detection_service import FaceDetectionService

        face_service = FaceDetectionService()
        samples = []
        sample_count = 0

        if not self.start_camera():
            print("Failed to start camera")
            return samples

        print(f"Capturing {required_samples} samples. Press SPACE to capture, ESC to cancel.")

        while sample_count < required_samples:
            frame = self.capture_frame()
            if frame is None:
                print("Failed to capture frame")
                break

            faces = face_service.detect_faces(frame)

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show different messages based on face detection
            if faces:
                cv2.putText(frame, f"Face detected! Press SPACE to capture ({sample_count}/{required_samples})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Position yourself properly and press SPACE",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "No face detected - Please position your face in view",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Samples captured: {sample_count}/{required_samples}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Enrollment - Position your face and press SPACE', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space key
                if faces:  # FIX: Check if faces list is not empty
                    try:
                        face_roi = face_service.extract_face_roi(frame, faces[0])
                        samples.append(face_roi)
                        sample_count += 1
                        print(f"Sample {sample_count} captured successfully")
                    except Exception as e:
                        print(f"Error capturing face sample: {e}")
                else:
                    print("No face detected. Please position your face in the camera view and try again.")
            elif key == 27:  # ESC key
                print("Enrollment cancelled by user")
                break

        self.stop_camera()

        if sample_count < required_samples:
            print(f"Warning: Only {sample_count} out of {required_samples} samples were captured")

        return samples
