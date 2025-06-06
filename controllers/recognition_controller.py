import cv2
from services.face_detection_service import FaceDetectionService
from services.camera_service import CameraService
from services.file_service import FileService
from repositories.user_repository import UserRepository
from typing import Dict


class RecognitionController:

    def __init__(self, user_repository: UserRepository, file_service: FileService):
        self.user_repository = user_repository
        self.file_service = file_service
        self.face_service = FaceDetectionService()
        self.camera_service = CameraService()
        self.recognizer_trained = False

    def start_recognition(self) -> None:
        users = self.user_repository.get_all_users()

        # FIX: Allow camera to work even without enrolled faces
        if not users:
            print("No faces enrolled yet. Camera will work in detection-only mode.")
            self.recognizer_trained = False
        else:
            # Train recognizer only if we have users
            self.recognizer_trained = self._train_recognizer()
            if not self.recognizer_trained:
                print("Failed to train recognizer. Camera will work in detection-only mode.")

        if not self.camera_service.start_camera():
            print("Failed to start camera")
            return

        print("Starting face recognition. Press 'q' to quit.")
        if not self.recognizer_trained:
            print("Detection-only mode: Will show faces but cannot identify them")

        try:
            while True:
                frame = self.camera_service.capture_frame()
                if frame is None:
                    print("Failed to capture frame")
                    break

                # Process frame for recognition - works with or without trained recognizer
                self._process_recognition_frame(frame)

                cv2.imshow('Face Recognition - Press Q to quit', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Error during recognition: {e}")
        finally:
            self.camera_service.stop_camera()

    def _train_recognizer(self) -> bool:
        faces = []
        labels = []

        try:
            for user_id, user in self.user_repository.get_all_users().items():
                face_images = self.file_service.load_user_face_images(user.face_files)
                for face_img in face_images:
                    if face_img is not None:
                        faces.append(face_img)
                        labels.append(user_id)

            if not faces:
                print("No face images found for training")
                return False

            success = self.face_service.train_recognizer(faces, labels)
            if success:
                print(f"Recognizer trained with {len(faces)} face samples from {len(set(labels))} users")
            return success
        except Exception as e:
            print(f"Error training recognizer: {e}")
            return False

    def _process_recognition_frame(self, frame) -> None:
        try:
            faces = self.face_service.detect_faces(frame)

            if not faces:
                # No face detected - show detection mode status
                self._draw_frame_border(frame, (128, 128, 128))
                cv2.putText(frame, "No face detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

                if not self.recognizer_trained:
                    cv2.putText(frame, "Detection-only mode (No faces enrolled)", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                else:
                    cv2.putText(frame, "Please position your face in view", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                return

            # Process each detected face
            for face_coords in faces:
                if self.recognizer_trained:
                    self._process_single_face_with_recognition(frame, face_coords)
                else:
                    self._process_single_face_detection_only(frame, face_coords)

        except Exception as e:
            print(f"Error processing recognition frame: {e}")
            cv2.putText(frame, "Recognition Error", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def _process_single_face_detection_only(self, frame, face_coords) -> None:
        try:
            x, y, w, h = face_coords

            # Blue border for detection-only mode
            self._draw_frame_border(frame, (255, 255, 0))  # Cyan border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            cv2.putText(frame, "Face Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "No faces enrolled yet", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, "Enroll faces to enable recognition", (x, y + h + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        except Exception as e:
            print(f"Error in detection-only mode: {e}")

    def _process_single_face_with_recognition(self, frame, face_coords) -> None:
        try:
            x, y, w, h = face_coords

            if w <= 0 or h <= 0:
                print("Invalid face coordinates detected")
                return

            face_roi = self.face_service.extract_face_roi(frame, face_coords)

            if face_roi is None or face_roi.size == 0:
                print("Failed to extract face ROI")
                return

            user_id, confidence = self.face_service.recognize_face(face_roi)

            # FIX: More strict recognition criteria
            if (user_id != -1 and
                    self.face_service.is_face_recognized(confidence) and
                    self._is_user_valid(user_id)):
                self._draw_recognized_face(frame, face_coords, user_id, confidence)
            else:
                self._draw_unknown_face(frame, face_coords, confidence)

        except Exception as e:
            print(f"Error processing single face: {e}")
            x, y, w, h = face_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Error", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _is_user_valid(self, user_id: int) -> bool:
        try:
            user = self.user_repository.get_user(user_id)
            return user is not None
        except Exception as e:
            print(f"Error checking user validity: {e}")
            return False

    def _draw_recognized_face(self, frame, face_coords, user_id: int, confidence: float) -> None:
        try:
            x, y, w, h = face_coords
            user = self.user_repository.get_user(user_id)

            if not user:
                print(f"User with ID {user_id} not found in database")
                self._draw_unknown_face(frame, face_coords, confidence)
                return

            # Green border for recognized face
            self._draw_frame_border(frame, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display user information
            cv2.putText(frame, f"Name: {user.full_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {user.age}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {round(confidence, 1)}", (x, y + h + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"ID: {user_id}", (x, y + h + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        except Exception as e:
            print(f"Error drawing recognized face: {e}")

    def _draw_unknown_face(self, frame, face_coords, confidence: float) -> None:
        try:
            x, y, w, h = face_coords

            # Red border for unknown face
            self._draw_frame_border(frame, (0, 0, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(frame, "Unknown Person", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Confidence: {round(confidence, 1)}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "?", (x + w // 2 - 10, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        except Exception as e:
            print(f"Error drawing unknown face: {e}")

    def _draw_frame_border(self, frame, color) -> None:
        try:
            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), color, 3)
        except Exception as e:
            print(f"Error drawing frame border: {e}")
