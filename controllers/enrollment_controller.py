from services.face_detection_service import FaceDetectionService
from services.camera_service import CameraService
from services.file_service import FileService
from repositories.user_repository import UserRepository
from models.user_model import User
from typing import Optional


class EnrollmentController:

    def __init__(self, user_repository: UserRepository, file_service: FileService):
        self.user_repository = user_repository
        self.file_service = file_service
        self.camera_service = CameraService()

    def enroll_user(self, first_name: str, last_name: str, age: int) -> bool:
        try:
            # Validate input
            if not self._validate_user_input(first_name, last_name, age):
                return False

            print(f"Starting enrollment for {first_name} {last_name}")

            # Capture face samples
            face_samples = self.camera_service.capture_faces_for_enrollment()

            # FIX: Check minimum samples more strictly
            min_samples = 3
            if len(face_samples) < min_samples:
                print(f"Enrollment cancelled. Need at least {min_samples} samples, got {len(face_samples)}")
                return False

            # Generate user ID and save files
            user_id = self.user_repository.get_next_user_id()
            print(f"Saving face samples for user ID: {user_id}")

            face_files = self.file_service.save_face_samples(user_id, face_samples)

            if not face_files:  # FIX: Check if files were saved successfully
                print("Failed to save face samples")
                return False

            # Create and save user
            user = User.create(user_id, first_name, last_name, age, face_files)
            success = self.user_repository.add_user(user)

            if success:
                print(f"Face enrolled successfully for {user.full_name} with {len(face_samples)} samples")
                print(f"User ID: {user_id}")
            else:
                print("Failed to save user data")

            return success

        except Exception as e:
            print(f"Error during enrollment: {e}")
            return False

    def _validate_user_input(self, first_name: str, last_name: str, age: int) -> bool:
        try:
            if not first_name or not first_name.strip():
                print("Error: First name is required")
                return False

            if not last_name or not last_name.strip():
                print("Error: Last name is required")
                return False

            if not isinstance(age, int) or age <= 0 or age > 150:  # FIX: Add reasonable age limits
                print("Error: Age must be a positive number between 1 and 150")
                return False

            # FIX: Sanitize names (remove special characters that might cause file issues)
            if any(char in first_name + last_name for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
                print("Error: Names cannot contain special characters")
                return False

            return True
        except Exception as e:
            print(f"Error validating input: {e}")
            return False