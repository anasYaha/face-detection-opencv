import os
import shutil
import cv2
from typing import List


class FileService:

    def __init__(self, base_dir: str = "faces"):
        self.base_dir = base_dir
        self.ensure_directory_exists(base_dir)

    def ensure_directory_exists(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_face_samples(self, user_id: int, face_samples: List) -> List[str]:

        user_dir = os.path.join(self.base_dir, f"user_{user_id}")
        self.ensure_directory_exists(user_dir)

        face_files = []
        for i, face_img in enumerate(face_samples):
            filename = f"{user_dir}/sample_{i + 1}.jpg"
            cv2.imwrite(filename, face_img)
            face_files.append(filename)

        return face_files

    def delete_user_files(self, user_id: int) -> bool:
        user_dir = os.path.join(self.base_dir, f"user_{user_id}")
        try:
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
            return True
        except Exception as e:
            print(f"Error deleting user files: {e}")
            return False

    def load_user_face_images(self, face_files: List[str]) -> List:
        images = []
        for file_path in face_files:
            if os.path.exists(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (200, 200))
                    images.append(img)
        return images
