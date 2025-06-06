import os
import pickle
from typing import Dict, Optional
from models.user_model import User


class UserRepository:

    def __init__(self, data_file: str = "face_data.pkl"):
        self.data_file = data_file
        self._users: Dict[int, User] = {}
        self.load_users()

    def load_users(self) -> None:
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    # Convert dict data to User objects
                    for user_id, user_data in data.items():
                        self._users[user_id] = User(**user_data)
        except Exception as e:
            print(f"Error loading users: {e}")

    def save_users(self) -> bool:
        try:
            # Convert User objects to dict for pickle
            data = {uid: user.__dict__ for uid, user in self._users.items()}
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving users: {e}")
            return False

    def add_user(self, user: User) -> bool:
        self._users[user.id] = user
        return self.save_users()

    def get_user(self, user_id: int) -> Optional[User]:
        return self._users.get(user_id)

    def get_all_users(self) -> Dict[int, User]:
        return self._users.copy()

    def delete_user(self, user_id: int) -> bool:
        if user_id in self._users:
            del self._users[user_id]
            return self.save_users()
        return False

    def get_next_user_id(self) -> int:
        return max(self._users.keys(), default=0) + 1
