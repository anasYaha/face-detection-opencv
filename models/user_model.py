from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class User:
    id: int
    first_name: str
    last_name: str
    age: int
    face_files: List[str]
    enrolled_date: str

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @classmethod
    def create(cls, id: int, first_name: str, last_name: str, age: int, face_files: List[str]):
        return cls(
            id=id,
            first_name=first_name,
            last_name=last_name,
            age=age,
            face_files=face_files,
            enrolled_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )





