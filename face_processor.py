import numpy as np
import face_recognition
from typing import List, Tuple

class FaceProcessor:
    @staticmethod
    def read_image(filepath: str) -> np.ndarray:
        return face_recognition.load_image_file(filepath)

    @staticmethod
    def find_faces_and_encode(image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return list(zip(face_encodings, face_locations))
