from PIL import Image
import numpy as np
import face_recognition
from annoy import AnnoyIndex
from typing import List, Dict, Tuple

class CelebrityLookalike:
    def __init__(self, index_path: str, face_distance_threshold: float = 0.6):
        # Path to the pre-built Annoy index file containing the embeddings of celebrity faces
        self.index_path = index_path 
        # The maximum distance between embeddings for faces to be considered a match
        self.face_distance_threshold = face_distance_threshold 
        # An AnnoyIndex object initialized to expect 128-dimensional vectors 
        self.ann_index = AnnoyIndex(128, 'angular')  
        self.ann_index.load(index_path)  # Load pre-built Annoy index
        # Dictionary mapping from ID to celebrity information (name, image paths)
        self.id_to_celebrity_info = {}  

    def load_celebrity_info(self, id_to_celebrity_info: Dict[int, Dict[str, str]]):
        """
        Load and store a mapping of celebrity IDs to their respective information.
        
        Parameters:
        - id_to_celebrity_info (Dict[int, Dict[str, str]]): A dictionary where each key is a unique
        celebrity ID and each value is another dictionary containing information about the celebrity.
        """
        self.id_to_celebrity_info = id_to_celebrity_info

    def read_image(self, filepath: str) -> np.ndarray:
        """Reads an image from the provided file path and returns it as a NumPy array"""
        image = face_recognition.load_image_file(filepath)
        return image

    def find_faces_and_encode(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect faces in an image and return their encodings and bounding boxes. 
        Returns a list of tuples, each containing the face's embedding and its 
        bounding box coordinates. These embeddings represent the facial features 
        in a high-dimensional space
        """
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return list(zip(face_encodings, face_locations))

    def identify_lookalike(self, face_encoding: np.ndarray) -> List[Tuple[int, float]]:
        """
        Given the embedding of a face, uses the Annoy index to find 
        the nearest celebrity face embedding(s) in the dataset. 
        Returns the IDs of the closests matches and the distance to this match. 
        The distance is used to determine similarity, with smaller distances indicating closer matches.
        """
        nearest_neighbors = self.ann_index.get_nns_by_vector(face_encoding, 1, include_distances=True)
        return nearest_neighbors

    def process_image(self, filepath: str) -> List[Dict[str, any]]:
        """Process an input image, find faces, encode them, and identify celebrity lookalikes."""
        image = self.read_image(filepath)
        faces_and_encodings = self.find_faces_and_encode(image)
        results = []

        for encoding, bbox in faces_and_encodings:
            nearest_neighbors_ids, nearest_neighbors_distances = self.identify_lookalike(encoding)
            if nearest_neighbors_ids:  # Check if any nearest neighbors are found
                lookalike_id = nearest_neighbors_ids[0]
                distance = nearest_neighbors_distances[0]
                if distance < self.face_distance_threshold:
                    celebrity_info = self.id_to_celebrity_info.get(lookalike_id, {})
                    results.append({
                        "celebrity_id": lookalike_id,
                        "distance": distance,
                        "celebrity_name": celebrity_info.get("name", "Unknown"),
                        "celebrity_image_paths": celebrity_info.get("image_paths", []),
                        "face_bbox": bbox
                    })

        return results
