from dataset_manager import DatasetManager
from face_processor import FaceProcessor
from typing import Dict, List, Any

class CelebrityMatchFinder:
    def __init__(self, index_path: str, face_distance_threshold: float = 0.6):
        self.index_path = index_path
        self.face_distance_threshold = face_distance_threshold
        self.dataset_manager = DatasetManager(index_path)
        self.dataset_manager.load_annoy_index()
        self.id_to_celebrity_info = {}

    def load_celebrity_info(self, id_to_celebrity_info: Dict[int, Dict[str, str]]):
        self.id_to_celebrity_info = id_to_celebrity_info

    def process_image(self, filepath: str) -> List[Dict[str, Any]]:
        image = FaceProcessor.read_image(filepath)
        faces_and_encodings = FaceProcessor.find_faces_and_encode(image)
        results = []

        for encoding, bbox in faces_and_encodings:
            nearest_neighbors = self.dataset_manager.ann_index.get_nns_by_vector(encoding, 1, include_distances=True)
            if nearest_neighbors[0]:  # If any nearest neighbors are found
                lookalike_id = nearest_neighbors[0][0]
                distance = nearest_neighbors[1][0]
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
