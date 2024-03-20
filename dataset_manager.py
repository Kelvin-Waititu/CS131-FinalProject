import face_recognition
from annoy import AnnoyIndex
from typing import Dict, List

class DatasetManager:
    def __init__(self, index_path: str, vector_length: int = 128, metric: str = 'angular'):
        self.index_path = index_path
        self.vector_length = vector_length
        self.metric = metric
        self.ann_index = AnnoyIndex(vector_length, metric)

    def create_annoy_index(self, celebrity_info: Dict[int, Dict[str, List[str]]], num_trees: int = 10):
        for celeb_id, info in celebrity_info.items():
            for img_path in info['image_paths']:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.ann_index.add_item(celeb_id, encodings[0])

        self.ann_index.build(num_trees)
        self.ann_index.save(self.index_path)
        print(f"Annoy index built and saved to {self.index_path}.")

    def load_annoy_index(self):
        self.ann_index.load(self.index_path)
        print("Annoy index loaded from", self.index_path)
