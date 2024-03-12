import os
import face_recognition
from annoy import AnnoyIndex
from typing import Dict, List

def create_annoy_index(celebrity_info: Dict[int, Dict[str, List[str]]], index_path: str, num_trees: int = 10):
    """
    Creates an Annoy index from celebrity images and saves it to the specified path.

    Parameters:
    - celebrity_info: A dictionary mapping celebrity IDs to their information,
      which includes a list of image paths under the 'image_paths' key.
    - index_path: Path to save the Annoy index file.
    - num_trees: The number of trees to use in the Annoy index. More trees increase
      build time and index size but improve query precision.
    """
    vector_length = 128  # Length of the face embedding vectors
    ann_index = AnnoyIndex(vector_length, 'angular')  # Initialize Annoy index

    for celeb_id, info in celebrity_info.items():
        for img_path in info['image_paths']:
            image = face_recognition.load_image_file(img_path)
            # Handle cases where an image might contain multiple faces
            encodings = face_recognition.face_encodings(image)
            if encodings:
                ann_index.add_item(celeb_id, encodings[0])  # Assuming one face per image and taking the first encoding

    ann_index.build(num_trees)  # Build the Annoy index
    ann_index.save(index_path)  # Save the index to disk
    # Get the size of the index
    index_size = ann_index.get_n_items()

    print(f"Annoy index built and saved to {index_path}.")


