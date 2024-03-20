from dataset_manager import DatasetManager
# Assuming CelebrityInfoGenerator is in a module named celebrity_info
from celebrity_info import CelebrityInfoGenerator

def build_annoy_index(index_path: str, num_trees: int = 10):
    """
    Build and save an Annoy index for the celebrity dataset.

    Parameters:
    - index_path: Path where the Annoy index file will be saved.
    - num_trees: The number of trees for the Annoy index. Higher numbers offer better precision.
    """
    # Create an instance of the celebrity info generator
    info_generator = CelebrityInfoGenerator()
    # Generate the celebrity info dictionary
    celebrity_info = info_generator.get_celebrity_info()

    # Initialize DatasetManager and create the Annoy index
    dataset_manager = DatasetManager(index_path)
    dataset_manager.create_annoy_index(celebrity_info, num_trees=num_trees)

    print(f"Annoy index has been built and saved to {index_path}.")

if __name__ == "__main__":
    # Define the path to save the Annoy index
    index_path = "annoy_index.ann"
    # Optionally, adjust the number of trees for the index
    num_trees = 10
    # Build the index
    build_annoy_index(index_path, num_trees)
