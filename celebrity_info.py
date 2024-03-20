import os
from typing import Dict, List

class CelebrityInfoGenerator:
    def __init__(self, base_path: str = "celeb_data"):
        """
        Initializes the generator with the path to the directory containing celebrity data.
        
        Parameters:
        - base_path: The path to the directory where celebrity images are stored.
        """
        self.base_path = base_path

    def get_celebrity_info(self) -> Dict[int, Dict[str, List[str]]]:
        """
        Generates a dictionary where each key is a unique celebrity ID (integer) and
        each value is another dictionary containing 'name' and 'image_paths' keys.

        Returns:
        A dictionary mapping celebrity IDs to their names and list of image paths.
        """
        celebrity_info = {}
        celeb_id = 0

        for celeb_name in os.listdir(self.base_path):
            celeb_dir = os.path.join(self.base_path, celeb_name)
            if os.path.isdir(celeb_dir):
                image_paths = [os.path.join(celeb_dir, img) for img in os.listdir(celeb_dir) if self.is_image_file(img)]
                if image_paths:  # Ensure there's at least one image
                    formatted_name = celeb_name.replace("_", " ")
                    celebrity_info[celeb_id] = {
                        "name": formatted_name,
                        "image_paths": image_paths
                    }
                    celeb_id += 1

        return celebrity_info

    @staticmethod
    def is_image_file(filename: str) -> bool:
        """
        Determines whether a file is an image based on its extension.

        Parameters:
        - filename: The name of the file to check.

        Returns:
        True if the file is an image, False otherwise.
        """
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))

if __name__ == "__main__":
    generator = CelebrityInfoGenerator()
    celeb_info = generator.get_celebrity_info()
    print(celeb_info)
