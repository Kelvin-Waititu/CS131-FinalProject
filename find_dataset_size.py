import os

def count_images_and_subfolders(root_folder):
    # Supported image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    num_subfolders = 0
    num_images = 0
    
    for root, dirs, files in os.walk(root_folder):
        if dirs:  # If there are subdirectories, count them
            num_subfolders += len(dirs)
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                num_images += 1
    
    return num_subfolders, num_images


root_folder = 'celeb_data'
num_subfolders, num_images = count_images_and_subfolders(root_folder)
print(f"Number of subfolders: {num_subfolders}")
print(f"Total number of images: {num_images}")
