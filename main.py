import cv2
from celebrity_lookalike import CelebrityLookalike 
from annoy_index import create_annoy_index

def load_celebrity_info():
    # Return a dictionary mapping IDs to celebrity info
    # We are manually creating this right now but we will download a dataset later on
    # TODO find a datset of celebrity images and names and use that dataset to create the 
    # dictionary 
    return {
        0: {"name": "Zendaya", "image_paths": ["face_matching/zendaya-1.jpeg"]},
        1: {"name": "Zendaya", "image_paths": ["face_matching/zendaya-2.jpeg"]},
        2: {"name": "Zendaya", "image_paths": ["face_matching/zendaya-young-1.png"]},
        3: {"name": "Zendaya", "image_paths": ["face_matching/zendaya-young-2.jpeg"]},
        4: {"name": "Storm Reid", "image_paths": ["face_matching/storm-reid-1.jpeg"]},
        5: {"name": "Storm Reid", "image_paths": ["face_matching/storm-reid-2.jpeg"]},
        6: {"name": "Michael B Jordan", "image_paths": ["face_matching/michael-1.jpg"]},
        7: {"name": "Michael B Jordan", "image_paths": ["face_matching/michael-2.jpeg"]},
        8: {"name": "Nick Cannon", "image_paths": ["face_matching/nick-cannon-1.jpg"]},
        9: {"name": "Nick Cannon", "image_paths": ["face_matching/nick-cannon-2.jpeg"]},
    }

def annotate_image(image_path, results):
    """
    Annotate the image with bounding boxes around detected faces and 
    labels with the celebrity names.

    image_path: A string specifying the path to the image file that you want to annotate. 
    results: A list of dictionaries, where each dictionary contains information about a detected 
             face and its corresponding celebrity lookalike. 
    """
    image = cv2.imread(image_path)
    for result in results:
        bbox = result['face_bbox']
        # Adjusting from (top, right, bottom, left) to the format 
        # expected by cv2.rectangle: (left, top, right, bottom)
        start_point = (bbox[3], bbox[0])
        end_point = (bbox[1], bbox[2])
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(image, result['celebrity_name'], (bbox[3], bbox[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.imshow("Lookalike Results", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    celebrity_info = load_celebrity_info()
    index_path = "annoy_index.ann" 

    # Create Annoy index 
    create_annoy_index(celebrity_info, index_path)

    # Initialize CelebrityLookalike with the created Annoy index
    lookalike_app = CelebrityLookalike(index_path)
    lookalike_app.load_celebrity_info(celebrity_info)

    # Process a test image and get results
    test_image_path = "face_matching/michael-3.webp" 
    results = lookalike_app.process_image(test_image_path)

    # Annotate and display the image with the lookalike results
    # TODO this currently draws a box around the face in the input image 
    # and it will label the face with the name of the celebrity lookalike 
    # We will need to change this to instead show the image of the celebrity lookalike instead 
    annotate_image(test_image_path, results)

if __name__ == "__main__":
    main()
