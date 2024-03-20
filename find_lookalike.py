import cv2
import os
from celebrity_match_finder import CelebrityMatchFinder
from celebrity_info import CelebrityInfoGenerator
import numpy as np

def resize_and_pad(img, size, pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  

    # compute scaling and pad sizing
    if aspect > 1:  # wide image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)//2
        pad_top, pad_bot = pad_vert, pad_vert
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # tall image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)//2
        pad_left, pad_right = pad_horz, pad_horz
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):  # color image but only one color provided
        pad_color = [pad_color]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img

def visualize_results(test_image_path, results):
    test_image = cv2.imread(test_image_path)

    for result in results:
        celeb_images = result['celebrity_image_paths']
        if celeb_images:  # if there are images available
            celeb_image_path = celeb_images[0]
            celeb_image = cv2.imread(celeb_image_path)

            # Resize while maintaining aspect ratio
            desired_size = (500, 600)  # Desired width and height
            test_image_resized = resize_and_pad(test_image, desired_size)
            celeb_image_resized = resize_and_pad(celeb_image, desired_size)

            # Concatenate images horizontally
            combined_image = cv2.hconcat([test_image_resized, celeb_image_resized])

            # Prepare text for overlay
            text = f"Lookalike: {result['celebrity_name']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (255, 255, 255)  # white
            font_thickness = 2
            # Calculate text size
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # Set position for the text (at the top of the lookalike image)
            text_x = test_image_resized.shape[1] + (celeb_image_resized.shape[1] - text_size[0]) // 2
            text_y = max(30, text_size[1])  # 30 pixels from the top or text height, whichever is larger

            # Put text on the combined image
            cv2.putText(combined_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # Show the combined image
            cv2.imshow("Result", combined_image)
            cv2.waitKey(0)  # wait for a key press to proceed
            cv2.destroyAllWindows()



def main(test_image_path):
    # Path to the Annoy index file that should have been previously created
    index_path = "annoy_index.ann"
    # Check if the Annoy index exists
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No Annoy index found at {index_path}. Please run build_index.py first.")

    # Initialize the celebrity info generator and get the celebrity info
    info_generator = CelebrityInfoGenerator()
    celebrity_info = info_generator.get_celebrity_info()

    # Initialize the CelebrityMatchFinder application
    lookalike_app = CelebrityMatchFinder(index_path)
    # Load the celebrity information into the application
    lookalike_app.load_celebrity_info(celebrity_info)

    # Process the test image and get the results
    results = lookalike_app.process_image(test_image_path)

    # Visualize the results
    visualize_results(test_image_path, results)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python find_lookalike.py <path_to_test_image>")
        sys.exit(1)
    test_image_path = sys.argv[1]
    main(test_image_path)
