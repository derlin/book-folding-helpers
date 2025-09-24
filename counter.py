import cv2
import numpy as np


def count_segments_and_lines(image_path):
    """
    Calculates the total number of black segments and the number of
    unique vertical black lines in an image.
    """
    try:
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Error: Image not found at {image_path}")

        # Invert the image so the black lines become white
        img_inverted = cv2.bitwise_not(img)

        # Apply a threshold to get a binary image
        _, thresh = cv2.threshold(img_inverted, 128, 255, cv2.THRESH_BINARY)

        # --- Part 1: Count segments (rectangles) ---
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        total_segments = len(contours)

        # --- Part 2: Count unique vertical black lines ---
        # We need a robust way to count the lines regardless of their thickness.
        # A simple way to do this is to check for a single pixel along the horizontal axis,
        # but the previous method was flawed. A more robust way is to use a morphological
        # operation called a "hit-or-miss" or a simple scan of the image after thinning.

        # Here's a simpler and more effective approach: we will find the horizontal
        # profile of the image and count the number of distinct "black" regions.

        # Sum the pixels vertically to get a 1D array of horizontal pixel counts.
        horizontal_profile = np.sum(thresh, axis=0)

        # Now, count the number of distinct non-zero regions in this profile.
        num_vertical_lines = 0
        in_line = False
        for pixel_sum in horizontal_profile:
            if pixel_sum > 0 and not in_line:
                num_vertical_lines += 1
                in_line = True
            elif pixel_sum == 0:
                in_line = False

        print(
            f"Lines (cut): {num_vertical_lines} / {num_vertical_lines * 2} pages"
        )
        print(
            f"Segments (fold): {total_segments} / {total_segments * 2} pages"
        )

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <FILE>")
    else:
        count_segments_and_lines(sys.argv[1])
