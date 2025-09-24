import numpy as np
import click
from pathlib import Path
import cv2


def process(image_path: Path, output_path: Path, new_line_width: int):
    """
    Transforms each vertical segment into a solid rectangle, preserving all gaps.

    Args:
        image_path (Path): The path to the input image.
        output_path (Path): The path to save the output image.
        new_line_width (int): The desired width of the new rectangular lines.
    """
    try:
        # Load the image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Invert the image to make black lines white for contour detection
        _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

        # Find all individual black segments (contours) in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No black segments detected in the image.")
            return

        # Create a new blank white canvas
        new_img = np.full(img.shape, 255, dtype=np.uint8)

        # Iterate over each detected segment and draw a new solid rectangle
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Draw the new rectangle at the original position but with the new width
            cv2.rectangle(new_img,
                          (x, y),
                          (x + new_line_width, y + h),
                          (0, 0, 0),  # Black color
                          -1)        # Fill the rectangle

        # Save the new image
        cv2.imwrite(str(output_path), new_img)
        print(f"Image successfully processed and saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

@click.command()
@click.option("-o", "--output", default=None, type=click.Path(writable=True, dir_okay=False), help="Output path, default to <input>-cleaned.<ext>")
@click.option("-w", "--width", default=1, help="Width of the new gaps in pixels")
@click.argument("FILE", type=click.Path(exists=True, dir_okay=False))
def main(file, output, width):
    file = Path(file)
    output = file.with_name(f"{file.stem}-cleaned{file.suffix}") if not output else Path(output)
    print(file)
    process(file, output, width)

if __name__ == "__main__":
    main()
