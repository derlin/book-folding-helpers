import numpy as np
import click
from pathlib import Path
import cv2
from collections import defaultdict
import attrs

# BGR
GRAY = (200, 200, 200)
RED = (0, 0, 200)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


@attrs.define
class Options:
    # Width of the new black segments in pixels
    segment_width: int = 2
    # Width of the new white gaps in pixels
    gap_width: int = 5
    # Whether to draw helper lines every 2 and 10 segments
    draw_helper_lines: bool = True
    # Skip segments if their total height is below this value in pixels
    skip_if_pixels_below: int = 20


@attrs.define
class NewImage:
    size: tuple[int, int]
    _img: np.ndarray = attrs.field(init=False)

    def __attrs_post_init__(self):
        # an image with 3 channels (BGR)
        self._img = np.full(
            [self.size[0], self.size[1], 3], 255, dtype=np.uint8
        )

    def draw_helper_line(self, x: int, color: tuple[int, int, int] = GRAY):
        self.draw_vertical_line(x, 1, color)

    def draw_vertical_line(self, x: int, w: int, color: tuple[int, int, int]):
        cv2.rectangle(self._img, (x, 0), (x + w, self.size[1]), color, -1)

    def draw_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color: tuple[int, int, int] = BLACK,
    ):
        # Double check we are not writing over an existing segment
        rect = self._img[y : y + h, x : x + w]
        if np.any(np.all(rect == color, axis=2)):
            click.secho(f"Drawing over an existing segment! {x, y}", fg="red")
            color = RED

        cv2.rectangle(
            self._img,
            (x, y),
            (x + w, y + h),
            color,
            -1,  # Fill the rectangle
        )

    def save(self, path: Path):
        cv2.imwrite(str(path), self._img)
        print(f"Image successfully processed and saved to: {path}")


def process(
    image_path: Path,
    output_path: Path,
    options: Options,
):
    print("Using options", options)
    # Load the image in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found at {image_path}")

    # Invert the image to make black lines white for contour detection
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find all individual black segments (contours) in the image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        click.secho("No black segments detected in the image.", fg="red")
        return
    click.echo(f"Found {len(contours)} segments in the original image.")

    # Group each contour to a start x position, so we can process
    # lines by line vertically
    rects = defaultdict(list)
    for contour in contours:
        x, y, _w, h = cv2.boundingRect(contour)
        rects[x].append((y, h))

    start_x = min(rects.keys())
    new_width = (
        2 * start_x
        + (len(rects) * options.segment_width)
        + (len(rects) * options.gap_width)
    )
    # Create a new blank white canvas
    new_img = NewImage((img.shape[0], new_width))

    # Increment the next x position, and draw all segments in this vertical line
    # The initial width and x doesn't matter, only the horizontal placement of each segment
    x = start_x
    lines, segments = 0, 0
    for idx, key in enumerate(sorted(rects.keys())):
        if options.draw_helper_lines and idx > 0:
            if idx % 10 == 0:
                new_img.draw_helper_line(x, RED)
            elif idx % 2 == 0:
                new_img.draw_helper_line(x, GRAY)

        if sum(t[1] for t in rects[key]) < options.skip_if_pixels_below:
            click.secho(
                f"Skipping line at x={key} with total height {sum(t[1] for t in rects[key])}",
                fg="yellow",
            )
            continue

        for y, h in rects[key]:
            segments += 1
            # Draw the new rectangle at the original position but with the new width
            new_img.draw_rect(x, y, w=options.segment_width, h=h)
        lines += 1
        x += options.segment_width + options.gap_width

    # Save the new image
    new_img.save(output_path)
    click.echo(f"Lines: {lines} / {2 * lines} pages")
    click.echo(f"Segments: {segments} / {2 * segments} pages")


_OPTS = Options()  # Get the defaults for click


@click.command()
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(writable=True, dir_okay=False),
    help="Output path, default to <input>-cleaned.<ext>",
)
@click.option(
    "-w",
    "--width",
    default=_OPTS.segment_width,
    help="Width of the new black lines in pixels",
)
@click.option(
    "-s",
    "--space",
    default=_OPTS.gap_width,
    help="Width of the new white gaps in pixels",
)
@click.option(
    "--skip-if",
    default=_OPTS.skip_if_pixels_below,
    help="Skip if the total height of the segments in a vertical less is less than this value in pixels",
)
@click.option(
    "--helper-lines/--no-helper-lines",
    is_flag=True,
    default=_OPTS.draw_helper_lines,
    help="Do not draw helper lines",
)
@click.argument("FILE", type=click.Path(exists=True, dir_okay=False))
def main(file, output, width, space, helper_lines, skip_if):
    file = Path(file)
    output = (
        file.with_name(f"{file.stem}-cleaned{file.suffix}")
        if not output
        else Path(output)
    )
    process(file, output, Options(width, space, helper_lines, skip_if))


if __name__ == "__main__":
    main()
