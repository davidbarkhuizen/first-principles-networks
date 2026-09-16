from indrajala_ml.capture_common import stamp_brush

GRID_SIZE = 8

# matches the actual reference work this codebase's bundled training data comes from (see
# sklearn.datasets.load_digits()'s own dataset description): "32x32 bitmaps are divided into
# nonoverlapping blocks of 4x4 and the number of on pixels are counted in each block" -
# producing the 8x8, 0-16-graded grid data/digits/digits.csv actually contains. Capturing at
# this same 32x32 resolution and genuinely counting blocks down, rather than approximating
# grading with an ad-hoc brush/falloff heuristic, is what gives a mouse-painted digit the same
# soft, anti-aliased edges real training examples have.
CAPTURE_GRID_SIZE = 32
BLOCK_SIZE = CAPTURE_GRID_SIZE // GRID_SIZE
MAX_BLOCK_VALUE = BLOCK_SIZE * BLOCK_SIZE  # 16 - matches digits_data.py's own /16.0 normalization


def tile_grid_to_state(grid: list[list[float]]) -> tuple[float, ...]:
    """
    Flattens an 8x8 tile grid into the same 64-value, row-major, [0.0, 1.0]-normalized state
    shape digits_data.load_digits_dataset produces (row 0 -> indices 0-7, row 1 -> 8-15, ...) -
    matching exactly how sklearn.datasets.load_digits().data flattens its own 8x8 images, since
    that's what data/digits/digits.csv was extracted from. Kept separate from the interactive
    capture UI (indrajala_ml/demos/demo_uci_digit_capture.py) because it's the one piece of that
    tool's logic that's actually a pure function, and so the one piece worth unit-testing the
    same way as everything else in this codebase - the tkinter mouse/canvas code itself isn't
    meaningfully testable without a real or virtual display.
    """

    assert len(grid) == GRID_SIZE, f"grid must have {GRID_SIZE} rows; got {len(grid)}"
    assert all(len(row) == GRID_SIZE for row in grid), f"every row must have {GRID_SIZE} columns"

    return tuple(value for row in grid for value in row)


def pixel_to_tile(x: int, y: int, tile_size: int) -> tuple[int, int]:
    """
    Maps a canvas-relative pixel coordinate to the (row, col) tile it falls in - simple integer
    division, since every tile is the same fixed size. Doesn't clamp/validate against any
    particular grid size - a mouse-drag event firing just outside a canvas can produce a
    negative or too-large row/col, and it's the caller's job to decide whether to ignore that
    (see demo_uci_digit_capture.py's _handle_paint_event), not this function's. Used for both the
    32x32 capture grid and (implicitly, via its fixed tile size) the 8x8 preview grid.
    """

    return y // tile_size, x // tile_size


def downsample_to_target_grid(capture_grid: list[list[float]]) -> list[list[float]]:
    """
    Reproduces the UCI hand-written digits dataset's own preprocessing exactly: a
    CAPTURE_GRID_SIZE x CAPTURE_GRID_SIZE binary bitmap (each cell 0.0 or 1.0 - "on" or "off",
    mirroring NIST's thresholded scan of a pen-on-paper form) is divided into BLOCK_SIZE x
    BLOCK_SIZE nonoverlapping blocks, and each block's "on" pixels are counted - out of
    MAX_BLOCK_VALUE (16) possible - then normalized to [0.0, 1.0] the same way
    digits_data.load_digits_dataset normalizes the bundled training data's own 0-16 pixel
    values (divide by 16). A stroke covering only part of a block registers as partial
    intensity, not a hard on/off toggle - this is what gives a captured digit the same soft,
    anti-aliased edges real training examples have, genuinely counted rather than approximated.
    """

    assert len(capture_grid) == CAPTURE_GRID_SIZE, f"capture_grid must have {CAPTURE_GRID_SIZE} rows"
    assert all(
        len(row) == CAPTURE_GRID_SIZE for row in capture_grid
    ), f"every row must have {CAPTURE_GRID_SIZE} columns"

    target_grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    for target_row in range(GRID_SIZE):
        for target_col in range(GRID_SIZE):
            on_pixel_count = sum(
                capture_grid[target_row * BLOCK_SIZE + block_row][target_col * BLOCK_SIZE + block_col]
                for block_row in range(BLOCK_SIZE)
                for block_col in range(BLOCK_SIZE)
            )
            target_grid[target_row][target_col] = on_pixel_count / MAX_BLOCK_VALUE

    return target_grid


CAPTURE_BRUSH_RADIUS = 2


def paint_brush_stroke(
    grid: list[list[float]], row: int, col: int, radius: int = CAPTURE_BRUSH_RADIUS
) -> list[list[float]]:
    """
    Needed because a single mouse-driven cell toggle only ever paints a stroke 1 capture-cell
    wide - once downsample_to_target_grid block-counts that down, it comes out far fainter
    (measured directly: ~25% of full intensity at best) than any real training example, which
    are consistently near-full intensity (~14-16 out of 16) throughout their stroked regions -
    real NIST pen strokes are proportionally thick relative to the 32x32 capture resolution,
    not single-pixel-thin, and a classifier fed input this faint from what it actually trained
    on will misclassify almost everything, usually toward whichever class's decision region
    happens to catch faint, ambiguous input (observed directly: nearly always "7").

    This brush size (radius=2, a 5x5 stamp) was picked empirically, not guessed: it reliably
    reaches near-full block intensity along a stroke's core while still leaving room for a real
    digit's negative space - e.g. an unfilled "0"'s hole. A radius of 3 or more starts filling
    that hole in too, which visibly hurts classification (a hand-simulated "0" correctly
    classified at radius=2 misclassified as "4" at radius=3 and radius=4, in testing before
    this constant was chosen). The actual stamping is shared with mnist_capture.paint_brush_stroke
    via capture_common.stamp_brush - only this size-specific validation lives here.
    """

    assert len(grid) == CAPTURE_GRID_SIZE and all(
        len(r) == CAPTURE_GRID_SIZE for r in grid
    ), "grid must be CAPTURE_GRID_SIZE x CAPTURE_GRID_SIZE"

    return stamp_brush(grid, row, col, radius)


def intensity_to_color(intensity: float) -> str:
    """
    Maps a [0.0, 1.0]-normalized tile intensity to a grayscale hex color for rendering
    (0.0 -> "#000000", 1.0 -> "#ffffff") - the same linear scale digits_data.py's
    normalization uses, just inverted back to a displayable 0-255 color channel.
    """

    assert 0.0 <= intensity <= 1.0, f"intensity must be in [0.0, 1.0]; got {intensity}"

    level = round(intensity * 255)
    return f"#{level:02x}{level:02x}{level:02x}"
