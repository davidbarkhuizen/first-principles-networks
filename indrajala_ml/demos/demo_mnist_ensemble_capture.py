from indrajala_ml.demos.capture_app import CaptureConfig, run_capture_demo
from indrajala_ml.mnist_capture import (
    CANVAS_SIZE,
    CAPTURE_BRUSH_RADIUS,
    CAPTURE_GRID_SIZE,
    paint_brush_stroke,
    preprocess_capture,
)
from indrajala_ml.model.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork

MODEL_PATH = "data/mnist/trained_model.json"


def _grid_to_state(grid: list[list[float]]) -> tuple[float, ...]:
    return tuple(value for row in grid for value in row)


# Captures a digit the same way real MNIST source data was itself produced (see
# mnist_capture.preprocess_capture): the user paints a binary CAPTURE_GRID_SIZE x
# CAPTURE_GRID_SIZE bitmap by mouse (each stroke stamped CAPTURE_BRUSH_RADIUS cells wide, via
# mnist_capture.paint_brush_stroke - a single-cell-wide mouse line is far thinner than any real
# digit stroke once cropped and scaled down), which is then genuinely cropped to its own bounding
# box, aspect-preserving-anti-aliased-scaled to fit a 20px box, and center-of-mass-placed into a
# 28x28 field. See indrajala_ml/demos/capture_app.py for the shared capture UI this config drives.
CONFIG = CaptureConfig(
    capture_grid_size=CAPTURE_GRID_SIZE,
    capture_tile_size=8,
    preview_grid_size=CANVAS_SIZE,
    preview_tile_size=10,
    brush_radius=CAPTURE_BRUSH_RADIUS,
    paint_brush_stroke=paint_brush_stroke,
    preprocess=preprocess_capture,
    to_state=_grid_to_state,
)


def main() -> None:
    run_capture_demo(
        title="MNIST capture",
        model_path=MODEL_PATH,
        load_classifier=EnsembleBackpropClassifierNetwork.load,
        train_demo_title="MNIST ensemble recognition",
        config=CONFIG,
    )


if __name__ == "__main__":
    main()
