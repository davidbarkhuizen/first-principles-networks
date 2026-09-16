from indrajala_ml.demos.capture_app import CaptureConfig, run_capture_demo
from indrajala_ml.digit_capture import (
    CAPTURE_BRUSH_RADIUS,
    CAPTURE_GRID_SIZE,
    GRID_SIZE,
    downsample_to_target_grid,
    paint_brush_stroke,
    tile_grid_to_state,
)
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork

MODEL_PATH = "data/digits/trained_model.json"

# Captures a digit the same way the reference work behind the bundled training data did (see
# digit_capture.downsample_to_target_grid): the user paints a binary CAPTURE_GRID_SIZE x
# CAPTURE_GRID_SIZE bitmap by mouse (mirroring NIST's own thresholded pen-on-paper scan) - each
# stroke stamped CAPTURE_BRUSH_RADIUS cells wide (see digit_capture.paint_brush_stroke), since a
# single-cell-wide mouse line comes out far fainter after downsampling than any real training
# stroke - which is then genuinely block-counted down to the GRID_SIZE x GRID_SIZE, 0-16-graded
# shape the model was actually trained on. See indrajala_ml/demos/capture_app.py for the shared
# capture UI this config drives.
CONFIG = CaptureConfig(
    capture_grid_size=CAPTURE_GRID_SIZE,
    capture_tile_size=10,
    preview_grid_size=GRID_SIZE,
    preview_tile_size=35,
    brush_radius=CAPTURE_BRUSH_RADIUS,
    paint_brush_stroke=paint_brush_stroke,
    preprocess=downsample_to_target_grid,
    to_state=tile_grid_to_state,
)


def main() -> None:
    run_capture_demo(
        title="digit capture",
        model_path=MODEL_PATH,
        load_classifier=MultiClassBackpropClassifierNetwork.load,
        train_demo_title="UCI digit recognition",
        config=CONFIG,
    )


if __name__ == "__main__":
    main()
