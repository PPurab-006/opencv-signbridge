import cv2
import numpy as np
from backend.vision import VisionPipeline, extract_features

def test_extract_features_no_hand():
    # Empty black image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    features, mask, contour = extract_features(img)
    assert features is None
    assert contour is None

def test_vision_pipeline_no_frame():
    pipeline = VisionPipeline()
    frame, pred, conf = pipeline.process_frame(None)
    assert frame is None
    assert pred == "Error: No frame"
    assert conf == 0.0

def test_vision_pipeline_smoothing():
    pipeline = VisionPipeline()
    # Force history
    pipeline.history.extend(["HELLO", "HELLO", "PEACE", "HELLO"])
    # Next frame empty
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, pred, _ = pipeline.process_frame(img)
    # The current frame will yield UNKNOWN, so history becomes:
    # ["HELLO", "PEACE", "HELLO", "UNKNOWN"] -> HELLO should still win
    assert pred == "HELLO"

def test_vision_pipeline_with_synthetic_gesture():
    from scripts.generate_data import generate_shape_image
    pipeline = VisionPipeline()
    # Generate HELLO shape
    img, _ = generate_shape_image("HELLO", noise=False)
    processed_frame, pred, conf = pipeline.process_frame(img)
    assert processed_frame is not None
    assert isinstance(conf, float)
    assert pred in ["HELLO", "PEACE", "OKAY", "REST/UNKNOWN", "UNCERTAIN", "UNKNOWN"]

def test_vision_pipeline_no_model():
    pipeline = VisionPipeline()
    pipeline.clf = None
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, pred, conf = pipeline.process_frame(img)
    assert pred == "UNKNOWN"
    assert conf == 0.0
