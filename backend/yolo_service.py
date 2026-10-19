"""Module 2.4 - Object detection with pretrained YOLOv8 Nano (COCO dataset, 80 classes).
Based on the professor's CNN_YOLO.ipynb (Reference B)."""
import base64
import threading

import cv2
from ultralytics import YOLO

_model = None
_lock = threading.Lock()


def _load():
    """Lazy-load YOLOv8n once (about 6 MB of weights, downloaded on first use)."""
    global _model
    with _lock:
        if _model is None:
            _model = YOLO("yolov8n.pt")
    return _model


def detect(img, conf: float = 0.25) -> dict:
    model = _load()
    results = model(img, conf=conf, device="cpu", verbose=False)
    r = results[0]

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "box": [round(float(v), 1) for v in box.xyxy[0].tolist()],   # x1, y1, x2, y2
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # r.plot() draws the boxes and returns a BGR image (OpenCV color order).
    # The notebook converts BGR -> RGB to show it with matplotlib; cv2.imencode
    # expects BGR, so here we can encode it directly as JPEG.
    annotated_bgr = r.plot()
    ok, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode the annotated image.")
    return {"count": len(detections), "detections": detections,
            "annotated_image_base64": base64.b64encode(buf.tobytes()).decode()}
