"""
smart_fridge_webcam.py
======================
Real-time Smart Fridge pipeline using a laptop webcam.

Models required (place in the same directory or update paths below):
  • yolo_fruit3.pt       — YOLOv8n detector  (apple / banana / orange)
  • freshness_model.pt   — MobileNetV3-Small freshness classifier (Fresh / Spoiled → 3 runtime labels)

Controls:
  'c'  — capture current frame and run the full inference pipeline
  'q'  — quit

Usage:
  python smart_fridge_webcam.py
  python smart_fridge_webcam.py --yolo yolo_fruit3.pt --freshness freshness_model.pt
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import models, transforms
from ultralytics import YOLO

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Freshness thresholds (must match training notebook) ────────────────────────
FRESH_THRESH = 0.6
SPOILED_THRESH = 0.6
    
# ── Shelf-life baselines (days at ~4 °C) ───────────────────────────────────────
BASELINE_SHELF_LIFE = {
    "apple":   42,
    "banana":   5,
    "orange":  21,
    "unknown":  5,
}

# ── Visualisation colours (BGR) ────────────────────────────────────────────────
VIS_COLORS = {
    "Fresh":    (0, 200, 0),    # green
    "Moderate": (0, 165, 255),  # orange
    "Spoiled":  (0, 0, 220),    # red
}

# ── Crop padding (pixels) ──────────────────────────────────────────────────────
BBOX_PAD = 8

# ── Inference transform (mirrors val_tf in notebook) ──────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_infer_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ═══════════════════════════════════════════════════════════════════════════════
# 1 · Model Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_models(yolo_path: str, freshness_path: str):
    """
    Load and return the YOLO detector and MobileNetV3 freshness classifier.

    Parameters
    ----------
    yolo_path      : path to yolo_fruit3.pt
    freshness_path : path to freshness_model.pt (state-dict only)

    Returns
    -------
    yolo_model     : ultralytics YOLO instance, ready for inference
    fresh_model    : torch.nn.Module, set to eval(), moved to DEVICE
    """
    # ── YOLO detector ─────────────────────────────────────────────────────────
    print(f"[INFO] Loading YOLO detector from: {yolo_path}")
    yolo_model = YOLO(yolo_path)
    print(f"[INFO] YOLO classes: {yolo_model.names}")

    # ── MobileNetV3-Small freshness classifier ─────────────────────────────────
    print(f"[INFO] Loading freshness classifier from: {freshness_path}")
    backbone = models.mobilenet_v3_small(weights=None)
    in_features = backbone.classifier[3].in_features
    backbone.classifier[3] = nn.Linear(in_features, 2)   # 2-class head: Fresh / Spoiled

    state = torch.load(freshness_path, map_location=DEVICE)
    backbone.load_state_dict(state)
    backbone.to(DEVICE)
    backbone.eval()
    print(f"[INFO] Freshness classifier loaded  (device={DEVICE})")

    return yolo_model, backbone


# ═══════════════════════════════════════════════════════════════════════════════
# 2 · Core Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_crop(crop_bgr: np.ndarray) -> torch.Tensor:
    """BGR numpy crop → (1, 3, 224, 224) normalised tensor."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    return _infer_tf(pil).unsqueeze(0)   # (1, 3, 224, 224)


def _classify_freshness(fresh_model: nn.Module, crop_bgr: np.ndarray) -> dict:
    """
    Run the freshness classifier on a single BGR crop.

    Returns
    -------
    dict with keys:
        freshness_label : 'Fresh' | 'Moderate' | 'Spoiled'
        freshness_score : float = P(Fresh)
        probabilities   : {'Fresh': float, 'Spoiled': float}
    """
    tensor = _preprocess_crop(crop_bgr).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(fresh_model(tensor), dim=1).squeeze(0)  # (2,)
    fresh_p   = float(probs[0])   # index 0 = Fresh (alphabetical: Fresh < Spoiled)
    spoiled_p = float(probs[1])

    if fresh_p >= 0.6:
        label = "Fresh"
    elif spoiled_p >= 0.8:
        label = "Spoiled"
    else:
        label = "Near Spoilage"

    return {
        "freshness_label": label,
        "freshness_score": round(fresh_p, 4),
        "probabilities":   {"Fresh": round(fresh_p, 4), "Spoiled": round(spoiled_p, 4)},
    }


def _score_to_modifier(score: float) -> float:
    """Piecewise-linear map: P(Fresh) → shelf-life multiplier [0, 1]."""
    score = min(max(score, 0.0), 1.0)
    breakpoints = [
        (0.00, 0.00),
        (0.25, 0.05),
        (0.50, 0.30),
        (0.65, 0.55),
        (0.75, 0.75),
        (1.00, 1.00),
    ]
    for i in range(1, len(breakpoints)):
        s0, m0 = breakpoints[i - 1]
        s1, m1 = breakpoints[i]
        if s0 <= score <= s1:
            t = (score - s0) / (s1 - s0)
            return m0 + t * (m1 - m0)
    return 1.0


def _estimate_shelf_life(item_name: str, freshness_score: float, freshness_label: str) -> int:
    """Return estimated days remaining (0 if Spoiled)."""
    if freshness_label == "Spoiled" or freshness_score < 0.10:
        return 0
    baseline = BASELINE_SHELF_LIFE.get(item_name.lower(), BASELINE_SHELF_LIFE["unknown"])
    return max(0, math.floor(baseline * _score_to_modifier(freshness_score)))


# ═══════════════════════════════════════════════════════════════════════════════
# 2 · run_pipeline(frame)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    frame: np.ndarray,
    yolo_model: YOLO,
    fresh_model: nn.Module,
    det_conf: float = 0.4,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run the full detect → crop → classify → shelf-life pipeline on one BGR frame.

    Parameters
    ----------
    frame       : BGR numpy array (from cv2.VideoCapture)
    yolo_model  : loaded YOLO instance
    fresh_model : loaded MobileNetV3 freshness model
    det_conf    : YOLO detection confidence threshold

    Returns
    -------
    annotated   : BGR frame with bounding boxes and labels drawn
    results     : list of per-detection dicts
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()
    results   = []

    # ── Step 1: Detect ─────────────────────────────────────────────────────────
    det_out = yolo_model.predict(frame, conf=det_conf, device=DEVICE, verbose=False,iou=0.5)[0]

    if len(det_out.boxes) == 0:
        _put_text(annotated, "No fruits detected", (10, 30), (200, 200, 200))
        return annotated, results

    # ── Step 2–4: For each detection, crop → classify → shelf life ─────────────
    for box in det_out.boxes:
        cls_name = yolo_model.names[int(box.cls[0])]
        det_conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Padded crop (clamped to frame bounds)
        cx1 = max(0, x1 - BBOX_PAD)
        cy1 = max(0, y1 - BBOX_PAD)
        cx2 = min(w, x2 + BBOX_PAD)
        cy2 = min(h, y2 + BBOX_PAD)
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            continue

        fresh_result = _classify_freshness(fresh_model, crop)
        days         = _estimate_shelf_life(
            cls_name,
            fresh_result["freshness_score"],
            fresh_result["freshness_label"],
        )

        result = {
            "class_name":      cls_name,
            "det_confidence":  round(det_conf_val, 3),
            "freshness_label": fresh_result["freshness_label"],
            "freshness_score": fresh_result["freshness_score"],
            "probabilities":   fresh_result["probabilities"],
            "days_remaining":  days,
            "bbox":            (x1, y1, x2, y2),
        }
        results.append(result)

        # ── Step 5: Draw annotation ────────────────────────────────────────────
        color = VIS_COLORS.get(fresh_result["freshness_label"], (200, 200, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label_line1 = f"{cls_name}  {fresh_result['freshness_label']}"
        label_line2 = f"P={fresh_result['freshness_score']:.2f}  {days}d left"

        _put_text(annotated, label_line1, (x1, max(y1 - 22, 14)), color)
        _put_text(annotated, label_line2, (x1, max(y1 -  4, 30)), color, scale=0.45)

    return annotated, results


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: text rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _put_text(img, text, pos, color, scale=0.55, thickness=2):
    """Draw text with a dark shadow for readability on any background."""
    x, y = pos
    # shadow
    cv2.putText(img, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # foreground
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# 3 · Webcam Loop
# ═══════════════════════════════════════════════════════════════════════════════

def webcam_loop(yolo_model: YOLO, fresh_model: nn.Module, camera_index: int = 0):
    """
    Live webcam loop.

    Keys
    ----
    'c'  — capture current frame and run inference
    'q'  — quit
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index={camera_index}). "
              "Try a different --camera index.")
        sys.exit(1)

    # Increase buffer to 1 so we always read the latest frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\n[INFO] Webcam opened.")
    print("       Press  'c'  to capture and run inference.")
    print("       Press  'q'  to quit.\n")

    last_annotated = None   # holds the last inference result overlay

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame. Retrying...")
            continue

        # Flip horizontally for a mirror-view feel
        frame = cv2.flip(frame, 1)

        # Build display: live feed with a subtle status bar
        display = frame.copy()
        _put_text(display, "Press 'c' to capture  |  'q' to quit",
                  (10, display.shape[0] - 10), (220, 220, 220), scale=0.5, thickness=1)

        # Show live feed
        cv2.imshow("Smart Fridge — Live Feed", display)

        # Show last inference result (separate window so it doesn't flicker)
        if last_annotated is not None:
            cv2.imshow("Smart Fridge — Last Inference", last_annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Quit requested.")
            break

        elif key == ord('c'):
            print("[INFO] Capturing frame ...")
            annotated, results = run_pipeline(frame, yolo_model, fresh_model)
            last_annotated = annotated

            # Console summary
            if not results:
                print("       → No fruits detected.\n")
            else:
                print(f"       → {len(results)} detection(s):")
                for r in results:
                    print(f"          {r['class_name']:<8} "
                          f"det={r['det_confidence']:.2f}  "
                          f"{r['freshness_label']:<10} "
                          f"P(Fresh)={r['freshness_score']:.3f}  "
                          f"days={r['days_remaining']}")
                print()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Goodbye.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry-point
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(description="Smart Fridge real-time webcam demo")
    parser.add_argument(
        "--yolo",
        default="yolo_fruit3.pt",
        help="Path to the YOLOv8 detection model weights (default: yolo_fruit3.pt)",
    )
    parser.add_argument(
        "--freshness",
        default="freshness_model.pt",
        help="Path to the MobileNetV3 freshness classifier weights (default: freshness_model.pt)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0 = built-in laptop camera)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="YOLO detection confidence threshold (default: 0.30)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Validate model paths before opening the camera
    for label, path in [("YOLO", args.yolo), ("Freshness", args.freshness)]:
        if not Path(path).exists():
            print(f"[ERROR] {label} model not found: {path}")
            print("        Pass the correct path with --yolo / --freshness.")
            sys.exit(1)

    yolo_model, fresh_model = load_models(args.yolo, args.freshness)
    webcam_loop(yolo_model, fresh_model, camera_index=args.camera)
