import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time
from datetime import datetime
import os
from severity import analyze_rust

# Configuration
MODEL_PATH = "../models/best.onnx"
DETECTIONS_FOLDER = "../detections"
CONFIDENCE_THRESHOLD = 0.75
INPUT_SIZE = 640

# Create detections folder
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)

# Class names — order MUST match the model's output index ordering.
# This model was trained with the names {0: 'high', 1: 'low', 2: 'medium'}.
CLASS_NAMES = ["high", "low", "medium"]

# Per-class drawing color (BGR) for detection boxes
CLASS_COLORS = {
    "high":   (0,   0, 255),   # red
    "medium": (0, 200, 255),   # amber
    "low":    (0, 200,   0),   # green
}

# Settings panel definition: (settings_key, display_label)
SETTING_LABELS = [
    ("show_inference_time", "Show Inference Time"),
    ("show_detection_count", "Show Detection Count"),
    ("show_fps", "Show FPS"),
    ("show_model", "Show Model Name"),
    ("auto_save", "Auto-Save (skip review)"),
]

def on_mouse(event, x, y, _flags, ui):
    """Handle mouse clicks for settings button and panel."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    rects = ui["rects"]

    # Settings button click → toggle panel
    if "settings_btn" in rects:
        bx1, by1, bx2, by2 = rects["settings_btn"]
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            ui["show_settings"] = not ui["show_settings"]
            return

    # Option row click → toggle setting
    if ui["show_settings"]:
        for key, rect in rects.items():
            if key.startswith("opt_"):
                rx1, ry1, rx2, ry2 = rect
                if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                    setting_key = key[4:]
                    ui["settings"][setting_key] = not ui["settings"][setting_key]
                    return
        # Threshold +/- buttons
        if "threshold_minus" in rects:
            tx1, ty1, tx2, ty2 = rects["threshold_minus"]
            if tx1 <= x <= tx2 and ty1 <= y <= ty2:
                ui["settings"]["threshold"] = round(max(0.05, ui["settings"]["threshold"] - 0.05), 2)
                return
        if "threshold_plus" in rects:
            tx1, ty1, tx2, ty2 = rects["threshold_plus"]
            if tx1 <= x <= tx2 and ty1 <= y <= ty2:
                ui["settings"]["threshold"] = round(min(0.99, ui["settings"]["threshold"] + 0.05), 2)
                return
        # Model row click → cycle to next model
        if "model_row" in rects:
            mrx1, mry1, mrx2, mry2 = rects["model_row"]
            if mrx1 <= x <= mrx2 and mry1 <= y <= mry2:
                ui["model_idx"] = (ui["model_idx"] + 1) % len(ui["model_paths"])
                ui["model_changed"] = True
                return
        # Click outside the panel → close it
        if "panel" in rects:
            px1, py1, px2, py2 = rects["panel"]
            if not (px1 <= x <= px2 and py1 <= y <= py2):
                ui["show_settings"] = False

def draw_settings_ui(frame, ui):
    """Draw the settings button (and panel if open). Updates ui['rects']."""
    w = frame.shape[1]
    rects = {}

    # Settings button — top-right corner
    btn_x1, btn_y1 = w - 115, 5
    btn_x2, btn_y2 = w - 5, 32
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (50, 50, 50), -1)
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (160, 160, 160), 1)
    cv2.putText(frame, "* Settings", (btn_x1 + 5, btn_y2 - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    rects["settings_btn"] = (btn_x1, btn_y1, btn_x2, btn_y2)

    if ui["show_settings"]:
        row_h = 34
        panel_w = 260
        panel_h = 15 + len(SETTING_LABELS) * row_h + 12 + row_h + row_h + 8
        panel_x = w - panel_w - 5
        panel_y = btn_y2 + 4

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (130, 130, 130), 1)
        rects["panel"] = (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h)

        # Toggle rows
        for i, (key, label) in enumerate(SETTING_LABELS):
            ry = panel_y + 10 + i * row_h
            rx1, ry1 = panel_x + 8, ry
            rx2, ry2 = panel_x + panel_w - 8, ry + row_h - 5
            enabled = ui["settings"][key]
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2),
                         (0, 110, 0) if enabled else (55, 55, 55), -1)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (140, 140, 140), 1)
            status = "ON" if enabled else "OFF"
            cv2.putText(frame, f"{label}  [{status}]", (rx1 + 6, ry1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            rects[f"opt_{key}"] = (rx1, ry1, rx2, ry2)

        # Separator
        sep_y = panel_y + 10 + len(SETTING_LABELS) * row_h + 4
        cv2.line(frame, (panel_x + 8, sep_y), (panel_x + panel_w - 8, sep_y), (100, 100, 100), 1)

        # Threshold row
        thr = ui["settings"]["threshold"]
        ty = sep_y + 6
        tx1, ty1 = panel_x + 8, ty
        tx2, ty2 = panel_x + panel_w - 8, ty + row_h - 5
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (30, 30, 30), -1)
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (140, 140, 140), 1)
        # Minus button
        btn_w = 28
        cv2.rectangle(frame, (tx1, ty1), (tx1 + btn_w, ty2), (80, 40, 40), -1)
        cv2.putText(frame, "-", (tx1 + 8, ty1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 180), 2)
        rects["threshold_minus"] = (tx1, ty1, tx1 + btn_w, ty2)
        # Plus button
        cv2.rectangle(frame, (tx2 - btn_w, ty1), (tx2, ty2), (40, 80, 40), -1)
        cv2.putText(frame, "+", (tx2 - btn_w + 6, ty1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
        rects["threshold_plus"] = (tx2 - btn_w, ty1, tx2, ty2)
        # Label
        cv2.putText(frame, f"Threshold: {thr:.2f}", (tx1 + btn_w + 8, ty1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Model selector row
        model_name = os.path.splitext(os.path.basename(ui["model_paths"][ui["model_idx"]]))[0]
        my = ty2 + 6
        mx1, my1 = panel_x + 8, my
        mx2, my2 = panel_x + panel_w - 8, my + row_h - 5
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (40, 60, 100), -1)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (140, 140, 140), 1)
        cv2.putText(frame, f"Model: {model_name} >", (mx1 + 6, my1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 255), 1)
        rects["model_row"] = (mx1, my1, mx2, my2)

    ui["rects"] = rects

def preprocess_image(image, input_size):
    """Preprocess image for YOLO input"""
    # Convert RGBA to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    img_height, img_width = image.shape[:2]
    
    # Calculate scale
    scale = min(input_size / img_height, input_size / img_width)
    new_height = int(img_height * scale)
    new_width = int(img_width * scale)
    
    # Resize
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image (gray background)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    
    # Calculate padding
    pad_top = (input_size - new_height) // 2
    pad_left = (input_size - new_width) // 2
    
    # Place resized image in center
    padded[pad_top:pad_top+new_height, pad_left:pad_left+new_width, :] = resized
    
    # Normalize to [0, 1]
    input_image = padded.astype(np.float32) / 255.0
    
    # Transpose HWC to CHW
    input_image = np.transpose(input_image, (2, 0, 1))
    
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    
    return input_image, scale, pad_top, pad_left

def postprocess_detections(outputs, img_width, img_height, scale, pad_top, pad_left, conf_threshold):
    """Post-process multi-class YOLOv8 output: [1, 4+nc, 8400] → boxes/scores/class_ids."""
    predictions = np.squeeze(outputs[0])
    predictions = np.transpose(predictions)  # → [8400, 4+nc]

    class_scores = predictions[:, 4:]
    per_row_conf = class_scores.max(axis=1)
    per_row_cls  = class_scores.argmax(axis=1)

    max_conf = float(per_row_conf.max()) if per_row_conf.size else 0.0
    num_above = int((per_row_conf >= conf_threshold).sum())
    print(f"Max confidence: {max_conf:.3f}, Predictions above {conf_threshold}: {num_above}")

    boxes = []
    scores = []
    class_ids = []

    for pred, confidence, cls_id in zip(predictions, per_row_conf, per_row_cls):
        if confidence < conf_threshold:
            continue

        # Pixel-space center/size (model uses 0–INPUT_SIZE coords, not normalized)
        x_center, y_center, w, h = pred[:4]

        x1 = (x_center - w / 2 - pad_left) / scale
        y1 = (y_center - h / 2 - pad_top) / scale
        width  = w / scale
        height = h / scale

        x1 = max(0, x1)
        y1 = max(0, y1)
        width  = min(width,  img_width  - x1)
        height = min(height, img_height - y1)

        if width > 5 and height > 5:
            boxes.append([int(x1), int(y1), int(width), int(height)])
            scores.append(float(confidence))
            class_ids.append(int(cls_id))

    if len(boxes) == 0:
        return [], [], []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
    if len(indices) == 0:
        return [], [], []

    final_boxes, final_scores, final_ids = [], [], []
    for i in indices.flatten():
        b = boxes[i]
        final_boxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
        final_scores.append(scores[i])
        final_ids.append(class_ids[i])

    print(f"✓ {len(final_boxes)} detections after NMS")
    return final_boxes, final_scores, final_ids

def draw_detections(image, boxes, scores, class_ids):
    """Draw bounding boxes colored by severity class."""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        class_name = CLASS_NAMES[class_id]
        color = CLASS_COLORS.get(class_name, (0, 0, 255))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if y1 - label_h - 5 >= 0:
            cv2.rectangle(image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.rectangle(image, (x1, y1), (x1 + label_w, y1 + label_h + 5), color, -1)
            cv2.putText(image, label, (x1, y1 + label_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

def main():
    print("=== RustWatch Corrosion Detection System ===")
    
    session = None
    picam2 = None
    
    try:
        # Load model
        print(f"Loading model: {MODEL_PATH}")
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        print("✓ Model loaded successfully!")
        
        # Initialize camera
        print("Initializing camera...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        print("✓ Camera started!")
        
        time.sleep(2)
        
        # Attempt to enable continuous autofocus (Camera Module 3 only; ignored on fixed-focus cameras)
        try:
            picam2.set_controls({"AfMode": 2})  # 2 = Continuous AF
            print("✓ Continuous autofocus enabled")
        except Exception:
            print("  Autofocus not supported on this camera (fixed-focus)")

        reviewing = False
        boxes, scores, class_ids = [], [], []
        inference_time = 0
        last_result_frame = None
        blur_score = 999.0
        rust_analysis = {"severity": "NONE", "num_patches": 0, "coverage_ratio": 0.0, "suspicious": False}
        fps, fps_counter, fps_timer = 0, 0, time.time()

        # Scan for available models in the models folder
        model_dir = os.path.dirname(MODEL_PATH)
        model_paths = sorted([
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir) if f.endswith(".onnx")
        ]) or [MODEL_PATH]
        model_idx = next(
            (i for i, p in enumerate(model_paths) if os.path.abspath(p) == os.path.abspath(MODEL_PATH)),
            0
        )

        ui = {
            "show_settings": False,
            "settings": {
                "show_inference_time": True,
                "show_detection_count": True,
                "show_fps": False,
                "show_model": False,
                "auto_save": False,
                "threshold": CONFIDENCE_THRESHOLD,
            },
            "model_paths": model_paths,
            "model_idx": model_idx,
            "model_changed": False,
            "rects": {}
        }

        cv2.namedWindow("RustWatch - Corrosion Detection", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("RustWatch - Corrosion Detection", on_mouse, ui)

        print("\n=== DETECTION STARTED ===")
        print("Press 'q'         - quit")
        print("Press SPACE       - capture & detect")
        print("After capture:")
        print("  Press 's'       - save and return to preview")
        print("  Press 'r'       - retake (back to live preview)\n")

        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # FPS counter
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            if reviewing:
                # --- REVIEW STATE: show frozen capture with detection results ---
                display_frame = last_result_frame.copy()
                h, w = display_frame.shape[:2]
                # Bottom banner
                cv2.rectangle(display_frame, (0, h - 50), (w, h), (0, 0, 0), -1)
                cv2.putText(display_frame, "[s] Save   [r] Retake   [q] Quit",
                           (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                hud = "REVIEW"
                if ui["settings"]["show_detection_count"]:
                    hud += f"  |  Detections: {len(boxes)}"
                if ui["settings"]["show_inference_time"]:
                    hud += f"  |  Inference: {inference_time*1000:.1f}ms"
                cv2.putText(display_frame, hud, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # Severity line
                severity = rust_analysis["severity"]
                sev_color = {"NONE": (180, 180, 180), "LOW": (0, 200, 0),
                             "MEDIUM": (0, 200, 200), "HIGH": (0, 0, 255)}.get(severity, (255, 255, 255))
                sev_text = (f"Severity: {severity}  |  "
                            f"Patches: {rust_analysis['num_patches']}  |  "
                            f"Coverage: {rust_analysis['coverage_ratio']*100:.1f}%")
                cv2.putText(display_frame, sev_text, (10, 58),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, sev_color, 2)
                # Warnings for low-quality captures
                warn_y = 86
                if blur_score < 100.0:
                    cv2.putText(display_frame, f"! BLURRY IMAGE (score: {blur_score:.0f})",
                               (10, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
                    warn_y += 24
                if rust_analysis.get("suspicious"):
                    cv2.putText(display_frame, "! BOX COVERS FULL FRAME — POSSIBLE FALSE POSITIVE",
                               (10, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            else:
                # --- LIVE PREVIEW STATE ---
                display_frame = frame_bgr.copy()
                cv2.putText(display_frame, "LIVE  |  [SPACE] Capture", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                if ui["settings"]["show_fps"]:
                    cv2.putText(display_frame, f"FPS: {fps}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if ui["settings"]["show_model"]:
                hf = display_frame.shape[0]
                model_label = os.path.splitext(os.path.basename(ui["model_paths"][ui["model_idx"]]))[0]
                label_y = hf - 55 if reviewing else hf - 10
                cv2.putText(display_frame, f"Model: {model_label}", (10, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            draw_settings_ui(display_frame, ui)

            # if not manual_mode:
            #     # --- AUTO MODE: run inference on every frame ---
            #     input_image, scale, pad_top, pad_left = preprocess_image(frame_rgb, INPUT_SIZE)
            #
            #     inference_start = time.time()
            #     outputs = session.run(None, {input_name: input_image})
            #     inference_time = time.time() - inference_start
            #
            #     boxes, scores, class_ids = postprocess_detections(
            #         outputs, img_width, img_height, scale, pad_top, pad_left, CONFIDENCE_THRESHOLD
            #     )
            #
            #     display_frame = frame_bgr.copy()
            #     if len(boxes) > 0:
            #         display_frame = draw_detections(display_frame, boxes, scores, class_ids)
            #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #         save_path = os.path.join(DETECTIONS_FOLDER, f"corrosion_{timestamp}.jpg")
            #         cv2.imwrite(save_path, display_frame)
            #         print(f"✓ Auto-saved: {save_path}")
            #
            #     # FPS counter (auto mode only)
            #     fps_frame_count += 1
            #     if time.time() - fps_start_time >= 1.0:
            #         fps = fps_frame_count
            #         fps_frame_count = 0
            #         fps_start_time = time.time()
            #
            #     cv2.putText(display_frame, f"FPS: {fps}", (10, 30),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #     cv2.putText(display_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 60),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #     cv2.putText(display_frame, f"Detections: {len(boxes)}", (10, 90),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #     cv2.putText(display_frame, "MODE: AUTO", (10, 120),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("RustWatch - Corrosion Detection", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break

            elif key == ord(' ') and not reviewing:
                # Capture & run inference, then enter review state
                print("Capturing...")
                capture_rgb = picam2.capture_array()
                capture_bgr = cv2.cvtColor(capture_rgb, cv2.COLOR_RGB2BGR)
                cap_height, cap_width = capture_rgb.shape[:2]

                input_image, scale, pad_top, pad_left = preprocess_image(capture_rgb, INPUT_SIZE)

                inference_start = time.time()
                outputs = session.run(None, {input_name: input_image})
                inference_time = time.time() - inference_start

                boxes, scores, class_ids = postprocess_detections(
                    outputs, cap_width, cap_height, scale, pad_top, pad_left, ui["settings"]["threshold"]
                )

                det_dicts = [
                    {"bbox": box, "class": CLASS_NAMES[cls_id]}
                    for box, cls_id in zip(boxes, class_ids)
                ]
                rust_analysis = analyze_rust(det_dicts, (cap_height, cap_width), capture_bgr)
                print(f"Severity: {rust_analysis['severity']}  |  "
                      f"Patches: {rust_analysis['num_patches']}  |  "
                      f"Coverage: {rust_analysis['coverage_ratio']*100:.1f}%")

                # Blur score: Laplacian variance — low value = blurry image
                gray = cv2.cvtColor(capture_bgr, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

                last_result_frame = capture_bgr.copy()
                if len(boxes) > 0:
                    last_result_frame = draw_detections(last_result_frame, boxes, scores, class_ids)

                if ui["settings"]["auto_save"] and len(boxes) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(DETECTIONS_FOLDER, f"corrosion_{timestamp}.jpg")
                    cv2.imwrite(save_path, last_result_frame)
                    print(f"✓ Auto-saved: {save_path}")
                    boxes, scores, class_ids = [], [], []
                else:
                    if len(boxes) > 0:
                        print(f"✓ {len(boxes)} detection(s) found — press [s] to save or [r] to retake")
                    else:
                        print("No corrosion detected — press [s] to save anyway or [r] to retake")
                    reviewing = True

            elif key == ord('s') and reviewing:
                # Save captured frame and return to live preview
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = "corrosion" if len(boxes) > 0 else "capture"
                save_path = os.path.join(DETECTIONS_FOLDER, f"{prefix}_{timestamp}.jpg")
                cv2.imwrite(save_path, last_result_frame)
                print(f"✓ Saved: {save_path}")
                reviewing = False
                boxes, scores, class_ids = [], [], []
                blur_score = 999.0
                rust_analysis = {"severity": "NONE", "num_patches": 0, "coverage_ratio": 0.0, "suspicious": False}

            elif key == ord('r') and reviewing:
                # Discard capture and return to live preview
                print("Retaking — back to live preview")
                reviewing = False
                boxes, scores, class_ids = [], [], []
                blur_score = 999.0
                rust_analysis = {"severity": "NONE", "num_patches": 0, "coverage_ratio": 0.0, "suspicious": False}

            # Reload model if changed via settings panel
            if ui["model_changed"]:
                ui["model_changed"] = False
                new_path = ui["model_paths"][ui["model_idx"]]
                print(f"Switching model: {os.path.basename(new_path)}")
                try:
                    session = ort.InferenceSession(new_path)
                    input_name = session.get_inputs()[0].name
                    print(f"✓ Model loaded: {os.path.basename(new_path)}")
                except Exception as e:
                    print(f"Failed to load model: {e}")
                    ui["model_idx"] = (ui["model_idx"] - 1) % len(ui["model_paths"])
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        cv2.destroyAllWindows()
        if picam2 is not None:
            picam2.stop()
        print("✓ System stopped")

if __name__ == "__main__":
    main()
