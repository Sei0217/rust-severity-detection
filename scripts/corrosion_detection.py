import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time
from datetime import datetime
import os

# Configuration
MODEL_PATH = "../models/development/yolov8n_static_int8.onnx"
DETECTIONS_FOLDER = "../detections"
CONFIDENCE_THRESHOLD = 0.75
INPUT_SIZE = 640

# Create detections folder
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)

# Class names
CLASS_NAMES = ["corrosion"]

# Settings panel definition: (settings_key, display_label)
SETTING_LABELS = [
    ("show_inference_time", "Show Inference Time"),
    ("show_detection_count", "Show Detection Count"),
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
        panel_w = 240
        panel_h = 15 + len(SETTING_LABELS) * row_h + 8
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
    """Post-process YOLO outputs with correct scaling and NMS format"""
    predictions = np.squeeze(outputs[0])
    predictions = np.transpose(predictions)
    
    # Debug output
    max_conf = np.max(predictions[:, 4])
    num_above = np.sum(predictions[:, 4] >= conf_threshold)
    print(f"Max confidence: {max_conf:.3f}, Predictions above {conf_threshold}: {num_above}")
    
    boxes = []
    scores = []
    class_ids = []
    
    # YOLOv8 outputs: [8400, 5 or 6]
    # Each row: [x_center, y_center, width, height, confidence, (optional class)]
    for pred in predictions:
        confidence = pred[4]
        
        if confidence >= conf_threshold:
            # Coordinates are already in pixels (0-640), NOT normalized
            x_center, y_center, w, h = pred[:4]
            
            # Adjust for padding and scale to original image size
            x1 = (x_center - w / 2 - pad_left) / scale
            y1 = (y_center - h / 2 - pad_top) / scale
            width = w / scale
            height = h / scale
            
            # Clip to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            width = min(width, img_width - x1)
            height = min(height, img_height - y1)
            
            # Only add valid boxes
            if width > 5 and height > 5:
                # NMS expects [x, y, width, height]
                boxes.append([int(x1), int(y1), int(width), int(height)])
                scores.append(float(confidence))
                class_ids.append(0)
    
    # Apply NMS
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
        if len(indices) > 0:
            final_boxes = []
            final_scores = []
            final_ids = []
            
            for i in indices.flatten():
                b = boxes[i]
                # Convert [x, y, w, h] to [x1, y1, x2, y2] for drawing
                final_boxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
                final_scores.append(scores[i])
                final_ids.append(0)
            
            print(f"✓ {len(final_boxes)} detections after NMS")
            return final_boxes, final_scores, final_ids
    
    return [], [], []

def draw_detections(image, boxes, scores, class_ids):
    """Draw bounding boxes on image"""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        # Red box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Label
        label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_h - 5), (x1 + label_w, y1), (0, 0, 255), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
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
        
        reviewing = False
        boxes, scores, class_ids = [], [], []
        inference_time = 0
        last_result_frame = None
        ui = {
            "show_settings": False,
            "settings": {
                "show_inference_time": True,
                "show_detection_count": True,
                "auto_save": False,
            },
            "rects": {}
        }

        cv2.namedWindow("RustWatch - Corrosion Detection")
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
            else:
                # --- LIVE PREVIEW STATE ---
                display_frame = frame_bgr.copy()
                cv2.putText(display_frame, "LIVE  |  [SPACE] Capture", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

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
                    outputs, cap_width, cap_height, scale, pad_top, pad_left, CONFIDENCE_THRESHOLD
                )

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

            elif key == ord('r') and reviewing:
                # Discard capture and return to live preview
                print("Retaking — back to live preview")
                reviewing = False
                boxes, scores, class_ids = [], [], []
    
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
