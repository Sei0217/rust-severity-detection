import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time
from datetime import datetime
import os

# Configuration
MODEL_PATH = "../models/yolov8n.onnx"
DETECTIONS_FOLDER = "../detections"
CONFIDENCE_THRESHOLD = 0.75
INPUT_SIZE = 640

# Create detections folder
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)

# Class names
CLASS_NAMES = ["corrosion"]

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
        
        boxes, scores, class_ids = [], [], []
        inference_time = 0
        last_result_frame = None

        print("\n=== DETECTION STARTED ===")
        print("Press 'q'     - quit")
        print("Press SPACE   - capture & detect")
        print("Press 's'     - save current frame\n")

        while True:
            # Always grab a live frame for the preview
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

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
            #
            # --- MANUAL MODE: show live preview, overlay last result ---
            display_frame = frame_bgr.copy()
            if last_result_frame is not None:
                # Blend last detection overlay onto current live view
                display_frame = last_result_frame.copy()

            cv2.putText(display_frame, f"Detections: {len(boxes)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "MODE: MANUAL  [SPACE] to capture", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            cv2.imshow("RustWatch - Corrosion Detection", display_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break

            elif key == ord(' '):
                # Manual capture: grab a fresh frame and run inference
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
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(DETECTIONS_FOLDER, f"corrosion_{timestamp}.jpg")
                    cv2.imwrite(save_path, last_result_frame)
                    print(f"✓ Corrosion detected & saved: {save_path}")
                else:
                    print("No corrosion detected in capture")

            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(DETECTIONS_FOLDER, f"manual_{timestamp}.jpg")
                cv2.imwrite(save_path, display_frame)
                print(f"✓ Saved: {save_path}")
    
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
