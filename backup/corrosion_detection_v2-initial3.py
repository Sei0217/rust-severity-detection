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
CONFIDENCE_THRESHOLD = 0.25
INPUT_SIZE = 640

# Create detections folder if it doesn't exist
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)

# Class names
CLASS_NAMES = ["corrosion"]

def preprocess_image(image, input_size):
    """Preprocess image for YOLO input - matches YOLOv8 expectations"""
    # Image should be RGB format (from picamera2)
    img_height, img_width = image.shape[:2]
    
    # Calculate scale to fit image into input_size x input_size
    scale = min(input_size / img_height, input_size / img_width)
    new_height = int(img_height * scale)
    new_width = int(img_width * scale)
    
    # Ensure dimensions are at least 1
    new_height = max(1, new_height)
    new_width = max(1, new_width)
    
    # Resize image maintaining RGB format
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image (filled with gray 114)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets to center the image
    pad_top = (input_size - new_height) // 2
    pad_left = (input_size - new_width) // 2
    
    # Place resized image in center
    padded[pad_top:pad_top+new_height, pad_left:pad_left+new_width, :] = resized
    
    # Convert to float32 and normalize to [0, 1]
    input_image = padded.astype(np.float32) / 255.0
    
    # Transpose from HWC to CHW (YOLOv8 expects channels first)
    input_image = np.transpose(input_image, (2, 0, 1))
    
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    
    return input_image, scale, pad_top, pad_left

def postprocess_detections(outputs, img_width, img_height, scale, pad_top, pad_left, conf_threshold):
    """Post-process YOLO outputs to get bounding boxes"""
    predictions = outputs[0]
    
    # YOLOv8 ONNX output: (1, 6, 8400) for single class
    # Transpose to (8400, 6)
    predictions = np.squeeze(predictions)  # (6, 8400)
    predictions = np.transpose(predictions)  # (8400, 6)
    
    print(f"\n=== DEBUG ===")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0]}")
    print(f"Max confidence in batch: {np.max(predictions[:, 4])}")
    print (f"Predictions above {conf_threshold}: {np.sum(predictions[:, 4] >= conf_threshold)}")
    
    boxes = []
    scores = []
    class_ids = []
    
    for pred in predictions:
        # Format: [x_center, y_center, width, height, confidence, class]
        if len(pred) == 6:
            x_center, y_center, width, height, confidence, class_prob = pred
        elif len(pred) == 5:
            x_center, y_center, width, height, confidence = pred
        else:
            continue
        
        if confidence >= conf_threshold:
            # YOLOv8 outputs are normalized [0,1] relative to 640x640 input
            # Convert to pixels in 640x640 space
            x_center_px = x_center * INPUT_SIZE
            y_center_px = y_center * INPUT_SIZE
            width_px = width * INPUT_SIZE
            height_px = height * INPUT_SIZE
            
            # Convert to corner format
            x1 = x_center_px - width_px / 2
            y1 = y_center_px - height_px / 2
            x2 = x_center_px + width_px / 2
            y2 = y_center_px + height_px / 2
            
            # Adjust for padding
            x1 = x1 - pad_left
            y1 = y1 - pad_top
            x2 = x2 - pad_left
            y2 = y2 - pad_top
            
            # Scale back to original image size
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale
            
            # Clip to image boundaries
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            # Only add if box has reasonable size
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(confidence))
                class_ids.append(0)
    
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
        if len(indices) > 0:
            indices = indices.flatten()
            return [boxes[i] for i in indices], [scores[i] for i in indices], [class_ids[i] for i in indices]
    
    return [], [], []

def draw_detections(image, boxes, scores, class_ids):
    """Draw bounding boxes on image"""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_height - 5), (x1 + label_width, y1), (0, 0, 255), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image

def main():
    print("Initializing Corrosion Detection System...")
    
    # Initialize variables
    session = None
    picam2 = None
    
    try:
        print(f"Loading model: {MODEL_PATH}")
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        print("Model loaded successfully!")
        
        print("Initializing camera...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        print("Camera started!")
        
        time.sleep(2)
        
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        print("\n=== CORROSION DETECTION STARTED ===")
        print("Press 'q' to quit")
        print("Press 's' to manually save current frame\n")
        
        while True:
            # Capture frame (RGB from picamera2)
            frame_rgb = picam2.capture_array()
    
            # Create BGR copy ONLY for OpenCV display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            img_height, img_width = frame_rgb.shape[:2]

            # Preprocess RGB frame for model
            input_image, scale, pad_top, pad_left = preprocess_image(frame_rgb, INPUT_SIZE)

            # Run inference
            inference_start = time.time()
            outputs = session.run(None, {input_name: input_image})
            inference_time = time.time() - inference_start

            # Post-process
            boxes, scores, class_ids = postprocess_detections(
                outputs, img_width, img_height, scale, pad_top, pad_left, CONFIDENCE_THRESHOLD
            )

            # Draw on BGR frame for display
            display_frame = frame_bgr.copy()
            
            if len(boxes) > 0:
                display_frame = draw_detections(display_frame, boxes, scores, class_ids)
                
                # Auto-save detected images
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(DETECTIONS_FOLDER, f"corrosion_{timestamp}.jpg")
                cv2.imwrite(save_path, display_frame)
                print(f"Detection saved: {save_path}")
            
            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_frame_count
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Display overlays
            cv2.putText(display_frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Detections: {len(boxes)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Corrosion Detection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(DETECTIONS_FOLDER, f"manual_{timestamp}.jpg")
                cv2.imwrite(save_path, display_frame)
                print(f"Manually saved: {save_path}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nShutting down...")
        cv2.destroyAllWindows()
        if picam2 is not None:
            picam2.stop()
        print("System stopped.")

if __name__ == "__main__":
    main()
