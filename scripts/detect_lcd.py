#!/usr/bin/env python3
"""
detect_lcd.py — headless corrosion detection rendered straight to the 3.5"
SPI LCD framebuffer (/dev/fb1). No desktop / X11 / cv2 window required.

It reuses the inference + drawing pipeline from corrosion_detection.py, but
instead of cv2.imshow() it converts each annotated frame to the framebuffer's
pixel format (RGB565) and writes it directly to /dev/fb1.

Run from a Pi Connect Remote Shell (so console text stays off the panel):
    cd <repo>/scripts
    python3 detect_lcd.py
    python3 detect_lcd.py --no-save        # don't auto-save detections
    python3 detect_lcd.py --conf 0.6       # lower confidence threshold
    python3 detect_lcd.py --fb /dev/fb0    # target a different framebuffer

Quit with Ctrl+C. Detected frames auto-save to ../detections (5s cooldown).

Requirements:
- The LCD overlay must be loaded (piscreen -> /dev/fb1). Run `switch-lcd` first.
- The user must be in the 'video' group to write /dev/fb1 without sudo.
- If the colors look red/blue swapped, change COLOR_BGR2BGR565 below to
  COLOR_RGB2BGR565 (some panels expect the opposite channel order).
"""

import argparse
import contextlib
import io
import os
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

import corrosion_detection as cd

SAVE_COOLDOWN = 5.0  # seconds between auto-saves


def get_fb_geometry(fb_path):
    """Read resolution + bits-per-pixel for /dev/fbN from sysfs."""
    idx = fb_path.rstrip("/").split("fb")[-1]
    base = f"/sys/class/graphics/fb{idx}"
    with open(f"{base}/virtual_size") as f:
        xres, yres = (int(v) for v in f.read().strip().split(","))
    with open(f"{base}/bits_per_pixel") as f:
        bpp = int(f.read().strip())
    return xres, yres, bpp


def letterbox(img, tw, th):
    """Resize img into a tw x th canvas, preserving aspect ratio (black bars)."""
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def make_fb_writer(fb_path):
    """Open the framebuffer once and return a function that blits a BGR frame."""
    xres, yres, bpp = get_fb_geometry(fb_path)
    print(f"Framebuffer {fb_path}: {xres}x{yres} @ {bpp}bpp")
    if bpp == 16:
        convert = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2BGR565)
    elif bpp == 32:
        convert = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2BGRA)
    else:
        raise RuntimeError(f"Unsupported framebuffer depth: {bpp} bpp")

    fb = open(fb_path, "r+b")

    def write(frame_bgr):
        buf = convert(letterbox(frame_bgr, xres, yres))
        fb.seek(0)
        fb.write(buf.tobytes())
        fb.flush()

    return write


def main():
    parser = argparse.ArgumentParser(description="Headless corrosion detection on the SPI LCD.")
    parser.add_argument("--fb", default="/dev/fb1", help="Framebuffer device (default /dev/fb1)")
    parser.add_argument("--no-save", action="store_true", help="Do not auto-save detected frames")
    parser.add_argument("--conf", type=float, default=cd.CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default {cd.CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()

    if not os.path.exists(args.fb):
        raise SystemExit(f"{args.fb} not found — is the LCD overlay loaded? Run switch-lcd first.")

    write_fb = make_fb_writer(args.fb)

    print(f"Loading model: {cd.MODEL_PATH}")
    session = ort.InferenceSession(cd.MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print("Model loaded.")

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()
    time.sleep(2)
    try:
        picam2.set_controls({"AfMode": 2})  # continuous AF (Camera Module 3 only)
    except Exception:
        pass

    sev_colors = {"NONE": (180, 180, 180), "LOW": (0, 200, 0),
                  "MEDIUM": (0, 200, 200), "HIGH": (0, 0, 255)}
    last_save = 0.0
    print("Detection running — Ctrl+C to stop.")

    try:
        while True:
            rgb = picam2.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = rgb.shape[:2]

            inp, scale, pad_top, pad_left = cd.preprocess_image(rgb, cd.INPUT_SIZE)
            t0 = time.time()
            outputs = session.run(None, {input_name: inp})
            infer_ms = (time.time() - t0) * 1000

            # postprocess_detections prints debug lines per call; silence them
            with contextlib.redirect_stdout(io.StringIO()):
                boxes, scores, ids = cd.postprocess_detections(
                    outputs, w, h, scale, pad_top, pad_left, args.conf)

            frame = bgr.copy()
            if boxes:
                frame = cd.draw_detections(frame, boxes, scores, ids)

            det_dicts = [{"bbox": b, "class": cd.CLASS_NAMES[c]} for b, c in zip(boxes, ids)]
            analysis = cd.analyze_rust(det_dicts, (h, w))
            sev = analysis["severity"]

            hud = f"{sev}  det:{len(boxes)}  {infer_ms:.0f}ms"
            cv2.putText(frame, hud, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        sev_colors.get(sev, (255, 255, 255)), 2)
            if analysis.get("suspicious"):
                cv2.putText(frame, "! full-frame box", (8, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

            write_fb(frame)

            if boxes and not args.no_save and time.time() - last_save > SAVE_COOLDOWN:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(cd.DETECTIONS_FOLDER, f"corrosion_{ts}.jpg")
                cv2.imwrite(path, frame)
                print(f"Saved {path}  (severity {sev}, {len(boxes)} det)")
                last_save = time.time()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
