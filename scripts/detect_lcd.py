#!/usr/bin/env python3
"""
detect_lcd.py — corrosion detection rendered straight to the 3.5" SPI LCD
framebuffer (/dev/fb1). No desktop / X11 / cv2 window required.

It reuses the inference + drawing pipeline from corrosion_detection.py, but
instead of cv2.imshow() it converts each frame to the framebuffer's pixel
format (RGB565) and writes it directly to /dev/fb1.

Capture-on-demand (autoscan disabled): the LCD shows a LIVE preview, and
detection only runs when you trigger a capture — by TAPPING THE TOUCHSCREEN
or pressing Enter in the terminal.

Controls:
    Tap panel / Enter      capture the current frame + run detection
    Tap panel / Enter      (while reviewing) go back to live preview
    q + Enter              quit

Run from a desktop terminal or a Pi Connect Remote Shell:
    cd <repo>/scripts
    python3 detect_lcd.py
    python3 detect_lcd.py --no-save        # don't save captures
    python3 detect_lcd.py --conf 0.6       # lower confidence threshold
    python3 detect_lcd.py --no-touch       # keyboard only
    python3 detect_lcd.py --touch /dev/input/event3   # force touch device

Requirements:
- The LCD overlay must be loaded (piscreen -> /dev/fb1). Run `lcd-on` first.
- Must be in the 'video' group to write /dev/fb1, and the 'input' group to
  read the touchscreen (else touch is skipped and only the keyboard works):
      sudo usermod -aG video,input $USER   # then log out/in
- If the colors look red/blue swapped, change COLOR_BGR2BGR565 below to
  COLOR_RGB2BGR565 (some panels expect the opposite channel order).
"""

import argparse
import contextlib
import io
import os
import select
import struct
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

import corrosion_detection as cd

SEV_COLORS = {"NONE": (180, 180, 180), "LOW": (0, 200, 0),
              "MEDIUM": (0, 200, 200), "HIGH": (0, 0, 255)}

# Linux input_event: struct timeval (2 longs) + type,code (u16) + value (s32)
EV_FORMAT = "llHHi"
EV_SIZE = struct.calcsize(EV_FORMAT)
EV_KEY = 0x01
BTN_TOUCH = 0x14a
TOUCH_DEBOUNCE = 0.4  # seconds, ignore repeat touch-downs within this window


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


def find_touch_device(override=None):
    """Locate the touchscreen's /dev/input/eventN via /proc/bus/input/devices."""
    if override:
        return override
    try:
        with open("/proc/bus/input/devices") as f:
            blocks = f.read().split("\n\n")
    except OSError:
        return None
    for block in blocks:
        low = block.lower()
        if "ads7846" in low or "touchscreen" in low:
            for line in block.splitlines():
                if line.startswith("H:"):
                    for tok in line.split():
                        if tok.startswith("event"):
                            return f"/dev/input/{tok}"
    return None


def open_touch(path):
    """Open the touch device non-blocking; return fd or None (with a message)."""
    if not path:
        print("No touch device found — keyboard only.")
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        print(f"Touch: {path} (tap to capture)")
        return fd
    except PermissionError:
        print(f"Touch found at {path} but permission denied — add yourself to 'input':")
        print("  sudo usermod -aG input $USER   (then log out/in)")
    except OSError as e:
        print(f"Could not open touch device {path}: {e}")
    return None


def touch_pressed(fd):
    """Drain pending touch events; return True if a touch-DOWN occurred."""
    pressed = False
    try:
        while True:
            data = os.read(fd, EV_SIZE)
            if len(data) < EV_SIZE:
                break
            _, _, etype, code, value = struct.unpack(EV_FORMAT, data)
            if etype == EV_KEY and code == BTN_TOUCH and value == 1:
                pressed = True
    except BlockingIOError:
        pass
    return pressed


def poll_trigger(touch_fd, last_touch):
    """Non-blocking check of stdin + touch.
    Returns (quit_requested, capture_triggered, last_touch)."""
    watch = [sys.stdin]
    if touch_fd is not None:
        watch.append(touch_fd)
    ready, _, _ = select.select(watch, [], [], 0)
    quit_req = trigger = False
    for r in ready:
        if r is sys.stdin:
            line = sys.stdin.readline().strip().lower()
            if line == "q":
                quit_req = True
            else:
                trigger = True
        elif touch_pressed(touch_fd):
            now = time.time()
            if now - last_touch > TOUCH_DEBOUNCE:
                trigger = True
                last_touch = now
    return quit_req, trigger, last_touch


def capture_and_detect(picam2, session, input_name, conf):
    """Grab a frame, run inference, and return (annotated_bgr, boxes, analysis, ms)."""
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
            outputs, w, h, scale, pad_top, pad_left, conf)

    frame = bgr.copy()
    if boxes:
        frame = cd.draw_detections(frame, boxes, scores, ids)

    det_dicts = [{"bbox": b, "class": cd.CLASS_NAMES[c]} for b, c in zip(boxes, ids)]
    analysis = cd.analyze_rust(det_dicts, (h, w))
    return frame, boxes, analysis, infer_ms


def main():
    parser = argparse.ArgumentParser(description="Capture-on-demand corrosion detection on the SPI LCD.")
    parser.add_argument("--fb", default="/dev/fb1", help="Framebuffer device (default /dev/fb1)")
    parser.add_argument("--no-save", action="store_true", help="Do not save captured frames")
    parser.add_argument("--conf", type=float, default=cd.CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default {cd.CONFIDENCE_THRESHOLD})")
    parser.add_argument("--touch", default=None, help="Touch input device (default: auto-detect)")
    parser.add_argument("--no-touch", action="store_true", help="Disable touch, keyboard only")
    args = parser.parse_args()

    if not os.path.exists(args.fb):
        raise SystemExit(f"{args.fb} not found — is the LCD overlay loaded? Run lcd-on first.")

    write_fb = make_fb_writer(args.fb)

    touch_fd = None if args.no_touch else open_touch(find_touch_device(args.touch))

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

    trig = "Tap/Enter" if touch_fd is not None else "Enter"
    print("\n=== Capture-on-demand (autoscan OFF) ===")
    print(f"  {trig}      capture + detect / back to live")
    print("  q + Enter  quit\n")

    reviewing = False
    last_touch = 0.0
    try:
        while True:
            quit_req, trigger, last_touch = poll_trigger(touch_fd, last_touch)
            if quit_req:
                break
            if trigger:
                if not reviewing:
                    # Capture + detect, then freeze the result on the LCD
                    frame, boxes, analysis, infer_ms = capture_and_detect(
                        picam2, session, input_name, args.conf)
                    h = frame.shape[0]
                    sev = analysis["severity"]
                    cv2.putText(frame, f"{sev}  det:{len(boxes)}  {infer_ms:.0f}ms",
                                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                SEV_COLORS.get(sev, (255, 255, 255)), 2)
                    if analysis.get("suspicious"):
                        cv2.putText(frame, "! full-frame box", (8, h - 34),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
                    cv2.putText(frame, f"{trig}=new  q=quit", (8, h - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    write_fb(frame)

                    print(f"Captured: severity {sev}, {len(boxes)} detection(s), {infer_ms:.0f}ms")
                    if not args.no_save:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        prefix = "corrosion" if boxes else "capture"
                        path = os.path.join(cd.DETECTIONS_FOLDER, f"{prefix}_{ts}.jpg")
                        cv2.imwrite(path, frame)
                        print(f"Saved {path}")
                    reviewing = True
                else:
                    # Trigger while reviewing returns to live preview
                    reviewing = False

            if not reviewing:
                # LIVE preview (no inference)
                rgb = picam2.capture_array()
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f"LIVE  {trig}=capture", (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
                write_fb(frame)
            else:
                # Reviewing: hold the frozen result, just poll for input
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()
        if touch_fd is not None:
            os.close(touch_fd)
        print("Camera stopped.")


if __name__ == "__main__":
    main()
