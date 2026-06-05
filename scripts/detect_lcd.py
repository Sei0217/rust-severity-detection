#!/usr/bin/env python3
"""
detect_lcd.py — corrosion detection rendered straight to the 3.5" SPI LCD
framebuffer (/dev/fb1). No desktop / X11 / cv2 window required.

Reuses the inference + drawing pipeline from corrosion_detection.py, but
instead of cv2.imshow() it converts each frame to the framebuffer's pixel
format (RGB565) and writes it directly to /dev/fb1.

Capture-on-demand (autoscan disabled) with a touch UI:
    LIVE     : short-tap / Enter        -> capture + detect
    REVIEW   : short-tap / Enter        -> back to live
    anywhere : long-press / 's'+Enter   -> open/close Settings
    quit     : 'q' + Enter

SETTINGS (tap-zones, no calibration) — 4 full-width rows; tap a row's
LEFT half to decrease/prev, RIGHT half to increase/next/toggle:
    Threshold   |  Model  |  Auto-save  |  Show FPS

Run from a desktop terminal or a Pi Connect Remote Shell:
    cd <repo>/scripts
    python3 detect_lcd.py
    python3 detect_lcd.py --conf 0.6
    python3 detect_lcd.py --no-touch                 # keyboard only
    python3 detect_lcd.py --swap-xy --flip-y         # fix touch orientation
    python3 detect_lcd.py --touch-debug              # print tap coords

Touch orientation: if a tapped row/side is mirrored, toggle --swap-xy /
--flip-x / --flip-y until the on-screen marker lands under your finger.

Requirements:
- LCD overlay loaded (piscreen -> /dev/fb1). Run `lcd-on` first.
- Be in 'video' (write /dev/fb1) and 'input' (read touch) groups:
      sudo usermod -aG video,input $USER   # then log out/in
- If colors look red/blue swapped, change COLOR_BGR2BGR565 -> COLOR_RGB2BGR565.
"""

import argparse
import contextlib
import fcntl
import io
import json
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

# Linux input_event: struct timeval (2 longs) + type, code (u16) + value (s32)
EV_FORMAT = "llHHi"
EV_SIZE = struct.calcsize(EV_FORMAT)
EV_KEY = 0x01
EV_ABS = 0x03
ABS_X = 0x00
ABS_Y = 0x01
BTN_TOUCH = 0x14a
LONG_PRESS = 0.6          # seconds held = "long press"
TOUCH_CAL_FILE = os.path.join(os.path.expanduser("~"), ".detect_lcd_touch.json")

SETTING_ROWS = ["threshold", "model", "auto_save", "show_fps"]
ROW_TOP = 0.18            # rows occupy this fraction..0.98 of the screen height


# ----------------------------------------------------------------------------
# Framebuffer
# ----------------------------------------------------------------------------
def get_fb_geometry(fb_path):
    idx = fb_path.rstrip("/").split("fb")[-1]
    base = f"/sys/class/graphics/fb{idx}"
    with open(f"{base}/virtual_size") as f:
        xres, yres = (int(v) for v in f.read().strip().split(","))
    with open(f"{base}/bits_per_pixel") as f:
        bpp = int(f.read().strip())
    return xres, yres, bpp


def letterbox(img, tw, th):
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def make_fb_writer(fb_path):
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

    return write, xres, yres


# ----------------------------------------------------------------------------
# Touchscreen (ADS7846) — gesture detection, no calibration
# ----------------------------------------------------------------------------
def _eviocgabs(axis):
    return (2 << 30) | (struct.calcsize("6i") << 16) | (ord("E") << 8) | (0x40 + axis)


class Touch:
    def __init__(self, fd, swap_xy=False, flip_x=False, flip_y=False, debug=False):
        self.fd = fd
        self.swap_xy, self.flip_x, self.flip_y, self.debug = swap_xy, flip_x, flip_y, debug
        self.x = self.y = 0
        self.down_time = None
        # Affine correction (corrected = a*raw + b) from 2-tap calibration
        self.cal_ax, self.cal_bx, self.cal_ay, self.cal_by = 1.0, 0.0, 1.0, 0.0
        self.xmin, self.xmax = self._range(ABS_X)
        self.ymin, self.ymax = self._range(ABS_Y)

    def _range(self, axis, default=(0, 4095)):
        try:
            buf = bytearray(struct.calcsize("6i"))
            fcntl.ioctl(self.fd, _eviocgabs(axis), buf, True)
            _, mn, mx, *_ = struct.unpack("6i", bytes(buf))
            if mx > mn:
                return mn, mx
        except OSError:
            pass
        return default

    def _norm(self, x, y):
        nx = (x - self.xmin) / max(1, self.xmax - self.xmin)
        ny = (y - self.ymin) / max(1, self.ymax - self.ymin)
        if self.swap_xy:
            nx, ny = ny, nx
        if self.flip_x:
            nx = 1.0 - nx
        if self.flip_y:
            ny = 1.0 - ny
        nx = self.cal_ax * nx + self.cal_bx
        ny = self.cal_ay * ny + self.cal_by
        return min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny))

    def poll(self):
        """Drain events; return list of gestures: ('tap', nx, ny) / ('long', nx, ny)."""
        gestures = []
        try:
            while True:
                data = os.read(self.fd, EV_SIZE)
                if len(data) < EV_SIZE:
                    break
                _, _, etype, code, value = struct.unpack(EV_FORMAT, data)
                if etype == EV_ABS:
                    if code == ABS_X:
                        self.x = value
                    elif code == ABS_Y:
                        self.y = value
                elif etype == EV_KEY and code == BTN_TOUCH:
                    if value == 1:
                        self.down_time = time.time()
                    elif value == 0 and self.down_time is not None:
                        dur = time.time() - self.down_time
                        nx, ny = self._norm(self.x, self.y)
                        kind = "long" if dur >= LONG_PRESS else "tap"
                        if self.debug:
                            print(f"touch {kind}: nx={nx:.2f} ny={ny:.2f} (raw {self.x},{self.y})")
                        gestures.append((kind, nx, ny))
                        self.down_time = None
        except BlockingIOError:
            pass
        return gestures


def find_touch_device(override=None):
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
    if not path:
        print("No touch device found — keyboard only.")
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        print(f"Touch: {path}")
        return fd
    except PermissionError:
        print(f"Touch at {path} but permission denied — add yourself to 'input':")
        print("  sudo usermod -aG input $USER   (then log out/in)")
    except OSError as e:
        print(f"Could not open touch device {path}: {e}")
    return None


# ----------------------------------------------------------------------------
# Detection + drawing
# ----------------------------------------------------------------------------
def capture_and_detect(picam2, session, input_name, conf):
    rgb = picam2.capture_array()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]
    inp, scale, pad_top, pad_left = cd.preprocess_image(rgb, cd.INPUT_SIZE)
    t0 = time.time()
    outputs = session.run(None, {input_name: inp})
    infer_ms = (time.time() - t0) * 1000
    with contextlib.redirect_stdout(io.StringIO()):
        boxes, scores, ids = cd.postprocess_detections(
            outputs, w, h, scale, pad_top, pad_left, conf)
    frame = bgr.copy()
    if boxes:
        frame = cd.draw_detections(frame, boxes, scores, ids)
    det_dicts = [{"bbox": b, "class": cd.CLASS_NAMES[c]} for b, c in zip(boxes, ids)]
    analysis = cd.analyze_rust(det_dicts, (h, w))
    return frame, boxes, analysis, infer_ms


def model_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def draw_settings(xres, yres, settings, models, model_idx, last_tap):
    """Render the settings screen at native fb size (no letterbox)."""
    c = np.zeros((yres, xres, 3), dtype=np.uint8)
    cv2.putText(c, "SETTINGS  (long-press = back)", (8, int(0.12 * yres)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    rows = [
        ("Threshold", f"< {settings['threshold']:.2f} >"),
        ("Model", f"< {model_name(models[model_idx])} >"),
        ("Auto-save", "ON" if settings["auto_save"] else "OFF"),
        ("Show FPS", "ON" if settings["show_fps"] else "OFF"),
    ]
    row_h = (0.98 - ROW_TOP) / len(rows)
    for i, (label, value) in enumerate(rows):
        y1 = int((ROW_TOP + i * row_h) * yres)
        y2 = int((ROW_TOP + (i + 1) * row_h) * yres) - 4
        cv2.rectangle(c, (6, y1), (xres - 6, y2), (60, 60, 60), -1)
        cv2.line(c, (xres // 2, y1), (xres // 2, y2), (90, 90, 90), 1)
        ty = (y1 + y2) // 2 + 6
        cv2.putText(c, label, (14, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        (vw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(c, value, (xres - 14 - vw, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 220, 255), 1)

    # Marker showing where the last tap landed (helps fix orientation flags)
    if last_tap is not None:
        mx, my = int(last_tap[0] * xres), int(last_tap[1] * yres)
        cv2.circle(c, (mx, my), 9, (0, 165, 255), 2)
    return c


def settings_hit(nx, ny):
    """Map a normalized tap to (row_index, side L/R), or (None, None)."""
    if ny < ROW_TOP or ny > 0.98:
        return None, None
    row = int((ny - ROW_TOP) / ((0.98 - ROW_TOP) / len(SETTING_ROWS)))
    row = min(row, len(SETTING_ROWS) - 1)
    return row, ("L" if nx < 0.5 else "R")


def apply_setting(row, side, settings, models, model_idx, reload_model):
    """Adjust the given setting; returns possibly-updated model_idx."""
    key = SETTING_ROWS[row]
    if key == "threshold":
        step = -0.05 if side == "L" else 0.05
        settings["threshold"] = round(min(0.95, max(0.05, settings["threshold"] + step)), 2)
    elif key == "model":
        if len(models) > 1:
            model_idx = (model_idx + (-1 if side == "L" else 1)) % len(models)
            reload_model(models[model_idx])
    elif key == "auto_save":
        settings["auto_save"] = not settings["auto_save"]
    elif key == "show_fps":
        settings["show_fps"] = not settings["show_fps"]
    return model_idx


# ----------------------------------------------------------------------------
# Touch calibration (2-tap) + persistence
# ----------------------------------------------------------------------------
def load_cal():
    try:
        with open(TOUCH_CAL_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_cal(d):
    try:
        with open(TOUCH_CAL_FILE, "w") as f:
            json.dump(d, f)
    except OSError as e:
        print(f"Could not save calibration: {e}")


def wait_for_tap(touch):
    while True:
        for kind, nx, ny in touch.poll():
            if kind in ("tap", "long"):
                return nx, ny
        time.sleep(0.02)


def run_calibration(touch, write_fb, xres, yres):
    """Show two corner targets, capture taps, return affine (ax,bx,ay,by) or None."""
    print("Calibration: tap each target on the LCD.")
    touch.cal_ax, touch.cal_bx, touch.cal_ay, touch.cal_by = 1.0, 0.0, 1.0, 0.0  # identity
    targets = [("TOP-LEFT", (16, 22)), ("BOTTOM-RIGHT", (xres - 16, yres - 22))]
    obs = []
    for name, (tx, ty) in targets:
        c = np.zeros((yres, xres, 3), dtype=np.uint8)
        cv2.line(c, (tx - 16, ty), (tx + 16, ty), (0, 255, 255), 1)
        cv2.line(c, (tx, ty - 16), (tx, ty + 16), (0, 255, 255), 1)
        cv2.circle(c, (tx, ty), 11, (0, 255, 255), 2)
        cv2.putText(c, f"Tap the {name} cross", (xres // 2 - 120, yres // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        write_fb(c)
        obs.append(wait_for_tap(touch))
        time.sleep(0.5)  # debounce between targets
    (ox0, oy0), (ox1, oy1) = obs
    if abs(ox1 - ox0) < 0.05 or abs(oy1 - oy0) < 0.05:
        print("Calibration points too close together — ignored.")
        return None
    # Map observed taps to the crosses' actual screen fractions, so corrected
    # coords equal screen fractions (matching where draw_settings puts the rows).
    tfx0, tfy0 = targets[0][1][0] / xres, targets[0][1][1] / yres
    tfx1, tfy1 = targets[1][1][0] / xres, targets[1][1][1] / yres
    ax = (tfx1 - tfx0) / (ox1 - ox0)
    ay = (tfy1 - tfy0) / (oy1 - oy0)
    return ax, tfx0 - ax * ox0, ay, tfy0 - ay * oy0


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Capture-on-demand corrosion detection on the SPI LCD.")
    p.add_argument("--fb", default="/dev/fb1")
    p.add_argument("--conf", type=float, default=cd.CONFIDENCE_THRESHOLD)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--touch", default=None, help="Touch device (default: auto-detect)")
    p.add_argument("--no-touch", action="store_true")
    p.add_argument("--swap-xy", action="store_true", help="Swap touch X/Y (rotated panels)")
    p.add_argument("--flip-x", action="store_true")
    p.add_argument("--flip-y", action="store_true")
    p.add_argument("--touch-debug", action="store_true", help="Print tap coordinates")
    p.add_argument("--calibrate", action="store_true",
                   help="Run 2-tap touch calibration (with your --flip/--swap flags) and save it")
    args = p.parse_args()

    if not os.path.exists(args.fb):
        raise SystemExit(f"{args.fb} not found — is the LCD overlay loaded? Run lcd-on first.")

    write_fb, xres, yres = make_fb_writer(args.fb)

    touch = None
    if not args.no_touch:
        fd = open_touch(find_touch_device(args.touch))
        if fd is not None:
            touch = Touch(fd, args.swap_xy, args.flip_x, args.flip_y, args.touch_debug)

    # Touch calibration: run it (and save), or load a previously saved one.
    if touch is not None:
        if args.calibrate:
            cal = run_calibration(touch, write_fb, xres, yres)
            if cal:
                touch.cal_ax, touch.cal_bx, touch.cal_ay, touch.cal_by = cal
                save_cal({"swap_xy": args.swap_xy, "flip_x": args.flip_x, "flip_y": args.flip_y,
                          "ax": cal[0], "bx": cal[1], "ay": cal[2], "by": cal[3]})
                print(f"Saved touch calibration to {TOUCH_CAL_FILE}")
        else:
            saved = load_cal()
            if saved and "ax" in saved:
                touch.swap_xy = saved.get("swap_xy", touch.swap_xy)
                touch.flip_x = saved.get("flip_x", touch.flip_x)
                touch.flip_y = saved.get("flip_y", touch.flip_y)
                touch.cal_ax, touch.cal_bx = saved["ax"], saved["bx"]
                touch.cal_ay, touch.cal_by = saved["ay"], saved["by"]
                print("Loaded saved touch calibration.")
            elif saved:
                print("Old calibration format — please re-run:  python3 detect_lcd.py --calibrate --flip-x")

    # Model list (for the Model setting), like corrosion_detection.py
    model_dir = os.path.dirname(cd.MODEL_PATH)
    models = sorted(os.path.join(model_dir, f) for f in os.listdir(model_dir)
                    if f.endswith(".onnx")) or [cd.MODEL_PATH]
    model_idx = next((i for i, m in enumerate(models)
                      if os.path.abspath(m) == os.path.abspath(cd.MODEL_PATH)), 0)

    print(f"Loading model: {models[model_idx]}")
    session = ort.InferenceSession(models[model_idx])
    input_name = session.get_inputs()[0].name
    print("Model loaded.")

    def reload_model(path):
        nonlocal session, input_name
        try:
            session = ort.InferenceSession(path)
            input_name = session.get_inputs()[0].name
            print(f"Model -> {model_name(path)}")
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    settings = {"threshold": args.conf, "auto_save": not args.no_save, "show_fps": False}

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()
    time.sleep(2)
    try:
        picam2.set_controls({"AfMode": 2})
    except Exception:
        pass

    trig = "Tap/Enter" if touch else "Enter"
    print("\n=== Capture-on-demand (autoscan OFF) ===")
    print(f"  {trig}            capture / back")
    print("  long-press / s    open/close Settings")
    print("  q + Enter         quit\n")

    state = "live"          # live | review | settings
    last_tap = None
    fps, fps_n, fps_t = 0, 0, time.time()

    try:
        while True:
            # --- gather input (keyboard + touch), non-blocking ---
            timeout = 0.0 if state == "live" else 0.08
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            kbd = sys.stdin.readline().strip().lower() if r else None
            gestures = touch.poll() if touch else []

            if kbd == "q":
                break
            kbd_enter = kbd is not None and kbd != "s" and kbd != "q"
            kbd_settings = kbd == "s"

            # --- map inputs to actions for the current state ---
            long_press = any(g[0] == "long" for g in gestures)
            taps = [g for g in gestures if g[0] == "tap"]

            if state == "settings":
                if long_press or kbd_settings or kbd_enter:
                    state = "live"
                else:
                    for _, nx, ny in taps:
                        last_tap = (nx, ny)
                        row, side = settings_hit(nx, ny)
                        if row is not None:
                            model_idx = apply_setting(row, side, settings, models, model_idx, reload_model)
                c = draw_settings(xres, yres, settings, models, model_idx, last_tap)
                write_fb(c)
                continue

            if long_press or kbd_settings:
                state = "settings"
                last_tap = None
                continue

            triggered = bool(taps) or kbd_enter

            if state == "review":
                if triggered:
                    state = "live"
                else:
                    time.sleep(0.03)
                continue

            # --- LIVE ---
            if triggered:
                frame, boxes, analysis, infer_ms = capture_and_detect(
                    picam2, session, input_name, settings["threshold"])
                h = frame.shape[0]
                sev = analysis["severity"]
                cv2.putText(frame, f"{sev}  det:{len(boxes)}  {infer_ms:.0f}ms",
                            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            SEV_COLORS.get(sev, (255, 255, 255)), 2)
                if analysis.get("suspicious"):
                    cv2.putText(frame, "! full-frame box", (8, h - 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
                cv2.putText(frame, f"{trig}=new  long=settings", (8, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                write_fb(frame)
                print(f"Captured: severity {sev}, {len(boxes)} detection(s), {infer_ms:.0f}ms")
                if settings["auto_save"]:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    prefix = "corrosion" if boxes else "capture"
                    path = os.path.join(cd.DETECTIONS_FOLDER, f"{prefix}_{ts}.jpg")
                    cv2.imwrite(path, frame)
                    print(f"Saved {path}")
                state = "review"
                continue

            # live preview frame
            rgb = picam2.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n, 0, time.time()
            cv2.putText(frame, f"LIVE  {trig}=capture", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            if settings["show_fps"]:
                cv2.putText(frame, f"FPS: {fps}", (8, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            write_fb(frame)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()
        if touch is not None:
            os.close(touch.fd)
        print("Camera stopped.")


if __name__ == "__main__":
    main()
