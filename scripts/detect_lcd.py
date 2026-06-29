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
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".detect_lcd_settings.json")

SETTINGS_LAYOUT = ["threshold", "model", "auto_save", "show_fps", "gallery", "shutdown"]
ROW_TOP = 0.18            # rows occupy this fraction..0.98 of the screen height
ZOOM_LEVELS = [1.0, 2.0, 4.0]   # digital zoom factors cycled by the live button


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
ROTATE_OPS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
              270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_frame(img, deg):
    """Rotate a captured frame 0/90/180/270 deg in software (angled camera)."""
    op = ROTATE_OPS.get(deg)
    return cv2.rotate(img, op) if op is not None else img


def capture_and_detect(picam2, session, input_name, conf, rotate=0):
    rgb = picam2.capture_array()
    if rotate:
        rgb = rotate_frame(rgb, rotate)
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
    # Blur score: Laplacian variance on the raw capture — low value = blurry
    blur_score = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return frame, boxes, scores, ids, analysis, infer_ms, blur_score


def model_name(path):
    return os.path.splitext(os.path.basename(path))[0]


# ----------------------------------------------------------------------------
# Digital zoom (ScalerCrop) — centered crop of the sensor, scaled by the ISP
# ----------------------------------------------------------------------------
def get_full_crop(picam2):
    """Full usable sensor rectangle (x, y, w, h) for ScalerCrop."""
    props = picam2.camera_properties
    rect = props.get("ScalerCropMaximum")
    if rect and rect[2] > 0 and rect[3] > 0:
        return tuple(int(v) for v in rect)
    size = props.get("PixelArraySize")
    if size:
        return (0, 0, int(size[0]), int(size[1]))
    return (0, 0, 3280, 2464)  # IMX219 / Camera Module 2 fallback


def apply_zoom(picam2, factor, full_rect):
    """Set a centered ScalerCrop for the given zoom factor (1.0 = full frame)."""
    fx, fy, fw, fh = full_rect
    cw, ch = int(fw / factor), int(fh / factor)
    cx = fx + (fw - cw) // 2
    cy = fy + (fh - ch) // 2
    try:
        picam2.set_controls({"ScalerCrop": (cx, cy, cw, ch)})
    except Exception as e:
        print(f"Zoom set failed: {e}")


def draw_zoom_button(frame, factor):
    """Small circular zoom indicator/button at the bottom-center of the frame."""
    h, w = frame.shape[:2]
    cx, cy, r = w // 2, int(h * 0.90), 20
    cv2.circle(frame, (cx, cy), r, (40, 40, 40), -1)
    cv2.circle(frame, (cx, cy), r, (220, 220, 220), 1)
    label = f"{factor:g}x"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(frame, label, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def _centered(c, text, rect, scale, color, thick):
    x1, y1, x2, y2 = rect
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    tx = x1 + ((x2 - x1) - tw) // 2
    ty = y1 + ((y2 - y1) + th) // 2
    cv2.putText(c, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_settings(xres, yres, settings, models, model_idx, last_tap):
    """Render the settings screen at native fb size (no letterbox).

    Adjustable rows show tappable [-]/[+] (or [<]/[>]) edge buttons; the whole
    left/right half is the actual hit zone. Toggle rows flip on any tap.
    """
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    c = np.zeros((yres, xres, 3), dtype=np.uint8)
    cv2.putText(c, "SETTINGS   (long-press = back)", (8, int(0.12 * yres)),
                FONT, 0.5, (0, 255, 255), 1)

    # kind, label, value-string, left-btn, right-btn
    rows = [
        ("adjust", "Threshold", f"{settings['threshold']:.2f}", "-", "+"),
        ("adjust", "Model", model_name(models[model_idx]), "<", ">"),
        ("toggle", "Auto-save", "ON" if settings["auto_save"] else "OFF", None, None),
        ("toggle", "Show FPS", "ON" if settings["show_fps"] else "OFF", None, None),
        ("action", "Gallery", "view captures >", None, None),
        ("action", "Shutdown", "power off Pi >", None, None),
    ]
    row_h = (0.98 - ROW_TOP) / len(rows)
    bw = int(xres * 0.16)
    for i, (kind, label, value, lbtn, rbtn) in enumerate(rows):
        y1 = int((ROW_TOP + i * row_h) * yres)
        y2 = int((ROW_TOP + (i + 1) * row_h) * yres) - 4
        cv2.rectangle(c, (6, y1), (xres - 6, y2), (45, 45, 45), -1)
        cv2.rectangle(c, (6, y1), (xres - 6, y2), (90, 90, 90), 1)

        if kind == "adjust":  # same-colored edge buttons, label/value between
            cv2.rectangle(c, (6, y1), (6 + bw, y2), (75, 75, 75), -1)
            _centered(c, lbtn, (6, y1, 6 + bw, y2), 1.0, (255, 255, 255), 2)
            cv2.rectangle(c, (xres - 6 - bw, y1), (xres - 6, y2), (75, 75, 75), -1)
            _centered(c, rbtn, (xres - 6 - bw, y1, xres - 6, y2), 1.0, (255, 255, 255), 2)
            cv2.putText(c, label, (6 + bw + 14, (y1 + y2) // 2 + 5),
                        FONT, 0.5, (210, 210, 210), 1)
            (vw, _), _ = cv2.getTextSize(value, FONT, 0.6, 2)
            cv2.putText(c, value, (xres - 6 - bw - vw - 14, (y1 + y2) // 2 + 7),
                        FONT, 0.6, (160, 220, 255), 2)
        elif kind == "toggle":  # label left, colored ON/OFF pill right
            on = value == "ON"
            cv2.putText(c, label, (16, (y1 + y2) // 2 + 5), FONT, 0.5, (210, 210, 210), 1)
            pill = (xres - 6 - bw, y1 + 6, xres - 12, y2 - 6)
            cv2.rectangle(c, pill[:2], pill[2:], (0, 140, 0) if on else (70, 70, 70), -1)
            _centered(c, value, pill, 0.6, (255, 255, 255), 2)
        else:  # action row (Gallery / Shutdown): full-width button, tap anywhere
            danger = label == "Shutdown"
            bg = (0, 0, 90) if danger else (55, 55, 80)
            fg = (170, 170, 255) if danger else (200, 220, 255)
            cv2.rectangle(c, (6, y1), (xres - 6, y2), bg, -1)
            _centered(c, f"{label}    {value}", (6, y1, xres - 6, y2), 0.55, fg, 1)

    if last_tap is not None:
        mx, my = int(last_tap[0] * xres), int(last_tap[1] * yres)
        cv2.circle(c, (mx, my), 8, (0, 165, 255), 2)
    return c


def draw_confirm(xres, yres, title, action_label):
    """Full-screen confirm at native fb size. Left half = Cancel, right half =
    `action_label`. If action_label is None, just show the title (e.g. status)."""
    c = np.zeros((yres, xres, 3), dtype=np.uint8)
    _centered(c, title, (0, int(yres * 0.16), xres, int(yres * 0.40)), 0.7, (255, 255, 255), 2)
    if action_label is None:
        return c
    y1b, y2b = int(yres * 0.50), int(yres * 0.76)
    cv2.rectangle(c, (int(xres * 0.08), y1b), (int(xres * 0.46), y2b), (70, 70, 70), -1)
    _centered(c, "Cancel", (int(xres * 0.08), y1b, int(xres * 0.46), y2b), 0.6, (255, 255, 255), 2)
    cv2.rectangle(c, (int(xres * 0.54), y1b), (int(xres * 0.92), y2b), (0, 0, 170), -1)
    _centered(c, action_label, (int(xres * 0.54), y1b, int(xres * 0.92), y2b), 0.6, (255, 255, 255), 2)
    return c


def settings_hit(nx, ny):
    """Map a normalized tap to (row_index, side L/R), or (None, None)."""
    if ny < ROW_TOP or ny > 0.98:
        return None, None
    row = int((ny - ROW_TOP) / ((0.98 - ROW_TOP) / len(SETTINGS_LAYOUT)))
    row = min(row, len(SETTINGS_LAYOUT) - 1)
    return row, ("L" if nx < 0.5 else "R")


def apply_setting(key, side, settings, models, model_idx, reload_model):
    """Adjust the setting named `key`; returns possibly-updated model_idx."""
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
# Gallery (browse saved captures)
# ----------------------------------------------------------------------------
def load_gallery():
    """Return saved capture image paths, newest first.

    Sort by modification time, not filename — otherwise the prefix dominates
    ('corrosion_' always sorts ahead of 'capture_'), burying recent no-detection
    captures below older detected ones regardless of date.
    """
    try:
        files = [os.path.join(cd.DETECTIONS_FOLDER, f)
                 for f in os.listdir(cd.DETECTIONS_FOLDER)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    except OSError:
        return []
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


GALLERY_BAR = 0.86        # bottom toolbar occupies ny in [GALLERY_BAR, 1.0]


def render_gallery(write_fb, xres, yres, files, idx, confirm=False):
    """Draw the gallery at native fb size (no letterbox) so tap-zones map directly.
    Bottom toolbar: [< Prev] [DELETE] [Next >]. confirm shows a Cancel/Delete overlay."""
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    c = np.zeros((yres, xres, 3), dtype=np.uint8)
    if not files:
        _centered(c, "No captures yet", (0, 0, xres, yres), 0.7, (200, 200, 200), 2)
        _centered(c, "long-press = back", (0, int(yres * 0.7), xres, yres), 0.5, (0, 255, 255), 1)
        write_fb(c)
        return

    img = cv2.imread(files[idx])
    if img is not None:
        c = letterbox(img, xres, yres)
    else:
        _centered(c, "cannot read image", (0, 0, xres, yres), 0.6, (0, 0, 255), 2)

    # top bar: index + filename
    cv2.rectangle(c, (0, 0), (xres, 24), (0, 0, 0), -1)
    cv2.putText(c, f"{idx + 1}/{len(files)}  {os.path.basename(files[idx])}",
                (6, 17), FONT, 0.45, (0, 255, 255), 1)

    # bottom toolbar with three zones
    by = int(yres * GALLERY_BAR)
    t = xres // 3
    cv2.rectangle(c, (0, by), (xres, yres), (0, 0, 0), -1)
    _centered(c, "< Prev", (0, by, t, yres), 0.5, (0, 255, 255), 1)
    _centered(c, "DELETE", (t, by, 2 * t, yres), 0.55, (80, 120, 255), 2)
    _centered(c, "Next >", (2 * t, by, xres, yres), 0.5, (0, 255, 255), 1)
    cv2.line(c, (t, by), (t, yres), (80, 80, 80), 1)
    cv2.line(c, (2 * t, by), (2 * t, yres), (80, 80, 80), 1)

    if confirm:
        ov = c.copy()
        cv2.rectangle(ov, (0, 0), (xres, yres), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.6, c, 0.4, 0, c)
        _centered(c, "Delete this capture?", (0, int(yres * 0.26), xres, int(yres * 0.44)),
                  0.65, (255, 255, 255), 2)
        y1b, y2b = int(yres * 0.52), int(yres * 0.74)
        cv2.rectangle(c, (int(xres * 0.08), y1b), (int(xres * 0.46), y2b), (70, 70, 70), -1)
        _centered(c, "Cancel", (int(xres * 0.08), y1b, int(xres * 0.46), y2b), 0.6, (255, 255, 255), 2)
        cv2.rectangle(c, (int(xres * 0.54), y1b), (int(xres * 0.92), y2b), (0, 0, 170), -1)
        _centered(c, "Delete", (int(xres * 0.54), y1b, int(xres * 0.92), y2b), 0.6, (255, 255, 255), 2)

    write_fb(c)


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
# Live MJPEG stream — mirrors whatever we draw to the LCD, to a browser.
# Capture-on-demand stays unchanged: the live view is raw, and detection boxes
# appear on the stream exactly when they appear on the LCD (on capture). No
# extra inference, so the Pi load is unchanged.
# ----------------------------------------------------------------------------
class FrameBus:
    """Thread-safe holder for the most recent frame to stream (reference only)."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def update(self, frame_bgr):
        with self._lock:
            self._frame = frame_bgr

    def get(self):
        with self._lock:
            return self._frame


_STREAM_INDEX = (
    b"<!doctype html><title>RustWatch RPi5 Live</title>"
    b"<body style='margin:0;background:#111;text-align:center'>"
    b"<img src='/video_feed' style='max-width:100%;height:auto'></body>"
)


def start_stream_server(bus, port, fps, quality):
    """Start a background MJPEG server (daemon thread). Returns the server."""
    boundary = "FRAME"
    frame_interval = 1.0 / max(1, fps)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # silence per-request logging

        def do_GET(self):
            if self.path.rstrip("/") in ("", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_STREAM_INDEX)
                return
            if not self.path.startswith("/video_feed"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=%s" % boundary)
            self.end_headers()
            try:
                while True:
                    frame = bus.get()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    ok, buf = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if not ok:
                        continue
                    data = buf.tobytes()
                    self.wfile.write(("--%s\r\n" % boundary).encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n" % len(data)).encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    time.sleep(frame_interval)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser disconnected — normal

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


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
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                   help="Rotate the camera image in software (for a physically angled camera)")
    p.add_argument("--threads", type=int, default=0,
                   help="Cap ONNX Runtime threads to flatten the inference power spike "
                        "(0=default/all cores; try 2 on a weak power supply)")
    p.add_argument("--calibrate", action="store_true",
                   help="Run 2-tap touch calibration (with your --flip/--swap flags) and save it")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable the live MJPEG web stream")
    p.add_argument("--stream-port", type=int, default=8000,
                   help="Port for the live MJPEG web stream (default 8000)")
    p.add_argument("--stream-fps", type=int, default=15,
                   help="Max frames/sec sent to the web stream (default 15)")
    p.add_argument("--stream-quality", type=int, default=70,
                   help="JPEG quality 1-100 for the web stream (default 70)")
    args = p.parse_args()

    if not os.path.exists(args.fb):
        raise SystemExit(f"{args.fb} not found — is the LCD overlay loaded? Run lcd-on first.")

    base_write_fb, xres, yres = make_fb_writer(args.fb)

    # Optional live MJPEG mirror of whatever we draw to the LCD
    stream_bus = None
    if not args.no_stream:
        stream_bus = FrameBus()
        try:
            start_stream_server(stream_bus, args.stream_port,
                                args.stream_fps, args.stream_quality)
            print(f"Live stream on port {args.stream_port} "
                  f"→ http://<this-pi-ip>:{args.stream_port}/video_feed")
        except OSError as e:
            print(f"Live stream disabled (port {args.stream_port}): {e}")
            stream_bus = None

    def write_fb(frame_bgr):
        base_write_fb(frame_bgr)
        if stream_bus is not None:
            stream_bus.update(frame_bgr)

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

    # Persisted settings (survive reboots); CLI args are the fallback defaults.
    settings = {"threshold": args.conf, "auto_save": not args.no_save, "show_fps": False}
    try:
        with open(SETTINGS_FILE) as _f:
            _saved = json.load(_f)
        for _k in ("threshold", "auto_save", "show_fps"):
            if _k in _saved:
                settings[_k] = _saved[_k]
        if _saved.get("model"):
            model_idx = next((i for i, p in enumerate(models)
                              if model_name(p) == _saved["model"]), model_idx)
        print("Loaded saved settings.")
    except (OSError, ValueError):
        pass

    def save_settings():
        try:
            with open(SETTINGS_FILE, "w") as _f:
                json.dump({"threshold": settings["threshold"],
                           "auto_save": settings["auto_save"],
                           "show_fps": settings["show_fps"],
                           "model": model_name(models[model_idx])}, _f)
        except OSError as e:
            print(f"Could not save settings: {e}")

    def make_session(path):
        so = ort.SessionOptions()
        if args.threads and args.threads > 0:
            so.intra_op_num_threads = args.threads
            so.inter_op_num_threads = args.threads
        return ort.InferenceSession(path, sess_options=so)

    print(f"Loading model: {models[model_idx]}")
    session = make_session(models[model_idx])
    input_name = session.get_inputs()[0].name
    print(f"Model loaded (threads={args.threads or 'default'}).")

    def reload_model(path):
        nonlocal session, input_name
        try:
            session = make_session(path)
            input_name = session.get_inputs()[0].name
            print(f"Model -> {model_name(path)}")
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()
    time.sleep(2)
    try:
        picam2.set_controls({"AfMode": 2})
    except Exception:
        pass

    full_crop = get_full_crop(picam2)
    zoom_idx = 0
    apply_zoom(picam2, ZOOM_LEVELS[zoom_idx], full_crop)

    trig = "Tap/Enter" if touch else "Enter"
    print("\n=== Capture-on-demand (autoscan OFF) ===")
    print(f"  {trig}            capture / back")
    print("  long-press / s    open/close Settings (Gallery is a row there)")
    print("  q + Enter         quit\n")

    state = "live"          # live | review | settings | gallery
    last_tap = None
    gallery_files, gallery_idx = [], 0
    gallery_confirm, gallery_dirty = False, False
    shutdown_confirm = False
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
                if shutdown_confirm:
                    # confirm sub-screen: tap right half / 'y' = power off, else cancel
                    decided = None
                    if taps:
                        nx, ny = taps[-1][1], taps[-1][2]
                        last_tap = (nx, ny)
                        decided = "yes" if nx >= 0.5 else "no"
                    elif kbd is not None:
                        decided = "yes" if kbd in ("y", "d") else "no"
                    if decided == "yes":
                        write_fb(draw_confirm(xres, yres, "Shutting down...", None))
                        os.system("sudo shutdown -h now")
                        break
                    if decided == "no":
                        shutdown_confirm = False
                        write_fb(draw_settings(xres, yres, settings, models, model_idx, last_tap))
                    else:
                        write_fb(draw_confirm(xres, yres, "Shut down the Pi?", "Shutdown"))
                    continue
                if long_press or kbd_settings or kbd_enter:
                    state = "live"
                else:
                    for _, nx, ny in taps:
                        last_tap = (nx, ny)
                        row, side = settings_hit(nx, ny)
                        if row is None:
                            continue
                        key = SETTINGS_LAYOUT[row]
                        if key == "gallery":
                            gallery_files = load_gallery()
                            gallery_idx, gallery_confirm, gallery_dirty = 0, False, True
                            state = "gallery"
                            break
                        if key == "shutdown":
                            shutdown_confirm = True
                            break
                        model_idx = apply_setting(key, side, settings, models, model_idx, reload_model)
                        save_settings()
                if state == "settings":
                    if shutdown_confirm:
                        write_fb(draw_confirm(xres, yres, "Shut down the Pi?", "Shutdown"))
                    else:
                        write_fb(draw_settings(xres, yres, settings, models, model_idx, last_tap))
                continue

            if state == "gallery":
                if long_press or kbd_settings:
                    state = "live"
                    gallery_confirm = False
                    continue

                if gallery_confirm:
                    decided = None
                    if taps:
                        nx, ny = taps[-1][1], taps[-1][2]
                        last_tap = (nx, ny)
                        decided = "delete" if nx >= 0.5 else "cancel"
                    elif kbd is not None:
                        decided = "delete" if kbd in ("y", "d") else "cancel"
                    if decided is not None:
                        if decided == "delete" and gallery_files:
                            try:
                                os.remove(gallery_files[gallery_idx])
                                print(f"Deleted {gallery_files[gallery_idx]}")
                            except OSError as e:
                                print(f"Delete failed: {e}")
                            gallery_files = load_gallery()
                            if gallery_idx >= len(gallery_files):
                                gallery_idx = max(0, len(gallery_files) - 1)
                        gallery_confirm = False
                        gallery_dirty = True
                else:
                    if taps:
                        nx, ny = taps[-1][1], taps[-1][2]
                        last_tap = (nx, ny)
                        if ny > GALLERY_BAR:                 # bottom toolbar
                            if nx < 1 / 3:
                                gallery_idx -= 1
                            elif nx < 2 / 3:
                                gallery_confirm = True
                            else:
                                gallery_idx += 1
                        else:                               # image area
                            gallery_idx += -1 if nx < 0.5 else 1
                        gallery_dirty = True
                    elif kbd == "d":
                        gallery_confirm = True
                        gallery_dirty = True
                    elif kbd is not None:
                        gallery_idx += 1
                        gallery_dirty = True
                    if gallery_files:
                        gallery_idx %= len(gallery_files)

                if gallery_dirty:
                    render_gallery(write_fb, xres, yres, gallery_files, gallery_idx, gallery_confirm)
                    gallery_dirty = False
                else:
                    time.sleep(0.08)
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
            do_cap = kbd_enter
            if taps:
                nx, ny = taps[-1][1], taps[-1][2]
                last_tap = (nx, ny)
                if ny > 0.85 and 0.40 < nx < 0.60:          # bottom-center zoom button
                    zoom_idx = (zoom_idx + 1) % len(ZOOM_LEVELS)
                    apply_zoom(picam2, ZOOM_LEVELS[zoom_idx], full_crop)
                else:
                    do_cap = True

            if do_cap:
                frame, boxes, scores, ids, analysis, infer_ms, blur_score = capture_and_detect(
                    picam2, session, input_name, settings["threshold"], args.rotate)
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
                    # Send the saved capture to the website (best-effort; never blocks)
                    cd.upload_to_website(frame, boxes, scores, ids,
                                         analysis.get("severity", "NONE"),
                                         blur_score, infer_ms / 1000.0)
                state = "review"
                continue

            # live preview frame
            rgb = picam2.capture_array()
            if args.rotate:
                rgb = rotate_frame(rgb, args.rotate)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n, 0, time.time()
            cv2.putText(frame, f"LIVE  {trig}=capture", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            if settings["show_fps"]:
                cv2.putText(frame, f"FPS: {fps}", (8, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            draw_zoom_button(frame, ZOOM_LEVELS[zoom_idx])
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
