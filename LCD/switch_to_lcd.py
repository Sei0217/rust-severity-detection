#!/usr/bin/env python3
"""
switch_to_lcd.py

Switch the Raspberry Pi 5 to the 3.5" ILI9486 SPI LCD as a TEXT CONSOLE.

Run:
  sudo python3 switch_to_lcd.py                 # switch + reboot
  sudo python3 switch_to_lcd.py --no-reboot     # apply, don't reboot
  sudo python3 switch_to_lcd.py --rotate 270    # other landscape orientation

What it does:
  - backs up config.txt and cmdline.txt
  - enables SPI and loads the piscreen overlay
    (fb_ili9486 driver -> /dev/fb1, 480x320)
  - maps the kernel console to the LCD (fbcon=map:1) so you can log in and
    type on the panel
  - boots to the text console (no desktop) with autologin for USER
  - reboots

HDMI is intentionally left ENABLED. Disabling it (video=HDMI-A-*:d) renumbers
the framebuffers so the LCD can become fb0, which breaks `fbcon=map:1` and
leaves the panel blank. Keeping HDMI on keeps the LCD as a stable /dev/fb1
(the monitor just stays blank in console mode).

Confirmed working on a Raspberry Pi 5 (kernel 6.18, Bookworm) with a generic
PiScreen-compatible ILI9486 480x320 SPI panel.

Rotation (LCD_ROTATE / --rotate): 90 or 270 = landscape, 0 or 180 = portrait.
"""

from pathlib import Path
from datetime import datetime
import argparse
import os
import pwd
import re
import shutil
import subprocess
import sys

CONFIG = Path("/boot/firmware/config.txt")
CMDLINE = Path("/boot/firmware/cmdline.txt")

USER = "eaglekim"
LCD_ROTATE = 90           # 90/270 = landscape, 0/180 = portrait
LCD_SPEED = 16000000      # 16 MHz - stable for ILI9486 (80 MHz is out of spec)

HDMI_DISABLE_TOKENS = ["video=HDMI-A-1:d", "video=HDMI-A-2:d"]
FBCON_TOKEN = "fbcon=map:1"

BEGIN = "# ===== DISPLAY SWITCH MANAGED BEGIN ====="
END = "# ===== DISPLAY SWITCH MANAGED END ====="

GETTY_DIR = Path("/etc/systemd/system/getty@tty1.service.d")
GETTY_AUTOCONF = GETTY_DIR / "autologin.conf"

def require_root():
    if os.geteuid() != 0:
        print("Run with sudo:")
        print("  sudo python3 switch_to_lcd.py")
        sys.exit(1)

def backup(path: Path):
    if path.exists():
        b = path.with_name(path.name + ".backup-lcd-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        b.write_text(path.read_text(errors="ignore"))
        print(f"Backup saved: {b}")

def strip_managed_block(s: str) -> str:
    return re.sub(
        r"\n?# ===== DISPLAY SWITCH MANAGED BEGIN =====.*?# ===== DISPLAY SWITCH MANAGED END =====\n?",
        "\n",
        s,
        flags=re.S,
    )

def clean_old_config(s: str) -> str:
    """Remove our previous managed block and the junk left by the old
    home-grown script (duplicated decorative headers, the non-existent
    ili9486 overlay, legacy hotplug-ignore, stray piscreen lines)."""
    s = strip_managed_block(s)
    # Old decorative section headers (the old script appended these every run)
    s = re.sub(r"^[ \t]*# ===== ILI9486 LCD Configuration =====[ \t]*$\n?", "", s, flags=re.M)
    s = re.sub(r"^[ \t]*# ===== HDMI Monitor Configuration =====[ \t]*$\n?", "", s, flags=re.M)
    # Neutralize lines we manage ourselves / that don't work
    s = re.sub(r"^[ \t]*dtoverlay=ili9486.*$", lambda m: "# " + m.group(0), s, flags=re.M)
    s = re.sub(r"^[ \t]*dtoverlay=piscreen.*$", lambda m: "# " + m.group(0), s, flags=re.M)
    s = re.sub(r"^[ \t]*hdmi_ignore_hotplug=1[ \t]*$", "# hdmi_ignore_hotplug=1", s, flags=re.M)
    # Collapse runs of blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def ensure_spi(s: str) -> str:
    if re.search(r"^[ \t]*#[ \t]*dtparam=spi=on[ \t]*$", s, flags=re.M):
        return re.sub(r"^[ \t]*#[ \t]*dtparam=spi=on[ \t]*$", "dtparam=spi=on", s, flags=re.M)
    if not re.search(r"^[ \t]*dtparam=spi=on[ \t]*$", s, flags=re.M):
        return s + "\ndtparam=spi=on\n"
    return s

def enable_console_autologin():
    """Boot to the text console (no desktop) and auto-login USER on tty1."""
    try:
        pwd.getpwnam(USER)
    except KeyError:
        print(f"Warning: user {USER} not found, skipping autologin.")
        return

    print(f"Enabling console autologin for {USER}")

    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "set-default", "multi-user.target"], check=False)
    if shutil.which("raspi-config"):
        # B2 = console + autologin
        subprocess.run(["raspi-config", "nonint", "do_boot_behaviour", "B2"], check=False)

    GETTY_DIR.mkdir(parents=True, exist_ok=True)
    GETTY_AUTOCONF.write_text(f"""[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin {USER} --noclear %I $TERM
""")

    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "daemon-reload"], check=False)

def main():
    parser = argparse.ArgumentParser(description="Switch the Pi to the SPI LCD text console.")
    parser.add_argument("--rotate", type=int, default=LCD_ROTATE,
                        help="Panel rotation: 90/270 landscape, 0/180 portrait (default 90)")
    parser.add_argument("--keep-hdmi", action="store_true",
                        help="(deprecated, no effect) HDMI is always kept enabled now")
    parser.add_argument("--no-reboot", action="store_true",
                        help="Apply changes but do not reboot")
    args = parser.parse_args()

    require_root()

    if not CONFIG.exists():
        print(f"Missing {CONFIG}")
        sys.exit(1)

    backup(CONFIG)
    if CMDLINE.exists():
        backup(CMDLINE)

    s = CONFIG.read_text()
    s = clean_old_config(s)
    s = ensure_spi(s)

    s = s.rstrip() + f"""

{BEGIN}
# 3.5" ILI9486 SPI LCD as a text console (fb_ili9486 -> /dev/fb1, 480x320)
dtoverlay=piscreen,speed={LCD_SPEED},rotate={args.rotate}
{END}
"""
    CONFIG.write_text(s)

    # cmdline.txt: map the console to the LCD framebuffer. We also strip any
    # HDMI-disable tokens from a previous run, because disabling HDMI renumbers
    # the framebuffers and breaks fbcon=map:1 (see header note).
    if CMDLINE.exists():
        cmd = CMDLINE.read_text().replace("\n", " ").strip()
        drop = set(HDMI_DISABLE_TOKENS) | {FBCON_TOKEN}
        parts = [p for p in cmd.split() if p not in drop]
        parts.append(FBCON_TOKEN)
        CMDLINE.write_text(" ".join(parts) + "\n")

    enable_console_autologin()

    print("Switched to LCD console mode.")
    if args.no_reboot:
        print("Skipping reboot (--no-reboot). Reboot manually to apply changes.")
        return
    print("Rebooting now...")
    subprocess.run(["sync"], check=False)
    subprocess.run(["reboot"], check=False)

if __name__ == "__main__":
    main()
