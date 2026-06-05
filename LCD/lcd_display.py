#!/usr/bin/env python3
"""
lcd_display.py — enable/disable the 3.5" SPI LCD as a SECONDARY display,
without touching the desktop.

This only toggles the piscreen overlay so /dev/fb1 appears (--on) or goes
away (--off). It does NOT change the boot target, console mapping, or HDMI,
so the desktop stays on the monitor (Pi Connect screen sharing intact).

Use this when you want to run detect_lcd.py (which paints onto /dev/fb1)
while keeping the normal desktop on the HDMI monitor — no console mode needed.

Run:
  sudo python3 lcd_display.py --on              # enable /dev/fb1 + reboot
  sudo python3 lcd_display.py --off             # disable + reboot
  sudo python3 lcd_display.py --on --no-reboot  # apply without rebooting

After --on reboots, confirm with: ls /dev/fb1
"""

from pathlib import Path
from datetime import datetime
import argparse
import os
import re
import subprocess
import sys

CONFIG = Path("/boot/firmware/config.txt")
LCD_ROTATE = 90           # 90/270 = landscape, 0/180 = portrait
LCD_SPEED = 16000000      # 16 MHz - stable for ILI9486
OVERLAY_LINE = f"dtoverlay=piscreen,speed={LCD_SPEED},rotate={LCD_ROTATE}"

def require_root():
    if os.geteuid() != 0:
        print("Run with sudo:")
        print("  sudo python3 lcd_display.py --on")
        sys.exit(1)

def backup(path: Path):
    if path.exists():
        b = path.with_name(path.name + ".backup-lcddisp-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        b.write_text(path.read_text(errors="ignore"))
        print(f"Backup saved: {b}")

def ensure_spi(s: str) -> str:
    if re.search(r"^[ \t]*#[ \t]*dtparam=spi=on[ \t]*$", s, flags=re.M):
        return re.sub(r"^[ \t]*#[ \t]*dtparam=spi=on[ \t]*$", "dtparam=spi=on", s, flags=re.M)
    if not re.search(r"^[ \t]*dtparam=spi=on[ \t]*$", s, flags=re.M):
        return s.rstrip() + "\ndtparam=spi=on\n"
    return s

def main():
    parser = argparse.ArgumentParser(
        description="Toggle the SPI LCD as a secondary display (/dev/fb1), keeping the desktop.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--on", action="store_true", help="Enable the LCD overlay (default)")
    group.add_argument("--off", action="store_true", help="Disable the LCD overlay")
    parser.add_argument("--no-reboot", action="store_true", help="Apply but do not reboot")
    args = parser.parse_args()

    require_root()

    if not CONFIG.exists():
        print(f"Missing {CONFIG}")
        sys.exit(1)

    backup(CONFIG)
    s = CONFIG.read_text()

    if args.off:
        # Comment out any active piscreen overlay line
        s = re.sub(r"^[ \t]*dtoverlay=piscreen.*$", lambda m: "# " + m.group(0), s, flags=re.M)
        print("LCD overlay disabled (/dev/fb1 will be gone after reboot).")
    else:
        s = ensure_spi(s)
        if re.search(r"^[ \t]*dtoverlay=piscreen", s, flags=re.M):
            print("LCD overlay already enabled.")
        elif re.search(r"^[ \t]*#[ \t]*dtoverlay=piscreen", s, flags=re.M):
            # Re-enable the first commented piscreen line (normalized)
            s = re.sub(r"^[ \t]*#[ \t]*dtoverlay=piscreen.*$", OVERLAY_LINE, s, count=1, flags=re.M)
            print("LCD overlay re-enabled.")
        else:
            s = s.rstrip() + f"\n{OVERLAY_LINE}\n"
            print("LCD overlay added.")

    CONFIG.write_text(s)

    if args.no_reboot:
        print("Skipping reboot (--no-reboot). Reboot manually to apply changes.")
        return
    print("Rebooting now...")
    subprocess.run(["sync"], check=False)
    subprocess.run(["reboot"], check=False)

if __name__ == "__main__":
    main()
