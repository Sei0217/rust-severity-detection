#!/usr/bin/env python3
"""
switch_to_monitor.py

Switch the Raspberry Pi 5 back to the HDMI monitor with the normal desktop.

Run:
  sudo python3 switch_to_monitor.py               # switch + reboot
  sudo python3 switch_to_monitor.py --no-reboot   # apply, don't reboot

What it does:
  - backs up config.txt and cmdline.txt
  - disables the LCD (piscreen) overlay
  - removes the LCD console mapping and HDMI-disable tokens from cmdline.txt
  - keeps vc4-kms-v3d enabled
  - boots back to the graphical desktop with autologin for USER
  - reboots
"""

from pathlib import Path
from datetime import datetime
import os
import pwd
import re
import shutil
import subprocess
import sys

CONFIG = Path("/boot/firmware/config.txt")
CMDLINE = Path("/boot/firmware/cmdline.txt")

USER = "eaglekim"
HDMI_DISABLE_TOKENS = ["video=HDMI-A-1:d", "video=HDMI-A-2:d"]
FBCON_TOKEN = "fbcon=map:1"

BEGIN = "# ===== DISPLAY SWITCH MANAGED BEGIN ====="
END = "# ===== DISPLAY SWITCH MANAGED END ====="

LIGHTDM_DIR = Path("/etc/lightdm/lightdm.conf.d")
LIGHTDM_AUTOCONF = LIGHTDM_DIR / "99-display-switch-autologin.conf"
GETTY_DIR = Path("/etc/systemd/system/getty@tty1.service.d")
GETTY_AUTOCONF = GETTY_DIR / "autologin.conf"

def require_root():
    if os.geteuid() != 0:
        print("Run with sudo:")
        print("  sudo python3 switch_to_monitor.py")
        sys.exit(1)

def backup(path: Path):
    if path.exists():
        b = path.with_name(path.name + ".backup-monitor-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        b.write_text(path.read_text(errors="ignore"))
        print(f"Backup saved: {b}")

def strip_managed_block(s: str) -> str:
    return re.sub(
        r"\n?# ===== DISPLAY SWITCH MANAGED BEGIN =====.*?# ===== DISPLAY SWITCH MANAGED END =====\n?",
        "\n",
        s,
        flags=re.S,
    )

def enable_desktop_autologin():
    """Boot to the graphical desktop and auto-login USER."""
    try:
        pwd.getpwnam(USER)
    except KeyError:
        print(f"Warning: user {USER} not found, skipping autologin.")
        return

    print(f"Enabling desktop autologin for {USER}")

    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "set-default", "graphical.target"], check=False)
        subprocess.run(["systemctl", "enable", "lightdm.service"], check=False)
    if shutil.which("raspi-config"):
        # B4 = desktop + autologin
        subprocess.run(["raspi-config", "nonint", "do_boot_behaviour", "B4"], check=False)

    for group in ("autologin", "nopasswdlogin"):
        subprocess.run(["groupadd", "-f", group], check=False)
        subprocess.run(["usermod", "-a", "-G", group, USER], check=False)

    LIGHTDM_DIR.mkdir(parents=True, exist_ok=True)
    LIGHTDM_AUTOCONF.write_text(f"""[Seat:*]
autologin-user={USER}
autologin-user-timeout=0
""")

    # Remove the console-only autologin left by switch_to_lcd.py
    if GETTY_AUTOCONF.exists():
        GETTY_AUTOCONF.unlink()

    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "daemon-reload"], check=False)

def main():
    no_reboot = "--no-reboot" in sys.argv

    require_root()

    if not CONFIG.exists():
        print(f"Missing {CONFIG}")
        sys.exit(1)

    backup(CONFIG)
    if CMDLINE.exists():
        backup(CMDLINE)

    s = CONFIG.read_text()
    s = strip_managed_block(s)

    # Disable the LCD overlay
    s = re.sub(r"^[ \t]*dtoverlay=piscreen.*$", lambda m: "# " + m.group(0), s, flags=re.M)

    # Keep KMS enabled (re-enable if it was commented)
    s = re.sub(r"^[ \t]*#[ \t]*dtoverlay=vc4-kms-v3d[ \t]*$", "dtoverlay=vc4-kms-v3d", s, flags=re.M)
    if not re.search(r"^[ \t]*dtoverlay=vc4-kms-v3d[ \t]*$", s, flags=re.M):
        s += "\ndtoverlay=vc4-kms-v3d\n"

    CONFIG.write_text(s)

    # Remove the LCD console mapping and HDMI-disable tokens from cmdline.txt
    if CMDLINE.exists():
        cmd = CMDLINE.read_text().replace("\n", " ").strip()
        drop = set(HDMI_DISABLE_TOKENS) | {FBCON_TOKEN}
        parts = [p for p in cmd.split() if p not in drop]
        CMDLINE.write_text(" ".join(parts) + "\n")

    enable_desktop_autologin()

    print("Switched to monitor mode.")
    if no_reboot:
        print("Skipping reboot (--no-reboot). Reboot manually to apply changes.")
        return
    print("Rebooting now...")
    subprocess.run(["sync"], check=False)
    subprocess.run(["reboot"], check=False)

if __name__ == "__main__":
    main()
