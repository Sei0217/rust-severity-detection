#!/usr/bin/env python3
"""
switch_to_monitor.py

Switch Raspberry Pi 5 back to HDMI monitor mode.

Run:
  sudo python3 switch_to_monitor.py

What it does:
  - backs up /boot/firmware/config.txt
  - backs up /boot/firmware/cmdline.txt
  - disables LCD overlays
  - removes HDMI disable tokens from cmdline.txt
  - keeps vc4-kms-v3d enabled
  - enables desktop autologin for eaglekim
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

BEGIN = "# ===== DISPLAY SWITCH MANAGED BEGIN ====="
END = "# ===== DISPLAY SWITCH MANAGED END ====="

LIGHTDM_DIR = Path("/etc/lightdm/lightdm.conf.d")
LIGHTDM_AUTOCONF = LIGHTDM_DIR / "99-display-switch-autologin.conf"
GETTY_DIR = Path("/etc/systemd/system/getty@tty1.service.d")
GETTY_AUTOCONF = GETTY_DIR / "autologin.conf"

# X11/fbdev snippet written by switch_to_lcd.py for the SPI panel.
XORG_LCD_CONF = Path("/usr/share/X11/xorg.conf.d/99-lcd-fbdev.conf")

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

def enable_autologin():
    try:
        pwd.getpwnam(USER)
    except KeyError:
        print(f"Warning: user {USER} not found, skipping autologin.")
        return

    print(f"Enabling autologin for {USER}")

    if shutil.which("raspi-config"):
        subprocess.run(["raspi-config", "nonint", "do_boot_behaviour", "B4"], check=False)

    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "set-default", "graphical.target"], check=False)
        subprocess.run(["systemctl", "enable", "lightdm.service"], check=False)

    for group in ("autologin", "nopasswdlogin"):
        subprocess.run(["groupadd", "-f", group], check=False)
        subprocess.run(["usermod", "-a", "-G", group, USER], check=False)

    LIGHTDM_DIR.mkdir(parents=True, exist_ok=True)
    LIGHTDM_AUTOCONF.write_text(f"""[Seat:*]
autologin-user={USER}
autologin-user-timeout=0
""")

    GETTY_DIR.mkdir(parents=True, exist_ok=True)
    GETTY_AUTOCONF.write_text(f"""[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin {USER} --noclear %I $TERM
""")

    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "daemon-reload"], check=False)

def restore_monitor_desktop():
    """Undo the LCD desktop config: remove the fbdev X snippet and put the
    desktop session back on Wayland (labwc, the Pi 5 default) for HDMI."""
    if XORG_LCD_CONF.exists():
        XORG_LCD_CONF.unlink()
        print(f"Removed {XORG_LCD_CONF}")

    # Restore Wayland (W3 = labwc). Harmless if already on Wayland.
    if shutil.which("raspi-config"):
        subprocess.run(["raspi-config", "nonint", "do_wayland", "W3"], check=False)

def main():
    require_root()

    if not CONFIG.exists():
        print(f"Missing {CONFIG}")
        sys.exit(1)

    backup(CONFIG)
    if CMDLINE.exists():
        backup(CMDLINE)

    s = CONFIG.read_text()

    # Remove previous LCD managed switch block
    s = strip_managed_block(s)

    # Disable LCD overlays
    for pat in [
        r"^[ \t]*(dtoverlay=piscreen.*)$",
        r"^[ \t]*(dtoverlay=ili9486.*)$",
        r"^[ \t]*(dtoverlay=fbtft.*)$",
        r"^[ \t]*(dtoverlay=pitft.*)$",
    ]:
        s = re.sub(pat, r"# \1", s, flags=re.M)

    # Make sure HDMI is not ignored
    s = re.sub(r"^[ \t]*hdmi_ignore_hotplug=1[ \t]*$", "# hdmi_ignore_hotplug=1", s, flags=re.M)

    # Keep KMS enabled
    s = re.sub(
        r"^[ \t]*#[ \t]*dtoverlay=vc4-kms-v3d[ \t]*$",
        "dtoverlay=vc4-kms-v3d",
        s,
        flags=re.M,
    )
    if not re.search(r"^[ \t]*dtoverlay=vc4-kms-v3d[ \t]*$", s, flags=re.M):
        s += "\ndtoverlay=vc4-kms-v3d\n"

    CONFIG.write_text(s)

    # Remove HDMI disable tokens from cmdline.txt
    if CMDLINE.exists():
        cmd = CMDLINE.read_text().replace("\n", " ").strip()
        parts = [p for p in cmd.split() if p not in HDMI_DISABLE_TOKENS]
        CMDLINE.write_text(" ".join(parts) + "\n")

    restore_monitor_desktop()
    enable_autologin()

    print("Switched to monitor mode.")
    if "--no-reboot" in sys.argv:
        print("Skipping reboot (--no-reboot). Reboot manually to apply changes.")
        return
    print("Rebooting now...")
    subprocess.run(["sync"], check=False)
    subprocess.run(["reboot"], check=False)

if __name__ == "__main__":
    main()
