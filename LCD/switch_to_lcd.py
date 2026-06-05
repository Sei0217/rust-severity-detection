#!/usr/bin/env python3
"""
switch_to_lcd.py

Switch Raspberry Pi 5 to the 3.5 inch MPI3501 / ILI9486 LCD mode.

Run:
  sudo python3 switch_to_lcd.py

What it does:
  - backs up /boot/firmware/config.txt
  - backs up /boot/firmware/cmdline.txt
  - enables SPI
  - enables vc4-kms-v3d
  - disables old/wrong LCD overlay lines
  - enables the piscreen DRM overlay
  - disables HDMI in cmdline.txt so the desktop should use the LCD
  - enables desktop autologin for eaglekim
  - reboots

If the LCD orientation is wrong, edit LCD_ROTATE below:
  90 or 270 = landscape choices
  0 or 180 = portrait choices
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
# Panel rotation. The piscreen panel is natively 320x480 (portrait).
#   90 or 270 = landscape   |   0 or 180 = portrait
# If the orientation is wrong, change this value and re-run.
LCD_ROTATE = 90
LCD_SPEED = 18000000

HDMI_DISABLE_TOKENS = ["video=HDMI-A-1:d", "video=HDMI-A-2:d"]

BEGIN = "# ===== DISPLAY SWITCH MANAGED BEGIN ====="
END = "# ===== DISPLAY SWITCH MANAGED END ====="

LIGHTDM_DIR = Path("/etc/lightdm/lightdm.conf.d")
LIGHTDM_AUTOCONF = LIGHTDM_DIR / "99-display-switch-autologin.conf"
GETTY_DIR = Path("/etc/systemd/system/getty@tty1.service.d")
GETTY_AUTOCONF = GETTY_DIR / "autologin.conf"

# The Wayland compositor on the Pi (vc4 GPU) cannot render onto the SPI
# panel, so the desktop must run under X11 with the fbdev driver pointed at
# the panel's framebuffer. This snippet does that.
XORG_CONF_DIR = Path("/usr/share/X11/xorg.conf.d")
XORG_LCD_CONF = XORG_CONF_DIR / "99-lcd-fbdev.conf"
FB_DEVICE = "/dev/fb1"

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

def configure_lcd_desktop():
    """Make the X11 desktop render to the SPI panel's framebuffer.

    Wayland on the Pi cannot drive the SPI panel, so the session is forced
    to X11 and an xorg snippet points the fbdev driver at /dev/fb1.
    Rotation is handled by the overlay's rotate= value (see LCD_ROTATE).
    """
    # Force the desktop session to X11 (W1). Wayland can't use the SPI panel.
    if shutil.which("raspi-config"):
        subprocess.run(["raspi-config", "nonint", "do_wayland", "W1"], check=False)

    # The fbdev X driver is required to target /dev/fb1.
    if shutil.which("apt-get"):
        subprocess.run(
            ["apt-get", "install", "-y", "xserver-xorg-video-fbdev"],
            check=False,
        )

    XORG_CONF_DIR.mkdir(parents=True, exist_ok=True)
    XORG_LCD_CONF.write_text(f"""Section "Device"
    Identifier "LCD"
    Driver "fbdev"
    Option "fbdev" "{FB_DEVICE}"
EndSection
""")
    print(f"Wrote {XORG_LCD_CONF} (X11 desktop on {FB_DEVICE})")

def main():
    require_root()

    if not CONFIG.exists():
        print(f"Missing {CONFIG}")
        sys.exit(1)

    backup(CONFIG)
    if CMDLINE.exists():
        backup(CMDLINE)

    s = CONFIG.read_text()

    # Remove previous managed switch block
    s = strip_managed_block(s)

    # Enable KMS/DRM
    s = re.sub(
        r"^[ \t]*#[ \t]*dtoverlay=vc4-kms-v3d[ \t]*$",
        "dtoverlay=vc4-kms-v3d",
        s,
        flags=re.M,
    )
    if not re.search(r"^[ \t]*dtoverlay=vc4-kms-v3d[ \t]*$", s, flags=re.M):
        s += "\ndtoverlay=vc4-kms-v3d\n"

    # Ensure SPI is on
    if re.search(r"^[ \t]*#[ \t]*dtparam=spi=on[ \t]*$", s, flags=re.M):
        s = re.sub(r"^[ \t]*#[ \t]*dtparam=spi=on[ \t]*$", "dtparam=spi=on", s, flags=re.M)
    elif not re.search(r"^[ \t]*dtparam=spi=on[ \t]*$", s, flags=re.M):
        s += "\ndtparam=spi=on\n"

    # Comment old/wrong LCD overlay lines
    for pat in [
        r"^[ \t]*dtoverlay=ili9486.*$",
        r"^[ \t]*dtoverlay=fbtft.*$",
        r"^[ \t]*dtoverlay=piscreen.*$",
    ]:
        s = re.sub(pat, lambda m: "# " + m.group(0), s, flags=re.M)

    # Add LCD block
    s += f"""

{BEGIN}
# MPI3501 / 3.5 inch / ILI9486 / 320x480 SPI LCD
# LCD-only mode. HDMI is disabled in cmdline.txt (video=HDMI-A-*:d).
dtoverlay=piscreen,drm,rotate={LCD_ROTATE},speed={LCD_SPEED},xohms=100
{END}
"""

    CONFIG.write_text(s)

    # Disable HDMI outputs at KMS/kernel level too
    if CMDLINE.exists():
        cmd = CMDLINE.read_text().replace("\n", " ").strip()
        parts = [p for p in cmd.split() if p not in HDMI_DISABLE_TOKENS]
        parts.extend(HDMI_DISABLE_TOKENS)
        CMDLINE.write_text(" ".join(parts) + "\n")

    configure_lcd_desktop()
    enable_autologin()

    print("Switched to LCD mode.")
    if "--no-reboot" in sys.argv:
        print("Skipping reboot (--no-reboot). Reboot manually to apply changes.")
        return
    print("Rebooting now...")
    subprocess.run(["sync"], check=False)
    subprocess.run(["reboot"], check=False)

if __name__ == "__main__":
    main()
