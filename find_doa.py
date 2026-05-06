"""Real-time DOA monitor – helps you find the device's physical 0° reference.

Run this script, then move a sound source (phone playing a tone) around the
device.  The DOA reading tells you which direction the device *thinks* the
sound is coming from.  Find the physical position where DOA ≈ 0° – that is
your reference direction.  30° clockwise from there is your target.

Press Ctrl+C to exit.
"""

import math
import struct
import sys
import time

import usb.core
import usb.util
import libusb_package


def main() -> int:
    dev = libusb_package.find(idVendor=0x2886, idProduct=0x001A)
    if dev is None:
        print("XVF3800 not found. Check USB connection.")
        return 1

    try:
        usb.util.claim_interface(dev, 3)
    except usb.core.USBError:
        pass

    print("DOA Monitor – move a sound source around the device")
    print("Press Ctrl+C to stop.\n")
    print(f"{'Time':>6s}  {'DOA':>6s}  {'Speech':>6s}  {'Direction hint':s}")
    print("-" * 55)

    t0 = time.monotonic()
    try:
        while True:
            try:
                resp = dev.ctrl_transfer(
                    usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                    0, 0x80 | 18, 20, 2 * 2 + 1, 3000,
                )
                if resp[0] == 0:
                    raw = resp.tobytes()
                    doa = int.from_bytes(raw[1:3], "little")
                    speech = int.from_bytes(raw[3:5], "little")

                    # Direction hint for common angles
                    if doa < 10 or doa > 350:
                        hint = "<<< 0° (FORWARD)"
                    elif 25 <= doa <= 35:
                        hint = "<<< 30° (TARGET)"
                    elif 80 <= doa <= 100:
                        hint = "90° (SIDE)"
                    elif 170 <= doa <= 190:
                        hint = "180° (BEHIND)"
                    elif 200 <= doa <= 220:
                        hint = "210° (OPPOSITE BEAM)"
                    elif 260 <= doa <= 280:
                        hint = "270° (SIDE)"
                    else:
                        hint = ""

                    elapsed = time.monotonic() - t0
                    bar = "#" * min(int(speech * 10), 10) if speech else ""
                    print(
                        f"{elapsed:5.1f}s  {doa:4d}°   "
                        f"{'YES' if speech else 'no':>6s}   {hint}"
                    )
                else:
                    # Retry silently
                    pass
            except usb.core.USBError:
                pass
            time.sleep(0.15)

    except KeyboardInterrupt:
        print("\nDone.")

    usb.util.release_interface(dev, 3)
    usb.util.dispose_resources(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
