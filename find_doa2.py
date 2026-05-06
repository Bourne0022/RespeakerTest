"""True independent DOA monitor – reads the XVF3800's processed speaker direction.

Unlike DOA_VALUE (which reports fixed-beam activity), this reads
AUDIO_MGR_SELECTED_AZIMUTHS which gives the independently-estimated
speaker direction using speech energy across all beams.

Returns NAN when no clear speech source is detected.

Press Ctrl+C to exit.
"""

import math
import struct
import sys
import time

import usb.core
import usb.util
import libusb_package


def read_with_retry(dev, resid, cmdid, count, vtype, max_retry=10):
    """Read a parameter with retry on firmware busy."""
    read_cmdid = 0x80 | cmdid
    if vtype in ("float", "radians"):
        data_len = count * 4 + 1
    elif vtype == "uint16":
        data_len = count * 2 + 1
    else:
        data_len = count + 1

    for _ in range(max_retry):
        try:
            resp = dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, read_cmdid, resid, data_len, 3000,
            )
        except usb.core.USBError:
            time.sleep(0.01)
            continue

        if resp[0] == 0:  # CONTROL_SUCCESS
            raw = resp.tobytes()
            if vtype in ("float", "radians"):
                return struct.unpack("<" + "f" * count, raw[1 : 1 + 4 * count])
            elif vtype == "uint16":
                return tuple(
                    int.from_bytes(raw[i : i + 2], "little")
                    for i in range(1, len(raw), 2)
                )
        elif resp[0] == 64:  # SERVICER_COMMAND_RETRY
            time.sleep(0.01)
            continue
        else:
            time.sleep(0.01)
            continue
    return None


def main() -> int:
    dev = libusb_package.find(idVendor=0x2886, idProduct=0x001A)
    if dev is None:
        print("XVF3800 not found. Check USB connection.")
        return 1

    try:
        usb.util.claim_interface(dev, 3)
    except usb.core.USBError:
        pass

    print("Independent DOA Monitor")
    print("Reads the device's actual speaker-direction estimate.")
    print("'NAN' = no clear speech source detected.")
    print("Move a sound source around to see real-time DOA.\n")
    print(f"{'Time':>6s}  {'ProcDOA':>10s}  {'AutoSel':>10s}  {'Energy0':>10s}  {'Energy1':>10s}")
    print("-" * 65)

    t0 = time.monotonic()
    try:
        while True:
            # Read processed speaker DOA (independent estimate)
            doa_result = read_with_retry(dev, 35, 11, 2, "radians", max_retry=5)
            # Read speech energy on both fixed beams
            energy_result = read_with_retry(dev, 33, 80, 4, "float", max_retry=5)

            proc_doa = doa_result[0] if doa_result else float("nan")
            auto_sel = doa_result[1] if doa_result else float("nan")
            e0 = energy_result[0] if energy_result else 0.0
            e1 = energy_result[1] if energy_result else 0.0

            elapsed = time.monotonic() - t0

            if math.isnan(proc_doa):
                doa_str = "NAN"
            else:
                doa_str = f"{math.degrees(proc_doa):.1f}°"

            if math.isnan(auto_sel):
                sel_str = "NAN"
            else:
                sel_str = f"{math.degrees(auto_sel):.1f}°"

            print(
                f"{elapsed:5.1f}s  {doa_str:>10s}  {sel_str:>10s}  "
                f"{e0:10.6f}  {e1:10.6f}"
            )

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nDone.")

    usb.util.release_interface(dev, 3)
    usb.util.dispose_resources(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
