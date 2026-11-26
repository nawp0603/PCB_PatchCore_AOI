"""Send a software stop to a Dobot after the AOI worker releases the serial port."""
from __future__ import annotations

import argparse
import time


def _packet(command_id: int, control: int, params: bytes = b"") -> bytes:
    body = bytes((command_id, control)) + params
    return b"\xaa\xaa" + bytes((len(body),)) + body + bytes(((-sum(body)) & 0xFF,))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    args = parser.parse_args()

    # The UI terminates its worker first, then this process acquires the serial port.
    time.sleep(0.4)
    import serial

    with serial.Serial(args.port, 115200, timeout=0.1, write_timeout=1.0) as device:
        # Stop/clear queued motion, then release the suction cup without DobotLab/pydobot.
        for command in (_packet(242, 0x01), _packet(241, 0x01),
                        _packet(245, 0x01), _packet(62, 0x01, b"\x01\x00")):
            device.write(command)
        device.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
