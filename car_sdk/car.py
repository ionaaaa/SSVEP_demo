"""Minimal serial SDK for the supplied STM32 Bluetooth firmware."""
import threading
import time

import serial


class Car:
    """Open a caller-selected Bluetooth serial port. No command acknowledgments."""

    def __init__(self, port: str):
        if not isinstance(port, str) or not port.strip():
            raise ValueError("port must be a non-empty serial port name, e.g. COM5")
        self._lock = threading.RLock()
        self._last_write = None
        self._closed = False
        self._serial = serial.Serial(
            port=port, baudrate=9600, bytesize=8, parity="N", stopbits=1,
            timeout=1, write_timeout=1, xonxoff=False, rtscts=False, dsrdtr=False)

    def _send(self, command: bytes) -> None:
        with self._lock:
            if self._closed or not self._serial.is_open:
                raise serial.SerialException("Car connection is closed")
            # ponytail: conservative pacing for firmware's 100ms blocking loop;
            # reliable framing/ACKs require a firmware protocol upgrade.
            if self._last_write is not None:
                remaining = 0.15 - (time.monotonic() - self._last_write)
                if remaining > 0:
                    time.sleep(remaining)
            try:
                if self._serial.write(command) != len(command):
                    raise serial.SerialException("Incomplete command write; car state unknown")
            finally:
                self._last_write = time.monotonic()

    def forward(self) -> None:
        """Continue forward until another command is received."""
        self._send(b"ONA")

    def backward(self) -> None:
        """Continue backward until another command is received."""
        self._send(b"ONB")

    def left(self) -> None:
        """Turn left by driving the right side."""
        self._send(b"ONC")

    def right(self) -> None:
        """Turn right by driving the left side."""
        self._send(b"OND")

    def stop(self) -> None:
        """Send ONF. Successful write does not confirm a physical stop."""
        self._send(b"ONF")

    def close(self) -> None:
        """Attempt a stop then release the port, including on write failure."""
        with self._lock:
            if self._closed:
                return
            try:
                self.stop()
                # Allow the tiny queued stop command time to leave before closing.
                # ponytail: best effort only; an ACK-capable firmware is the upgrade.
                time.sleep(0.15)
            finally:
                self._closed = True
                self._serial.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"Car cleanup also failed; stop unconfirmed: {cleanup_error}")
        return False
