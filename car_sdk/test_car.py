"""Run: python -m unittest -v (no hardware needed)."""
import unittest
from unittest.mock import patch

import serial

try:
    from car import Car
except ImportError:
    Car = None


class Port:
    def __init__(self):
        self.is_open = True
        self.sent = []
        self.error = None
        self.partial = False

    def write(self, data):
        if self.error:
            raise self.error
        self.sent.append(data)
        return 1 if self.partial else len(data)

    def close(self):
        self.is_open = False


class CarTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(Car, "Car SDK has not been implemented")
        self.port = Port()
        self.factory = patch("car.serial.Serial", return_value=self.port).start()
        self.addCleanup(patch.stopall)

    def test_configuration_and_exact_commands(self):
        car = Car("COM5")
        self.factory.assert_called_once_with(
            port="COM5", baudrate=9600, bytesize=8, parity="N", stopbits=1,
            timeout=1, write_timeout=1, xonxoff=False, rtscts=False, dsrdtr=False)
        for action in (car.forward, car.backward, car.left, car.right, car.stop):
            action()
        self.assertEqual(self.port.sent, [b"ONA", b"ONB", b"ONC", b"OND", b"ONF"])
        car.close()

    def test_gap_before_next_command(self):
        with patch("car.time.monotonic", return_value=10), patch("car.time.sleep") as sleep:
            car = Car("COM5")
            car.forward()
            sleep.assert_not_called()
            car.stop()
            self.assertAlmostEqual(sleep.call_args.args[0], 0.15)

    def test_invalid_port_before_open(self):
        for value in (None, "", "  ", 5):
            with self.assertRaises(ValueError):
                Car(value)
        self.factory.assert_not_called()

    def test_open_failure_propagates(self):
        self.factory.side_effect = serial.SerialException("busy")
        with self.assertRaisesRegex(serial.SerialException, "busy"):
            Car("COM5")

    def test_context_stops_and_closes_on_business_exception(self):
        with self.assertRaisesRegex(RuntimeError, "business"):
            with Car("COM5") as car:
                car.forward()
                raise RuntimeError("business")
        self.assertEqual(self.port.sent, [b"ONA", b"ONF"])
        self.assertFalse(self.port.is_open)

    def test_close_failure_still_releases_port(self):
        car = Car("COM5")
        self.port.error = serial.SerialTimeoutException("disconnected")
        with self.assertRaises(serial.SerialTimeoutException):
            car.close()
        self.assertFalse(self.port.is_open)
        car.close()

    def test_cleanup_failure_does_not_mask_business_exception(self):
        with self.assertRaisesRegex(RuntimeError, "business"):
            with Car("COM5"):
                self.port.error = serial.SerialException("disconnected")
                raise RuntimeError("business")
        self.assertFalse(self.port.is_open)

    def test_partial_write_is_failure(self):
        car = Car("COM5")
        self.port.partial = True
        with self.assertRaises(serial.SerialException):
            car.forward()

    def test_closed_car_rejects_commands_and_close_is_idempotent(self):
        car = Car("COM5")
        car.close()
        car.close()
        self.assertEqual(self.port.sent, [b"ONF"])
        with self.assertRaises(serial.SerialException):
            car.forward()


if __name__ == "__main__":
    unittest.main()
