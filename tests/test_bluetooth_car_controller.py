from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import SimpleNamespace
from typing import Callable

import pytest

from ssvep_demo.bluetooth_car_controller import (
    BluetoothCarController,
    BluetoothControllerError,
    BluetoothControllerState,
    COMMAND_PAYLOADS,
    _SdkBluetoothBackend,
)
from ssvep_demo.control import ControlCommand


class FakeAsyncBackend:
    """Deterministic async backend; no WinRT, Bluetooth, or sleeps."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes | None]] = []
        self.thread_ids: list[int] = []
        self.created_thread_id: int | None = None
        self._condition = threading.Condition()
        self.connect_error: Exception | None = None
        self.send_error: Exception | None = None
        self.close_error: Exception | None = None
        self.block_connect = False
        self.block_close = False
        self.block_payload: bytes | None = None
        self.connect_started = threading.Event()
        self.close_started = threading.Event()
        self.send_started = threading.Event()
        self._connect_loop: asyncio.AbstractEventLoop | None = None
        self._connect_release: asyncio.Event | None = None
        self._send_loop: asyncio.AbstractEventLoop | None = None
        self._send_release: asyncio.Event | None = None

    async def connect(self) -> tuple[str, str]:
        self._record("connect")
        self.connect_started.set()
        if self.block_connect:
            self._connect_loop = asyncio.get_running_loop()
            self._connect_release = asyncio.Event()
            await self._connect_release.wait()
        if self.connect_error is not None:
            raise self.connect_error
        return "Fake car", "fake:01"

    async def send(self, payload: bytes) -> None:
        self._record("send", payload)
        if payload == self.block_payload and self._send_release is None:
            self._send_loop = asyncio.get_running_loop()
            self._send_release = asyncio.Event()
            self.send_started.set()
            await self._send_release.wait()
        if self.send_error is not None:
            error, self.send_error = self.send_error, None
            raise error

    async def stop(self) -> None:
        self._record("stop", b"ONF")

    async def close(self) -> None:
        self._record("close")
        if self.block_close:
            self.close_started.set()
            await asyncio.Event().wait()
        if self.close_error is not None:
            raise self.close_error

    def release_send(self) -> None:
        assert self._send_loop is not None and self._send_release is not None
        self._send_loop.call_soon_threadsafe(self._send_release.set)

    def release_connect(self) -> None:
        assert self._connect_loop is not None and self._connect_release is not None
        if not self._connect_loop.is_closed():
            self._connect_loop.call_soon_threadsafe(self._connect_release.set)

    def wait_for(self, predicate: Callable[[list[tuple[str, bytes | None]]], bool], timeout: float = 1.0) -> None:
        with self._condition:
            assert self._condition.wait_for(lambda: predicate(self.calls), timeout), self.calls

    def _record(self, name: str, payload: bytes | None = None) -> None:
        with self._condition:
            self.calls.append((name, payload))
            self.thread_ids.append(threading.get_ident())
            self._condition.notify_all()


def make_controller(
    fake: FakeAsyncBackend, *, io_timeout_s: float = 1.0
) -> BluetoothCarController:
    def factory() -> FakeAsyncBackend:
        fake.created_thread_id = threading.get_ident()
        return fake

    return BluetoothCarController(backend_factory=factory, io_timeout_s=io_timeout_s)


def connected_controller(
    fake: FakeAsyncBackend | None = None,
) -> tuple[BluetoothCarController, FakeAsyncBackend]:
    fake = fake or FakeAsyncBackend()
    controller = make_controller(fake)
    controller.connect(timeout_s=1.0)
    assert controller.state is BluetoothControllerState.READY
    return controller, fake


def close_safely(controller: BluetoothCarController, fake: FakeAsyncBackend) -> None:
    if fake.send_started.is_set() and fake._send_release is not None:
        fake.release_send()
    if fake.connect_started.is_set() and fake._connect_release is not None:
        fake.release_connect()
    try:
        controller.close(timeout_s=1.0)
    except TimeoutError:
        pass


def movement_payloads(fake: FakeAsyncBackend) -> list[bytes]:
    return [payload for name, payload in fake.calls if name == "send" and payload is not None]


def test_module_import_does_not_load_bluetooth_sdk() -> None:
    sys.modules.pop("bluetooth_car", None)
    assert "bluetooth_car" not in sys.modules


def test_command_mapping_is_exact() -> None:
    assert COMMAND_PAYLOADS == {
        ControlCommand.FORWARD: b"ONA",
        ControlCommand.BACKWARD: b"ONB",
        ControlCommand.LEFT: b"ONC",
        ControlCommand.RIGHT: b"OND",
        ControlCommand.STOP: b"ONF",
    }


def test_duplicate_device_names_require_an_explicit_device_id() -> None:
    backend = object.__new__(_SdkBluetoothBackend)
    backend._requested_name = "BT04-A"
    backend._requested_id = None
    devices = [
        SimpleNamespace(name="BT04-A", address="device-1"),
        SimpleNamespace(name="BT04-A", address="device-2"),
    ]

    with pytest.raises(ConnectionError, match="Multiple Bluetooth devices.*device_id"):
        backend._select_device(devices)


def test_initial_hardware_command_after_connect_is_stop() -> None:
    controller, fake = connected_controller()
    try:
        assert fake.calls[:2] == [("connect", None), ("stop", b"ONF")]
        assert controller.device_name == "Fake car"
        assert controller.device_id == "fake:01"
    finally:
        close_safely(controller, fake)


def test_execute_returns_while_bluetooth_write_is_blocked() -> None:
    fake = FakeAsyncBackend()
    fake.block_payload = b"ONA"
    controller, fake = connected_controller(fake)
    try:
        controller.execute(ControlCommand.FORWARD)
        assert fake.send_started.wait(1.0)
        started = time.monotonic()
        controller.execute(ControlCommand.LEFT)
        assert time.monotonic() - started < 0.1
    finally:
        close_safely(controller, fake)


def test_backend_factory_and_all_async_calls_use_one_worker_thread() -> None:
    controller, fake = connected_controller()
    caller_thread = threading.get_ident()
    try:
        controller.execute(ControlCommand.RIGHT)
        fake.wait_for(lambda calls: ("send", b"OND") in calls)
        controller.close(timeout_s=1.0)
        assert fake.created_thread_id is not None
        assert set(fake.thread_ids) == {fake.created_thread_id}
        assert fake.created_thread_id != caller_thread
    finally:
        close_safely(controller, fake)


def test_pending_motion_is_latest_wins() -> None:
    fake = FakeAsyncBackend()
    fake.block_payload = b"ONA"
    controller, fake = connected_controller(fake)
    try:
        controller.execute(ControlCommand.FORWARD)
        assert fake.send_started.wait(1.0)
        controller.execute(ControlCommand.BACKWARD)
        controller.execute(ControlCommand.LEFT)
        controller.execute(ControlCommand.RIGHT)
        fake.release_send()
        fake.wait_for(lambda calls: ("send", b"OND") in calls)
        assert movement_payloads(fake) == [b"ONA", b"OND"]
    finally:
        close_safely(controller, fake)


def test_full_motion_mailbox_never_blocks_execute() -> None:
    fake = FakeAsyncBackend()
    fake.block_payload = b"ONA"
    controller, fake = connected_controller(fake)
    try:
        controller.execute(ControlCommand.FORWARD)
        assert fake.send_started.wait(1.0)
        started = time.monotonic()
        for index in range(2_000):
            command = ControlCommand.LEFT if index % 2 else ControlCommand.RIGHT
            controller.execute(command)
        assert time.monotonic() - started < 0.5
    finally:
        close_safely(controller, fake)


def test_stop_clears_pending_motion_and_is_next_after_current_write() -> None:
    fake = FakeAsyncBackend()
    fake.block_payload = b"ONA"
    controller, fake = connected_controller(fake)
    try:
        controller.execute(ControlCommand.FORWARD)
        assert fake.send_started.wait(1.0)
        controller.execute(ControlCommand.LEFT)
        controller.stop()
        fake.release_send()
        fake.wait_for(lambda calls: calls.count(("stop", b"ONF")) >= 2)
        assert movement_payloads(fake) == [b"ONA"]
        assert fake.calls[-1] == ("stop", b"ONF")
    finally:
        close_safely(controller, fake)


def test_stop_cannot_be_displaced_by_a_full_motion_mailbox() -> None:
    fake = FakeAsyncBackend()
    fake.block_payload = b"ONA"
    controller, fake = connected_controller(fake)
    try:
        controller.execute(ControlCommand.FORWARD)
        assert fake.send_started.wait(1.0)
        for command in (ControlCommand.LEFT, ControlCommand.RIGHT, ControlCommand.BACKWARD):
            controller.execute(command)
        controller.stop()
        fake.release_send()
        fake.wait_for(lambda calls: calls.count(("stop", b"ONF")) >= 2)
        assert movement_payloads(fake) == [b"ONA"]
    finally:
        close_safely(controller, fake)


def test_repeated_stop_is_safe_and_does_not_restore_old_motion() -> None:
    fake = FakeAsyncBackend()
    fake.block_payload = b"ONA"
    controller, fake = connected_controller(fake)
    try:
        controller.execute(ControlCommand.FORWARD)
        assert fake.send_started.wait(1.0)
        controller.execute(ControlCommand.BACKWARD)
        controller.stop()
        controller.stop()
        controller.stop()
        fake.release_send()
        fake.wait_for(lambda calls: calls.count(("stop", b"ONF")) >= 2)
        assert movement_payloads(fake) == [b"ONA"]
    finally:
        close_safely(controller, fake)


def test_unknown_command_fails_safe_without_motion() -> None:
    controller, fake = connected_controller()
    try:
        with pytest.raises(ValueError, match="known ControlCommand"):
            controller.execute(ControlCommand.UNKNOWN)
        fake.wait_for(lambda calls: calls.count(("stop", b"ONF")) >= 2)
        assert movement_payloads(fake) == []
    finally:
        close_safely(controller, fake)


def test_non_ready_controller_rejects_motion() -> None:
    fake = FakeAsyncBackend()
    controller = make_controller(fake)
    with pytest.raises(BluetoothControllerError, match="DISCONNECTED"):
        controller.execute(ControlCommand.FORWARD)
    assert fake.calls == []
    controller.close(timeout_s=1.0)


def test_connection_failure_enters_fault() -> None:
    fake = FakeAsyncBackend()
    fake.connect_error = ConnectionError("radio unavailable")
    controller = make_controller(fake)
    try:
        with pytest.raises(ConnectionError, match="radio unavailable"):
            controller.connect(timeout_s=1.0)
        assert controller.state is BluetoothControllerState.FAULT
        assert controller.last_error == "ConnectionError: radio unavailable"
    finally:
        close_safely(controller, fake)


def test_connection_timeout_is_finite_and_enters_fault() -> None:
    fake = FakeAsyncBackend()
    fake.block_connect = True
    controller = make_controller(fake)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            controller.connect(timeout_s=0.05)
        assert time.monotonic() - started < 0.75
        assert controller.state is BluetoothControllerState.FAULT
    finally:
        close_safely(controller, fake)


def test_write_failure_enters_fault_and_rejects_later_motion() -> None:
    fake = FakeAsyncBackend()
    controller, fake = connected_controller(fake)
    fake.send_error = OSError("write failed")
    try:
        controller.execute(ControlCommand.LEFT)
        fake.wait_for(lambda calls: calls.count(("stop", b"ONF")) >= 2)
        assert controller.state is BluetoothControllerState.FAULT
        assert controller.last_error == "OSError: write failed"
        with pytest.raises(BluetoothControllerError, match="FAULT"):
            controller.execute(ControlCommand.RIGHT)
        assert b"OND" not in movement_payloads(fake)
    finally:
        close_safely(controller, fake)


def test_close_stops_releases_backend_stops_loop_and_joins_thread() -> None:
    controller, fake = connected_controller()
    thread = controller._thread
    controller.close(timeout_s=1.0)
    assert fake.calls[-2:] == [("stop", b"ONF"), ("close", None)]
    assert controller.state is BluetoothControllerState.CLOSED
    assert thread is not None and not thread.is_alive()


def test_close_is_idempotent() -> None:
    controller, fake = connected_controller()
    controller.close(timeout_s=1.0)
    calls = list(fake.calls)
    controller.close(timeout_s=1.0)
    assert fake.calls == calls


def test_close_failure_is_reported_as_fault_but_thread_is_reclaimed() -> None:
    fake = FakeAsyncBackend()
    fake.close_error = OSError("close failed")
    controller, fake = connected_controller(fake)
    thread = controller._thread
    controller.close(timeout_s=1.0)
    assert controller.state is BluetoothControllerState.FAULT
    assert controller.last_error == "OSError: close failed"
    assert thread is not None and not thread.is_alive()


def test_close_timeout_is_explicit_and_does_not_leave_worker_alive() -> None:
    fake = FakeAsyncBackend()
    fake.block_close = True
    controller, fake = connected_controller(fake)
    thread = controller._thread
    with pytest.raises(TimeoutError, match="did not close"):
        controller.close(timeout_s=0.05)
    assert controller.state is BluetoothControllerState.FAULT
    assert controller.last_error is not None and "TimeoutError" in controller.last_error
    assert thread is not None and not thread.is_alive()


def test_worker_operation_exception_cannot_hang_process() -> None:
    fake = FakeAsyncBackend()
    controller, fake = connected_controller(fake)
    fake.send_error = RuntimeError("worker boom")
    controller.execute(ControlCommand.BACKWARD)
    fake.wait_for(lambda calls: calls.count(("stop", b"ONF")) >= 2)
    controller.close(timeout_s=1.0)
    assert controller._thread is not None and not controller._thread.is_alive()


def test_real_controller_creation_has_clear_non_windows_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = sys.modules["ssvep_demo.bluetooth_car_controller"]
    monkeypatch.setattr(module.sys, "platform", "linux")
    with pytest.raises(BluetoothControllerError, match="Windows 10/11 and the WinRT dependencies"):
        BluetoothCarController()
