"""Non-blocking Bluetooth car controller with thread-confined WinRT I/O.

The public controller methods are synchronous so they can be called from a
PsychoPy frame loop.  All Bluetooth discovery, construction, and asynchronous
I/O happen on one dedicated worker thread and its single asyncio event loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections import deque
from dataclasses import dataclass
from enum import Enum
import importlib
import logging
import math
import sys
import threading
import time
from typing import Callable, Protocol

from .control import ControlCommand


logger = logging.getLogger(__name__)

WINDOWS_REQUIREMENT = (
    "Bluetooth car control requires Windows 10/11 and the WinRT dependencies."
)

# This is the one command mapping for the controller boundary.  The production
# adapter translates these bytes back to the SDK's command names because the
# SDK deliberately owns the actual RFCOMM write and its 150 ms rate limit.
COMMAND_PAYLOADS: dict[ControlCommand, bytes] = {
    ControlCommand.FORWARD: b"ONA",
    ControlCommand.BACKWARD: b"ONB",
    ControlCommand.LEFT: b"ONC",
    ControlCommand.RIGHT: b"OND",
    ControlCommand.STOP: b"ONF",
}
_PAYLOAD_TO_CONTROL_COMMAND = {payload: command for command, payload in COMMAND_PAYLOADS.items()}
_PAYLOAD_TO_SDK_COMMAND = {
    payload: command.value.lower() for command, payload in COMMAND_PAYLOADS.items()
}


@dataclass(frozen=True)
class ControllerEvent:
    """Immutable worker-to-main-thread controller telemetry."""

    sequence: int
    event_type: str
    monotonic_s: float
    command: str | None
    payload_ascii: str | None
    state: str
    success: bool | None
    error: str | None


_CRITICAL_EVENT_TYPES = {
    "controller_fault",
    "controller_command_failed",
    "controller_close_failed",
    "controller_closed",
    "controller_stop_requested",
}


class BluetoothControllerState(str, Enum):
    """Thread-safe lifecycle state visible without touching WinRT objects."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    READY = "READY"
    FAULT = "FAULT"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


class BluetoothControllerError(RuntimeError):
    """Base error for lifecycle and worker failures."""


class AsyncBluetoothBackend(Protocol):
    """Small injection boundary implemented by the SDK adapter and test fakes."""

    async def connect(self) -> tuple[str | None, str | None]:
        """Connect and return ``(device_name, device_id)``."""

    async def send(self, payload: bytes) -> None:
        """Send one mapped motion payload."""

    async def stop(self) -> None:
        """Send the hardware STOP command."""

    async def close(self) -> None:
        """Stop if possible and release all transport resources."""


BackendFactory = Callable[[], AsyncBluetoothBackend]


def _format_error(exc: BaseException) -> str:
    reason = str(exc).strip()
    return type(exc).__name__ if not reason else f"{type(exc).__name__}: {reason}"


class _SdkBluetoothBackend:
    """Thin adapter around the installed ``bluetooth_car`` SDK module.

    Instances are created only by :meth:`BluetoothCarController._worker_main`,
    so importing WinRT and constructing ``BluetoothCar`` happen on the worker.
    """

    def __init__(self, requested_name: str | None, requested_id: str | None) -> None:
        try:
            module = importlib.import_module("bluetooth_car")
        except (ImportError, OSError) as exc:
            raise BluetoothControllerError(WINDOWS_REQUIREMENT) from exc
        self._module = module
        self._car = module.BluetoothCar()
        self._requested_name = requested_name
        self._requested_id = requested_id

    async def connect(self) -> tuple[str | None, str | None]:
        devices = await self._module.discover_devices()
        device = self._select_device(devices)
        await self._car.connect(device)
        return getattr(device, "name", None), self._device_id(device)

    async def send(self, payload: bytes) -> None:
        try:
            command = _PAYLOAD_TO_SDK_COMMAND[payload]
        except KeyError as exc:
            raise ValueError(f"unsupported Bluetooth car payload: {payload!r}") from exc
        await self._car.send(command)

    async def stop(self) -> None:
        await self._car.stop()

    async def close(self) -> None:
        await self._car.disconnect()

    def _select_device(self, devices: object) -> object:
        candidates = list(devices)  # type: ignore[arg-type]
        if self._requested_id is not None:
            requested = self._requested_id.casefold()
            match = next(
                (
                    device
                    for device in candidates
                    if (self._device_id(device) or "").casefold() == requested
                ),
                None,
            )
        elif self._requested_name is not None:
            requested = self._requested_name.casefold()
            matches = [
                device
                for device in candidates
                if str(getattr(device, "name", "")).casefold() == requested
            ]
            if len(matches) > 1:
                identities = ", ".join(self._device_id(device) or "unknown-id" for device in matches)
                raise ConnectionError(
                    f"Multiple Bluetooth devices are named {self._requested_name!r}: {identities}. "
                    "Set controller.bluetooth.device_id explicitly."
                )
            match = matches[0] if matches else None
        else:
            raise ValueError("Bluetooth controller requires an explicit device_id or device_name")
        if match is None:
            requested = self._requested_id or self._requested_name or "the only available device"
            available = ", ".join(
                str(getattr(device, "label", getattr(device, "name", device)))
                for device in candidates
            )
            suffix = f" Available devices: {available}" if available else " No devices were found."
            raise ConnectionError(f"Bluetooth car {requested!r} was not found.{suffix}")
        return match

    @staticmethod
    def _device_id(device: object) -> str | None:
        value = getattr(device, "address", getattr(device, "id", None))
        return None if value is None else str(value)


class BluetoothCarController:
    """Synchronous controller facade backed by one asyncio worker thread.

    ``execute`` and ``stop`` only update a bounded mailbox and wake the worker;
    they never wait for Bluetooth I/O.  ``connect`` and ``close`` are lifecycle
    boundaries and may wait, but both are bounded by caller-provided timeouts.
    """

    def __init__(
        self,
        *,
        device_name: str | None = "BT04-A",
        device_id: str | None = None,
        backend_factory: BackendFactory | None = None,
        io_timeout_s: float = 3.0,
        event_capacity: int = 256,
    ) -> None:
        if backend_factory is None and sys.platform != "win32":
            raise BluetoothControllerError(WINDOWS_REQUIREMENT)
        if (
            not isinstance(io_timeout_s, (int, float))
            or isinstance(io_timeout_s, bool)
            or not math.isfinite(float(io_timeout_s))
            or io_timeout_s <= 0
        ):
            raise ValueError("io_timeout_s must be a positive number")
        if device_name is not None and not isinstance(device_name, str):
            raise ValueError("device_name must be a string or None")
        if device_id is not None and not isinstance(device_id, str):
            raise ValueError("device_id must be a string or None")
        if isinstance(event_capacity, bool) or not isinstance(event_capacity, int) or event_capacity < 8:
            raise ValueError("event_capacity must be an integer of at least 8")

        self._requested_device_name = device_name
        self._requested_device_id = device_id
        self._backend_factory = backend_factory or (
            lambda: _SdkBluetoothBackend(device_name, device_id)
        )
        self._io_timeout_s = float(io_timeout_s)

        self._lock = threading.Lock()
        self._state = BluetoothControllerState.DISCONNECTED
        self._last_error: str | None = None
        self._device_name: str | None = None
        self._device_id: str | None = None
        self._event_capacity = event_capacity
        self._events: deque[ControllerEvent] = deque()
        self._event_sequence = 0
        self._event_overflow_count = 0

        # Bounded mailbox: one replaceable motion slot and one reserved STOP
        # flag.  STOP therefore cannot be displaced by motion or queue fullness.
        self._pending_motion: bytes | None = None
        self._stop_pending = False
        self._stop_inflight = False
        self._close_requested = False
        self._connect_request: tuple[float, concurrent.futures.Future[None]] | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake_event: asyncio.Event | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._thread_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._close_future: concurrent.futures.Future[None] | None = None

    @property
    def state(self) -> BluetoothControllerState:
        with self._lock:
            return self._state

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def device_name(self) -> str | None:
        with self._lock:
            return self._device_name

    @property
    def device_id(self) -> str | None:
        with self._lock:
            return self._device_id

    @property
    def event_overflow_count(self) -> int:
        with self._lock:
            return self._event_overflow_count

    def poll_events(self) -> list[ControllerEvent]:
        """Return currently available events without waiting for Bluetooth I/O."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def connect(self, timeout_s: float = 10.0) -> None:
        """Connect on the worker and wait for its confirmed initial STOP."""
        timeout_s = self._positive_timeout(timeout_s, "timeout_s")
        with self._lock:
            if self._state is not BluetoothControllerState.DISCONNECTED:
                raise BluetoothControllerError(
                    f"connect requires DISCONNECTED state, got {self._state.value}"
                )
            self._set_state_locked(BluetoothControllerState.CONNECTING)
            self._last_error = None
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            self._connect_request = (timeout_s, future)

        self._start_worker()
        self._wake_worker()
        try:
            # The worker's own wait_for is authoritative.  This small scheduling
            # allowance prevents a completed timeout from racing Future.result.
            future.result(timeout=timeout_s + 0.5)
        except concurrent.futures.TimeoutError as exc:
            error = TimeoutError(f"Bluetooth connection did not finish within {timeout_s:.3g}s")
            self._record_fault(error, request_stop=False)
            self._request_close()
            raise error from exc

    def execute(self, command: ControlCommand) -> None:
        """Submit a movement without waiting; newer pending motion wins."""
        if not isinstance(command, ControlCommand) or command is ControlCommand.UNKNOWN:
            self.stop()
            raise ValueError("BluetoothCarController.execute requires a known ControlCommand")
        if command is ControlCommand.STOP:
            self.stop()
            return

        payload = COMMAND_PAYLOADS[command]
        with self._lock:
            if self._state is not BluetoothControllerState.READY:
                raise BluetoothControllerError(
                    f"movement rejected while controller is {self._state.value}"
                )
            self._emit_event_locked("controller_command_requested", payload=payload)
            if self._pending_motion is not None:
                self._emit_event_locked(
                    "controller_command_coalesced",
                    payload=self._pending_motion,
                    success=False,
                    error="superseded_by_latest_motion",
                )
            self._pending_motion = payload
            self._emit_event_locked("controller_command_queued", payload=payload, success=True)
        self._wake_worker()

    def stop(self) -> None:
        """Request priority STOP without waiting for the current I/O call."""
        should_wake = False
        with self._lock:
            self._emit_event_locked("controller_stop_requested", payload=COMMAND_PAYLOADS[ControlCommand.STOP])
            if self._pending_motion is not None:
                self._emit_event_locked(
                    "controller_command_coalesced",
                    payload=self._pending_motion,
                    success=False,
                    error="superseded_by_stop",
                )
            self._pending_motion = None
            if self._state in {
                BluetoothControllerState.READY,
                BluetoothControllerState.FAULT,
                BluetoothControllerState.CLOSING,
            } and not self._stop_pending and not self._stop_inflight:
                self._stop_pending = True
                should_wake = True
                self._emit_event_locked(
                    "controller_command_queued",
                    payload=COMMAND_PAYLOADS[ControlCommand.STOP],
                    success=True,
                )
        if should_wake:
            self._wake_worker()

    def close(self, timeout_s: float = 3.0) -> None:
        """Best-effort STOP, backend release, loop shutdown, and thread join."""
        timeout_s = self._positive_timeout(timeout_s, "timeout_s")
        started = time.monotonic()
        with self._lock:
            thread = self._thread
            if self._state is BluetoothControllerState.CLOSED:
                return
            self._emit_event_locked("controller_close_started")
            if thread is None or not thread.is_alive():
                self._pending_motion = None
                if self._state is not BluetoothControllerState.FAULT:
                    self._set_state_locked(BluetoothControllerState.CLOSED)
                    self._emit_event_locked("controller_closed", success=True)
                return
            if self._close_future is None:
                self._close_future = concurrent.futures.Future()
            close_future = self._close_future
            self._set_state_locked(BluetoothControllerState.CLOSING)
            self._pending_motion = None
            self._close_requested = True
            if not self._stop_pending and not self._stop_inflight:
                self._stop_pending = True

        self._wake_worker()
        timeout_error: TimeoutError | None = None
        try:
            close_future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            timeout_error = TimeoutError(
                f"Bluetooth worker did not close within {timeout_s:.3g}s"
            )
            self._record_fault(timeout_error, request_stop=False)
            self._cancel_worker()

        remaining = max(0.0, timeout_s - (time.monotonic() - started))
        if timeout_error is not None:
            # Give cancellation a small, still-bounded opportunity to unwind
            # the coroutine and close the loop before reporting the timeout.
            remaining = min(0.5, max(0.05, timeout_s))
        thread.join(remaining)
        if thread.is_alive():
            error = TimeoutError(f"Bluetooth worker thread did not join within {timeout_s:.3g}s")
            self._emit_event("controller_close_failed", success=False, error=_format_error(error))
            self._record_fault(error, request_stop=False)
            raise error
        if timeout_error is not None:
            self._emit_event("controller_close_failed", success=False, error=_format_error(timeout_error))
            raise timeout_error

    def _start_worker(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            thread = threading.Thread(
                target=self._thread_main,
                name="BluetoothCarIO",
                daemon=True,
            )
            self._thread = thread
        thread.start()
        if not self._thread_ready.wait(1.0):
            error = RuntimeError("Bluetooth worker event loop failed to start")
            self._record_fault(error, request_stop=False)
            raise BluetoothControllerError(str(error))

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        wake_event = asyncio.Event()
        with self._lock:
            self._loop = loop
            self._wake_event = wake_event
        self._thread_ready.set()
        try:
            task = loop.create_task(self._worker_main())
            with self._lock:
                self._worker_task = task
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            self._record_fault(exc, request_stop=False)
            logger.error("Bluetooth I/O worker terminated: %s", _format_error(exc))
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            with self._lock:
                self._wake_event = None
                self._loop = None
                self._worker_task = None
            loop.close()
            self._finish_close_future()

    async def _worker_main(self) -> None:
        backend: AsyncBluetoothBackend | None = None
        cleanup_error: BaseException | None = None
        try:
            while True:
                wake_event = self._wake_event
                if wake_event is None:
                    raise RuntimeError("Bluetooth worker wake event is unavailable")
                await wake_event.wait()
                action = self._next_action(wake_event)
                if action is None:
                    continue
                kind, value = action
                if kind == "connect":
                    timeout_s, future = value
                    try:
                        backend = self._backend_factory()
                        name, device_id = await asyncio.wait_for(
                            self._connect_and_stop(backend), timeout=timeout_s
                        )
                    except Exception as exc:
                        self._record_fault(exc, request_stop=False)
                        if not future.done():
                            future.set_exception(exc)
                    else:
                        with self._lock:
                            self._device_name = name
                            self._device_id = device_id
                            if self._state is BluetoothControllerState.CONNECTING:
                                self._set_state_locked(BluetoothControllerState.READY)
                        if not future.done():
                            future.set_result(None)
                    continue
                if kind == "motion":
                    if backend is None:
                        self._record_fault(RuntimeError("Bluetooth backend is unavailable"))
                        continue
                    self._emit_event("controller_command_send_started", payload=value)
                    try:
                        await asyncio.wait_for(
                            backend.send(value), timeout=self._io_timeout_s
                        )
                    except Exception as exc:
                        self._emit_event(
                            "controller_command_failed", payload=value, success=False, error=_format_error(exc)
                        )
                        self._record_fault(exc, request_stop=True)
                    else:
                        self._emit_event("controller_command_sent", payload=value, success=True)
                    continue
                if kind == "stop":
                    stop_payload = COMMAND_PAYLOADS[ControlCommand.STOP]
                    self._emit_event("controller_command_send_started", payload=stop_payload)
                    try:
                        if backend is not None:
                            await asyncio.wait_for(
                                backend.stop(), timeout=self._io_timeout_s
                            )
                    except Exception as exc:
                        self._emit_event(
                            "controller_command_failed", payload=stop_payload, success=False, error=_format_error(exc)
                        )
                        self._record_fault(exc, request_stop=False)
                    else:
                        self._emit_event("controller_command_sent", payload=stop_payload, success=True)
                    finally:
                        with self._lock:
                            self._stop_inflight = False
                    continue
                if kind == "close":
                    break
        finally:
            if backend is not None:
                try:
                    await asyncio.wait_for(backend.close(), timeout=self._io_timeout_s)
                except Exception as exc:
                    cleanup_error = exc
                    self._emit_event("controller_close_failed", success=False, error=_format_error(exc))
                    self._record_fault(exc, request_stop=False)
            with self._lock:
                if cleanup_error is None and self._state is BluetoothControllerState.CLOSING:
                    self._set_state_locked(BluetoothControllerState.CLOSED)
                    self._emit_event_locked("controller_closed", success=True)

    async def _connect_and_stop(
        self, backend: AsyncBluetoothBackend
    ) -> tuple[str | None, str | None]:
        name, device_id = await backend.connect()
        # BluetoothCar.connect already sends ONF.  This explicit, awaited STOP
        # makes READY independent of that SDK detail and is harmless because
        # the SDK preserves its existing 150 ms minimum interval.
        stop_payload = COMMAND_PAYLOADS[ControlCommand.STOP]
        self._emit_event("controller_command_send_started", payload=stop_payload)
        try:
            await backend.stop()
        except Exception as exc:
            self._emit_event(
                "controller_command_failed", payload=stop_payload, success=False, error=_format_error(exc)
            )
            raise
        self._emit_event("controller_command_sent", payload=stop_payload, success=True)
        return name, device_id

    def _next_action(self, wake_event: asyncio.Event) -> tuple[str, object] | None:
        with self._lock:
            if self._connect_request is not None:
                request = self._connect_request
                self._connect_request = None
                return "connect", request
            if self._stop_pending:
                self._stop_pending = False
                self._stop_inflight = True
                return "stop", None
            if self._close_requested:
                return "close", None
            if self._pending_motion is not None and self._state is BluetoothControllerState.READY:
                payload = self._pending_motion
                self._pending_motion = None
                return "motion", payload
            wake_event.clear()
            return None

    def _wake_worker(self) -> None:
        with self._lock:
            loop = self._loop
            event = self._wake_event
        if loop is None or event is None:
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError as exc:
            self._record_fault(exc, request_stop=False)

    def _request_close(self) -> None:
        with self._lock:
            self._pending_motion = None
            self._close_requested = True
        self._wake_worker()

    def _cancel_worker(self) -> None:
        with self._lock:
            loop = self._loop
            task = self._worker_task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    def _record_fault(self, exc: BaseException, *, request_stop: bool = True) -> None:
        should_wake = False
        with self._lock:
            self._last_error = _format_error(exc)
            self._set_state_locked(BluetoothControllerState.FAULT)
            self._pending_motion = None
            self._emit_event_locked(
                "controller_fault", success=False, error=self._last_error
            )
            if request_stop and not self._stop_pending and not self._stop_inflight:
                self._stop_pending = True
                should_wake = True
        if should_wake:
            self._wake_worker()

    def _set_state_locked(self, state: BluetoothControllerState) -> None:
        if self._state is state:
            return
        self._state = state
        self._emit_event_locked("controller_state_changed")

    def _emit_event(
        self,
        event_type: str,
        *,
        payload: bytes | None = None,
        success: bool | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._emit_event_locked(
                event_type, payload=payload, success=success, error=error
            )

    def _emit_event_locked(
        self,
        event_type: str,
        *,
        payload: bytes | None = None,
        success: bool | None = None,
        error: str | None = None,
    ) -> None:
        self._event_sequence += 1
        command = _PAYLOAD_TO_CONTROL_COMMAND.get(payload) if payload is not None else None
        event = ControllerEvent(
            sequence=self._event_sequence,
            event_type=event_type,
            monotonic_s=time.monotonic(),
            command=command.value if command is not None else None,
            payload_ascii=payload.decode("ascii") if payload is not None else None,
            state=self._state.value,
            success=success,
            error=error,
        )
        if len(self._events) >= self._event_capacity:
            noncritical_index = next(
                (
                    index
                    for index, queued in enumerate(self._events)
                    if queued.event_type not in _CRITICAL_EVENT_TYPES
                ),
                None,
            )
            if event_type not in _CRITICAL_EVENT_TYPES and noncritical_index is None:
                self._event_overflow_count += 1
                return
            if noncritical_index is None:
                self._events.popleft()
            else:
                del self._events[noncritical_index]
            self._event_overflow_count += 1
        self._events.append(event)

    def _finish_close_future(self) -> None:
        with self._lock:
            future = self._close_future
        if future is not None and not future.done():
            future.set_result(None)

    @staticmethod
    def _positive_timeout(value: float, name: str) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive number")
        return float(value)


__all__ = [
    "AsyncBluetoothBackend",
    "BluetoothCarController",
    "BluetoothControllerError",
    "BluetoothControllerState",
    "ControllerEvent",
    "COMMAND_PAYLOADS",
]
