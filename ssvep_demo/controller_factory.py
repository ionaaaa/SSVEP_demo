"""Central controller selection and lifecycle helpers for demo runners."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Protocol

from .config import BluetoothControllerConfig, ControllerConfig
from .control import Controller, SafeCommandDispatcher
from .virtual_car import VirtualCarController


class ControllerStartupError(RuntimeError):
    """A selected controller could not become ready; no fallback is allowed."""

    def __init__(
        self,
        message: str,
        *,
        runtime: "ControllerRuntime | None" = None,
        connect_latency_ms: float | None = None,
    ) -> None:
        super().__init__(message)
        self.runtime = runtime
        self.connect_latency_ms = connect_latency_ms


class ControllerFaultError(RuntimeError):
    """The active hardware controller entered its fail-safe FAULT state."""


class EventLogger(Protocol):
    def event(self, event: str, **payload: Any) -> None: ...


BluetoothFactory = Callable[..., Controller]


def controller_config_with_overrides(
    config: ControllerConfig,
    *,
    controller_type: str | None = None,
    device_name: str | None = None,
    device_id: str | None = None,
    connect_timeout_s: float | None = None,
    io_timeout_s: float | None = None,
    close_timeout_s: float | None = None,
) -> ControllerConfig:
    """Apply only explicit CLI values over the validated YAML configuration."""
    selected_type = config.type if controller_type is None else controller_type
    if selected_type not in {"virtual", "bluetooth"}:
        raise ValueError("controller type must be 'virtual' or 'bluetooth'")
    current = config.bluetooth
    values = {
        "connect_timeout_s": current.connect_timeout_s if connect_timeout_s is None else connect_timeout_s,
        "io_timeout_s": current.io_timeout_s if io_timeout_s is None else io_timeout_s,
        "close_timeout_s": current.close_timeout_s if close_timeout_s is None else close_timeout_s,
    }
    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"controller.bluetooth.{name} must be a positive finite number")
    resolved_name = current.device_name if device_name is None else device_name
    resolved_id = current.device_id if device_id is None else device_id
    for name, value in (("device_name", resolved_name), ("device_id", resolved_id)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"controller.bluetooth.{name} must be a non-empty string or null")
    if selected_type == "bluetooth" and resolved_name is None and resolved_id is None:
        raise ValueError("Bluetooth controller requires device_id or device_name")
    return ControllerConfig(
        type=selected_type,
        bluetooth=BluetoothControllerConfig(
            device_name=resolved_name,
            device_id=resolved_id,
            connect_timeout_s=float(values["connect_timeout_s"]),
            io_timeout_s=float(values["io_timeout_s"]),
            close_timeout_s=float(values["close_timeout_s"]),
        ),
    )


def _state_value(controller: object) -> str:
    state = getattr(controller, "state", "UNKNOWN")
    return str(getattr(state, "value", state))


@dataclass
class ControllerRuntime:
    """The selected controller plus display/logging lifecycle information."""

    controller: Controller
    config: ControllerConfig
    virtual_car: VirtualCarController | None = None
    connected: bool = False
    connect_latency_ms: float | None = None
    initial_state: str | None = None

    @property
    def type(self) -> str:
        return self.config.type

    @property
    def is_bluetooth(self) -> bool:
        return self.type == "bluetooth"

    @property
    def state(self) -> str:
        return "READY" if self.virtual_car is not None else _state_value(self.controller)

    @property
    def device_name(self) -> str | None:
        value = getattr(self.controller, "device_name", None)
        return value if isinstance(value, str) else self.config.bluetooth.device_name

    @property
    def device_id(self) -> str | None:
        value = getattr(self.controller, "device_id", None)
        return value if isinstance(value, str) else self.config.bluetooth.device_id

    @property
    def last_error(self) -> str | None:
        value = getattr(self.controller, "last_error", None)
        return value if isinstance(value, str) else None

    def ensure_healthy(self) -> None:
        if self.is_bluetooth and self.state != "READY":
            reason = self.last_error or f"controller is {self.state}, not READY"
            raise ControllerFaultError(f"Bluetooth controller fault: {reason}")

    def snapshot(self) -> dict[str, Any]:
        if self.virtual_car is not None:
            return {"controller_type": "virtual", "controller_state": "READY", **self.virtual_car.state()}
        return {
            "controller_type": "bluetooth",
            "controller_state": self.state,
            "device_name": self.device_name,
            "device_id": self.device_id,
            "last_error": self.last_error,
            "position_feedback_available": False,
        }

    def position(self) -> dict[str, float | None]:
        """Return virtual coordinates only; Bluetooth has no position feedback."""
        if self.virtual_car is None:
            return {"x": None, "y": None, "heading_degrees": None}
        return {
            "x": self.virtual_car.x,
            "y": self.virtual_car.y,
            "heading_degrees": self.virtual_car.heading_degrees,
        }

    def update_virtual(self, delta_time_s: float) -> None:
        if self.virtual_car is not None:
            self.virtual_car.update(delta_time_s)

    def stop_virtual(self) -> None:
        if self.virtual_car is not None:
            self.virtual_car.stop()

    def motion_active(self, dispatcher: SafeCommandDispatcher) -> bool:
        if self.virtual_car is not None:
            return self.virtual_car.is_moving
        return dispatcher.motion_deadline_s is not None

    def trial_controller_fields(self, decision: Any | None) -> dict[str, Any]:
        if decision is None:
            requested = None
            delivery = "not_decoded"
        elif decision.action == "executed":
            requested = decision.effective_command.value
            delivery = "queued" if self.is_bluetooth else "executed_virtual"
        elif decision.action == "stopped":
            requested = "STOP"
            delivery = "queued" if self.is_bluetooth else "executed_virtual"
        else:
            requested = None
            delivery = "not_requested"
        return {
            "controller_type": self.type,
            "controller_state_at_decision": self.state,
            "controller_command_requested": requested,
            "controller_command_delivery_status": delivery,
            "controller_failure_reason": self.last_error,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "controller_type": self.type,
            "car_device_name": self.device_name if self.is_bluetooth else None,
            "car_device_id": self.device_id if self.is_bluetooth else None,
            "car_connect_timeout_s": (
                self.config.bluetooth.connect_timeout_s if self.is_bluetooth else None
            ),
            "controller_connected": self.connected or self.virtual_car is not None,
            "controller_connect_latency_ms": self.connect_latency_ms,
            "controller_initial_state": self.initial_state or self.state,
        }

    def close(self) -> None:
        if self.virtual_car is not None:
            self.virtual_car.stop()
            return
        close = getattr(self.controller, "close", None)
        if callable(close):
            close(timeout_s=self.config.bluetooth.close_timeout_s)
        if self.state == "FAULT":
            reason = self.last_error or "Bluetooth controller entered FAULT while closing"
            raise ControllerFaultError(reason)


def create_controller(
    config: ControllerConfig,
    *,
    mode: str,
    bluetooth_factory: BluetoothFactory | None = None,
    logger: EventLogger | None = None,
) -> ControllerRuntime:
    """Create the selected controller and synchronously ready Bluetooth.

    This function must be called before constructing a PsychoPy window.
    """
    if config.type == "virtual":
        car = VirtualCarController()
        return ControllerRuntime(
            car, config, virtual_car=car, connected=True, connect_latency_ms=0.0, initial_state="READY"
        )
    if mode == "replay":
        raise ControllerStartupError("Bluetooth controller is disabled in replay mode for safety.")

    if bluetooth_factory is None:
        # Deliberately lazy: virtual mode never imports the WinRT-facing module.
        from .bluetooth_car_controller import BluetoothCarController

        bluetooth_factory = BluetoothCarController
    bluetooth = config.bluetooth
    target = bluetooth.device_id or bluetooth.device_name
    print(f"Controller: Bluetooth; connecting to car device {target!r}...")
    connect_started_s = time.monotonic()
    _safe_event(
        logger,
        "controller_connect_started",
        controller_type="bluetooth",
        car_device_name=bluetooth.device_name,
        car_device_id=bluetooth.device_id,
        event_monotonic_s=connect_started_s,
    )
    controller: Controller | None = None
    runtime: ControllerRuntime | None = None
    try:
        controller = bluetooth_factory(
            device_name=bluetooth.device_name,
            device_id=bluetooth.device_id,
            io_timeout_s=bluetooth.io_timeout_s,
        )
        runtime = ControllerRuntime(controller, config, initial_state=_state_value(controller))
        connect = getattr(controller, "connect")
        connect(timeout_s=bluetooth.connect_timeout_s)
        state = _state_value(controller)
        if state != "READY":
            raise ControllerStartupError(
                f"Bluetooth controller for {target!r} did not become READY (state={state})"
            )
    except BaseException as exc:
        latency_ms = (time.monotonic() - connect_started_s) * 1000.0
        if runtime is not None:
            runtime.connect_latency_ms = latency_ms
            drain_controller_events(runtime, logger)
        try:
            close = getattr(controller, "close", None) if controller is not None else None
            if callable(close):
                close(timeout_s=bluetooth.close_timeout_s)
        except BaseException:
            pass
        if runtime is not None:
            drain_controller_events(runtime, logger)
        _safe_event(
            logger,
            "controller_connect_failed",
            controller_type="bluetooth",
            car_device_name=bluetooth.device_name,
            car_device_id=bluetooth.device_id,
            controller_state=runtime.state if runtime is not None else "UNAVAILABLE",
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            connect_latency_ms=latency_ms,
        )
        if isinstance(exc, ControllerStartupError):
            exc.runtime = runtime
            exc.connect_latency_ms = latency_ms
            raise
        raise ControllerStartupError(
            f"Bluetooth connection to {target!r} failed: {type(exc).__name__}: {exc}",
            runtime=runtime,
            connect_latency_ms=latency_ms,
        ) from exc
    assert controller is not None
    latency_ms = (time.monotonic() - connect_started_s) * 1000.0
    runtime = runtime or ControllerRuntime(controller, config)
    runtime.connected = True
    runtime.connect_latency_ms = latency_ms
    drain_controller_events(runtime, logger)
    print(f"Controller: Bluetooth; Car device: {runtime.device_name or target}; Car state: {runtime.state}")
    return runtime


def log_controller_connected(logger: EventLogger, runtime: ControllerRuntime) -> None:
    """Record selection/connection once, never once per frame."""
    update_metadata = getattr(logger, "update_metadata", None)
    if callable(update_metadata):
        update_metadata(**runtime.metadata())
    logger.event(
        "controller_connected",
        controller_type=runtime.type,
        car_device_name=runtime.device_name if runtime.is_bluetooth else None,
        car_device_id=runtime.device_id if runtime.is_bluetooth else None,
        controller_state=runtime.state,
        connect_latency_ms=runtime.connect_latency_ms,
    )


def drain_controller_events(
    runtime: ControllerRuntime | None,
    logger: EventLogger | None,
) -> int:
    """Non-blockingly transfer worker events to the main-thread session logger."""
    if runtime is None or logger is None:
        return 0
    poll = getattr(runtime.controller, "poll_events", None)
    if not callable(poll):
        return 0
    events = poll()
    consume = getattr(logger, "controller_event", None)
    for event in events:
        try:
            if callable(consume):
                consume(event, controller_type=runtime.type)
            else:
                logger.event(event.event_type, controller_event_sequence=event.sequence)
        except BaseException:
            # Logging must never interfere with emergency STOP/close.
            break
    return len(events)


def _safe_event(logger: EventLogger | None, name: str, **payload: Any) -> None:
    if logger is not None:
        try:
            logger.event(name, **payload)
        except BaseException:
            pass


def cleanup_controller(
    dispatcher: SafeCommandDispatcher | None,
    runtime: ControllerRuntime | None,
    logger: EventLogger | None = None,
) -> BaseException | None:
    """Close dispatcher first, then controller, preserving the first error."""
    cleanup_error: BaseException | None = None

    def log_event(name: str, **payload: Any) -> None:
        nonlocal cleanup_error
        if logger is None:
            return
        try:
            logger.event(name, **payload)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc

    if dispatcher is not None:
        if runtime is None or not runtime.is_bluetooth:
            log_event("controller_stop_requested", reason="dispatcher_close")
        try:
            dispatcher.close()
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
            log_event(
                "controller_stop_failed",
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            )
        drain_controller_events(runtime, logger)
    if runtime is not None:
        try:
            runtime.close()
            if not runtime.is_bluetooth:
                log_event("controller_closed", controller_type=runtime.type, controller_state=runtime.state)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
            if not (
                runtime.is_bluetooth
                and callable(getattr(runtime.controller, "poll_events", None))
            ):
                log_event(
                    "controller_close_failed",
                    controller_type=runtime.type,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                )
        drain_controller_events(runtime, logger)
    return cleanup_error


def finalize_session(
    *,
    logger: Any,
    archive: Any | None,
    runtime: ControllerRuntime | None,
    dispatcher: SafeCommandDispatcher | None,
    status: str,
    error: BaseException | None,
    failure_stage: str | None,
    failure_reason: str | None = None,
    live_source: Any | None = None,
    window: Any | None = None,
) -> BaseException | None:
    """Run the shared safety-first cleanup sequence and close the logger last."""
    cleanup_errors: list[BaseException] = []

    def step_failed(step: str, exc: BaseException) -> None:
        cleanup_errors.append(exc)
        _safe_event(
            logger,
            "cleanup_step_failed",
            cleanup_step=step,
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
        )

    _safe_event(logger, "cleanup_started", session_status=status)
    controller_error = cleanup_controller(dispatcher, runtime, logger)
    if controller_error is not None:
        step_failed("controller", controller_error)

    if live_source is not None:
        try:
            live_source.abort_trial("runner_shutdown")
        except BaseException as exc:
            step_failed("live_source_abort", exc)
        try:
            live_source.close()
        except BaseException as exc:
            step_failed("live_source_close", exc)

    drain_controller_events(runtime, logger)
    if window is not None:
        try:
            window.close()
        except BaseException as exc:
            step_failed("psychopy_window_close", exc)

    overflow_count = 0
    if runtime is not None:
        overflow_count = int(getattr(runtime.controller, "event_overflow_count", 0))
    update_metadata = getattr(logger, "update_metadata", None)
    if callable(update_metadata):
        try:
            update_metadata(
                controller_event_overflow_count=overflow_count,
                cleanup_error_count=len(cleanup_errors),
                **(runtime.metadata() if runtime is not None else {}),
            )
        except BaseException as exc:
            step_failed("logger_metadata", exc)

    final_status = status
    final_error = error
    final_stage = failure_stage
    if cleanup_errors and error is None:
        final_status = "failed"
        final_error = cleanup_errors[0]
        final_stage = "cleanup"
    _safe_event(
        logger,
        "cleanup_finished",
        cleanup_error_count=len(cleanup_errors),
        session_status=final_status,
    )
    terminal_event = {
        "completed": "session_completed",
        "aborted": "session_aborted",
        "failed": "session_failed",
    }.get(final_status, "session_failed")
    _safe_event(
        logger,
        terminal_event,
        session_failure_stage=final_stage if final_status == "failed" else None,
        session_failure_reason=(
            failure_reason
            if final_status == "failed" and failure_reason is not None
            else str(final_error)[:500] if final_error is not None else None
        ),
    )
    try:
        logger.close(
            status=final_status,
            archived_eeg_windows=archive.count if archive is not None else 0,
            final_car_state=runtime.snapshot() if runtime is not None else None,
            error=final_error,
            failure_stage=final_stage,
            failure_reason=failure_reason,
            cleanup_error_count=len(cleanup_errors),
            controller_event_overflow_count=overflow_count,
        )
    except BaseException as exc:
        cleanup_errors.append(exc)
    return cleanup_errors[0] if cleanup_errors else None


__all__ = [
    "ControllerFaultError",
    "ControllerRuntime",
    "ControllerStartupError",
    "cleanup_controller",
    "controller_config_with_overrides",
    "create_controller",
    "drain_controller_events",
    "finalize_session",
    "log_controller_connected",
]
