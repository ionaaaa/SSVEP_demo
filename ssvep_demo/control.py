"""Hardware-independent command definitions and safety dispatching.

This module deliberately has no PsychoPy, EEG, decoder, or transport
dependency.  A future Bluetooth controller only needs to implement
``Controller``; the confirmation, timeout, stale-result, and STOP safeguards
remain here unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Callable, Mapping, Protocol

from .protocol import DecodeResult


class ControlCommand(str, Enum):
    """Commands understood by the common controller boundary."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FORWARD = "FORWARD"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def parse(cls, value: object) -> "ControlCommand":
        """Parse a configured command or raise instead of silently guessing."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError("control command must be a string")
        try:
            return cls(value.strip().upper())
        except ValueError as exc:
            allowed = ", ".join(command.value for command in cls)
            raise ValueError(f"unknown control command {value!r}; expected one of: {allowed}") from exc


class Controller(Protocol):
    """Minimal boundary implemented by virtual and future hardware cars.

    ``execute`` receives only movement commands.  The dispatcher converts STOP
    to ``stop`` and never permits UNKNOWN to reach a controller.
    """

    def execute(self, command: ControlCommand) -> None:
        """Start or update the requested movement command."""

    def stop(self) -> None:
        """Immediately stop movement; repeated calls must be safe."""


@dataclass(frozen=True)
class DispatchDecision:
    """Auditable result of one input submission or periodic safety check."""

    received_command: ControlCommand
    effective_command: ControlCommand
    action: str
    reason: str
    confirmation_count: int
    timestamp_s: float


@dataclass(frozen=True)
class ControlConfig:
    """Validated, hardware-agnostic dispatcher settings."""

    confidence_threshold: float = 0.60
    confirmations_required: int = 2
    max_confirmation_gap_s: float = 1.0
    command_duration_s: float = 0.5
    input_timeout_s: float = 1.0
    max_prediction_age_s: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.confidence_threshold, (int, float)) or isinstance(self.confidence_threshold, bool):
            raise ValueError("control.confidence_threshold must be a number")
        if not math.isfinite(float(self.confidence_threshold)) or not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise ValueError("control.confidence_threshold must be finite and between 0 and 1")
        if isinstance(self.confirmations_required, bool) or not isinstance(self.confirmations_required, int):
            raise ValueError("control.confirmations_required must be a positive integer")
        if self.confirmations_required <= 0:
            raise ValueError("control.confirmations_required must be a positive integer")
        for field_name in (
            "max_confirmation_gap_s",
            "command_duration_s",
            "input_timeout_s",
            "max_prediction_age_s",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"control.{field_name} must be a positive finite number")


# Keep the descriptive spelling as a compatibility alias for callers that use
# the dispatcher independently of the YAML configuration.
SafetyConfig = ControlConfig


class SafeCommandDispatcher:
    """Apply decoder outputs only after explicit, fail-safe policy checks.

    The injected clock is in the :func:`time.monotonic` domain.  It makes the
    class deterministic in tests and avoids any blocking timer or ``sleep``.
    """

    def __init__(
        self,
        controller: Controller,
        config: ControlConfig,
        commands: Mapping[float, ControlCommand | str],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.controller = controller
        self.config = config
        self.clock = clock
        self._commands = self._normalise_commands(commands)
        self._confirmation_command: ControlCommand | None = None
        self._confirmation_count = 0
        self._last_confirmation_timestamp_s: float | None = None
        self._last_prediction_timestamp_s: float | None = None
        self._last_valid_input_received_s: float | None = None
        self._active_command: ControlCommand | None = None
        self._motion_deadline_s: float | None = None
        self._input_timeout_fired = False
        self._closed = False

    @staticmethod
    def _normalise_commands(commands: Mapping[float, ControlCommand | str]) -> dict[float, ControlCommand]:
        normalised: dict[float, ControlCommand] = {}
        for raw_frequency, raw_command in commands.items():
            if isinstance(raw_frequency, bool) or not isinstance(raw_frequency, (int, float)) or not math.isfinite(raw_frequency):
                raise ValueError("commands keys must be finite numeric frequencies")
            command = ControlCommand.parse(raw_command)
            if command is ControlCommand.UNKNOWN:
                raise ValueError("configuration command mappings must not contain UNKNOWN")
            frequency = float(raw_frequency)
            if frequency in normalised:
                raise ValueError("commands must not contain duplicate frequency mappings")
            normalised[frequency] = command
        if not normalised:
            raise ValueError("commands must not be empty")
        return normalised

    def submit(self, result: DecodeResult) -> DispatchDecision:
        """Process one decoder result and return a reason suitable for logging."""
        now_s = self._now()
        received = self._received_command(result)

        # STOP is deliberately parsed before confidence, age, and ordering
        # checks: accepting an extra stop can only improve safety.
        if received is ControlCommand.STOP:
            self._stop_and_clear()
            if not self._closed:
                self._last_valid_input_received_s = now_s
                self._input_timeout_fired = False
            return self._decision(received, ControlCommand.STOP, "stopped", "stop_priority", now_s)

        if self._closed:
            return self._decision(received, ControlCommand.UNKNOWN, "rejected", "dispatcher_closed", now_s)

        timestamp_s = self._result_timestamp(result)
        if timestamp_s is None:
            self._stop_and_clear()
            return self._decision(received, ControlCommand.UNKNOWN, "rejected", "invalid_timestamp", now_s)
        if now_s - timestamp_s > self.config.max_prediction_age_s:
            self._stop_and_clear()
            return self._decision(received, ControlCommand.UNKNOWN, "stopped", "stale_prediction", now_s)
        if self._last_prediction_timestamp_s is not None and timestamp_s < self._last_prediction_timestamp_s:
            self._stop_and_clear()
            return self._decision(received, ControlCommand.UNKNOWN, "stopped", "out_of_order_prediction", now_s)

        self._last_prediction_timestamp_s = timestamp_s
        expected = self._expected_command(result)
        if received is ControlCommand.UNKNOWN or expected is None or received is not expected:
            self._stop_and_clear()
            return self._decision(received, ControlCommand.UNKNOWN, "stopped", "unknown_or_mismatched_command", now_s)

        confidence = getattr(result, "confidence", None)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            self._stop_and_clear()
            return self._decision(received, ControlCommand.UNKNOWN, "stopped", "invalid_confidence", now_s)
        if float(confidence) < self.config.confidence_threshold:
            self._stop_and_clear()
            return self._decision(received, ControlCommand.UNKNOWN, "stopped", "low_confidence", now_s)

        # A valid, fresh motion prediction is the only input that resets the
        # input-timeout watchdog.
        self._last_valid_input_received_s = now_s
        self._input_timeout_fired = False
        return self._confirm_or_execute(expected, timestamp_s, now_s)

    def tick(self, now_s: float | None = None) -> DispatchDecision | None:
        """Run non-blocking duration and input-watchdog checks."""
        now_s = self._now() if now_s is None else self._validate_now(now_s)
        if self._closed:
            return None
        if (
            self._last_valid_input_received_s is not None
            and not self._input_timeout_fired
            and now_s - self._last_valid_input_received_s >= self.config.input_timeout_s
        ):
            self._stop_and_clear()
            self._input_timeout_fired = True
            return self._decision(
                ControlCommand.UNKNOWN, ControlCommand.STOP, "stopped", "input_timeout", now_s
            )
        if self._motion_deadline_s is not None and now_s >= self._motion_deadline_s:
            self._stop_and_clear(reset_timeout=False)
            return self._decision(
                ControlCommand.UNKNOWN, ControlCommand.STOP, "stopped", "command_duration_elapsed", now_s
            )
        return None

    def close(self) -> None:
        """Safely stop once; no later moving command may be dispatched."""
        if self._closed:
            return
        self._closed = True
        self._stop_and_clear()

    def _confirm_or_execute(
        self, command: ControlCommand, prediction_timestamp_s: float, now_s: float
    ) -> DispatchDecision:
        is_contiguous = (
            command is self._confirmation_command
            and self._last_confirmation_timestamp_s is not None
            and prediction_timestamp_s - self._last_confirmation_timestamp_s <= self.config.max_confirmation_gap_s
        )
        if is_contiguous:
            self._confirmation_count += 1
        else:
            self._confirmation_command = command
            self._confirmation_count = 1
        self._last_confirmation_timestamp_s = prediction_timestamp_s

        if self._confirmation_count < self.config.confirmations_required:
            return self._decision(command, command, "pending", "awaiting_confirmation", now_s)

        # Preserve the original controller exception while still making a best
        # effort to stop the controller before it escapes to the caller.
        try:
            self.controller.execute(command)
        except BaseException:
            self._active_command = None
            self._motion_deadline_s = None
            self._clear_confirmations()
            try:
                self.controller.stop()
            except BaseException:
                pass
            raise
        self._active_command = command
        self._motion_deadline_s = now_s + self.config.command_duration_s
        self._clear_confirmations()
        return self._decision(command, command, "executed", "confirmed", now_s)

    def _stop_and_clear(self, *, reset_timeout: bool = True) -> None:
        self.controller.stop()
        self._active_command = None
        self._motion_deadline_s = None
        self._clear_confirmations()
        if reset_timeout:
            self._input_timeout_fired = False

    def _clear_confirmations(self) -> None:
        self._confirmation_command = None
        self._confirmation_count = 0
        self._last_confirmation_timestamp_s = None

    def _received_command(self, result: DecodeResult) -> ControlCommand:
        try:
            return ControlCommand.parse(getattr(result, "command", None))
        except ValueError:
            return ControlCommand.UNKNOWN

    def _expected_command(self, result: DecodeResult) -> ControlCommand | None:
        frequency = getattr(result, "predicted_frequency_hz", None)
        if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or not math.isfinite(float(frequency)):
            return None
        return self._commands.get(float(frequency))

    @staticmethod
    def _result_timestamp(result: DecodeResult) -> float | None:
        timestamp_s = getattr(result, "timestamp_s", None)
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or not math.isfinite(float(timestamp_s)):
            return None
        return float(timestamp_s)

    def _now(self) -> float:
        return self._validate_now(self.clock())

    @staticmethod
    def _validate_now(now_s: float) -> float:
        if isinstance(now_s, bool) or not isinstance(now_s, (int, float)) or not math.isfinite(float(now_s)):
            raise ValueError("dispatcher clock must return a finite monotonic timestamp")
        return float(now_s)

    def _decision(
        self,
        received: ControlCommand,
        effective: ControlCommand,
        action: str,
        reason: str,
        timestamp_s: float,
    ) -> DispatchDecision:
        return DispatchDecision(
            received_command=received,
            effective_command=effective,
            action=action,
            reason=reason,
            confirmation_count=self._confirmation_count,
            timestamp_s=timestamp_s,
        )
