import math

import pytest

from ssvep_demo.control import ControlCommand, SafetyConfig, SafeCommandDispatcher
from ssvep_demo.protocol import DecodeResult


COMMANDS = {
    8.0: ControlCommand.LEFT,
    10.0: ControlCommand.RIGHT,
    12.0: ControlCommand.FORWARD,
    15.0: ControlCommand.BACKWARD,
}


class FakeClock:
    def __init__(self, now_s: float = 0.0) -> None:
        self.now_s = now_s

    def __call__(self) -> float:
        return self.now_s


class RecordingController:
    def __init__(self) -> None:
        self.executed: list[ControlCommand] = []
        self.stop_calls = 0
        self.is_moving = False

    def execute(self, command: ControlCommand) -> None:
        self.executed.append(command)
        self.is_moving = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_moving = False


def _result(command: ControlCommand, timestamp_s: float, confidence: float = 0.9) -> DecodeResult:
    frequency = next(frequency for frequency, mapped in COMMANDS.items() if mapped is command)
    return DecodeResult(frequency, command, {frequency: 1.0}, confidence, timestamp_s)


def _dispatcher(clock: FakeClock, controller: RecordingController, **settings) -> SafeCommandDispatcher:
    defaults = dict(
        confidence_threshold=0.6,
        confirmations_required=2,
        max_confirmation_gap_s=1.0,
        command_duration_s=0.5,
        input_timeout_s=1.0,
        max_prediction_age_s=0.5,
    )
    defaults.update(settings)
    return SafeCommandDispatcher(controller, SafetyConfig(**defaults), COMMANDS, clock=clock)


@pytest.mark.parametrize("invalid_confidence", [math.nan, math.inf])
def test_low_or_invalid_confidence_becomes_unknown_and_stops(invalid_confidence: float) -> None:
    clock, controller = FakeClock(), RecordingController()
    dispatcher = _dispatcher(clock, controller)

    low = dispatcher.submit(_result(ControlCommand.LEFT, 0.0, confidence=0.2))
    assert (low.effective_command, low.reason, controller.executed, controller.stop_calls) == (
        ControlCommand.UNKNOWN,
        "low_confidence",
        [],
        1,
    )

    invalid_result = _result(ControlCommand.LEFT, 0.0)
    object.__setattr__(invalid_result, "confidence", invalid_confidence)
    invalid = dispatcher.submit(invalid_result)
    assert invalid.reason == "invalid_confidence"
    assert invalid.effective_command is ControlCommand.UNKNOWN
    assert controller.stop_calls == 2


def test_motion_requires_contiguous_confirmations_and_resets_on_other_command_or_gap() -> None:
    clock, controller = FakeClock(), RecordingController()
    dispatcher = _dispatcher(clock, controller)

    first = dispatcher.submit(_result(ControlCommand.LEFT, 0.0))
    assert (first.action, first.confirmation_count) == ("pending", 1)
    clock.now_s = 0.1
    second = dispatcher.submit(_result(ControlCommand.LEFT, 0.1))
    assert (second.action, controller.executed) == ("executed", [ControlCommand.LEFT])

    clock.now_s = 0.2
    assert dispatcher.submit(_result(ControlCommand.LEFT, 0.2)).confirmation_count == 1
    clock.now_s = 0.3
    assert dispatcher.submit(_result(ControlCommand.RIGHT, 0.3)).confirmation_count == 1
    clock.now_s = 0.4
    assert dispatcher.submit(_result(ControlCommand.LEFT, 0.4)).confirmation_count == 1
    assert controller.executed == [ControlCommand.LEFT]

    clock.now_s = 2.0
    assert dispatcher.submit(_result(ControlCommand.LEFT, 2.0)).confirmation_count == 1
    clock.now_s = 3.2
    after_gap = dispatcher.submit(_result(ControlCommand.LEFT, 3.2))
    assert (after_gap.action, after_gap.confirmation_count) == ("pending", 1)


def test_stop_is_immediate_and_clears_pending_and_active_motion() -> None:
    clock, controller = FakeClock(), RecordingController()
    dispatcher = _dispatcher(clock, controller, confirmations_required=2)
    assert dispatcher.submit(_result(ControlCommand.FORWARD, 0.0)).action == "pending"
    clock.now_s = 0.01
    assert dispatcher.submit(_result(ControlCommand.FORWARD, 0.01)).action == "executed"
    clock.now_s = 0.02
    assert dispatcher.submit(_result(ControlCommand.LEFT, 0.02)).action == "pending"
    assert controller.is_moving
    clock.now_s = 0.1
    stopped = dispatcher.submit(DecodeResult(15.0, ControlCommand.STOP, {15.0: 1.0}, 0.0, 0.1))
    assert (stopped.action, stopped.reason, stopped.confirmation_count) == ("stopped", "stop_priority", 0)
    assert not controller.is_moving
    assert controller.stop_calls >= 1


def test_duration_and_input_timeout_stop_without_sleep() -> None:
    clock, controller = FakeClock(), RecordingController()
    dispatcher = _dispatcher(clock, controller, confirmations_required=1, command_duration_s=0.5, input_timeout_s=2.0)
    dispatcher.submit(_result(ControlCommand.FORWARD, 0.0))
    clock.now_s = 0.49
    assert dispatcher.tick() is None
    clock.now_s = 0.5
    assert dispatcher.tick().reason == "command_duration_elapsed"
    assert not controller.is_moving

    clock, controller = FakeClock(), RecordingController()
    dispatcher = _dispatcher(clock, controller, confirmations_required=1, command_duration_s=3.0, input_timeout_s=1.0)
    dispatcher.submit(_result(ControlCommand.FORWARD, 0.0))
    clock.now_s = 1.0
    assert dispatcher.tick().reason == "input_timeout"
    assert dispatcher.tick() is None
    assert not controller.is_moving


def test_stale_and_out_of_order_results_stop_and_do_not_execute() -> None:
    clock, controller = FakeClock(10.0), RecordingController()
    dispatcher = _dispatcher(clock, controller, confirmations_required=1)
    stale = dispatcher.submit(_result(ControlCommand.LEFT, 0.0))
    assert (stale.reason, controller.executed) == ("stale_prediction", [])

    clock.now_s = 10.0
    dispatcher.submit(_result(ControlCommand.RIGHT, 10.0))
    clock.now_s = 10.1
    ordered = dispatcher.submit(_result(ControlCommand.LEFT, 9.9))
    assert ordered.reason == "out_of_order_prediction"
    assert controller.executed == [ControlCommand.RIGHT]


def test_close_and_controller_failures_are_safe_and_idempotent() -> None:
    clock, controller = FakeClock(), RecordingController()
    dispatcher = _dispatcher(clock, controller, confirmations_required=1)
    dispatcher.submit(_result(ControlCommand.LEFT, 0.0))
    dispatcher.close()
    dispatcher.close()
    rejected = dispatcher.submit(_result(ControlCommand.RIGHT, 0.0))
    assert rejected.reason == "dispatcher_closed"
    assert controller.executed == [ControlCommand.LEFT]
    assert controller.stop_calls == 1

    class FailingController(RecordingController):
        def execute(self, command: ControlCommand) -> None:
            self.is_moving = True
            raise RuntimeError("transport failed")

    failing = FailingController()
    dispatcher = _dispatcher(FakeClock(), failing, confirmations_required=1)
    with pytest.raises(RuntimeError, match="transport failed"):
        dispatcher.submit(_result(ControlCommand.FORWARD, 0.0))
    assert not failing.is_moving
    assert failing.stop_calls == 1


def test_command_parser_rejects_unknown_configuration_values() -> None:
    assert ControlCommand.parse(" left ") is ControlCommand.LEFT
    with pytest.raises(ValueError, match="unknown control command"):
        ControlCommand.parse("WARP")
