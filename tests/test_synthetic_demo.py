import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from ssvep_demo.config import load_config
from ssvep_demo.control import ControlCommand, DispatchDecision, SafeCommandDispatcher
from ssvep_demo.protocol import DecodeResult, EEGWindow
from ssvep_demo.stimulus import build_trial_schedule
from ssvep_demo.synthetic import SyntheticEEGSource
from ssvep_demo.synthetic_demo import (
    SSVEPSyntheticDemoRunner,
    SyntheticDemoSessionLogger,
    SyntheticDemoState,
    SyntheticDemoStateMachine,
    SyntheticTrialCoordinator,
)
from ssvep_demo.virtual_car import VirtualCarController


CONFIG_PATH = Path(__file__).parents[1] / "config" / "ssvep_demo.yaml"
CONFIG = load_config(CONFIG_PATH)


class FakeClock:
    def __init__(self, now_s: float = 0.0) -> None:
        self.now_s = now_s

    def __call__(self) -> float:
        return self.now_s


class CountingSource:
    def __init__(self) -> None:
        self.calls = 0
        self.targets: list[float] = []

    def generate(self, *, target_frequency_hz: float, duration_s: float, sample_rate_hz: float, snr_db: float) -> EEGWindow:
        self.calls += 1
        self.targets.append(target_frequency_hz)
        samples = int(duration_s * sample_rate_hz)
        return EEGWindow(np.zeros((8, samples)), sample_rate_hz, list(CONFIG.acquisition.channels), 0.0, duration_s)


class CountingDecoder:
    def __init__(self, clock: FakeClock, frequency: float = 8.0) -> None:
        self.calls = 0
        self.clock = clock
        self.frequency = frequency

    def decode(self, window: EEGWindow) -> DecodeResult:
        self.calls += 1
        assert not hasattr(window, "target_frequency_hz")
        return DecodeResult(
            self.frequency,
            CONFIG.commands[self.frequency],
            {frequency: float(frequency == self.frequency) for frequency in CONFIG.stimulus.frequencies_hz},
            0.95,
            self.clock(),
        )


class SubmitCountingDispatcher:
    def __init__(self, delegate: SafeCommandDispatcher) -> None:
        self.delegate = delegate
        self.submit_calls = 0

    def submit(self, result: DecodeResult) -> DispatchDecision:
        self.submit_calls += 1
        return self.delegate.submit(result)


def _schedule():
    return build_trial_schedule(
        CONFIG.stimulus.frequencies_hz, 1, "fixed", 42, CONFIG.commands
    )


def test_synthetic_state_flow_and_pause_restart_semantics() -> None:
    machine = SyntheticDemoStateMachine(_schedule())
    assert machine.state is SyntheticDemoState.IDLE
    machine.start()
    machine.cue_complete()
    machine.stimulation_complete()
    machine.decoding_complete()
    machine.executing_complete()
    assert machine.state is SyntheticDemoState.REST
    machine.rest_complete()
    assert machine.state is SyntheticDemoState.CUE

    resting = SyntheticDemoStateMachine(_schedule())
    resting.start()
    resting.cue_complete()
    resting.stimulation_complete()
    resting.decoding_complete()
    resting.executing_complete()
    resting.pause()
    assert resting.state is SyntheticDemoState.PAUSED
    resting.resume()
    assert resting.state is SyntheticDemoState.REST

    restarted = SyntheticDemoStateMachine(_schedule())
    restarted.start()
    restarted.cue_complete()
    restarted.pause()
    assert restarted.aborted_trial_ids == [0]
    restarted.resume()
    assert restarted.state is SyntheticDemoState.CUE


def test_each_synthetic_trial_generates_decodes_and_submits_once_without_target_leakage() -> None:
    clock = FakeClock()
    car = VirtualCarController()
    dispatcher = SafeCommandDispatcher(car, replace(CONFIG.control, confirmations_required=1), CONFIG.commands, clock=clock)
    source = CountingSource()
    decoder = CountingDecoder(clock)
    counting_dispatcher = SubmitCountingDispatcher(dispatcher)
    coordinator = SyntheticTrialCoordinator(
        CONFIG, decoder, counting_dispatcher, snr_db=0.0, seed=42, source_factory=lambda _: source, clock=clock
    )

    computed = coordinator.process(_schedule()[0])
    assert source.calls == decoder.calls == counting_dispatcher.submit_calls == 1
    assert source.targets == [8.0]
    assert computed.decision.action == "executed"
    assert car.active_command is ControlCommand.LEFT


def test_synthetic_confirmation_is_one_but_default_safety_confirmation_remains_two() -> None:
    runner = SSVEPSyntheticDemoRunner(CONFIG, CONFIG_PATH, max_trials=1)
    assert CONFIG.control.confirmations_required == 2
    assert runner.control_config.confirmations_required == 1
    assert SSVEPSyntheticDemoRunner(CONFIG, CONFIG_PATH, max_trials=1, confirmations_required=2).control_config.confirmations_required == 2


def test_frequency_mapping_and_backward_motion_are_distinct_from_stop() -> None:
    assert CONFIG.commands == {
        8.0: ControlCommand.LEFT,
        10.0: ControlCommand.RIGHT,
        12.0: ControlCommand.FORWARD,
        15.0: ControlCommand.BACKWARD,
    }
    assert ControlCommand.STOP not in CONFIG.commands.values()
    car = VirtualCarController(heading_degrees=0.0, speed_units_s=0.2)
    car.execute(ControlCommand.BACKWARD)
    car.update(1.0)
    assert car.x == pytest.approx(-0.2)
    assert car.is_moving
    car.stop()
    assert not car.is_moving


def test_car_continuous_updates_are_frame_rate_independent_and_rest_is_static() -> None:
    one_frame = VirtualCarController(heading_degrees=0.0, speed_units_s=0.3, turn_rate_degrees_s=60.0)
    many_frames = VirtualCarController(heading_degrees=0.0, speed_units_s=0.3, turn_rate_degrees_s=60.0)
    one_frame.execute(ControlCommand.FORWARD)
    many_frames.execute(ControlCommand.FORWARD)
    one_frame.update(1.0)
    for _ in range(20):
        many_frames.update(0.05)
    assert one_frame.x == pytest.approx(many_frames.x)

    left = VirtualCarController(turn_rate_degrees_s=60.0)
    right = VirtualCarController(turn_rate_degrees_s=60.0)
    left.execute(ControlCommand.LEFT)
    right.execute(ControlCommand.RIGHT)
    left.update(0.5)
    right.update(0.5)
    assert left.heading_degrees == pytest.approx(120.0)
    assert right.heading_degrees == pytest.approx(60.0)
    before = left.state()
    left.stop()
    left.update(10.0)
    assert left.state() == {**before, "is_moving": False, "active_command": None}


def test_duration_and_low_confidence_stop_do_not_continue_previous_motion() -> None:
    clock = FakeClock()
    car = VirtualCarController(heading_degrees=0.0, speed_units_s=0.2)
    dispatcher = SafeCommandDispatcher(car, replace(CONFIG.control, confirmations_required=1), CONFIG.commands, clock=clock)
    high = DecodeResult(12.0, CONFIG.commands[12.0], {12.0: 1.0}, 0.9, 0.0)
    assert dispatcher.submit(high).action == "executed"
    car.update(0.2)
    moved_x = car.x
    clock.now_s = 0.2
    low = DecodeResult(8.0, CONFIG.commands[8.0], {8.0: 1.0}, 0.1, 0.2)
    low_decision = dispatcher.submit(low)
    assert low_decision.effective_command is ControlCommand.UNKNOWN
    car.update(1.0)
    assert car.x == moved_x and not car.is_moving

    clock = FakeClock()
    car = VirtualCarController(heading_degrees=0.0, speed_units_s=0.2)
    dispatcher = SafeCommandDispatcher(car, replace(CONFIG.control, confirmations_required=1, command_duration_s=0.5), CONFIG.commands, clock=clock)
    dispatcher.submit(DecodeResult(12.0, CONFIG.commands[12.0], {12.0: 1.0}, 0.9, 0.0))
    car.update(0.5)
    clock.now_s = 0.5
    assert dispatcher.tick().reason == "command_duration_elapsed"
    assert not car.is_moving


def test_same_seed_reproduces_window_and_prediction() -> None:
    clock = FakeClock()
    car_a, car_b = VirtualCarController(), VirtualCarController()
    dispatch_a = SafeCommandDispatcher(car_a, replace(CONFIG.control, confirmations_required=1), CONFIG.commands, clock=clock)
    dispatch_b = SafeCommandDispatcher(car_b, replace(CONFIG.control, confirmations_required=1), CONFIG.commands, clock=clock)
    from ssvep_demo.decoders import CCADecoder

    first = SyntheticTrialCoordinator(CONFIG, CCADecoder(CONFIG), dispatch_a, snr_db=5.0, seed=9, clock=clock).process(_schedule()[1])
    second = SyntheticTrialCoordinator(CONFIG, CCADecoder(CONFIG), dispatch_b, snr_db=5.0, seed=9, clock=clock).process(_schedule()[1])
    assert np.array_equal(first.window.data, second.window.data)
    assert first.result.predicted_frequency_hz == second.result.predicted_frequency_hz


def test_unified_synthetic_session_log_contains_required_trial_fields(tmp_path: Path) -> None:
    logger = SyntheticDemoSessionLogger(tmp_path, CONFIG_PATH, {"mode": "synthetic", "real_eeg_validated": False})
    logger.trial(
        {
            "trial_id": 0,
            "mode": "synthetic",
            "trial_status": "completed",
            "target_frequency_hz": 8.0,
            "target_command": "LEFT",
            "decoder_scores": json.dumps({"8.0": 1.0}),
            "predicted_frequency_hz": 8.0,
            "predicted_command": "LEFT",
            "confidence": 0.9,
            "car_start_x": 0.0,
            "car_end_x": 0.1,
        }
    )
    logger.close()
    with (logger.directory / "trials.csv").open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    metadata = json.loads((logger.directory / "session.json").read_text(encoding="utf-8"))
    assert row["decoder_scores"] == '{"8.0": 1.0}'
    assert {"target_frequency_hz", "predicted_command", "car_end_heading", "dropped_frame_count"}.issubset(row)
    assert metadata["mode"] == "synthetic"


def test_synthetic_demo_module_import_has_no_psychopy_window_side_effect() -> None:
    import ssvep_demo.synthetic_demo as module

    assert "psychopy" not in module.__dict__


@pytest.mark.skipif(True, reason="requires a local display and manual PsychoPy smoke run")
def test_synthetic_demo_gui_smoke_is_manual_only() -> None:
    pytest.skip("Run scripts/run_ssvep_demo.py manually with a display")
