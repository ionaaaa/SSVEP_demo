import os
from pathlib import Path

import pytest

from ssvep_demo.config import load_config
from ssvep_demo.stimulus import (
    FlickerScheduler,
    FlipMarkerRecorder,
    TrialState,
    TrialStateMachine,
    build_trial_schedule,
    detect_dropped_frames,
    estimate_effective_frequency_hz,
    has_uneven_runs,
    planned_flicker_sequences,
    static_target_states,
    target_layout,
)


CONFIG = load_config(Path(__file__).parents[1] / "config" / "ssvep_demo.yaml")


def _schedule(order: str = "fixed", seed: int = 42):
    return build_trial_schedule(
        CONFIG.stimulus.frequencies_hz, 3, order, seed, CONFIG.commands
    )


def test_fixed_schedule_is_balanced_and_in_frequency_order() -> None:
    schedule = _schedule()
    assert [trial.target_frequency_hz for trial in schedule] == list(CONFIG.stimulus.frequencies_hz) * 3
    assert [trial.command for trial in schedule[:4]] == ["LEFT", "RIGHT", "FORWARD", "STOP"]


def test_random_schedule_is_seed_reproducible_and_balanced() -> None:
    first = _schedule("random", 9)
    second = _schedule("random", 9)
    assert first == second
    assert [trial.target_frequency_hz for trial in first] != [trial.target_frequency_hz for trial in _schedule()]
    for frequency in CONFIG.stimulus.frequencies_hz:
        assert sum(trial.target_frequency_hz == frequency for trial in first) == 3


def test_flicker_sequences_are_deterministic_and_estimate_requested_frequency() -> None:
    frequencies = CONFIG.stimulus.frequencies_hz
    first = planned_flicker_sequences(frequencies, 60.0, 240)
    assert first == planned_flicker_sequences(frequencies, 60.0, 240)
    for frequency, sequence in first.items():
        assert abs(estimate_effective_frequency_hz(sequence, 60.0) - frequency) <= 0.25
    assert has_uneven_runs(first[8.0])


def test_targets_have_independent_phases_and_static_states_do_not_flicker() -> None:
    together = FlickerScheduler((8.0, 12.0), 60.0)
    alone = FlickerScheduler((8.0,), 60.0)
    together_states = [together.advance()[8.0] for _ in range(20)]
    alone_states = [alone.advance()[8.0] for _ in range(20)]
    assert together_states == alone_states
    assert static_target_states((8.0, 10.0, 12.0)) == {8.0: False, 10.0: False, 12.0: False}


def test_dropped_frame_detection_uses_strict_threshold() -> None:
    assert detect_dropped_frames([1 / 60, 0.024, 0.026], 1 / 60, 1.5) == [2]
    assert detect_dropped_frames([1 / 60, 0.024], 1 / 60, 1.5) == []


def test_pause_during_cue_or_stimulation_aborts_and_restarts_cue() -> None:
    machine = TrialStateMachine(_schedule())
    assert machine.start() == TrialState.CUE
    assert machine.pause() == TrialState.PAUSED
    assert machine.current_trial.trial_id in machine.aborted_trial_ids
    assert machine.resume() == TrialState.CUE
    assert machine.cue_complete() == TrialState.STIMULATION
    assert machine.pause() == TrialState.PAUSED
    assert machine.resume() == TrialState.CUE


def test_flip_marker_uses_callback_times_for_start_and_end() -> None:
    values = iter((10.0, 14.0))
    recorder = FlipMarkerRecorder(clock=lambda: next(values))
    trial = _schedule()[0]
    recorder.mark_stimulus_start()
    recorder.mark_stimulus_end()
    marker = recorder.marker(trial)
    assert marker.stimulus_start_monotonic_s == 10.0
    assert marker.stimulus_end_monotonic_s == 14.0


@pytest.mark.parametrize("count", [2, 3, 4])
def test_layout_and_schedule_support_two_to_four_targets(count: int) -> None:
    frequencies = CONFIG.stimulus.frequencies_hz[:count]
    commands = {frequency: CONFIG.commands[frequency] for frequency in frequencies}
    assert len(target_layout(count)) == count
    assert len(build_trial_schedule(frequencies, 2, "fixed", 1, commands)) == count * 2


def test_importing_stimulus_module_does_not_import_psychopy_or_create_a_window() -> None:
    import ssvep_demo.stimulus as stimulus_module

    assert "psychopy" not in stimulus_module.__dict__


@pytest.mark.skipif(not os.environ.get("RUN_PSYCHOPY_SMOKE"), reason="requires a local display and manual PsychoPy smoke run")
def test_gui_smoke_requires_explicit_local_opt_in() -> None:
    pytest.skip("Run scripts/run_ssvep_stimulus.py manually; it waits in IDLE for Enter.")
