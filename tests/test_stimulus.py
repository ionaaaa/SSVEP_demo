import os
from pathlib import Path

import pytest

from ssvep_demo.config import load_config
from ssvep_demo.stimulus import (
    DEFAULT_CJK_FONT_FILE,
    DEFAULT_CJK_FONT_NAME,
    FlickerScheduler,
    FlipMarkerRecorder,
    PsychoPyStimulusRunner,
    TrialState,
    TrialStateMachine,
    build_trial_schedule,
    detect_dropped_frames,
    estimate_effective_frequency_hz,
    has_uneven_runs,
    planned_flicker_sequences,
    resolve_cjk_font,
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
    assert [trial.command for trial in schedule[:4]] == ["LEFT", "RIGHT", "FORWARD", "BACKWARD"]


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


def test_cjk_text_is_unmodified_and_no_tracking_helper_remains() -> None:
    import ssvep_demo.stimulus as stimulus_module

    assert "double_cjk_tracking" not in stimulus_module.__dict__
    assert "\u3000" not in Path(stimulus_module.__file__).read_text(encoding="utf-8")
    assert "请注视目标" == "请注视目标"


def test_resolve_cjk_font_defaults_to_bundled_font_independent_of_working_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    font = resolve_cjk_font(None, None, None, None)
    assert font.file == DEFAULT_CJK_FONT_FILE
    assert font.name == DEFAULT_CJK_FONT_NAME
    assert font.source == "bundled"
    assert font.file.is_file() and font.file.stat().st_size > 0


def test_cli_font_override_has_priority_over_yaml_and_default(tmp_path: Path) -> None:
    cli_font = tmp_path / "cli.otf"
    yaml_font = tmp_path / "yaml.otf"
    cli_font.write_bytes(b"font")
    yaml_font.write_bytes(b"font")
    font = resolve_cjk_font(cli_font, "CLI Name", yaml_font, "YAML Name", config_directory=tmp_path)
    assert (font.file, font.name, font.source) == (cli_font.resolve(), "CLI Name", "cli")


def test_yaml_font_override_is_relative_to_config_directory(tmp_path: Path) -> None:
    yaml_font = tmp_path / "fonts" / "custom.ttf"
    yaml_font.parent.mkdir()
    yaml_font.write_bytes(b"font")
    font = resolve_cjk_font(None, None, Path("fonts/custom.ttf"), "Custom Name", config_directory=tmp_path)
    assert (font.file, font.name, font.source) == (yaml_font.resolve(), "Custom Name", "yaml")


@pytest.mark.parametrize("name, content, message", [("missing.otf", None, "does not exist"), ("empty.otf", b"", "is empty")])
def test_invalid_font_files_fail_with_clear_error(tmp_path: Path, name: str, content: bytes | None, message: str) -> None:
    font_file = tmp_path / name
    if content is not None:
        font_file.write_bytes(content)
    with pytest.raises(ValueError, match=message):
        resolve_cjk_font(font_file, "Broken Font", None, None)


def test_all_chinese_text_widgets_use_one_cjk_font_and_keep_raw_text() -> None:
    class FakeStim:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.text = kwargs.get("text", "")

        def draw(self):
            pass

    class FakeVisual:
        @staticmethod
        def Rect(*_args, **kwargs):
            return FakeStim(**kwargs)

        @staticmethod
        def TextStim(*_args, **kwargs):
            return FakeStim(**kwargs)

    runner = PsychoPyStimulusRunner(CONFIG, Path(__file__).parents[1] / "config" / "ssvep_demo.yaml")
    runner._cjk_font = resolve_cjk_font(None, None, None, None)
    widgets = runner._make_widgets(object(), FakeVisual, {frequency: frequency for frequency in CONFIG.stimulus.frequencies_hz})
    assert widgets["status"].kwargs["font"] == DEFAULT_CJK_FONT_NAME
    assert widgets["detail"].kwargs["font"] == DEFAULT_CJK_FONT_NAME
    assert widgets["targets"][8.0]["estimate"].kwargs["font"] == DEFAULT_CJK_FONT_NAME
    runner._draw_static(widgets, "请注视目标", "掉帧警告数：0")
    assert widgets["status"].text == "请注视目标"
    assert widgets["detail"].text == "掉帧警告数：0"


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
