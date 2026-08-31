"""Frame-synchronised PsychoPy SSVEP presentation and GUI-free timing logic.

PsychoPy is imported only by :meth:`PsychoPyStimulusRunner.run`, so importing
this module never opens a window and works in headless test environments.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import DemoConfig
from .protocol import TrialMarker


FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
DEFAULT_CJK_FONT_FILE = FONT_DIR / "NotoSansSC-Regular.otf"
DEFAULT_CJK_FONT_NAME = "Noto Sans SC"
DEFAULT_CJK_FONT_SHA256 = "faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9"


@dataclass(frozen=True)
class CJKFontSpec:
    """Resolved CJK font information, independent of PsychoPy or a window."""

    name: str
    file: Path
    source: str


def resolve_cjk_font(
    cli_file: str | Path | None,
    cli_name: str | None,
    config_file: str | Path | None,
    config_name: str | None,
    *,
    config_directory: str | Path | None = None,
) -> CJKFontSpec:
    """Resolve CLI, YAML, or bundled CJK font paths without GUI dependencies.

    CLI relative paths follow normal command-line semantics (the current working
    directory). YAML relative paths are resolved against ``config_directory``.
    The bundled font path is always resolved from this module's directory.
    """
    if cli_file is not None:
        path = Path(cli_file).expanduser().resolve()
        source = "cli"
    elif config_file is not None:
        configured = Path(config_file).expanduser()
        if configured.is_absolute():
            path = configured.resolve()
        else:
            if config_directory is None:
                raise ValueError("config_directory is required for a relative YAML cjk_font_file")
            path = (Path(config_directory).expanduser().resolve() / configured).resolve()
        source = "yaml"
    else:
        path = DEFAULT_CJK_FONT_FILE
        source = "bundled"
    name = cli_name or config_name or DEFAULT_CJK_FONT_NAME
    if not isinstance(name, str) or not name.strip():
        raise ValueError("CJK font name must be a non-empty string")
    if source == "bundled" and name != DEFAULT_CJK_FONT_NAME:
        raise ValueError(
            "A non-default CJK font name requires --cjk-font-file or ui.cjk_font_file; "
            f"the bundled font name is '{DEFAULT_CJK_FONT_NAME}'."
        )
    _validate_font_file(path, name)
    return CJKFontSpec(name=name, file=path, source=source)


def _validate_font_file(path: Path, font_name: str) -> None:
    guidance = "Specify a valid font with ui.cjk_font_file or --cjk-font-file and its matching name."
    if not path.exists() or not path.is_file():
        raise ValueError(f"CJK font file does not exist: {path} (font name: {font_name}). {guidance}")
    if path.stat().st_size == 0:
        raise ValueError(f"CJK font file is empty: {path} (font name: {font_name}). {guidance}")
    if path.suffix.lower() not in {".otf", ".ttf"}:
        raise ValueError(f"CJK font file must be a static .otf or .ttf: {path} (font name: {font_name}). {guidance}")
    if path == DEFAULT_CJK_FONT_FILE:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != DEFAULT_CJK_FONT_SHA256:
            raise ValueError(f"Bundled CJK font failed integrity validation: {path}. {guidance}")


def register_cjk_font(win: Any, visual: Any, font: CJKFontSpec) -> None:
    """Register an external font once, before persistent text is created."""
    try:
        visual.TextStim(win, text="", font=font.name, fontFiles=[str(font.file)], autoLog=False)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load CJK font file '{font.file}' as '{font.name}'. "
            "Use ui.cjk_font_file/ui.cjk_font_name or --cjk-font-file/--cjk-font-name to provide a valid font."
        ) from exc


def measure_refresh_rate(win: Any, refresh_rate_override_hz: float | None) -> tuple[float, str]:
    """Warm a PsychoPy window then measure its refresh rate or use an explicit override."""
    for _ in range(30):
        win.flip()
    if refresh_rate_override_hz is not None:
        if not math.isfinite(refresh_rate_override_hz) or refresh_rate_override_hz <= 0:
            raise ValueError("--refresh-rate must be a positive finite value")
        return float(refresh_rate_override_hz), "cli_override"
    try:
        rate = win.getActualFrameRate(nIdentical=60, nMaxFrames=240, nWarmUpFrames=30)
    except Exception as exc:
        raise RuntimeError("PsychoPy refresh-rate measurement failed; pass --refresh-rate to override it.") from exc
    if rate is None or not math.isfinite(rate) or rate <= 0:
        raise RuntimeError("PsychoPy could not measure refresh rate; pass --refresh-rate to override it.")
    return float(rate), "measured"


class TrialState(str, Enum):
    IDLE = "IDLE"
    CUE = "CUE"
    STIMULATION = "STIMULATION"
    REST = "REST"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class ScheduledTrial:
    trial_id: int
    target_frequency_hz: float
    command: str


def build_trial_schedule(
    frequencies_hz: Sequence[float],
    repetitions: int,
    trial_order: str,
    random_seed: int,
    commands: dict[float, str],
) -> list[ScheduledTrial]:
    """Build balanced fixed or seed-reproducible random target trial order."""
    if not 2 <= len(frequencies_hz) <= 4:
        raise ValueError("visual stimulus supports between two and four target frequencies")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if trial_order not in {"fixed", "random"}:
        raise ValueError("trial_order must be 'fixed' or 'random'")
    targets = [frequency for _ in range(repetitions) for frequency in frequencies_hz]
    if any(frequency not in commands for frequency in targets):
        raise ValueError("every target frequency must have a command mapping")
    if trial_order == "random":
        random.Random(random_seed).shuffle(targets)
    return [
        ScheduledTrial(index, frequency, getattr(commands[frequency], "value", commands[frequency]))
        for index, frequency in enumerate(targets)
    ]


class TrialStateMachine:
    """Pure state machine; GUI code supplies elapsed-time and keyboard events."""

    def __init__(self, schedule: Sequence[ScheduledTrial]) -> None:
        if not schedule:
            raise ValueError("schedule must not be empty")
        self.schedule = list(schedule)
        self.state = TrialState.IDLE
        self.index = 0
        self._resume_state: TrialState | None = None
        self.aborted_trial_ids: list[int] = []

    @property
    def current_trial(self) -> ScheduledTrial:
        return self.schedule[self.index]

    def start(self) -> TrialState:
        if self.state != TrialState.IDLE:
            raise ValueError("only IDLE can start a session")
        self.state = TrialState.CUE
        return self.state

    def cue_complete(self) -> TrialState:
        self._require(TrialState.CUE)
        self.state = TrialState.STIMULATION
        return self.state

    def stimulation_complete(self) -> TrialState:
        self._require(TrialState.STIMULATION)
        self.state = TrialState.REST
        return self.state

    def rest_complete(self) -> TrialState:
        self._require(TrialState.REST)
        self.index += 1
        self.state = TrialState.FINISHED if self.index == len(self.schedule) else TrialState.CUE
        return self.state

    def pause(self) -> TrialState:
        if self.state not in {TrialState.CUE, TrialState.STIMULATION, TrialState.REST}:
            raise ValueError("only an active trial state can be paused")
        self._resume_state = self.state
        if self.state in {TrialState.CUE, TrialState.STIMULATION}:
            self.aborted_trial_ids.append(self.current_trial.trial_id)
        self.state = TrialState.PAUSED
        return self.state

    def resume(self) -> TrialState:
        self._require(TrialState.PAUSED)
        # A cue/stimulation pause invalidates the trial and restarts it at CUE.
        self.state = TrialState.CUE if self._resume_state in {TrialState.CUE, TrialState.STIMULATION} else TrialState.REST
        self._resume_state = None
        return self.state

    def stop(self) -> TrialState:
        self.state = TrialState.STOPPED
        return self.state

    def _require(self, expected: TrialState) -> None:
        if self.state != expected:
            raise ValueError(f"expected state {expected.value}, got {self.state.value}")


class FlickerScheduler:
    """Independent phase accumulators for all targets, advanced once per flip."""

    def __init__(self, frequencies_hz: Sequence[float], refresh_rate_hz: float) -> None:
        if refresh_rate_hz <= 0 or not math.isfinite(refresh_rate_hz):
            raise ValueError("refresh_rate_hz must be positive and finite")
        if not frequencies_hz or any(frequency <= 0 or not math.isfinite(frequency) for frequency in frequencies_hz):
            raise ValueError("frequencies_hz must contain positive finite values")
        self.frequencies_hz = tuple(float(frequency) for frequency in frequencies_hz)
        self.refresh_rate_hz = float(refresh_rate_hz)
        self.reset()

    def reset(self) -> None:
        """Reset all targets to the deterministic phase zero at trial start."""
        self.phases = {frequency: 0.0 for frequency in self.frequencies_hz}
        self._frame_index = 0

    def advance(self) -> dict[float, bool]:
        """Advance all phases for one shared frame and return their on/off state."""
        states: dict[float, bool] = {}
        self._frame_index += 1
        for frequency in self.frequencies_hz:
            # Frame-index arithmetic is equivalent to repeatedly adding
            # frequency / refresh rate, while avoiding accumulated float drift.
            phase = (self._frame_index * frequency / self.refresh_rate_hz) % 1.0
            self.phases[frequency] = phase
            states[frequency] = phase < 0.5
        return states


def planned_flicker_sequences(
    frequencies_hz: Sequence[float], refresh_rate_hz: float, frame_count: int
) -> dict[float, tuple[bool, ...]]:
    """Generate deterministic target-wise on/off sequences without PsychoPy."""
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    scheduler = FlickerScheduler(frequencies_hz, refresh_rate_hz)
    sequences = {frequency: [] for frequency in scheduler.frequencies_hz}
    for _ in range(frame_count):
        for frequency, is_on in scheduler.advance().items():
            sequences[frequency].append(is_on)
    return {frequency: tuple(states) for frequency, states in sequences.items()}


def estimate_effective_frequency_hz(sequence: Sequence[bool], refresh_rate_hz: float) -> float:
    """Estimate frequency as off-to-on rising edges divided by planned duration."""
    if len(sequence) < 2 or refresh_rate_hz <= 0:
        raise ValueError("sequence needs at least two frames and a positive refresh rate")
    rises = sum(not before and after for before, after in zip(sequence, sequence[1:]))
    return rises / (len(sequence) / refresh_rate_hz)


def run_lengths(sequence: Sequence[bool]) -> dict[bool, list[int]]:
    """Return contiguous run lengths per visual state for refresh suitability checks."""
    if not sequence:
        return {True: [], False: []}
    result: dict[bool, list[int]] = {True: [], False: []}
    current = bool(sequence[0])
    count = 0
    for value in sequence:
        value = bool(value)
        if value != current:
            result[current].append(count)
            current, count = value, 1
        else:
            count += 1
    result[current].append(count)
    return result


def has_uneven_runs(sequence: Sequence[bool]) -> bool:
    """Whether either on or off state requires non-uniform frame run lengths."""
    runs = run_lengths(sequence)
    # The first/last run may be truncated by a trial boundary.  The interior
    # runs are the useful indicator of refresh-rate approximation unevenness.
    return any(len(set(lengths[1:-1])) > 1 for lengths in runs.values() if len(lengths) > 2)


def static_target_states(frequencies_hz: Sequence[float]) -> dict[float, bool]:
    """Return the all-off state used by REST and PAUSED screens."""
    return {float(frequency): False for frequency in frequencies_hz}


def detect_dropped_frames(
    frame_intervals_s: Sequence[float], nominal_interval_s: float, threshold_ratio: float
) -> list[int]:
    """Return indices whose software-measured interval exceeds the threshold."""
    if nominal_interval_s <= 0 or threshold_ratio <= 1:
        raise ValueError("nominal_interval_s must be positive and threshold_ratio must exceed 1")
    return [
        index
        for index, interval in enumerate(frame_intervals_s)
        if math.isfinite(interval) and interval > nominal_interval_s * threshold_ratio
    ]


def target_layout(target_count: int) -> list[tuple[float, float]]:
    """Return normalized positions for two, three, or four target rectangles."""
    layouts = {
        2: [(-0.5, 0.0), (0.5, 0.0)],
        3: [(0.0, 0.5), (-0.55, -0.35), (0.55, -0.35)],
        # Matches the requested order [left, right, top, bottom].
        4: [(-0.58, 0.0), (0.58, 0.0), (0.0, 0.53), (0.0, -0.53)],
    }
    try:
        return layouts[target_count]
    except KeyError as exc:
        raise ValueError("target layout supports two to four targets") from exc


@dataclass
class FlipMarkerRecorder:
    """Records marker boundaries only from callbacks executed by ``win.flip``."""

    clock: Callable[[], float] = time.monotonic
    stimulus_start_monotonic_s: float | None = None
    stimulus_end_monotonic_s: float | None = None

    def mark_stimulus_start(self) -> None:
        self.stimulus_start_monotonic_s = self.clock()

    def mark_stimulus_end(self) -> None:
        self.stimulus_end_monotonic_s = self.clock()

    def marker(self, trial: ScheduledTrial) -> TrialMarker:
        if self.stimulus_start_monotonic_s is None:
            raise ValueError("stimulus start has not been recorded on a flip")
        return TrialMarker(
            trial_id=trial.trial_id,
            target_frequency_hz=trial.target_frequency_hz,
            stimulus_start_monotonic_s=self.stimulus_start_monotonic_s,
            stimulus_end_monotonic_s=self.stimulus_end_monotonic_s,
        )


class _SessionLogger:
    """Incrementally flushed session files; UTC is only human-readable metadata."""

    trial_fields = [
        "trial_id", "target_frequency_hz", "command", "trial_status", "cue_start_monotonic_s",
        "stimulus_start_monotonic_s", "stimulus_end_monotonic_s", "rest_start_monotonic_s",
        "requested_frequency_hz", "estimated_frequency_hz", "measured_refresh_rate_hz",
        "refresh_rate_source", "planned_frame_count", "actual_frame_count", "dropped_frame_count",
        "max_frame_interval_s", "mean_frame_interval_s",
    ]

    def __init__(self, output_root: Path, config_path: Path, session_metadata: dict[str, Any]) -> None:
        utc_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = output_root / f"session_{utc_stamp}"
        suffix = 1
        while self.directory.exists():
            self.directory = output_root / f"session_{utc_stamp}_{suffix:02d}"
            suffix += 1
        self.directory.mkdir(parents=True)
        shutil.copy2(config_path, self.directory / "config.yaml")
        self.metadata = session_metadata
        self._write_session()
        self.events_file = (self.directory / "events.jsonl").open("a", encoding="utf-8")
        self.trials_file = (self.directory / "trials.csv").open("w", encoding="utf-8", newline="")
        self.trials = csv.DictWriter(self.trials_file, fieldnames=self.trial_fields)
        self.trials.writeheader()
        self.trials_file.flush()
        self.intervals_file = (self.directory / "frame_intervals.csv").open("w", encoding="utf-8", newline="")
        self.intervals = csv.DictWriter(
            self.intervals_file, fieldnames=["trial_id", "frame_index", "interval_s", "suspected_dropped"]
        )
        self.intervals.writeheader()
        self.intervals_file.flush()

    def _write_session(self) -> None:
        (self.directory / "session.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def event(self, event_type: str, **payload: Any) -> None:
        record = {
            "event": event_type,
            "monotonic_s": time.monotonic(),
            "utc_time": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self.events_file.write(json.dumps(record) + "\n")
        self.events_file.flush()

    def trial(self, record: dict[str, Any]) -> None:
        self.trials.writerow(record)
        self.trials_file.flush()

    def intervals_for_trial(self, trial_id: int, intervals_s: Sequence[float], dropped_indices: Sequence[int]) -> None:
        dropped = set(dropped_indices)
        for index, interval in enumerate(intervals_s):
            self.intervals.writerow(
                {"trial_id": trial_id, "frame_index": index, "interval_s": interval, "suspected_dropped": index in dropped}
            )
        self.intervals_file.flush()

    def close(self, **summary: Any) -> None:
        self.metadata.update(summary)
        self._write_session()
        self.events_file.close()
        self.trials_file.close()
        self.intervals_file.close()


class PsychoPyStimulusRunner:
    """Explicit-entry PsychoPy runner for visual-only SSVEP trial presentation."""

    def __init__(
        self,
        config: DemoConfig,
        config_path: str | Path,
        *,
        fullscreen: bool | None = None,
        screen_index: int | None = None,
        refresh_rate_hz: float | None = None,
        output_dir: str | Path = "outputs/ssvep_stimulus",
        cjk_font_file: str | Path | None = None,
        cjk_font_name: str | None = None,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path)
        if screen_index is not None and (isinstance(screen_index, bool) or not isinstance(screen_index, int) or screen_index < 0):
            raise ValueError("screen_index must be a non-negative integer")
        if fullscreen is not None and not isinstance(fullscreen, bool):
            raise ValueError("fullscreen override must be a boolean")
        self.fullscreen = config.stimulus.fullscreen if fullscreen is None else fullscreen
        self.screen_index = config.stimulus.screen_index if screen_index is None else screen_index
        self.refresh_rate_override_hz = refresh_rate_hz
        self.output_dir = Path(output_dir)
        self.cjk_font_file_override = cjk_font_file
        self.cjk_font_name_override = cjk_font_name
        self._cjk_font: CJKFontSpec | None = None
        self.schedule = build_trial_schedule(
            config.stimulus.frequencies_hz,
            config.stimulus.repetitions,
            config.stimulus.trial_order,
            config.stimulus.random_seed,
            config.commands,
        )
        self.state_machine = TrialStateMachine(self.schedule)
        self._drop_total = 0
        self._active_cue_start_monotonic_s: float | None = None

    def run(self) -> Path:
        """Create the PsychoPy window and run until Escape, completion, or close."""
        try:
            from psychopy import core, event, visual
            import psychopy
        except ImportError as exc:  # pragma: no cover - depends on local GUI setup
            raise RuntimeError("PsychoPy is required to run visual stimulation. Install the 'stimulus' extra.") from exc

        # ``core`` is deliberately imported with event/visual as the supported
        # PsychoPy runtime surface; internal protocol timestamps remain monotonic.
        del core
        win = visual.Window(
            size=self.config.stimulus.window_size,
            fullscr=self.fullscreen,
            screen=self.screen_index,
            color=self.config.stimulus.background_color,
            units="norm",
            waitBlanking=True,
        )
        logger: _SessionLogger | None = None
        try:
            self._cjk_font = resolve_cjk_font(
                self.cjk_font_file_override,
                self.cjk_font_name_override,
                self.config.ui.cjk_font_file,
                self.config.ui.cjk_font_name,
                config_directory=self.config_path.expanduser().resolve().parent,
            )
            self._register_cjk_font(win, visual, self._cjk_font)
            measured_hz, refresh_source = self._measure_refresh_rate(win)
            nominal_interval_s = 1.0 / measured_hz
            win.recordFrameIntervals = True
            win.refreshThreshold = nominal_interval_s * self.config.stimulus.dropped_frame_threshold_ratio
            frame_count = max(1, round(self.config.stimulus.trial_duration_s * measured_hz))
            plans = planned_flicker_sequences(self.config.stimulus.frequencies_hz, measured_hz, frame_count)
            estimates = {frequency: estimate_effective_frequency_hz(sequence, measured_hz) for frequency, sequence in plans.items()}
            uneven = {frequency: has_uneven_runs(sequence) for frequency, sequence in plans.items()}
            frequency_warnings = {
                frequency: self._frequency_warning(frequency, estimates[frequency], measured_hz, frame_count, uneven[frequency])
                for frequency in self.config.stimulus.frequencies_hz
            }
            logger = _SessionLogger(
                self.output_dir,
                self.config_path,
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "psychopy_version": psychopy.__version__,
                    "measured_refresh_rate_hz": measured_hz,
                    "refresh_rate_source": refresh_source,
                    "requested_frequencies_hz": list(self.config.stimulus.frequencies_hz),
                    "estimated_frequencies_hz": estimates,
                    "effective_frequency_estimation": "off_to_on_rising_edges / planned_duration",
                    "frequency_warnings": frequency_warnings,
                    "random_seed": self.config.stimulus.random_seed,
                    "cjk_font": {
                        "name": self._cjk_font.name,
                        "file": str(self._cjk_font.file),
                        "source": self._cjk_font.source,
                    },
                    "cli_overrides": {
                        "fullscreen": self.fullscreen,
                        "screen_index": self.screen_index,
                        "refresh_rate_hz": self.refresh_rate_override_hz,
                        "cjk_font_file": str(self.cjk_font_file_override) if self.cjk_font_file_override else None,
                        "cjk_font_name": self.cjk_font_name_override,
                    },
                },
            )
            print(f"PsychoPy {psychopy.__version__}")
            print(f"Refresh rate: {measured_hz:.4f} Hz ({refresh_source})")
            for frequency in self.config.stimulus.frequencies_hz:
                warning = f" [warning: {frequency_warnings[frequency]}]" if frequency_warnings[frequency] else ""
                print(f"{frequency:g} Hz requested; {estimates[frequency]:.4f} Hz estimated{warning}")
            print(f"Output session: {logger.directory}")
            self._run_loop(win, visual, event, logger, measured_hz, nominal_interval_s, frame_count, estimates)
            return logger.directory
        finally:  # Always preserve already-completed logs before the window closes.
            if logger is not None:
                logger.close(session_dropped_frame_count=self._drop_total, final_state=self.state_machine.state.value)
            win.close()

    @staticmethod
    def _frequency_warning(
        requested_hz: float, estimated_hz: float, refresh_hz: float, frame_count: int, uneven: bool
    ) -> str | None:
        if requested_hz >= refresh_hz / 2.0:
            return "requested frequency is at or above display Nyquist"
        if abs(requested_hz - estimated_hz) > refresh_hz / frame_count:
            return "estimated frequency differs by more than one planned-cycle bin"
        if uneven:
            return "non-uniform on/off frame runs approximate this frequency"
        return None

    def _measure_refresh_rate(self, win: Any) -> tuple[float, str]:
        return measure_refresh_rate(win, self.refresh_rate_override_hz)

    @staticmethod
    def _register_cjk_font(win: Any, visual: Any, font: CJKFontSpec) -> None:
        register_cjk_font(win, visual, font)

    def _run_loop(
        self, win: Any, visual: Any, event: Any, logger: _SessionLogger, refresh_hz: float,
        nominal_interval_s: float, frame_count: int, estimates: dict[float, float],
    ) -> None:
        widgets = self._make_widgets(win, visual, estimates)
        cue_start: float | None = None
        rest_start: float | None = None
        while self.state_machine.state not in {TrialState.FINISHED, TrialState.STOPPED}:
            state = self.state_machine.state
            if state == TrialState.IDLE:
                self._draw_static(widgets, "等待", "Enter 开始；P 暂停；Esc 安全停止")
                win.flip()
                keys = event.getKeys(keyList=["return", "escape"])
                if "escape" in keys:
                    self.state_machine.stop()
                elif "return" in keys:
                    self.state_machine.start()
                    logger.event("session_started")
                    cue_start = None
            elif state == TrialState.CUE:
                trial = self.state_machine.current_trial
                if cue_start is None:
                    cue_start = time.monotonic()
                    self._active_cue_start_monotonic_s = cue_start
                    logger.event("cue_started", trial_id=trial.trial_id, target_frequency_hz=trial.target_frequency_hz)
                self._draw_cue(widgets, trial)
                win.flip()
                if self._handle_runtime_keys(event, logger):
                    if self.state_machine.state == TrialState.PAUSED:
                        self._write_trial_record(logger, trial, "aborted", None, None, estimates, refresh_hz, frame_count, 0, [])
                    elif self.state_machine.state == TrialState.STOPPED:
                        self._write_trial_record(logger, trial, "stopped", None, None, estimates, refresh_hz, frame_count, 0, [])
                    cue_start = None
                    self._active_cue_start_monotonic_s = None
                    continue
                if time.monotonic() - cue_start >= self.config.stimulus.cue_duration_s:
                    self.state_machine.cue_complete()
            elif state == TrialState.STIMULATION:
                outcome, rest_start = self._present_stimulation(
                    win, event, widgets, logger, refresh_hz, nominal_interval_s, frame_count, estimates
                )
                cue_start = None
                if outcome == "completed":
                    self.state_machine.stimulation_complete()
                elif outcome == "paused":
                    rest_start = None
                self._active_cue_start_monotonic_s = None
            elif state == TrialState.REST:
                if rest_start is None:
                    rest_start = time.monotonic()
                    logger.event("rest_started", trial_id=self.state_machine.current_trial.trial_id)
                self._draw_static(widgets, "休息", "请放松并注视中央注视点；P 暂停；Esc 停止")
                win.flip()
                if self._handle_runtime_keys(event, logger):
                    continue
                if time.monotonic() - rest_start >= self.config.stimulus.rest_duration_s:
                    self.state_machine.rest_complete()
                    rest_start = None
            elif state == TrialState.PAUSED:
                self._draw_static(widgets, "已暂停", "按 P 从当前 trial 的提示阶段重新开始；Esc 停止")
                win.flip()
                keys = event.getKeys(keyList=["p", "escape"])
                if "escape" in keys:
                    self.state_machine.stop()
                    logger.event("session_stopped")
                elif "p" in keys:
                    self.state_machine.resume()
                    logger.event("resume", trial_id=self.state_machine.current_trial.trial_id)
                    cue_start, rest_start, self._active_cue_start_monotonic_s = None, None, None
        logger.event("session_finished" if self.state_machine.state == TrialState.FINISHED else "session_stopped")

    def _handle_runtime_keys(self, event: Any, logger: _SessionLogger) -> bool:
        keys = event.getKeys(keyList=["p", "escape"])
        if "escape" in keys:
            self.state_machine.stop()
            logger.event("session_stopped")
            return True
        if "p" in keys:
            trial = self.state_machine.current_trial
            state_before_pause = self.state_machine.state
            self.state_machine.pause()
            logger.event("pause", trial_id=trial.trial_id, paused_from=state_before_pause.value)
            if state_before_pause in {TrialState.CUE, TrialState.STIMULATION}:
                logger.event("trial_aborted", trial_id=trial.trial_id, reason="pause")
            return True
        return False

    def _present_stimulation(
        self, win: Any, event: Any, widgets: dict[str, Any], logger: _SessionLogger,
        refresh_hz: float, nominal_interval_s: float, frame_count: int, estimates: dict[float, float],
    ) -> tuple[str, float | None]:
        trial = self.state_machine.current_trial
        scheduler = FlickerScheduler(self.config.stimulus.frequencies_hz, refresh_hz)
        recorder = FlipMarkerRecorder()
        intervals_start = len(win.frameIntervals)
        logger.event("stimulation_scheduled", trial_id=trial.trial_id, planned_frame_count=frame_count)
        for frame_index in range(frame_count):
            keys = event.getKeys(keyList=["p", "escape"])
            if "escape" in keys:
                self.state_machine.stop()
                logger.event("session_stopped")
                intervals = list(win.frameIntervals)[intervals_start:]
                marker = recorder.marker(trial) if recorder.stimulus_start_monotonic_s is not None else None
                self._write_frame_intervals(logger, trial.trial_id, intervals, nominal_interval_s)
                self._write_trial_record(
                    logger, trial, "stopped", marker, None, estimates, refresh_hz, frame_count, frame_index, intervals
                )
                return "stopped", None
            if "p" in keys:
                self.state_machine.pause()
                logger.event("pause", trial_id=trial.trial_id, paused_from=TrialState.STIMULATION.value)
                logger.event("trial_aborted", trial_id=trial.trial_id, reason="pause")
                intervals = list(win.frameIntervals)[intervals_start:]
                marker = recorder.marker(trial) if recorder.stimulus_start_monotonic_s is not None else None
                self._write_frame_intervals(logger, trial.trial_id, intervals, nominal_interval_s)
                self._write_trial_record(
                    logger, trial, "aborted", marker, None, estimates, refresh_hz, frame_count, frame_index, intervals
                )
                return "paused", None
            self._draw_stimulation(widgets, scheduler.advance(), trial)
            if frame_index == 0:
                win.callOnFlip(recorder.mark_stimulus_start)
            win.flip()
        # The first static REST flip is the actual end boundary, not the Python
        # call immediately before it.
        self._draw_static(widgets, "休息", "请放松并注视中央注视点；P 暂停；Esc 停止")
        win.callOnFlip(recorder.mark_stimulus_end)
        win.flip()
        marker = recorder.marker(trial)
        intervals = list(win.frameIntervals)[intervals_start:]
        dropped = self._write_frame_intervals(logger, trial.trial_id, intervals, nominal_interval_s)
        self._write_trial_record(logger, trial, "completed", marker, marker.stimulus_end_monotonic_s, estimates, refresh_hz, frame_count, frame_count, intervals)
        logger.event("stimulation_completed", trial_id=trial.trial_id, dropped_frame_count=len(dropped))
        return "completed", marker.stimulus_end_monotonic_s

    def _write_frame_intervals(
        self, logger: _SessionLogger, trial_id: int, intervals: Sequence[float], nominal_interval_s: float
    ) -> list[int]:
        dropped = detect_dropped_frames(
            intervals, nominal_interval_s, self.config.stimulus.dropped_frame_threshold_ratio
        )
        self._drop_total += len(dropped)
        logger.intervals_for_trial(trial_id, intervals, dropped)
        return dropped

    def _write_trial_record(
        self, logger: _SessionLogger, trial: ScheduledTrial, status: str, marker: TrialMarker | None,
        rest_start: float | None, estimates: dict[float, float], refresh_hz: float, planned_frames: int,
        actual_frames: int, intervals: Sequence[float],
    ) -> None:
        dropped = detect_dropped_frames(
            intervals, 1.0 / refresh_hz, self.config.stimulus.dropped_frame_threshold_ratio
        ) if intervals else []
        logger.trial({
            "trial_id": trial.trial_id,
            "target_frequency_hz": trial.target_frequency_hz,
            "command": trial.command,
            "trial_status": status,
            "cue_start_monotonic_s": self._active_cue_start_monotonic_s or "",
            "stimulus_start_monotonic_s": marker.stimulus_start_monotonic_s if marker else "",
            "stimulus_end_monotonic_s": marker.stimulus_end_monotonic_s if marker else "",
            "rest_start_monotonic_s": rest_start if rest_start is not None else "",
            "requested_frequency_hz": trial.target_frequency_hz,
            "estimated_frequency_hz": estimates[trial.target_frequency_hz],
            "measured_refresh_rate_hz": refresh_hz,
            "refresh_rate_source": "cli_override" if self.refresh_rate_override_hz is not None else "measured",
            "planned_frame_count": planned_frames,
            "actual_frame_count": actual_frames,
            "dropped_frame_count": len(dropped),
            "max_frame_interval_s": max(intervals) if intervals else "",
            "mean_frame_interval_s": float(np.mean(intervals)) if intervals else "",
        })

    def _make_widgets(self, win: Any, visual: Any, estimates: dict[float, float]) -> dict[str, Any]:
        if self._cjk_font is None:
            raise RuntimeError("CJK font must be resolved and registered before creating widgets")
        positions = target_layout(len(self.config.stimulus.frequencies_hz))
        targets = {}
        for frequency, position in zip(self.config.stimulus.frequencies_hz, positions):
            targets[frequency] = {
                "rect": visual.Rect(win, width=0.32, height=0.24, pos=position, lineColor=(0.3, 0.3, 0.3)),
                # TextStim does not expose reliable per-letter kerning controls.
                # Keep each semantic line separate instead, so its visual gap is
                # explicit and scales with the chosen letter heights.
                "frequency": visual.TextStim(
                    win, pos=(position[0], position[1] - 0.165), height=0.052,
                    text=f"{frequency:g} Hz", bold=True,
                ),
                "command": visual.TextStim(
                    win, pos=(position[0], position[1] - 0.220), height=0.043,
                    text=self.config.commands[frequency], bold=True,
                ),
                "estimate": visual.TextStim(
                    win, pos=(position[0], position[1] - 0.270), height=0.030,
                    text=f"实际 {estimates[frequency]:.2f} Hz",
                    color=(0.75, 0.75, 0.75),
                    font=self._cjk_font.name,
                    wrapWidth=0.40,
                    alignText="center",
                    anchorHoriz="center",
                    anchorVert="center",
                ),
            }
        return {
            "targets": targets,
            # Status/detail include Chinese instructions and therefore use the
            # registered bundled/overridden CJK font. English-only target
            # frequency and command labels retain PsychoPy's default font.
            "status": visual.TextStim(
                win, pos=(0, 0.84), height=0.047, font=self._cjk_font.name,
                wrapWidth=1.80, alignText="center", anchorHoriz="center", anchorVert="center",
            ),
            "detail": visual.TextStim(
                win, pos=(0, -0.82), height=0.034, font=self._cjk_font.name,
                wrapWidth=1.72, alignText="center", anchorHoriz="center", anchorVert="center",
            ),
            "fixation": visual.TextStim(win, text="+", pos=(0, 0), height=0.12),
        }

    def _draw_static(self, widgets: dict[str, Any], status: str, detail: str) -> None:
        widgets["status"].text = status
        widgets["detail"].text = detail
        widgets["status"].draw()
        widgets["detail"].draw()
        widgets["fixation"].draw()
        # REST/PAUSED/IDLE deliberately draw all targets in the same off state;
        # no scheduler is advanced outside STIMULATION.
        for frequency in static_target_states(self.config.stimulus.frequencies_hz):
            target = widgets["targets"][frequency]
            target["rect"].fillColor = self.config.stimulus.stimulus_off_color
            target["rect"].lineColor = (0.3, 0.3, 0.3)
            target["rect"].draw()
            self._draw_target_labels(target)

    def _draw_cue(self, widgets: dict[str, Any], trial: ScheduledTrial) -> None:
        self._draw_static(
            widgets,
            f"提示  Trial {trial.trial_id + 1}/{len(self.schedule)}",
            f"请注视 {trial.target_frequency_hz:g} Hz / {trial.command}；P 暂停；Esc 停止",
        )
        for frequency, target in widgets["targets"].items():
            target["rect"].fillColor = self.config.stimulus.stimulus_off_color
            target["rect"].lineColor = "yellow" if frequency == trial.target_frequency_hz else (0.3, 0.3, 0.3)
            target["rect"].draw()
            self._draw_target_labels(target)

    def _draw_stimulation(self, widgets: dict[str, Any], states: dict[float, bool], trial: ScheduledTrial) -> None:
        dropped_text = f"掉帧警告数：{self._drop_total}"
        self._draw_static(
            widgets,
            f"刺激  Trial {trial.trial_id + 1}/{len(self.schedule)}",
            f"目标 {trial.target_frequency_hz:g} Hz / {trial.command}；P 暂停；Esc 停止；{dropped_text}",
        )
        for frequency, target in widgets["targets"].items():
            target["rect"].fillColor = (
                self.config.stimulus.stimulus_on_color if states[frequency] else self.config.stimulus.stimulus_off_color
            )
            target["rect"].lineColor = "yellow" if frequency == trial.target_frequency_hz else (0.3, 0.3, 0.3)
            target["rect"].draw()
            self._draw_target_labels(target)

    @staticmethod
    def _draw_target_labels(target: dict[str, Any]) -> None:
        """Draw pre-built labels without updating their text during flicker."""
        target["frequency"].draw()
        target["command"].draw()
        target["estimate"].draw()
