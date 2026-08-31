"""Integrated, explicitly simulated SSVEP closed-loop demonstration.

The module coordinates existing stimulus timing, synthetic EEG, decoders, and
the hardware-independent safety layer.  It contains no real EEG, Bluetooth,
or real-car implementation.  PsychoPy is imported only from ``run``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Protocol, Sequence

from .config import DemoConfig
from .control import ControlConfig, DispatchDecision, SafeCommandDispatcher
from .decoders import CCADecoder, FFTDecoder, SSVEPDecoder
from .protocol import DecodeResult, EEGWindow, TrialMarker
from .stimulus import (
    CJKFontSpec,
    FlickerScheduler,
    FlipMarkerRecorder,
    ScheduledTrial,
    build_trial_schedule,
    detect_dropped_frames,
    estimate_effective_frequency_hz,
    measure_refresh_rate,
    planned_flicker_sequences,
    register_cjk_font,
    resolve_cjk_font,
    static_target_states,
    target_layout,
)
from .synthetic import SyntheticEEGSource
from .virtual_car import VirtualCarController
from .eeg_archive import EEGWindowArchiveWriter
from .session_logging import UnifiedSessionLogger, resolved_config_dict


class SyntheticDemoState(str, Enum):
    IDLE = "IDLE"
    CUE = "CUE"
    STIMULATING = "STIMULATING"
    DECODING = "DECODING"
    EXECUTING = "EXECUTING"
    REST = "REST"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    STOPPED = "STOPPED"


class SyntheticDemoStateMachine:
    """Pure lifecycle for one simulated trial schedule."""

    def __init__(self, schedule: Sequence[ScheduledTrial]) -> None:
        if not schedule:
            raise ValueError("schedule must not be empty")
        self.schedule = list(schedule)
        self.index = 0
        self.state = SyntheticDemoState.IDLE
        self._resume_state: SyntheticDemoState | None = None
        self.aborted_trial_ids: list[int] = []

    @property
    def current_trial(self) -> ScheduledTrial:
        return self.schedule[self.index]

    def start(self) -> None:
        self._require(SyntheticDemoState.IDLE)
        self.state = SyntheticDemoState.CUE

    def cue_complete(self) -> None:
        self._require(SyntheticDemoState.CUE)
        self.state = SyntheticDemoState.STIMULATING

    def stimulation_complete(self) -> None:
        self._require(SyntheticDemoState.STIMULATING)
        self.state = SyntheticDemoState.DECODING

    def decoding_complete(self) -> None:
        self._require(SyntheticDemoState.DECODING)
        self.state = SyntheticDemoState.EXECUTING

    def executing_complete(self) -> None:
        self._require(SyntheticDemoState.EXECUTING)
        self.state = SyntheticDemoState.REST

    def rest_complete(self) -> None:
        self._require(SyntheticDemoState.REST)
        self.index += 1
        self.state = SyntheticDemoState.FINISHED if self.index == len(self.schedule) else SyntheticDemoState.CUE

    def pause(self) -> None:
        if self.state not in {
            SyntheticDemoState.CUE,
            SyntheticDemoState.STIMULATING,
            SyntheticDemoState.DECODING,
            SyntheticDemoState.EXECUTING,
            SyntheticDemoState.REST,
        }:
            raise ValueError("only an active synthetic-demo state can be paused")
        self._resume_state = self.state
        if self.state is not SyntheticDemoState.REST:
            self.aborted_trial_ids.append(self.current_trial.trial_id)
        self.state = SyntheticDemoState.PAUSED

    def resume(self) -> None:
        self._require(SyntheticDemoState.PAUSED)
        self.state = SyntheticDemoState.REST if self._resume_state is SyntheticDemoState.REST else SyntheticDemoState.CUE
        self._resume_state = None

    def stop(self) -> None:
        self.state = SyntheticDemoState.STOPPED

    def _require(self, expected: SyntheticDemoState) -> None:
        if self.state is not expected:
            raise ValueError(f"expected state {expected.value}, got {self.state.value}")


class _SyntheticSource(Protocol):
    def generate(
        self, *, target_frequency_hz: float, duration_s: float, sample_rate_hz: float, snr_db: float
    ) -> EEGWindow: ...


@dataclass(frozen=True)
class TrialComputation:
    """The one-generation, one-decode output for a simulated trial."""

    window: EEGWindow
    result: DecodeResult
    decision: DispatchDecision
    synthetic_seed: int
    generation_latency_ms: float
    decoding_latency_ms: float


class SyntheticTrialCoordinator:
    """GUI-free orchestrator that submits exactly one result per trial."""

    def __init__(
        self,
        config: DemoConfig,
        decoder: SSVEPDecoder,
        dispatcher: SafeCommandDispatcher,
        *,
        snr_db: float,
        seed: int,
        source_factory: Callable[[int], _SyntheticSource] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("synthetic seed must be a non-negative integer")
        if not isinstance(snr_db, (int, float)) or isinstance(snr_db, bool) or not math.isfinite(float(snr_db)):
            raise ValueError("snr_db must be finite")
        self.config = config
        self.decoder = decoder
        self.dispatcher = dispatcher
        self.snr_db = float(snr_db)
        self.seed = seed
        self.clock = clock
        self.source_factory = source_factory or (lambda trial_seed: SyntheticEEGSource(config, seed=trial_seed))

    def process(self, trial: ScheduledTrial) -> TrialComputation:
        """Generate once, decode once, and submit the resulting output once."""
        trial_seed = self.seed + trial.trial_id
        source = self.source_factory(trial_seed)
        generation_start_s = self.clock()
        window = source.generate(
            target_frequency_hz=trial.target_frequency_hz,
            duration_s=self.config.stimulus.trial_duration_s,
            sample_rate_hz=self.config.acquisition.sample_rate_hz,
            snr_db=self.snr_db,
        )
        generation_latency_ms = (self.clock() - generation_start_s) * 1000.0
        decoding_start_s = self.clock()
        # The decoder receives only EEGWindow.  The trial target is deliberately
        # absent from this call and from the protocol object itself.
        result = self.decoder.decode(window)
        decoding_latency_ms = (self.clock() - decoding_start_s) * 1000.0
        decision = self.dispatcher.submit(result)
        return TrialComputation(
            window=window,
            result=result,
            decision=decision,
            synthetic_seed=trial_seed,
            generation_latency_ms=generation_latency_ms,
            decoding_latency_ms=decoding_latency_ms,
        )


# Backwards-compatible import name; new synthetic and replay paths both use
# UnifiedSessionLogger rather than independent file layouts.
SyntheticDemoSessionLogger = UnifiedSessionLogger


class _SyntheticDemoView:
    """Persistent, minimal two-column PsychoPy drawing objects for the demo."""

    def __init__(self, win: Any, visual: Any, config: DemoConfig, font: CJKFontSpec, estimates: dict[float, float]) -> None:
        self.config = config
        self.estimates = estimates
        self.targets: dict[float, dict[str, Any]] = {}
        for frequency, position in zip(config.stimulus.frequencies_hz, target_layout(len(config.stimulus.frequencies_hz))):
            x, y = (-0.55 + position[0] * 0.35, position[1] * 0.55)
            self.targets[frequency] = {
                "rect": visual.Rect(win, pos=(x, y), width=0.25, height=0.20, fillColor=config.stimulus.stimulus_off_color),
                "freq": visual.TextStim(win, text=f"{frequency:g} Hz", pos=(x, y + 0.035), height=0.035, font=font.name),
                "command": visual.TextStim(win, text=config.commands[frequency].value, pos=(x, y - 0.025), height=0.027, font=font.name),
                "estimate": visual.TextStim(win, text=f"实际 {estimates[frequency]:.2f} Hz", pos=(x, y - 0.075), height=0.021, font=font.name),
            }
        self.simulation = visual.TextStim(
            win, text="SIMULATION MODE / 模拟模式\nEEG 由当前 trial 目标频率合成，不代表真实人体 SSVEP",
            pos=(0, 0.91), height=0.027, font=font.name, wrapWidth=1.85, alignText="center",
        )
        self.status = visual.TextStim(win, pos=(-0.45, -0.72), height=0.032, font=font.name, wrapWidth=1.0)
        self.details = visual.TextStim(win, pos=(-0.45, -0.84), height=0.024, font=font.name, wrapWidth=1.0)
        self.results = visual.TextStim(win, pos=(0.48, 0.48), height=0.027, font=font.name, wrapWidth=0.78, alignText="left")
        self.car_rect = visual.Rect(win, width=0.11, height=0.07, fillColor="royalblue", lineColor="white")
        self.heading = visual.Line(win, start=(0.45, 0), end=(0.45, 0.1), lineColor="yellow", lineWidth=3)
        self.car_text = visual.TextStim(win, pos=(0.48, -0.34), height=0.025, font=font.name, wrapWidth=0.78)

    def draw(
        self,
        *,
        state: SyntheticDemoState,
        trial: ScheduledTrial | None,
        target_states: dict[float, bool],
        car: VirtualCarController,
        result: DecodeResult | None,
        decision: DispatchDecision | None,
        trial_total: int,
        dropped_frame_count: int,
        remaining_s: float | None,
    ) -> None:
        self.simulation.draw()
        trial_text = "等待 Enter 开始" if trial is None else f"Trial {trial.trial_id + 1}/{trial_total}: {trial.target_frequency_hz:g} Hz / {trial.command}"
        self.status.text = f"{state.value}\n{trial_text}"
        self.details.text = f"掉帧警告数：{dropped_frame_count}" + (f"\n执行剩余：{remaining_s:.2f}s" if remaining_s is not None else "")
        self.status.draw()
        self.details.draw()
        for frequency, target in self.targets.items():
            target["rect"].fillColor = self.config.stimulus.stimulus_on_color if target_states[frequency] else self.config.stimulus.stimulus_off_color
            target["rect"].lineColor = "yellow" if trial and frequency == trial.target_frequency_hz else (0.3, 0.3, 0.3)
            target["rect"].draw()
            target["freq"].draw()
            target["command"].draw()
            target["estimate"].draw()
        scores = "--" if result is None else "\n".join(f"{frequency:g} Hz: {score:.4f}" for frequency, score in result.scores.items())
        prediction_command = None if result is None else getattr(result.command, "value", result.command)
        prediction = "--" if result is None else f"{result.predicted_frequency_hz:g} Hz / {prediction_command}\nconfidence: {result.confidence:.3f}"
        dispatch = "--" if decision is None else f"{decision.action}: {decision.reason}"
        self.results.text = f"Decoder results\n{scores}\n\nPrediction: {prediction}\nDispatch: {dispatch}"
        self.results.draw()
        car_x, car_y = (0.48 + car.x * 0.35, -0.02 + car.y * 0.38)
        angle = math.radians(car.heading_degrees)
        self.car_rect.pos = (car_x, car_y)
        self.heading.start = (car_x, car_y)
        self.heading.end = (car_x + 0.11 * math.cos(angle), car_y + 0.11 * math.sin(angle))
        self.car_rect.draw()
        self.heading.draw()
        active = car.active_command.value if car.active_command else "STOPPED"
        self.car_text.text = f"Virtual car\n({car.x:.2f}, {car.y:.2f})  {car.heading_degrees:.0f}°\n{active}"
        self.car_text.draw()


class SSVEPSyntheticDemoRunner:
    """PsychoPy presentation that connects the simulated closed-loop pipeline."""

    def __init__(
        self,
        config: DemoConfig,
        config_path: str | Path,
        *,
        decoder_type: str | None = None,
        snr_db: float | None = None,
        seed: int | None = None,
        fullscreen: bool | None = None,
        refresh_rate_hz: float | None = None,
        output_dir: str | Path = "outputs/ssvep_demo",
        max_trials: int | None = None,
        confirmations_required: int | None = None,
    ) -> None:
        if decoder_type is not None and decoder_type not in {"fft", "cca"}:
            raise ValueError("decoder_type must be 'fft' or 'cca'")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("seed override must be a non-negative integer")
        if max_trials is not None and (isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials <= 0):
            raise ValueError("max_trials must be a positive integer")
        if confirmations_required is not None and (
            isinstance(confirmations_required, bool) or not isinstance(confirmations_required, int) or confirmations_required <= 0
        ):
            raise ValueError("confirmations_required must be a positive integer")
        self.config = config
        self.config_path = Path(config_path)
        self.decoder_type = decoder_type or config.decoder.type
        self.snr_db = float(config.synthetic_demo.snr_db if snr_db is None else snr_db)
        if not math.isfinite(self.snr_db):
            raise ValueError("snr_db must be finite")
        self.seed = config.synthetic_demo.seed if seed is None else seed
        self.fullscreen = config.stimulus.fullscreen if fullscreen is None else fullscreen
        self.refresh_rate_override_hz = refresh_rate_hz
        self.output_dir = Path(output_dir)
        effective_confirmations = (
            config.synthetic_demo.confirmations_required if confirmations_required is None else confirmations_required
        )
        self.control_config: ControlConfig = replace(config.control, confirmations_required=effective_confirmations)
        schedule = build_trial_schedule(
            config.stimulus.frequencies_hz,
            config.stimulus.repetitions,
            config.stimulus.trial_order,
            config.stimulus.random_seed,
            config.commands,
        )
        self.schedule = schedule if max_trials is None else schedule[:max_trials]
        self.state_machine = SyntheticDemoStateMachine(self.schedule)
        self._drop_total = 0
        self._attempts: dict[int, int] = {}

    def _make_decoder(self) -> SSVEPDecoder:
        return FFTDecoder(self.config) if self.decoder_type == "fft" else CCADecoder(self.config)

    def run(self) -> Path:
        """Create a window only on explicit entry and run the simulated pipeline."""
        try:
            import psychopy
            from psychopy import event, visual
        except ImportError as exc:  # pragma: no cover - local GUI dependency
            raise RuntimeError("PsychoPy is required for the synthetic demo. Install the 'stimulus' extra.") from exc

        win = visual.Window(
            size=self.config.stimulus.window_size,
            fullscr=self.fullscreen,
            screen=self.config.stimulus.screen_index,
            color=self.config.stimulus.background_color,
            units="norm",
            waitBlanking=True,
        )
        logger: UnifiedSessionLogger | None = None
        archive: EEGWindowArchiveWriter | None = None
        dispatcher: SafeCommandDispatcher | None = None
        car = VirtualCarController()
        error: BaseException | None = None
        try:
            font = resolve_cjk_font(
                None,
                None,
                self.config.ui.cjk_font_file,
                self.config.ui.cjk_font_name,
                config_directory=self.config_path.expanduser().resolve().parent,
            )
            register_cjk_font(win, visual, font)
            refresh_hz, refresh_source = measure_refresh_rate(win, self.refresh_rate_override_hz)
            nominal_interval_s = 1.0 / refresh_hz
            win.recordFrameIntervals = True
            win.refreshThreshold = nominal_interval_s * self.config.stimulus.dropped_frame_threshold_ratio
            planned_frames = max(1, round(self.config.stimulus.trial_duration_s * refresh_hz))
            sequences = planned_flicker_sequences(self.config.stimulus.frequencies_hz, refresh_hz, planned_frames)
            estimates = {frequency: estimate_effective_frequency_hz(sequence, refresh_hz) for frequency, sequence in sequences.items()}
            effective_config = resolved_config_dict(
                self.config,
                decoder_type=self.decoder_type,
                synthetic_snr_db=self.snr_db,
                synthetic_seed=self.seed,
                confirmations_required=self.control_config.confirmations_required,
            )
            effective_config["stimulus"]["fullscreen"] = self.fullscreen
            effective_config["runtime"] = {"refresh_rate_hz": self.refresh_rate_override_hz}
            logger = UnifiedSessionLogger(
                self.output_dir,
                effective_config,
                {
                    "mode": "synthetic",
                    "synthetic_source_uses_trial_target": True,
                    "real_eeg_validated": False,
                    "real_car_validated": False,
                    "psychopy_version": psychopy.__version__,
                    "measured_refresh_rate_hz": refresh_hz,
                    "refresh_rate_source": refresh_source,
                    "decoder_type": self.decoder_type,
                    "snr_db": self.snr_db,
                    "synthetic_seed": self.seed,
                    "confirmations_required": self.control_config.confirmations_required,
                    "requested_frequencies_hz": list(self.config.stimulus.frequencies_hz),
                    "estimated_frequencies_hz": estimates,
                    "effective_frequency_estimation": "off_to_on_rising_edges / planned_duration",
                },
            )
            archive = EEGWindowArchiveWriter(
                logger.directory / "eeg_windows.npz",
                self.config.acquisition.channels,
                int(round(self.config.stimulus.trial_duration_s * self.config.acquisition.sample_rate_hz)),
            )
            print("SIMULATION MODE: EEG is synthesized from each trial target; this is not real human SSVEP.")
            print("No real EEG or real car has been validated.")
            print(f"PsychoPy {psychopy.__version__}; refresh {refresh_hz:.4f} Hz ({refresh_source})")
            print(f"Decoder: {self.decoder_type}; synthetic confirmations: {self.control_config.confirmations_required}")
            print(f"Output session: {logger.directory}")
            dispatcher = SafeCommandDispatcher(car, self.control_config, self.config.commands)
            coordinator = SyntheticTrialCoordinator(
                self.config, self._make_decoder(), dispatcher, snr_db=self.snr_db, seed=self.seed
            )
            view = _SyntheticDemoView(win, visual, self.config, font, estimates)
            self._run_loop(
                win, visual, event, view, logger, archive, coordinator, dispatcher, car, refresh_hz, nominal_interval_s
            )
            logger.event("session_finished" if self.state_machine.state is SyntheticDemoState.FINISHED else "session_stopped")
            return logger.directory
        except BaseException as exc:
            error = exc
            if logger is not None:
                logger.event("session_failed", error_type=type(exc).__name__, error_message=str(exc)[:500])
            raise
        finally:
            try:
                if dispatcher is not None:
                    dispatcher.close()
            finally:
                car.stop()
                if logger is not None:
                    logger.close(
                        status=("failed" if error else self.state_machine.state.value.lower()),
                        archived_eeg_windows=archive.count if archive is not None else 0,
                        final_car_state=car.state(),
                        error=error,
                    )
                win.close()

    def run_offline(self) -> Path:
        """Generate/decode/archive a deterministic synthetic session without PsychoPy.

        This is intentionally an acceptance/batch path: it does not claim to
        present a visual stimulus or create display-aligned markers.
        """
        effective_config = resolved_config_dict(
            self.config,
            decoder_type=self.decoder_type,
            synthetic_snr_db=self.snr_db,
            synthetic_seed=self.seed,
            confirmations_required=self.control_config.confirmations_required,
        )
        effective_config["runtime"] = {"no_gui": True}
        logger = UnifiedSessionLogger(
            self.output_dir,
            effective_config,
            {
                "mode": "synthetic",
                "synthetic_source_uses_trial_target": True,
                "real_eeg_validated": False,
                "real_car_validated": False,
                "decoder_type": self.decoder_type,
                "snr_db": self.snr_db,
                "synthetic_seed": self.seed,
                "confirmations_required": self.control_config.confirmations_required,
            },
        )
        archive = EEGWindowArchiveWriter(
            logger.directory / "eeg_windows.npz",
            self.config.acquisition.channels,
            int(round(self.config.stimulus.trial_duration_s * self.config.acquisition.sample_rate_hz)),
        )
        car = VirtualCarController()
        dispatcher = SafeCommandDispatcher(car, self.control_config, self.config.commands)
        coordinator = SyntheticTrialCoordinator(self.config, self._make_decoder(), dispatcher, snr_db=self.snr_db, seed=self.seed)
        error: BaseException | None = None
        try:
            logger.event("session_started")
            for trial in self.schedule:
                attempt_id = self._next_attempt_id(trial.trial_id)
                logger.event("cue_started", trial_id=trial.trial_id, attempt_id=attempt_id)
                logger.event("decode_started", trial_id=trial.trial_id, attempt_id=attempt_id)
                car_start = car.state()
                computation = coordinator.process(trial)
                result, decision = computation.result, computation.decision
                window_index = archive.add(
                    computation.window, trial_id=trial.trial_id, attempt_id=attempt_id,
                    target_frequency_hz=trial.target_frequency_hz, source_mode="synthetic",
                )
                execution_start_s = decision.timestamp_s if decision.action == "executed" else None
                execution_end_s = None
                if decision.action == "executed" and dispatcher.motion_deadline_s is not None:
                    deadline = dispatcher.motion_deadline_s
                    car.update(max(0.0, deadline - decision.timestamp_s))
                    stopped = dispatcher.tick(deadline)
                    execution_end_s = stopped.timestamp_s if stopped else deadline
                car.stop()
                logger.trial(
                    {
                        "trial_id": trial.trial_id, "attempt_id": attempt_id, "mode": "synthetic",
                        "trial_status": "completed", "target_frequency_hz": trial.target_frequency_hz,
                        "target_command": trial.command, "synthetic_seed": computation.synthetic_seed, "snr_db": self.snr_db,
                        "decoder_type": self.decoder_type,
                        "decoder_scores_json": json.dumps({str(key): value for key, value in result.scores.items()}),
                        "predicted_frequency_hz": result.predicted_frequency_hz,
                        "predicted_command": getattr(result.command, "value", result.command),
                        "effective_command": decision.effective_command.value, "confidence": result.confidence,
                        "correct": result.predicted_frequency_hz == trial.target_frequency_hz,
                        "decoder_latency_ms": computation.decoding_latency_ms,
                        "decoding_latency_ms": computation.decoding_latency_ms,
                        "generation_latency_ms": computation.generation_latency_ms,
                        "dropped_frames": 0, "dropped_frame_count": 0, "eeg_window_index": window_index,
                        "dispatcher_action": decision.action, "dispatcher_reason": decision.reason,
                        "confirmations_required": self.control_config.confirmations_required,
                        "execution_start_monotonic_s": execution_start_s, "execution_end_monotonic_s": execution_end_s,
                        "car_start_x": car_start["x"], "car_start_y": car_start["y"],
                        "car_start_heading": car_start["heading_degrees"],
                        "car_end_x": car.x, "car_end_y": car.y, "car_end_heading": car.heading_degrees,
                    }
                )
                logger.event("synthetic_eeg_generated", trial_id=trial.trial_id, attempt_id=attempt_id, eeg_window_index=window_index)
                logger.event("decode_completed", trial_id=trial.trial_id, attempt_id=attempt_id, prediction=result.predicted_frequency_hz)
                logger.event("command_dispatched", trial_id=trial.trial_id, attempt_id=attempt_id, action=decision.action, reason=decision.reason)
                logger.event("trial_completed", trial_id=trial.trial_id, attempt_id=attempt_id)
            logger.event("session_finished")
            return logger.directory
        except BaseException as exc:
            error = exc
            logger.event("session_failed", error_type=type(exc).__name__, error_message=str(exc)[:500])
            raise
        finally:
            dispatcher.close()
            car.stop()
            logger.close(
                status="failed" if error else "finished", archived_eeg_windows=archive.count,
                final_car_state=car.state(), error=error,
            )

    def _run_loop(
        self,
        win: Any,
        visual: Any,
        event: Any,
        view: _SyntheticDemoView,
        logger: UnifiedSessionLogger,
        archive: EEGWindowArchiveWriter,
        coordinator: SyntheticTrialCoordinator,
        dispatcher: SafeCommandDispatcher,
        car: VirtualCarController,
        refresh_hz: float,
        nominal_interval_s: float,
    ) -> None:
        cue_start_s: float | None = None
        rest_start_s: float | None = None
        stimulation_started_s: float | None = None
        marker: FlipMarkerRecorder | None = None
        scheduler: FlickerScheduler | None = None
        interval_start = 0
        record: dict[str, Any] | None = None
        computation: TrialComputation | None = None
        execution_rendered = False
        previous_frame_s = time.monotonic()

        while self.state_machine.state not in {SyntheticDemoState.FINISHED, SyntheticDemoState.STOPPED}:
            now_s = time.monotonic()
            frame_start_s = previous_frame_s
            delta_time_s = max(0.0, now_s - frame_start_s)
            previous_frame_s = now_s
            active_trial = (
                self.state_machine.current_trial if self.state_machine.state is not SyntheticDemoState.IDLE else None
            )
            keys = event.getKeys(keyList=["return", "p", "escape"])
            if "escape" in keys:
                active_attempt_id = record.get("attempt_id") if record else None
                dispatcher.stop(now_s, reason="escape")
                if record is not None:
                    self._finish_record(record, car, "stopped")
                    logger.trial(record)
                    record = None
                self.state_machine.stop()
                logger.event("session_stopped", trial_id=active_trial.trial_id if active_trial else None, attempt_id=active_attempt_id, reason="escape")
                continue
            if "p" in keys:
                if self.state_machine.state is SyntheticDemoState.PAUSED:
                    self.state_machine.resume()
                    cue_start_s = rest_start_s = stimulation_started_s = None
                    marker = None
                    computation = None
                    record = None
                    logger.event("resumed", trial_id=self.state_machine.current_trial.trial_id, attempt_id=None)
                elif self.state_machine.state not in {SyntheticDemoState.IDLE, SyntheticDemoState.FINISHED, SyntheticDemoState.STOPPED}:
                    if self.state_machine.state is not SyntheticDemoState.REST and record is not None:
                        self._finish_record(record, car, "aborted")
                        logger.trial(record)
                    dispatcher.stop(now_s, reason="paused")
                    was_rest = self.state_machine.state is SyntheticDemoState.REST
                    self.state_machine.pause()
                    logger.event("trial_aborted" if not was_rest else "paused", trial_id=self.state_machine.current_trial.trial_id, attempt_id=record.get("attempt_id") if record else None)
                    if not was_rest:
                        record = None
                continue

            state = self.state_machine.state
            trial = self.state_machine.current_trial if state is not SyntheticDemoState.IDLE else None
            if state is SyntheticDemoState.IDLE:
                car.stop()
                view.draw(
                    state=state, trial=None, target_states=static_target_states(self.config.stimulus.frequencies_hz), car=car,
                    result=None, decision=None, trial_total=len(self.schedule), dropped_frame_count=self._drop_total, remaining_s=None,
                )
                win.flip()
                if "return" in keys:
                    self.state_machine.start()
                    logger.event("session_started")
                continue
            if state is SyntheticDemoState.PAUSED:
                car.stop()
                view.draw(
                    state=state, trial=self.state_machine.current_trial,
                    target_states=static_target_states(self.config.stimulus.frequencies_hz), car=car, result=None, decision=None,
                    trial_total=len(self.schedule), dropped_frame_count=self._drop_total, remaining_s=None,
                )
                win.flip()
                continue
            assert trial is not None
            if state is SyntheticDemoState.CUE:
                car.stop()
                if cue_start_s is None:
                    cue_start_s = now_s
                    record = self._new_record(trial)
                    record["cue_start_monotonic_s"] = cue_start_s
                    logger.event("cue_started", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                view.draw(
                    state=state, trial=trial, target_states=static_target_states(self.config.stimulus.frequencies_hz), car=car,
                    result=None, decision=None, trial_total=len(self.schedule), dropped_frame_count=self._drop_total, remaining_s=None,
                )
                win.flip()
                if now_s - cue_start_s >= self.config.stimulus.cue_duration_s:
                    self.state_machine.cue_complete()
                    stimulation_started_s = None
                continue
            if state is SyntheticDemoState.STIMULATING:
                car.stop()
                if stimulation_started_s is None:
                    stimulation_started_s = now_s
                    marker = FlipMarkerRecorder()
                    scheduler = FlickerScheduler(self.config.stimulus.frequencies_hz, refresh_hz)
                    interval_start = len(win.frameIntervals)
                    logger.event("stimulation_started", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                assert marker is not None and scheduler is not None
                view.draw(
                    state=state, trial=trial, target_states=scheduler.advance(), car=car, result=None, decision=None,
                    trial_total=len(self.schedule), dropped_frame_count=self._drop_total, remaining_s=None,
                )
                if marker.stimulus_start_monotonic_s is None:
                    win.callOnFlip(marker.mark_stimulus_start)
                win.flip()
                if now_s - stimulation_started_s >= self.config.stimulus.trial_duration_s:
                    self.state_machine.stimulation_complete()
                    view.draw(
                        state=SyntheticDemoState.DECODING, trial=trial,
                        target_states=static_target_states(self.config.stimulus.frequencies_hz), car=car, result=None, decision=None,
                        trial_total=len(self.schedule), dropped_frame_count=self._drop_total, remaining_s=None,
                    )
                    win.callOnFlip(marker.mark_stimulus_end)
                    win.flip()  # A real DECODING frame precedes generation and decode work.
                    intervals = list(win.frameIntervals[interval_start:])
                    dropped = detect_dropped_frames(
                        intervals, nominal_interval_s, self.config.stimulus.dropped_frame_threshold_ratio
                    )
                    self._drop_total += len(dropped)
                    assert record is not None
                    trial_marker = marker.marker(trial)
                    record["stimulus_start_monotonic_s"] = trial_marker.stimulus_start_monotonic_s
                    record["stimulus_end_monotonic_s"] = trial_marker.stimulus_end_monotonic_s
                    record["dropped_frame_count"] = record["dropped_frames"] = len(dropped)
                    logger.frame_intervals(trial.trial_id, record["attempt_id"], intervals, dropped)
                    logger.event("stimulation_completed", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                continue
            if state is SyntheticDemoState.DECODING:
                assert record is not None
                car_snapshot = car.state()
                logger.event("decode_started", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                computation = coordinator.process(trial)
                result, decision = computation.result, computation.decision
                record["eeg_window_index"] = archive.add(
                    computation.window,
                    trial_id=trial.trial_id,
                    attempt_id=record["attempt_id"],
                    target_frequency_hz=trial.target_frequency_hz,
                    source_mode="synthetic",
                )
                record.update(
                    synthetic_seed=computation.synthetic_seed,
                    snr_db=self.snr_db,
                    decoder_type=self.decoder_type,
                    decoder_scores_json=json.dumps({str(key): value for key, value in result.scores.items()}),
                    predicted_frequency_hz=result.predicted_frequency_hz,
                    predicted_command=getattr(result.command, "value", result.command),
                    confidence=result.confidence,
                    dispatcher_action=decision.action,
                    dispatcher_reason=decision.reason,
                    effective_command=decision.effective_command.value,
                    confirmations_required=self.control_config.confirmations_required,
                    generation_latency_ms=computation.generation_latency_ms,
                    decoding_latency_ms=computation.decoding_latency_ms,
                    decoder_latency_ms=computation.decoding_latency_ms,
                    car_start_x=car_snapshot["x"], car_start_y=car_snapshot["y"],
                    car_start_heading=car_snapshot["heading_degrees"],
                    correct=result.predicted_frequency_hz == trial.target_frequency_hz,
                )
                self.state_machine.decoding_complete()
                execution_rendered = False
                if decision.action == "executed":
                    record["execution_start_monotonic_s"] = decision.timestamp_s
                else:
                    record["execution_end_monotonic_s"] = decision.timestamp_s
                logger.event("synthetic_eeg_generated", trial_id=trial.trial_id, attempt_id=record["attempt_id"], eeg_window_index=record["eeg_window_index"])
                logger.event("decode_completed", trial_id=trial.trial_id, attempt_id=record["attempt_id"], prediction=result.predicted_frequency_hz)
                logger.event("command_dispatched", trial_id=trial.trial_id, attempt_id=record["attempt_id"], action=decision.action, reason=decision.reason)
                continue
            if state is SyntheticDemoState.EXECUTING:
                assert computation is not None and record is not None
                result, decision = computation.result, computation.decision
                deadline = dispatcher.motion_deadline_s
                allowed_delta_s = delta_time_s
                if deadline is not None:
                    execution_start_s = float(record.get("execution_start_monotonic_s", now_s))
                    allowed_delta_s = max(
                        0.0,
                        min(now_s, deadline) - max(frame_start_s, execution_start_s),
                    )
                car.update(allowed_delta_s)
                timeout_decision = dispatcher.tick(now_s)
                if timeout_decision is not None:
                    decision = timeout_decision
                    computation = replace(computation, decision=decision)
                    record["dispatcher_action"] = decision.action
                    record["dispatcher_reason"] = decision.reason
                    record["effective_command"] = decision.effective_command.value
                    record["execution_end_monotonic_s"] = decision.timestamp_s
                    logger.event("car_stopped", trial_id=trial.trial_id, attempt_id=record["attempt_id"], reason=decision.reason)
                remaining_s = None if deadline is None else max(0.0, deadline - now_s)
                view.draw(
                    state=state, trial=trial, target_states=static_target_states(self.config.stimulus.frequencies_hz), car=car,
                    result=result, decision=decision, trial_total=len(self.schedule), dropped_frame_count=self._drop_total,
                    remaining_s=remaining_s,
                )
                win.flip()
                if execution_rendered and (timeout_decision is not None or not car.is_moving):
                    car.stop()
                    record.setdefault("execution_end_monotonic_s", time.monotonic())
                    self.state_machine.executing_complete()
                    rest_start_s = None
                execution_rendered = True
                continue
            if state is SyntheticDemoState.REST:
                car.stop()
                if rest_start_s is None:
                    rest_start_s = now_s
                    logger.event("rest_started", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                assert record is not None
                result = computation.result if computation else None
                decision = computation.decision if computation else None
                view.draw(
                    state=state, trial=trial, target_states=static_target_states(self.config.stimulus.frequencies_hz), car=car,
                    result=result, decision=decision, trial_total=len(self.schedule), dropped_frame_count=self._drop_total, remaining_s=None,
                )
                win.flip()
                if now_s - rest_start_s >= self.config.stimulus.rest_duration_s:
                    self._finish_record(record, car, "completed")
                    logger.trial(record)
                    logger.event("trial_completed", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                    self.state_machine.rest_complete()
                    cue_start_s = rest_start_s = stimulation_started_s = None
                    marker = scheduler = computation = record = None

    def _new_record(self, trial: ScheduledTrial) -> dict[str, Any]:
        return {
            "trial_id": trial.trial_id,
            "attempt_id": self._next_attempt_id(trial.trial_id),
            "mode": "synthetic",
            "trial_status": "running",
            "target_frequency_hz": trial.target_frequency_hz,
            "target_command": trial.command,
            "confirmations_required": self.control_config.confirmations_required,
            "snr_db": self.snr_db,
        }

    def _next_attempt_id(self, trial_id: int) -> int:
        attempt_id = self._attempts.get(trial_id, 0)
        self._attempts[trial_id] = attempt_id + 1
        return attempt_id

    @staticmethod
    def _finish_record(record: dict[str, Any], car: VirtualCarController, status: str) -> None:
        state = car.state()
        record.update(
            trial_status=status,
            car_end_x=state["x"], car_end_y=state["y"], car_end_heading=state["heading_degrees"],
        )
