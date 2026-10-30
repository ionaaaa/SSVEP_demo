"""PsychoPy SSVEP live-trial runner backed by the omniBCI WebSocket source."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from .config import ControllerConfig, DemoConfig
from .control import DispatchDecision, SafeCommandDispatcher
from .controller_factory import (
    ControllerFaultError,
    ControllerRuntime,
    ControllerStartupError,
    create_controller,
    drain_controller_events,
    finalize_session,
    log_controller_connected,
)
from .decoders import CCADecoder, FFTDecoder, SSVEPDecoder
from .eeg_archive import EEGWindowArchiveWriter
from .live_source import LiveEEGSource, LiveSourceStatus, OmniBCIWebSocketSource
from .protocol import DecodeResult, EEGWindow
from .session_logging import UnifiedSessionLogger, resolved_config_dict
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


class LiveDemoState(str, Enum):
    IDLE = "IDLE"
    CUE = "CUE"
    STIMULATING = "STIMULATING"
    COLLECTING_LIVE_WINDOW = "COLLECTING_LIVE_WINDOW"
    DECODING = "DECODING"
    EXECUTING = "EXECUTING"
    REST = "REST"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    STOPPED = "STOPPED"


def _window_close_requested(win: Any) -> bool:
    if bool(getattr(win, "closed", False)):
        return True
    handle = getattr(win, "winHandle", None)
    return bool(
        handle is not None
        and (getattr(handle, "has_exit", False) or getattr(handle, "_closed", False))
    )


class LiveDemoStateMachine:
    def __init__(self, schedule: Sequence[ScheduledTrial]) -> None:
        if not schedule:
            raise ValueError("schedule must not be empty")
        self.schedule = list(schedule)
        self.index = 0
        self.state = LiveDemoState.IDLE
        self._resume_state: LiveDemoState | None = None

    @property
    def current_trial(self) -> ScheduledTrial:
        return self.schedule[self.index]

    def start(self) -> None:
        self._require(LiveDemoState.IDLE)
        self.state = LiveDemoState.CUE

    def cue_complete(self) -> None:
        self._require(LiveDemoState.CUE)
        self.state = LiveDemoState.STIMULATING

    def stimulation_complete(self) -> None:
        self._require(LiveDemoState.STIMULATING)
        self.state = LiveDemoState.COLLECTING_LIVE_WINDOW

    def collection_complete(self) -> None:
        self._require(LiveDemoState.COLLECTING_LIVE_WINDOW)
        self.state = LiveDemoState.DECODING

    def collection_failed(self) -> None:
        self._require(LiveDemoState.COLLECTING_LIVE_WINDOW)
        self.state = LiveDemoState.REST

    def decoding_complete(self) -> None:
        self._require(LiveDemoState.DECODING)
        self.state = LiveDemoState.EXECUTING

    def executing_complete(self) -> None:
        self._require(LiveDemoState.EXECUTING)
        self.state = LiveDemoState.REST

    def rest_complete(self) -> None:
        self._require(LiveDemoState.REST)
        self.index += 1
        self.state = LiveDemoState.FINISHED if self.index == len(self.schedule) else LiveDemoState.CUE

    def pause(self) -> None:
        if self.state not in {
            LiveDemoState.CUE,
            LiveDemoState.STIMULATING,
            LiveDemoState.COLLECTING_LIVE_WINDOW,
            LiveDemoState.DECODING,
            LiveDemoState.EXECUTING,
            LiveDemoState.REST,
        }:
            raise ValueError("only an active live-demo state can be paused")
        self._resume_state = self.state
        self.state = LiveDemoState.PAUSED

    def resume(self) -> None:
        self._require(LiveDemoState.PAUSED)
        self.state = LiveDemoState.REST if self._resume_state is LiveDemoState.REST else LiveDemoState.CUE
        self._resume_state = None

    def stop(self) -> None:
        self.state = LiveDemoState.STOPPED

    def _require(self, expected: LiveDemoState) -> None:
        if self.state is not expected:
            raise ValueError(f"expected state {expected.value}, got {self.state.value}")


@dataclass(frozen=True)
class LiveTrialComputation:
    result: DecodeResult
    decision: DispatchDecision
    decoding_latency_ms: float


class LiveTrialCoordinator:
    """Decode and dispatch one already-complete unlabeled live EEG window."""

    def __init__(self, decoder: SSVEPDecoder, dispatcher: SafeCommandDispatcher) -> None:
        self.decoder = decoder
        self.dispatcher = dispatcher

    def process(self, window: EEGWindow) -> LiveTrialComputation:
        started_s = time.monotonic()
        result = self.decoder.decode(window)
        latency_ms = (time.monotonic() - started_s) * 1000.0
        return LiveTrialComputation(result, self.dispatcher.submit(result), latency_ms)


class _LiveDemoView:
    def __init__(
        self,
        win: Any,
        visual: Any,
        config: DemoConfig,
        font: CJKFontSpec,
        estimates: dict[float, float],
    ) -> None:
        self.config = config
        self.targets: dict[float, dict[str, Any]] = {}
        for frequency, position in zip(
            config.stimulus.frequencies_hz, target_layout(len(config.stimulus.frequencies_hz))
        ):
            x, y = (-0.55 + position[0] * 0.35, position[1] * 0.55)
            self.targets[frequency] = {
                "rect": visual.Rect(win, pos=(x, y), width=0.25, height=0.20, fillColor=config.stimulus.stimulus_off_color),
                "freq": visual.TextStim(win, text=f"{frequency:g} Hz", pos=(x, y + 0.035), height=0.035, font=font.name),
                "command": visual.TextStim(win, text=config.commands[frequency].value, pos=(x, y - 0.025), height=0.027, font=font.name),
                "estimate": visual.TextStim(win, text=f"实际 {estimates[frequency]:.2f} Hz", pos=(x, y - 0.075), height=0.021, font=font.name),
            }
        self.banner = visual.TextStim(
            win,
            text="LIVE EEG MODE / 实时脑电模式\nomniBCI WebSocket；人体诱发与设备级同步仍需现场验证",
            pos=(0, 0.91),
            height=0.027,
            font=font.name,
            wrapWidth=1.85,
            alignText="center",
        )
        self.status_text = visual.TextStim(win, pos=(-0.45, -0.70), height=0.028, font=font.name, wrapWidth=1.05)
        self.stream_text = visual.TextStim(win, pos=(-0.45, -0.84), height=0.021, font=font.name, wrapWidth=1.05)
        self.results = visual.TextStim(win, pos=(0.48, 0.48), height=0.027, font=font.name, wrapWidth=0.78, alignText="left")
        self.car_rect = visual.Rect(win, width=0.11, height=0.07, fillColor="royalblue", lineColor="white")
        self.heading = visual.Line(win, start=(0.45, 0), end=(0.45, 0.1), lineColor="yellow", lineWidth=3)
        self.car_text = visual.TextStim(win, pos=(0.48, -0.34), height=0.025, font=font.name, wrapWidth=0.78)

    def draw(
        self,
        *,
        state: LiveDemoState,
        trial: ScheduledTrial | None,
        target_states: dict[float, bool],
        controller: ControllerRuntime,
        result: DecodeResult | None,
        decision: DispatchDecision | None,
        trial_total: int,
        dropped_frames: int,
        live: LiveSourceStatus,
        remaining_s: float | None = None,
    ) -> None:
        self.banner.draw()
        trial_text = "等待 Enter 开始" if trial is None else f"Trial {trial.trial_id + 1}/{trial_total}: {trial.target_frequency_hz:g} Hz / {trial.command}"
        self.status_text.text = f"{state.value}\n{trial_text}\n掉帧警告数：{dropped_frames}"
        metadata = live.metadata
        rate = "--" if metadata is None else f"{metadata.sample_rate_hz:g} Hz"
        channels = "--" if metadata is None else ", ".join(metadata.mapped_channel_names)
        sequence = "--" if live.latest_sequence is None else str(live.latest_sequence)
        anomaly = live.recent_anomaly or "--"
        self.stream_text.text = (
            f"Live EEG: {live.connection_state}\n采样率：{rate}\n通道：{channels}\n"
            f"Sequence: {sequence}\nAlignment: {live.alignment_mode_used or live.alignment_mode_requested or '--'}\n"
            f"Samples: {live.collected_samples} / {live.expected_samples}\n最近异常：{anomaly}"
        )
        self.status_text.draw()
        self.stream_text.draw()
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
        car = controller.virtual_car
        suffix = "" if remaining_s is None else f" / {remaining_s:.2f}s"
        if car is not None:
            car_x, car_y = (0.48 + car.x * 0.35, -0.02 + car.y * 0.38)
            angle = math.radians(car.heading_degrees)
            self.car_rect.pos = (car_x, car_y)
            self.heading.start = (car_x, car_y)
            self.heading.end = (car_x + 0.11 * math.cos(angle), car_y + 0.11 * math.sin(angle))
            self.car_rect.draw()
            self.heading.draw()
            active = car.active_command.value if car.active_command else "STOPPED"
            self.car_text.text = f"Controller: Virtual\n({car.x:.2f}, {car.y:.2f})  {car.heading_degrees:.0f}°\n{active}{suffix}"
        else:
            requested = "--" if decision is None else decision.effective_command.value
            fault = "" if controller.last_error is None else f"\nFault: {controller.last_error}"
            self.car_text.text = (
                f"Controller: Bluetooth\nCar device: {controller.device_name or controller.device_id}\n"
                f"Car state: {controller.state}\nLast requested: {requested}{suffix}{fault}"
            )
        self.car_text.draw()


class SSVEPLiveDemoRunner:
    def __init__(
        self,
        config: DemoConfig,
        config_path: str | Path,
        *,
        server_url: str,
        decoder_type: str | None = None,
        fullscreen: bool | None = None,
        refresh_rate_hz: float | None = None,
        output_dir: str | Path = "outputs/ssvep_demo",
        max_trials: int | None = None,
        confirmations_required: int | None = None,
        live_source: LiveEEGSource | None = None,
        controller_runtime_factory: Callable[[], ControllerRuntime] | None = None,
    ) -> None:
        if config.stimulus.trial_duration_s != 4.0:
            raise ValueError("live mode requires stimulus.trial_duration_s to be exactly 4.0")
        if config.acquisition.sample_rate_hz != 250.0 or config.live.trial_samples != 1000:
            raise ValueError("live mode requires 250 Hz and exactly 1000 samples")
        if decoder_type is not None and decoder_type not in {"fft", "cca"}:
            raise ValueError("decoder_type must be 'fft' or 'cca'")
        if max_trials is not None and (isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials <= 0):
            raise ValueError("max_trials must be a positive integer")
        self.config = config
        self.config_path = Path(config_path)
        self.server_url = server_url
        self.decoder_type = decoder_type or config.decoder.type
        self.fullscreen = config.stimulus.fullscreen if fullscreen is None else fullscreen
        self.refresh_rate_override_hz = refresh_rate_hz
        self.output_dir = Path(output_dir)
        control = config.control if confirmations_required is None else replace(
            config.control, confirmations_required=confirmations_required
        )
        self.control_config = control
        schedule = build_trial_schedule(
            config.stimulus.frequencies_hz,
            config.stimulus.repetitions,
            config.stimulus.trial_order,
            config.stimulus.random_seed,
            config.commands,
        )
        self.schedule = schedule if max_trials is None else schedule[:max_trials]
        self.state_machine = LiveDemoStateMachine(self.schedule)
        self.live_source = live_source or OmniBCIWebSocketSource(server_url, config.live)
        self._attempts: dict[int, int] = {}
        self._drop_total = 0
        self._controller_runtime_factory = controller_runtime_factory

    def _decoder(self) -> SSVEPDecoder:
        return FFTDecoder(self.config) if self.decoder_type == "fft" else CCADecoder(self.config)

    def _create_controller_runtime(self, logger: UnifiedSessionLogger | None = None) -> ControllerRuntime:
        if self._controller_runtime_factory is not None:
            started = time.monotonic()
            if logger is not None:
                logger.event("controller_connect_started", controller_type=self.config.controller.type)
            try:
                runtime = self._controller_runtime_factory()
            except BaseException as exc:
                if logger is not None:
                    logger.event(
                        "controller_connect_failed",
                        controller_type=self.config.controller.type,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:500],
                    )
                raise
            runtime.connected = runtime.virtual_car is not None or runtime.state == "READY"
            runtime.connect_latency_ms = (time.monotonic() - started) * 1000.0
            runtime.initial_state = runtime.initial_state or runtime.state
            return runtime
        return create_controller(self.config.controller, mode="live", logger=logger)

    def _create_session(self) -> tuple[UnifiedSessionLogger, EEGWindowArchiveWriter]:
        effective_config = resolved_config_dict(
            self.config,
            decoder_type=self.decoder_type,
            confirmations_required=self.control_config.confirmations_required,
        )
        effective_config["stimulus"]["fullscreen"] = self.fullscreen
        effective_config["runtime"] = {
            "refresh_rate_hz": self.refresh_rate_override_hz,
            "source": "omnibci-websocket",
            "server_url": self.server_url,
        }
        bluetooth = self.config.controller.bluetooth
        logger = UnifiedSessionLogger(
            self.output_dir,
            effective_config,
            {
                "mode": "live",
                "source": "omnibci-websocket",
                "server_url": self.server_url,
                "decoder_type": self.decoder_type,
                "real_eeg_software_path_complete": True,
                "human_ssvep_validated": False,
                "hardware_sync_validated": False,
                "real_car_validated": False,
                "controller_type": self.config.controller.type,
                "car_device_name": bluetooth.device_name if self.config.controller.type == "bluetooth" else None,
                "car_device_id": bluetooth.device_id if self.config.controller.type == "bluetooth" else None,
                "car_connect_timeout_s": bluetooth.connect_timeout_s if self.config.controller.type == "bluetooth" else None,
                "controller_connected": False,
                "controller_initial_state": "DISCONNECTED" if self.config.controller.type == "bluetooth" else "READY",
            },
        )
        archive = EEGWindowArchiveWriter(
            logger.directory / "eeg_windows.npz",
            self.config.acquisition.channels,
            self.config.live.trial_samples,
        )
        return logger, archive

    def run(self) -> Path:
        logger, archive = self._create_session()
        controller_runtime: ControllerRuntime | None = None
        dispatcher: SafeCommandDispatcher | None = None
        win: Any | None = None
        error: BaseException | None = None
        failure_stage: str | None = None
        failure_reason: str | None = None
        status = "failed"
        try:
            failure_stage = "controller_connect"
            controller_runtime = self._create_controller_runtime(logger)
            log_controller_connected(logger, controller_runtime)
            failure_stage = "psychopy_import"
            try:
                import psychopy
                from psychopy import event, visual
            except ImportError as exc:  # pragma: no cover - GUI dependency
                raise RuntimeError("PsychoPy is required for live mode. Install the 'stimulus' extra.") from exc
            failure_stage = "window_create"
            win = visual.Window(
                size=self.config.stimulus.window_size,
                fullscr=self.fullscreen,
                screen=self.config.stimulus.screen_index,
                color=self.config.stimulus.background_color,
                units="norm",
                waitBlanking=True,
            )
            failure_stage = "runtime_setup"
            font = resolve_cjk_font(
                None, None, self.config.ui.cjk_font_file, self.config.ui.cjk_font_name,
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
            logger.update_metadata(
                psychopy_version=psychopy.__version__,
                measured_refresh_rate_hz=refresh_hz,
                refresh_rate_source=refresh_source,
            )
            dispatcher = SafeCommandDispatcher(controller_runtime.controller, self.control_config, self.config.commands)
            coordinator = LiveTrialCoordinator(self._decoder(), dispatcher)
            view = _LiveDemoView(win, visual, self.config, font, estimates)
            self.live_source.start()
            logger.event("live_source_started", server_url=self.server_url)
            logger.event("session_running", session_status="running")
            failure_stage = "trial_runtime"
            self._run_loop(
                win, event, view, logger, archive, coordinator, dispatcher, controller_runtime, refresh_hz, nominal_interval_s
            )
            failed_rows = [row for row in logger.rows if row.get("trial_status") == "failed"]
            if failed_rows:
                status = "failed"
                failure_stage = "live_eeg"
                failure_reason = str(
                    failed_rows[-1].get("live_window_failure_reason") or "one or more live EEG trials failed"
                )[:500]
            else:
                status = "completed" if self.state_machine.state is LiveDemoState.FINISHED else "aborted"
            return logger.directory
        except KeyboardInterrupt as exc:
            error = exc
            status = "aborted"
            failure_stage = "user_interrupt"
            raise
        except BaseException as exc:
            error = exc
            status = "failed"
            if isinstance(exc, ControllerStartupError):
                controller_runtime = controller_runtime or exc.runtime
                failure_stage = "controller_connect"
            elif isinstance(exc, ControllerFaultError):
                failure_stage = "controller_fault"
            else:
                failure_stage = getattr(exc, "session_failure_stage", failure_stage)
            raise
        finally:
            cleanup_error = finalize_session(
                logger=logger,
                archive=archive,
                runtime=controller_runtime,
                dispatcher=dispatcher,
                status=status,
                error=error,
                failure_stage=failure_stage,
                failure_reason=failure_reason,
                live_source=self.live_source,
                window=win,
            )
            if error is None and cleanup_error is not None:
                raise cleanup_error

    def _run_loop(
        self,
        win: Any,
        event: Any,
        view: _LiveDemoView,
        logger: UnifiedSessionLogger,
        archive: EEGWindowArchiveWriter,
        coordinator: LiveTrialCoordinator,
        dispatcher: SafeCommandDispatcher,
        controller: ControllerRuntime,
        refresh_hz: float,
        nominal_interval_s: float,
    ) -> None:
        # Preserve the pre-integration internal test/injection contract while
        # normal runners always pass the centralized runtime wrapper.
        if not isinstance(controller, ControllerRuntime):
            controller = ControllerRuntime(controller, ControllerConfig(), virtual_car=controller)
        cue_start_s: float | None = None
        rest_start_s: float | None = None
        marker: FlipMarkerRecorder | None = None
        scheduler: FlickerScheduler | None = None
        interval_start = 0
        record: dict[str, Any] | None = None
        computation: LiveTrialComputation | None = None
        execution_rendered = False
        previous_frame_s = time.monotonic()

        while self.state_machine.state not in {LiveDemoState.FINISHED, LiveDemoState.STOPPED}:
            drain_controller_events(controller, logger)
            try:
                controller.ensure_healthy()
            except BaseException as exc:
                self.live_source.abort_trial("controller_fault")
                try:
                    dispatcher.stop(reason="controller_fault")
                except BaseException:
                    pass
                if record is not None:
                    self._finish_record(
                        record, controller, self.live_source.status(), "failed", "controller_fault"
                    )
                    logger.trial(record)
                    record = None
                if not callable(getattr(controller.controller, "poll_events", None)):
                    logger.event(
                        "controller_fault", controller_state=controller.state, error_message=str(exc)[:500]
                    )
                raise
            self.live_source.poll()
            for source_event in self.live_source.take_events():
                exact_event_names = {
                    "stimulus_flip_sequence_snapshotted",
                    "trigger_request_queued",
                    "trigger_request_started",
                    "trigger_response_received",
                    "trigger_response_discarded",
                    "trigger_alignment_ready",
                    "trigger_alignment_failed",
                    "live_window_ready",
                }
                event_payload: dict[str, Any] = {
                    "source_monotonic_s": source_event.monotonic_s,
                    "reason": source_event.reason,
                    "detail": source_event.detail,
                }
                if source_event.detail:
                    try:
                        structured_detail = json.loads(source_event.detail)
                    except (json.JSONDecodeError, TypeError):
                        structured_detail = None
                    if isinstance(structured_detail, dict):
                        event_payload.update(structured_detail)
                logger.event(
                    source_event.kind if source_event.kind in exact_event_names else f"live_{source_event.kind}",
                    **event_payload,
                )
            live_status = self.live_source.status()
            now_s = time.monotonic()
            frame_start_s = previous_frame_s
            delta_s = max(0.0, now_s - frame_start_s)
            previous_frame_s = now_s
            keys = event.getKeys(keyList=["return", "p", "escape"])
            close_requested = _window_close_requested(win)
            if "escape" in keys or close_requested:
                stop_reason = "window_closed" if close_requested else "escape"
                self.live_source.abort_trial(stop_reason)
                dispatcher.stop(now_s, reason=stop_reason)
                controller.stop_virtual()
                if record is not None:
                    self._finish_record(record, controller, self.live_source.status(), "stopped")
                    logger.trial(record)
                    record = None
                self.state_machine.stop()
                logger.event("session_stop_requested", reason=stop_reason)
                continue
            if "p" in keys:
                if self.state_machine.state is LiveDemoState.PAUSED:
                    self.state_machine.resume()
                    cue_start_s = rest_start_s = None
                    marker = scheduler = None
                    computation = record = None
                    logger.event("resumed")
                elif self.state_machine.state not in {LiveDemoState.IDLE, LiveDemoState.FINISHED, LiveDemoState.STOPPED}:
                    was_rest = self.state_machine.state is LiveDemoState.REST
                    self.live_source.abort_trial("paused")
                    dispatcher.stop(now_s, reason="paused")
                    controller.stop_virtual()
                    if not was_rest and record is not None:
                        self._finish_record(record, controller, self.live_source.status(), "aborted", "paused")
                        logger.trial(record)
                        record = None
                    self.state_machine.pause()
                    logger.event("paused" if was_rest else "trial_aborted", reason="paused")
                continue

            state = self.state_machine.state
            trial = self.state_machine.current_trial if state is not LiveDemoState.IDLE else None
            if state is LiveDemoState.IDLE:
                controller.stop_virtual()
                view.draw(
                    state=state, trial=None, target_states=static_target_states(self.config.stimulus.frequencies_hz),
                    controller=controller, result=None, decision=None, trial_total=len(self.schedule),
                    dropped_frames=self._drop_total, live=live_status,
                )
                win.flip()
                if "return" in keys:
                    self.state_machine.start()
                    logger.event("session_started")
                continue
            if state is LiveDemoState.PAUSED:
                controller.stop_virtual()
                view.draw(
                    state=state, trial=self.state_machine.current_trial,
                    target_states=static_target_states(self.config.stimulus.frequencies_hz), controller=controller,
                    result=None, decision=None, trial_total=len(self.schedule), dropped_frames=self._drop_total,
                    live=live_status,
                )
                win.flip()
                continue
            assert trial is not None
            if state is LiveDemoState.CUE:
                controller.stop_virtual()
                if cue_start_s is None:
                    cue_start_s = now_s
                    record = self._new_record(trial)
                    record["cue_start_monotonic_s"] = cue_start_s
                    logger.event("cue_started", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                view.draw(
                    state=state, trial=trial, target_states=static_target_states(self.config.stimulus.frequencies_hz),
                    controller=controller, result=None, decision=None, trial_total=len(self.schedule),
                    dropped_frames=self._drop_total, live=live_status,
                )
                win.flip()
                if now_s - cue_start_s >= self.config.stimulus.cue_duration_s:
                    self.state_machine.cue_complete()
                    marker = scheduler = None
                continue
            if state is LiveDemoState.STIMULATING:
                controller.stop_virtual()
                if marker is None:
                    marker = FlipMarkerRecorder()
                    scheduler = FlickerScheduler(self.config.stimulus.frequencies_hz, refresh_hz)
                    interval_start = len(win.frameIntervals)
                    assert record is not None
                    logger.event("stimulation_started", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                assert scheduler is not None and record is not None
                view.draw(
                    state=state, trial=trial, target_states=scheduler.advance(), controller=controller, result=None, decision=None,
                    trial_total=len(self.schedule), dropped_frames=self._drop_total, live=live_status,
                )
                if marker.stimulus_start_monotonic_s is None:
                    def begin_live_trial() -> None:
                        assert marker is not None
                        marker.mark_stimulus_start()
                        assert marker.stimulus_start_monotonic_s is not None
                        self.live_source.mark_stimulus_flip(
                            trial_id=trial.trial_id,
                            attempt_id=record["attempt_id"],
                            stimulus_start_monotonic_s=marker.stimulus_start_monotonic_s,
                        )

                    win.callOnFlip(begin_live_trial)
                win.flip()
                if (
                    marker.stimulus_start_monotonic_s is not None
                    and time.monotonic() - marker.stimulus_start_monotonic_s >= self.config.stimulus.trial_duration_s
                ):
                    self.state_machine.stimulation_complete()
                    view.draw(
                        state=LiveDemoState.COLLECTING_LIVE_WINDOW, trial=trial,
                        target_states=static_target_states(self.config.stimulus.frequencies_hz), controller=controller,
                        result=None, decision=None, trial_total=len(self.schedule), dropped_frames=self._drop_total,
                        live=self.live_source.status(),
                    )
                    win.callOnFlip(marker.mark_stimulus_end)
                    win.flip()
                    intervals = list(win.frameIntervals[interval_start:])
                    dropped = detect_dropped_frames(
                        intervals, nominal_interval_s, self.config.stimulus.dropped_frame_threshold_ratio
                    )
                    self._drop_total += len(dropped)
                    trial_marker = marker.marker(trial)
                    record["stimulus_start_monotonic_s"] = trial_marker.stimulus_start_monotonic_s
                    record["stimulus_end_monotonic_s"] = trial_marker.stimulus_end_monotonic_s
                    record["dropped_frames"] = record["dropped_frame_count"] = len(dropped)
                    logger.frame_intervals(trial.trial_id, record["attempt_id"], intervals, dropped)
                    logger.event("stimulation_completed", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                continue
            if state is LiveDemoState.COLLECTING_LIVE_WINDOW:
                assert record is not None
                status = self.live_source.status()
                window = self.live_source.try_take_completed_window()
                if window is not None:
                    self._apply_live_fields(record, status, "complete", None)
                    record["eeg_window_index"] = archive.add(
                        window,
                        trial_id=trial.trial_id,
                        attempt_id=record["attempt_id"],
                        target_frequency_hz=trial.target_frequency_hz,
                        source_mode="live",
                        start_sequence=status.start_sequence,
                        end_sequence=status.end_sequence,
                        data_unit="uV",
                        alignment_mode=status.alignment_mode_used or "",
                    )
                    record["_window"] = window
                    self.state_machine.collection_complete()
                    logger.event("live_window_completed", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                    continue
                failure = status.trial_failure_reason
                end_s = record.get("stimulus_end_monotonic_s")
                if failure is None and isinstance(end_s, (int, float)) and now_s - end_s > self.config.live.max_collection_wait_s:
                    self.live_source.abort_trial("collection_timeout")
                    status = self.live_source.status()
                    failure = "collection_timeout"
                view.draw(
                    state=state, trial=trial, target_states=static_target_states(self.config.stimulus.frequencies_hz),
                    controller=controller, result=None, decision=None, trial_total=len(self.schedule), dropped_frames=self._drop_total,
                    live=status,
                )
                win.flip()
                if failure is not None:
                    dispatcher.stop(now_s, reason=failure)
                    controller.stop_virtual()
                    self._apply_live_fields(record, status, "incomplete_live_window", failure)
                    self._finish_record(record, controller, status, "failed", failure)
                    self.state_machine.collection_failed()
                    rest_start_s = None
                    logger.event("live_window_failed", trial_id=trial.trial_id, attempt_id=record["attempt_id"], reason=failure)
                continue
            if state is LiveDemoState.DECODING:
                assert record is not None
                window = record.pop("_window")
                car_start = controller.position()
                logger.event("decode_started", trial_id=trial.trial_id, attempt_id=record["attempt_id"])
                try:
                    computation = coordinator.process(window)
                except BaseException as exc:
                    try:
                        dispatcher.stop(reason="decoder_failure")
                    except BaseException:
                        pass
                    self.live_source.abort_trial("decoder_failure")
                    self._finish_record(
                        record, controller, self.live_source.status(), "failed", "decoder_failure"
                    )
                    logger.trial(record)
                    record = None
                    try:
                        controller.ensure_healthy()
                    except BaseException as fault:
                        drain_controller_events(controller, logger)
                        if not callable(getattr(controller.controller, "poll_events", None)):
                            logger.event(
                                "controller_fault",
                                controller_state=controller.state,
                                error_message=str(fault)[:500],
                            )
                        raise fault from exc
                    try:
                        setattr(exc, "session_failure_stage", "decoder")
                    except BaseException:
                        pass
                    raise
                result, decision = computation.result, computation.decision
                record.update(
                    decoder_type=self.decoder_type,
                    decoder_scores_json=json.dumps({str(key): value for key, value in result.scores.items()}),
                    predicted_frequency_hz=result.predicted_frequency_hz,
                    predicted_command=getattr(result.command, "value", result.command),
                    effective_command=decision.effective_command.value,
                    confidence=result.confidence,
                    correct=result.predicted_frequency_hz == trial.target_frequency_hz,
                    decoder_latency_ms=computation.decoding_latency_ms,
                    decoding_latency_ms=computation.decoding_latency_ms,
                    dispatcher_action=decision.action,
                    dispatcher_reason=decision.reason,
                    car_start_x=car_start["x"], car_start_y=car_start["y"],
                    car_start_heading=car_start["heading_degrees"],
                    **controller.trial_controller_fields(decision),
                )
                drain_controller_events(controller, logger)
                if decision.action == "executed":
                    record["execution_start_monotonic_s"] = decision.timestamp_s
                else:
                    record["execution_end_monotonic_s"] = decision.timestamp_s
                self.state_machine.decoding_complete()
                execution_rendered = False
                logger.event("decode_completed", trial_id=trial.trial_id, attempt_id=record["attempt_id"], prediction=result.predicted_frequency_hz)
                logger.event("command_dispatched", trial_id=trial.trial_id, attempt_id=record["attempt_id"], action=decision.action, reason=decision.reason)
                continue
            if state is LiveDemoState.EXECUTING:
                assert record is not None and computation is not None
                result, decision = computation.result, computation.decision
                deadline = dispatcher.motion_deadline_s
                allowed_delta_s = delta_s
                if deadline is not None:
                    execution_start = float(record.get("execution_start_monotonic_s", now_s))
                    allowed_delta_s = max(0.0, min(now_s, deadline) - max(frame_start_s, execution_start))
                controller.update_virtual(allowed_delta_s)
                timeout = dispatcher.tick(now_s)
                if timeout is not None:
                    decision = timeout
                    computation = replace(computation, decision=decision)
                    record["dispatcher_action"] = decision.action
                    record["dispatcher_reason"] = decision.reason
                    record["effective_command"] = decision.effective_command.value
                    record["execution_end_monotonic_s"] = decision.timestamp_s
                remaining = None if deadline is None else max(0.0, deadline - now_s)
                view.draw(
                    state=state, trial=trial, target_states=static_target_states(self.config.stimulus.frequencies_hz),
                    controller=controller, result=result, decision=decision, trial_total=len(self.schedule),
                    dropped_frames=self._drop_total, live=self.live_source.status(), remaining_s=remaining,
                )
                win.flip()
                if execution_rendered and (timeout is not None or not controller.motion_active(dispatcher)):
                    controller.stop_virtual()
                    record.setdefault("execution_end_monotonic_s", time.monotonic())
                    self.state_machine.executing_complete()
                    rest_start_s = None
                execution_rendered = True
                continue
            if state is LiveDemoState.REST:
                controller.stop_virtual()
                if rest_start_s is None:
                    rest_start_s = now_s
                    logger.event("rest_started", trial_id=trial.trial_id, attempt_id=record["attempt_id"] if record else None)
                result = computation.result if computation else None
                decision = computation.decision if computation else None
                view.draw(
                    state=state, trial=trial, target_states=static_target_states(self.config.stimulus.frequencies_hz),
                    controller=controller, result=result, decision=decision, trial_total=len(self.schedule),
                    dropped_frames=self._drop_total, live=self.live_source.status(),
                )
                win.flip()
                if now_s - rest_start_s >= self.config.stimulus.rest_duration_s:
                    assert record is not None
                    if record.get("trial_status") not in {"aborted", "failed", "stopped"}:
                        self._finish_record(record, controller, self.live_source.status(), "completed")
                    logger.trial(record)
                    logger.event("trial_completed", trial_id=trial.trial_id, attempt_id=record["attempt_id"], status=record["trial_status"])
                    record = None
                    computation = None
                    cue_start_s = rest_start_s = None
                    marker = scheduler = None
                    self.state_machine.rest_complete()

    def _new_record(self, trial: ScheduledTrial) -> dict[str, Any]:
        attempt_id = self._attempts.get(trial.trial_id, 0)
        self._attempts[trial.trial_id] = attempt_id + 1
        return {
            "trial_id": trial.trial_id,
            "attempt_id": attempt_id,
            "mode": "live",
            "source_mode": "live",
            "server_url": self.server_url,
            "target_frequency_hz": trial.target_frequency_hz,
            "target_command": trial.command,
            "expected_samples": self.config.live.trial_samples,
            "eeg_window_index": None,
            "alignment_mode_requested": self.config.live.alignment_mode,
            "raw_input_unit": "uV",
            "raw_input_preprocessing": "none",
        }

    def _apply_live_fields(
        self, record: dict[str, Any], status: LiveSourceStatus, window_status: str, reason: str | None
    ) -> None:
        stimulus_end = record.get("stimulus_end_monotonic_s")
        wait_ms = None
        if status.window_ready_monotonic_s is not None and isinstance(stimulus_end, (int, float)):
            wait_ms = max(0.0, (status.window_ready_monotonic_s - stimulus_end) * 1000.0)
        record.update(
            connection_state=status.connection_state,
            start_sequence=status.start_sequence,
            end_sequence=status.end_sequence,
            alignment_mode_requested=status.alignment_mode_requested or self.config.live.alignment_mode,
            alignment_mode_used=status.alignment_mode_used,
            stimulus_flip_monotonic_s=status.stimulus_flip_monotonic_s,
            sequence_at_flip=status.sequence_at_flip,
            trigger_request_monotonic_s=status.trigger_request_monotonic_s,
            trigger_response_monotonic_s=status.trigger_response_monotonic_s,
            trigger_http_latency_ms=status.trigger_http_latency_ms,
            trigger_sequence=status.trigger_sequence,
            trigger_sequence_semantics=status.trigger_sequence_semantics,
            window_start_sequence=status.window_start_sequence,
            window_end_sequence=status.window_end_sequence,
            raw_input_unit="uV",
            raw_input_preprocessing="none",
            received_samples=status.collected_samples,
            expected_samples=status.expected_samples,
            stream_gap_count=status.stream_gap_count,
            invalid_frame_count=status.invalid_frame_count,
            reconnect_count=status.reconnect_count,
            queue_overflow_count=status.queue_overflow_count,
            window_ready_monotonic_s=status.window_ready_monotonic_s,
            collection_wait_ms=wait_ms,
            live_window_status=window_status,
            live_window_failure_reason=reason,
        )

    def _finish_record(
        self,
        record: dict[str, Any],
        controller: ControllerRuntime,
        status: LiveSourceStatus,
        trial_status: str,
        failure_reason: str | None = None,
    ) -> None:
        record["trial_status"] = trial_status
        for key, value in controller.trial_controller_fields(None).items():
            record.setdefault(key, value)
        if failure_reason is not None:
            self._apply_live_fields(record, status, "incomplete_live_window", failure_reason)
        position = controller.position()
        record.update(
            car_end_x=position["x"],
            car_end_y=position["y"],
            car_end_heading=position["heading_degrees"],
        )
