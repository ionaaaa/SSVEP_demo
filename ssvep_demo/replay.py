"""Offline replay acceptance for archived EEG windows."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Sequence

from .config import DemoConfig
from .control import SafeCommandDispatcher
from .decoders import CCADecoder, FFTDecoder, SSVEPDecoder
from .eeg_archive import EEGWindowArchiveWriter, ReplayEEGSource
from .session_logging import UnifiedSessionLogger, resolved_config_dict
from .virtual_car import VirtualCarController


class ReplayDemoRunner:
    """Create a new, offline replay session without touching the source session."""

    def __init__(
        self,
        config: DemoConfig,
        replay_file: str | Path,
        *,
        decoder_type: str | None = None,
        output_dir: str | Path = "outputs/ssvep_demo",
        trial_ids: Sequence[int] | None = None,
        max_trials: int | None = None,
        no_gui: bool = False,
        confirmations_required: int | None = None,
    ) -> None:
        if decoder_type is not None and decoder_type not in {"fft", "cca"}:
            raise ValueError("decoder_type must be 'fft' or 'cca'")
        if max_trials is not None and (not isinstance(max_trials, int) or isinstance(max_trials, bool) or max_trials <= 0):
            raise ValueError("max_trials must be a positive integer")
        if trial_ids is not None and any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in trial_ids):
            raise ValueError("trial_ids must contain non-negative integers")
        self.config = config
        self.replay_file = Path(replay_file)
        self.decoder_type = decoder_type or config.decoder.type
        self.output_dir = Path(output_dir)
        self.trial_ids = tuple(trial_ids) if trial_ids is not None else None
        self.max_trials = max_trials
        self.no_gui = no_gui
        confirmations = config.synthetic_demo.confirmations_required if confirmations_required is None else confirmations_required
        self.control_config = replace(config.control, confirmations_required=confirmations)

    def _decoder(self) -> SSVEPDecoder:
        return FFTDecoder(self.config) if self.decoder_type == "fft" else CCADecoder(self.config)

    def run(self) -> Path:
        source = ReplayEEGSource(self.replay_file, expected_channels=self.config.acquisition.channels)
        effective_config = resolved_config_dict(
            self.config, decoder_type=self.decoder_type, confirmations_required=self.control_config.confirmations_required
        )
        effective_config["runtime"] = {"no_gui": self.no_gui}
        logger = UnifiedSessionLogger(
            self.output_dir,
            effective_config,
            {
                "mode": "replay",
                "source_replay_file": self.replay_file.name,
                "source_session": self.replay_file.parent.name,
                "decoder_type": self.decoder_type,
                "confirmations_required": self.control_config.confirmations_required,
                "real_eeg_validated": False,
                "real_car_validated": False,
            },
        )
        expected_samples = source.window(0).data.shape[1] if len(source) else int(
            round(self.config.stimulus.trial_duration_s * self.config.acquisition.sample_rate_hz)
        )
        archive = EEGWindowArchiveWriter(logger.directory / "eeg_windows.npz", source.channel_names, expected_samples)
        car = VirtualCarController()
        dispatcher = SafeCommandDispatcher(car, self.control_config, self.config.commands)
        decoder = self._decoder()
        error: BaseException | None = None
        try:
            indices = [index for index, _ in source.iter_windows(trial_ids=self.trial_ids)]
            if self.max_trials is not None:
                indices = indices[: self.max_trials]
            logger.event("replay_loaded", replay_windows=len(indices), no_gui=self.no_gui)
            for index in indices:
                window = source.window(index)
                # No archive label is read until after the decoder consumed its
                # EEGWindow, making target leakage structurally impossible here.
                trial_id, attempt_id = source.identity(index)
                logger.event("replay_window_started", trial_id=trial_id, attempt_id=attempt_id, eeg_window_index=index)
                start_s = time.monotonic()
                result = decoder.decode(window)
                latency_ms = (time.monotonic() - start_s) * 1000.0
                trial_id, attempt_id, target_frequency_hz, _ = source.evaluation_label(index)
                car_start = car.state()
                decision = dispatcher.submit(result)
                new_index = archive.add(
                    window,
                    trial_id=trial_id,
                    attempt_id=attempt_id,
                    target_frequency_hz=target_frequency_hz,
                    source_mode="replay",
                )
                execution_start_s = decision.timestamp_s if decision.action == "executed" else None
                execution_end_s = None
                if decision.action == "executed":
                    deadline = dispatcher.motion_deadline_s
                    if deadline is not None:
                        car.update(max(0.0, deadline - decision.timestamp_s))
                        stopped = dispatcher.tick(deadline)
                        execution_end_s = stopped.timestamp_s if stopped is not None else deadline
                car.stop()
                command = getattr(result.command, "value", result.command)
                logger.trial(
                    {
                        "trial_id": trial_id,
                        "attempt_id": attempt_id,
                        "mode": "replay",
                        "trial_status": "completed",
                        "target_frequency_hz": target_frequency_hz,
                        "target_command": getattr(self.config.commands.get(target_frequency_hz), "value", None),
                        "predicted_frequency_hz": result.predicted_frequency_hz,
                        "predicted_command": command,
                        "effective_command": decision.effective_command.value,
                        "confidence": result.confidence,
                        "correct": result.predicted_frequency_hz == target_frequency_hz,
                        "decoder_type": self.decoder_type,
                        "decoder_latency_ms": latency_ms,
                        "decoder_scores_json": json.dumps({str(key): value for key, value in result.scores.items()}),
                        "dropped_frames": 0,
                        "eeg_window_index": new_index,
                        "dispatcher_action": decision.action,
                        "dispatcher_reason": decision.reason,
                        "execution_start_monotonic_s": execution_start_s,
                        "execution_end_monotonic_s": execution_end_s,
                        "car_start_x": car_start["x"], "car_start_y": car_start["y"],
                        "car_start_heading": car_start["heading_degrees"],
                        "car_end_x": car.x, "car_end_y": car.y, "car_end_heading": car.heading_degrees,
                    }
                )
                logger.event("decode_completed", trial_id=trial_id, attempt_id=attempt_id, prediction=result.predicted_frequency_hz)
                logger.event("command_dispatched", trial_id=trial_id, attempt_id=attempt_id, action=decision.action, reason=decision.reason)
                logger.event("replay_window_completed", trial_id=trial_id, attempt_id=attempt_id, eeg_window_index=new_index)
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
                status="failed" if error else "finished",
                archived_eeg_windows=archive.count,
                final_car_state=car.state(),
                error=error,
            )
