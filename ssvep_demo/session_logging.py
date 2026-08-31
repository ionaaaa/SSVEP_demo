"""Shared UTF-8 session logging for synthetic generation and replay."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .config import DemoConfig


SESSION_SCHEMA_VERSION = 1

TRIAL_FIELDS = [
    "trial_id", "attempt_id", "mode", "trial_status", "target_frequency_hz", "target_command",
    "cue_start_monotonic_s", "stimulus_start_monotonic_s", "stimulus_end_monotonic_s",
    "predicted_frequency_hz", "predicted_command", "effective_command", "confidence", "correct",
    "decoder_type", "decoder_latency_ms", "decoding_latency_ms", "generation_latency_ms", "dropped_frames",
    "dropped_frame_count", "eeg_window_index", "synthetic_seed", "snr_db", "dispatcher_action",
    "dispatcher_reason", "car_start_x", "car_start_y", "car_start_heading", "car_end_x", "car_end_y",
    "car_end_heading", "decoder_scores_json", "decoder_scores", "execution_start_monotonic_s",
    "execution_end_monotonic_s",
]


def resolved_config_dict(
    config: DemoConfig,
    *,
    decoder_type: str | None = None,
    synthetic_snr_db: float | None = None,
    synthetic_seed: int | None = None,
    confirmations_required: int | None = None,
) -> dict[str, Any]:
    """Return a serialisable, effective YAML configuration without CLI paths."""
    return {
        "stimulus": {
            "frequencies": list(config.stimulus.frequencies_hz),
            "trial_duration_s": config.stimulus.trial_duration_s,
            "rest_duration_s": config.stimulus.rest_duration_s,
            "cue_duration_s": config.stimulus.cue_duration_s,
            "repetitions": config.stimulus.repetitions,
            "trial_order": config.stimulus.trial_order,
            "random_seed": config.stimulus.random_seed,
            "fullscreen": config.stimulus.fullscreen,
            "window_size": list(config.stimulus.window_size),
            "screen_index": config.stimulus.screen_index,
            "background_color": list(config.stimulus.background_color),
            "stimulus_on_color": list(config.stimulus.stimulus_on_color),
            "stimulus_off_color": list(config.stimulus.stimulus_off_color),
            "dropped_frame_threshold_ratio": config.stimulus.dropped_frame_threshold_ratio,
        },
        "acquisition": {"sample_rate_hz": config.acquisition.sample_rate_hz, "channels": list(config.acquisition.channels)},
        "decoder": {
            "type": decoder_type or config.decoder.type,
            "harmonics": config.decoder.harmonics,
            "bandpass_hz": list(config.decoder.bandpass_hz),
        },
        "commands": {str(frequency): getattr(command, "value", command) for frequency, command in config.commands.items()},
        "control": {
            "confidence_threshold": config.control.confidence_threshold,
            "confirmations_required": confirmations_required or config.control.confirmations_required,
            "max_confirmation_gap_s": config.control.max_confirmation_gap_s,
            "command_duration_s": config.control.command_duration_s,
            "input_timeout_s": config.control.input_timeout_s,
            "max_prediction_age_s": config.control.max_prediction_age_s,
        },
        "synthetic_demo": {
            "snr_db": config.synthetic_demo.snr_db if synthetic_snr_db is None else synthetic_snr_db,
            "seed": config.synthetic_demo.seed if synthetic_seed is None else synthetic_seed,
            "confirmations_required": config.synthetic_demo.confirmations_required,
        },
        "ui": {"cjk_font_file": config.ui.cjk_font_file, "cjk_font_name": config.ui.cjk_font_name},
    }


class UnifiedSessionLogger:
    """One incremental logger format shared by synthetic and replay runs."""

    fields = TRIAL_FIELDS

    def __init__(self, output_root: str | Path, effective_config: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = Path(output_root) / f"session_{stamp}"
        suffix = 1
        while self.directory.exists():
            self.directory = Path(output_root) / f"session_{stamp}_{suffix:02d}"
            suffix += 1
        self.directory.mkdir(parents=True)
        self.started_monotonic_s = time.monotonic()
        self.started_utc = datetime.now(timezone.utc)
        self.effective_config = dict(effective_config)
        (self.directory / "config.yaml").write_text(
            yaml.safe_dump(self.effective_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        self._events = (self.directory / "events.jsonl").open("a", encoding="utf-8")
        self._trials_file = (self.directory / "trials.csv").open("w", encoding="utf-8", newline="")
        self._trials = csv.DictWriter(self._trials_file, fieldnames=TRIAL_FIELDS)
        self._trials.writeheader()
        self._trials_file.flush()
        self._intervals_file = (self.directory / "frame_intervals.csv").open("w", encoding="utf-8", newline="")
        self._intervals = csv.DictWriter(
            self._intervals_file, fieldnames=["trial_id", "attempt_id", "frame_index", "interval_s", "suspected_dropped"]
        )
        self._intervals.writeheader()
        self._intervals_file.flush()
        self.rows: list[dict[str, Any]] = []
        self.metadata = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "mode": metadata.get("mode"),
            "started_utc": self.started_utc.isoformat(),
            "effective_config": self.effective_config,
            **dict(metadata),
        }

    def event(self, event: str, **payload: Any) -> None:
        record = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "event": event,
            "monotonic_s": time.monotonic(),
            "utc_time": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self._events.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
        self._events.flush()

    def trial(self, record: Mapping[str, Any]) -> None:
        normalised = {field: _csv_value(record.get(field)) for field in TRIAL_FIELDS}
        self._trials.writerow(normalised)
        self._trials_file.flush()
        self.rows.append(dict(record))

    def frame_intervals(
        self, trial_id: int, attempt_id: int, intervals_s: Sequence[float], dropped_indices: Sequence[int]
    ) -> None:
        dropped = set(dropped_indices)
        for index, interval in enumerate(intervals_s):
            self._intervals.writerow(
                {"trial_id": trial_id, "attempt_id": attempt_id, "frame_index": index, "interval_s": interval,
                 "suspected_dropped": index in dropped}
            )
        self._intervals_file.flush()

    def close(
        self,
        *,
        status: str,
        archived_eeg_windows: int,
        final_car_state: Mapping[str, Any] | None,
        error: BaseException | None = None,
        **extra: Any,
    ) -> None:
        summary = self._summary(status, archived_eeg_windows, final_car_state, error)
        summary.update(extra)
        (self.directory / "summary.json").write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._events.close()
        self._trials_file.close()
        self._intervals_file.close()

    def _summary(
        self, status: str, archived_eeg_windows: int, final_car_state: Mapping[str, Any] | None, error: BaseException | None
    ) -> dict[str, Any]:
        completed = [row for row in self.rows if row.get("trial_status") == "completed"]
        decoded = [row for row in self.rows if row.get("predicted_frequency_hz") is not None]
        correct_values = [bool(row.get("correct")) for row in decoded if row.get("correct") is not None]
        per_frequency: dict[str, float | None] = {}
        confusion: dict[str, dict[str, int]] = {}
        for row in decoded:
            target, prediction = row.get("target_frequency_hz"), row.get("predicted_frequency_hz")
            if target is None or prediction is None:
                continue
            target_key, prediction_key = str(target), str(prediction)
            confusion.setdefault(target_key, {})[prediction_key] = confusion.setdefault(target_key, {}).get(prediction_key, 0) + 1
        for target_key in {str(row.get("target_frequency_hz")) for row in decoded if row.get("target_frequency_hz") is not None}:
            target_rows = [row for row in decoded if str(row.get("target_frequency_hz")) == target_key and row.get("correct") is not None]
            per_frequency[target_key] = (sum(bool(row["correct"]) for row in target_rows) / len(target_rows)) if target_rows else None
        confidences = _finite_values(row.get("confidence") for row in decoded)
        latencies = _finite_values(row.get("decoder_latency_ms", row.get("decoding_latency_ms")) for row in decoded)
        return {
            **self.metadata,
            "source_replay_file": self.metadata.get("source_replay_file"),
            "source_session": self.metadata.get("source_session"),
            "status": status,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": time.monotonic() - self.started_monotonic_s,
            "total_attempts": len(self.rows),
            "completed_trials": len(completed),
            "aborted_trials": sum(row.get("trial_status") == "aborted" for row in self.rows),
            "stopped_trials": sum(row.get("trial_status") == "stopped" for row in self.rows),
            "archived_eeg_windows": archived_eeg_windows,
            "overall_accuracy": (sum(correct_values) / len(correct_values)) if correct_values else None,
            "per_frequency_accuracy": per_frequency,
            "confusion_matrix": confusion,
            "mean_confidence": float(np.mean(confidences)) if confidences else None,
            "decoder_latency_ms_mean": float(np.mean(latencies)) if latencies else None,
            "decoder_latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
            "decoder_latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
            "total_dropped_frames": int(sum(int(row.get("dropped_frames", row.get("dropped_frame_count", 0)) or 0) for row in self.rows)),
            "final_car_state": dict(final_car_state) if final_car_state is not None else None,
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error)[:500] if error else None,
        }


def _finite_values(values: Sequence[Any] | Any) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(_json_safe(value), ensure_ascii=False)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
