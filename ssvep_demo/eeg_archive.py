"""Strict, pickle-free EEG-window archive and replay source."""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
from typing import Iterator, Sequence

import numpy as np

from .protocol import CANONICAL_CHANNELS, EEGWindow


EEG_ARCHIVE_SCHEMA_VERSION = 1
_REQUIRED = {
    "schema_version", "data", "trial_id", "attempt_id", "sample_rate_hz", "channel_names",
    "window_start_monotonic_s", "window_end_monotonic_s", "target_frequency_hz", "source_mode", "data_unit",
}


class EEGWindowArchiveWriter:
    """Atomically rewrite a dense NPZ archive whenever a window is appended."""

    def __init__(self, path: str | Path, channel_names: Sequence[str], samples_per_window: int) -> None:
        self.path = Path(path)
        self.channel_names = tuple(channel_names)
        if self.channel_names != CANONICAL_CHANNELS:
            raise ValueError("EEG archive channel_names must exactly match canonical channels")
        if not isinstance(samples_per_window, int) or samples_per_window <= 0:
            raise ValueError("samples_per_window must be a positive integer")
        self.samples_per_window = samples_per_window
        self._windows: list[np.ndarray] = []
        self._trial_ids: list[int] = []
        self._attempt_ids: list[int] = []
        self._sample_rates: list[float] = []
        self._starts: list[float] = []
        self._ends: list[float] = []
        self._targets: list[float] = []
        self._source_modes: list[str] = []
        self.flush()

    def add(
        self,
        window: EEGWindow,
        *,
        trial_id: int,
        attempt_id: int,
        target_frequency_hz: float,
        source_mode: str,
    ) -> int:
        if window.data.shape != (len(self.channel_names), self.samples_per_window):
            raise ValueError("EEG window shape does not match archive dense-array schema")
        if tuple(window.channel_names) != self.channel_names:
            raise ValueError("EEG window channels do not match archive channel order")
        if not np.isfinite(window.data).all():
            raise ValueError("EEG archive cannot store non-finite EEG data")
        if not all(isinstance(value, int) and value >= 0 for value in (trial_id, attempt_id)):
            raise ValueError("trial_id and attempt_id must be non-negative integers")
        if not isinstance(target_frequency_hz, (int, float)) or not math.isfinite(float(target_frequency_hz)):
            raise ValueError("target_frequency_hz must be finite")
        if not isinstance(source_mode, str) or not source_mode:
            raise ValueError("source_mode must be a non-empty string")
        index = len(self._windows)
        self._windows.append(np.asarray(window.data, dtype=np.float32).copy())
        self._trial_ids.append(trial_id)
        self._attempt_ids.append(attempt_id)
        self._sample_rates.append(float(window.sample_rate_hz))
        self._starts.append(float(window.start_time_s))
        self._ends.append(float(window.end_time_s))
        self._targets.append(float(target_frequency_hz))
        self._source_modes.append(source_mode)
        self.flush()
        return index

    def flush(self) -> None:
        """Write a complete archive to a temporary NPZ then atomically replace it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        count = len(self._windows)
        data = (
            np.stack(self._windows, axis=0)
            if self._windows
            else np.empty((0, len(self.channel_names), self.samples_per_window), dtype=np.float32)
        )
        arrays = {
            "schema_version": np.asarray(EEG_ARCHIVE_SCHEMA_VERSION, dtype=np.int64),
            "data": data,
            "trial_id": np.asarray(self._trial_ids, dtype=np.int64),
            "attempt_id": np.asarray(self._attempt_ids, dtype=np.int64),
            "sample_rate_hz": np.asarray(self._sample_rates, dtype=np.float64),
            "channel_names": np.asarray(self.channel_names, dtype="U"),
            "window_start_monotonic_s": np.asarray(self._starts, dtype=np.float64),
            "window_end_monotonic_s": np.asarray(self._ends, dtype=np.float64),
            "target_frequency_hz": np.asarray(self._targets, dtype=np.float64),
            "source_mode": np.asarray(self._source_modes, dtype="U"),
            "data_unit": np.asarray("arbitrary_signal_unit", dtype="U"),
        }
        if data.shape[0] != count:  # Defensive check before exposing the file.
            raise RuntimeError("internal EEG archive length mismatch")
        with tempfile.NamedTemporaryFile(dir=self.path.parent, suffix=".npz", delete=False) as temp:
            temporary_path = Path(temp.name)
        try:
            np.savez_compressed(temporary_path, **arrays)
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @property
    def count(self) -> int:
        return len(self._windows)


class ReplayEEGSource:
    """Read validated EEG windows without exposing labels to decoder calls."""

    def __init__(self, path: str | Path, *, expected_channels: Sequence[str] = CANONICAL_CHANNELS) -> None:
        self.path = Path(path)
        try:
            archive = np.load(self.path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot load EEG archive {self.path}: {exc}") from exc
        try:
            missing = _REQUIRED - set(archive.files)
            if missing:
                raise ValueError(f"EEG archive missing required fields: {', '.join(sorted(missing))}")
            self._load_and_validate(archive, tuple(expected_channels))
        finally:
            archive.close()

    def _load_and_validate(self, archive: np.lib.npyio.NpzFile, expected_channels: tuple[str, ...]) -> None:
        try:
            schema_version = int(np.asarray(archive["schema_version"]).item())
        except (ValueError, TypeError) as exc:
            raise ValueError("EEG archive schema_version must be an integer scalar") from exc
        if schema_version != EEG_ARCHIVE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported EEG archive schema_version {schema_version}; expected {EEG_ARCHIVE_SCHEMA_VERSION}")
        arrays = {name: np.asarray(archive[name]) for name in _REQUIRED}
        if any(array.dtype == object for array in arrays.values()):
            raise ValueError("EEG archive object arrays are not supported")
        data = arrays["data"]
        if data.ndim != 3:
            raise ValueError("EEG archive data must have shape [N, C, T]")
        count, channels, samples = data.shape
        if channels != len(expected_channels) or samples <= 0:
            raise ValueError("EEG archive data channel/sample dimensions are incompatible")
        channel_names = tuple(str(value) for value in arrays["channel_names"].tolist())
        if channel_names != expected_channels:
            raise ValueError("EEG archive channel_names do not match the expected canonical order")
        per_window = (
            "trial_id", "attempt_id", "sample_rate_hz", "window_start_monotonic_s", "window_end_monotonic_s",
            "target_frequency_hz", "source_mode",
        )
        for name in per_window:
            if arrays[name].ndim != 1 or len(arrays[name]) != count:
                raise ValueError(f"EEG archive {name} must be a length-N one-dimensional array")
        if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise ValueError("EEG archive data must contain only finite numeric values")
        for name in ("sample_rate_hz", "window_start_monotonic_s", "window_end_monotonic_s", "target_frequency_hz"):
            if not np.issubdtype(arrays[name].dtype, np.number) or not np.isfinite(arrays[name]).all():
                raise ValueError(f"EEG archive {name} must contain finite numeric values")
        if np.any(arrays["sample_rate_hz"] <= 0):
            raise ValueError("EEG archive sample_rate_hz must be greater than zero")
        if np.any(arrays["window_end_monotonic_s"] < arrays["window_start_monotonic_s"]):
            raise ValueError("EEG archive window end time must not precede start time")
        if np.any(arrays["trial_id"] < 0) or np.any(arrays["attempt_id"] < 0):
            raise ValueError("EEG archive trial_id and attempt_id must be non-negative")
        pairs = list(zip(arrays["trial_id"].tolist(), arrays["attempt_id"].tolist()))
        if len(set(pairs)) != len(pairs):
            raise ValueError("EEG archive trial_id/attempt_id pairs must be unique")
        self._data = np.asarray(data, dtype=np.float32)
        self._trial_ids = np.asarray(arrays["trial_id"], dtype=np.int64)
        self._attempt_ids = np.asarray(arrays["attempt_id"], dtype=np.int64)
        self._sample_rates = np.asarray(arrays["sample_rate_hz"], dtype=float)
        self._starts = np.asarray(arrays["window_start_monotonic_s"], dtype=float)
        self._ends = np.asarray(arrays["window_end_monotonic_s"], dtype=float)
        self._targets = np.asarray(arrays["target_frequency_hz"], dtype=float)
        self._source_modes = tuple(str(value) for value in arrays["source_mode"].tolist())
        self.channel_names = list(channel_names)

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def window(self, eeg_window_index: int) -> EEGWindow:
        if isinstance(eeg_window_index, bool) or not isinstance(eeg_window_index, int) or not 0 <= eeg_window_index < len(self):
            raise ValueError("eeg_window_index is out of range")
        return EEGWindow(
            data=self._data[eeg_window_index].copy(),
            sample_rate_hz=float(self._sample_rates[eeg_window_index]),
            channel_names=list(self.channel_names),
            start_time_s=float(self._starts[eeg_window_index]),
            end_time_s=float(self._ends[eeg_window_index]),
        )

    def iter_windows(self, *, trial_ids: Sequence[int] | None = None) -> Iterator[tuple[int, EEGWindow]]:
        selected = None if trial_ids is None else set(trial_ids)
        for index, trial_id in enumerate(self._trial_ids.tolist()):
            if selected is None or trial_id in selected:
                yield index, self.window(index)

    def indices_for_trial(self, trial_id: int) -> list[int]:
        return [index for index, value in enumerate(self._trial_ids.tolist()) if value == trial_id]

    def evaluation_label(self, eeg_window_index: int) -> tuple[int, int, float, str]:
        """Return label metadata for reporting only; never pass it to a decoder."""
        if not 0 <= eeg_window_index < len(self):
            raise ValueError("eeg_window_index is out of range")
        return (
            int(self._trial_ids[eeg_window_index]), int(self._attempt_ids[eeg_window_index]),
            float(self._targets[eeg_window_index]), self._source_modes[eeg_window_index],
        )

    def identity(self, eeg_window_index: int) -> tuple[int, int]:
        """Return non-signal identifiers for lifecycle logging."""
        if not 0 <= eeg_window_index < len(self):
            raise ValueError("eeg_window_index is out of range")
        return int(self._trial_ids[eeg_window_index]), int(self._attempt_ids[eeg_window_index])
