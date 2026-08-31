"""Device-independent data objects shared between SSVEP demo stages.

All timestamps in these protocol objects are in the ``time.monotonic()`` clock
domain.  UTC timestamps, if a caller needs them for logging, must be kept as
separate metadata and must not be used for signal or trial alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CANONICAL_CHANNELS: tuple[str, ...] = (
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "O1",
    "Oz",
    "O2",
)


def _finite_number(value: float, field_name: str) -> float:
    """Return a finite numeric value or raise a clear protocol error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


@dataclass
class TrialMarker:
    """Monotonic-clock boundaries for one stimulus trial."""

    trial_id: int
    target_frequency_hz: float
    stimulus_start_monotonic_s: float
    stimulus_end_monotonic_s: float | None

    def __post_init__(self) -> None:
        if isinstance(self.trial_id, bool) or not isinstance(self.trial_id, int) or self.trial_id < 0:
            raise ValueError("trial_id must be a non-negative integer")
        self.target_frequency_hz = _finite_number(self.target_frequency_hz, "target_frequency_hz")
        if self.target_frequency_hz <= 0:
            raise ValueError("target_frequency_hz must be greater than zero")
        self.stimulus_start_monotonic_s = _finite_number(
            self.stimulus_start_monotonic_s, "stimulus_start_monotonic_s"
        )
        if self.stimulus_end_monotonic_s is not None:
            self.stimulus_end_monotonic_s = _finite_number(
                self.stimulus_end_monotonic_s, "stimulus_end_monotonic_s"
            )
            if self.stimulus_end_monotonic_s < self.stimulus_start_monotonic_s:
                raise ValueError("stimulus_end_monotonic_s must not precede stimulus_start_monotonic_s")


@dataclass
class EEGWindow:
    """A channel-major EEG segment, with ``data`` laid out as [channels, samples]."""

    data: np.ndarray
    sample_rate_hz: float
    channel_names: list[str]
    start_time_s: float
    end_time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray) or self.data.ndim != 2:
            raise ValueError("EEGWindow.data must be a two-dimensional numpy array [channels, samples]")
        self.sample_rate_hz = _finite_number(self.sample_rate_hz, "sample_rate_hz")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be greater than zero")
        if not isinstance(self.channel_names, list) or not all(
            isinstance(name, str) and name for name in self.channel_names
        ):
            raise ValueError("channel_names must be a list of non-empty strings")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must not contain duplicates")
        if self.data.shape[0] != len(self.channel_names):
            raise ValueError("EEGWindow.data channel dimension must match channel_names")
        self.start_time_s = _finite_number(self.start_time_s, "start_time_s")
        self.end_time_s = _finite_number(self.end_time_s, "end_time_s")
        if self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s must not precede start_time_s")


@dataclass
class DecodeResult:
    """A device-independent decoder output timestamped on the monotonic clock."""

    predicted_frequency_hz: float
    command: str
    scores: dict[float, float]
    confidence: float
    timestamp_s: float

    def __post_init__(self) -> None:
        self.predicted_frequency_hz = _finite_number(
            self.predicted_frequency_hz, "predicted_frequency_hz"
        )
        if self.predicted_frequency_hz <= 0:
            raise ValueError("predicted_frequency_hz must be greater than zero")
        if not isinstance(self.command, str) or not self.command:
            raise ValueError("command must be a non-empty string")
        if not isinstance(self.scores, dict):
            raise ValueError("scores must be a dictionary")
        self.scores = {
            _finite_number(frequency, "scores frequency"): _finite_number(score, "scores value")
            for frequency, score in self.scores.items()
        }
        self.confidence = _finite_number(self.confidence, "confidence")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.timestamp_s = _finite_number(self.timestamp_s, "timestamp_s")
