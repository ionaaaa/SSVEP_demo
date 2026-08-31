"""FFT and CCA decoders that consume only the stage-one EEG protocol."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import signal

from .config import DemoConfig
from .protocol import DecodeResult, EEGWindow


class SSVEPDecoder(Protocol):
    """A decoder receives only an unlabeled EEG window and returns a result."""

    def decode(self, window: EEGWindow) -> DecodeResult:
        """Decode an EEG window without access to a target label."""


def _confidence(scores: dict[float, float]) -> float:
    """Return normalized winner margin, robust for ties and scale differences.

    Confidence is ``(best - second_best) / (abs(best) + eps)``, clipped to
    ``[0, 1]``.  An exact tie therefore has confidence zero.
    """
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return 1.0
    best, second = ordered[:2]
    return float(np.clip((best - second) / (abs(best) + np.finfo(float).eps), 0.0, 1.0))


@dataclass
class _DecoderBase:
    config: DemoConfig

    def __post_init__(self) -> None:
        self._frequencies = self.config.stimulus.frequencies_hz
        self._sample_rate_hz = self.config.acquisition.sample_rate_hz
        self._channels = self.config.acquisition.channels
        self._harmonics = self.config.decoder.harmonics
        self._bandpass_hz = self.config.decoder.bandpass_hz
        self._validate_analysis_configuration()
        self._sos = signal.butter(
            4,
            self._bandpass_hz,
            btype="bandpass",
            fs=self._sample_rate_hz,
            output="sos",
        )

    def _validate_analysis_configuration(self) -> None:
        nyquist_hz = self._sample_rate_hz / 2.0
        for frequency in self._frequencies:
            if frequency >= nyquist_hz:
                raise ValueError(f"candidate frequency {frequency} reaches the Nyquist frequency")
            if not self._valid_harmonics(frequency):
                raise ValueError(f"candidate frequency {frequency} has no harmonic in the analysis band")

    def _valid_harmonics(self, frequency_hz: float) -> tuple[int, ...]:
        nyquist_hz = self._sample_rate_hz / 2.0
        low_hz, high_hz = self._bandpass_hz
        return tuple(
            harmonic
            for harmonic in range(1, self._harmonics + 1)
            if harmonic * frequency_hz < nyquist_hz and low_hz <= harmonic * frequency_hz <= high_hz
        )

    def _prepare_window(self, window: EEGWindow) -> np.ndarray:
        if not isinstance(window, EEGWindow):
            raise ValueError("window must be an EEGWindow")
        if not isinstance(window.data, np.ndarray) or window.data.ndim != 2:
            raise ValueError("EEGWindow.data must be a two-dimensional array [channels, samples]")
        if not np.issubdtype(window.data.dtype, np.number) or not np.isfinite(window.data).all():
            raise ValueError("EEGWindow.data must contain only finite values")
        if tuple(window.channel_names) != self._channels or window.data.shape[0] != len(self._channels):
            raise ValueError("EEGWindow channels must match the configured channel names and order")
        if not math.isclose(window.sample_rate_hz, self._sample_rate_hz, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("EEGWindow sample_rate_hz must match the configured acquisition sample rate")
        samples = window.data.shape[1]
        min_samples = math.ceil(self._sample_rate_hz / 0.25)
        if samples < min_samples:
            raise ValueError("EEGWindow is too short: at least four seconds are required for 0.25 Hz PSD resolution")
        expected_duration_s = samples / self._sample_rate_hz
        if abs((window.end_time_s - window.start_time_s) - expected_duration_s) > 1.5 / self._sample_rate_hz:
            raise ValueError("EEGWindow time range is inconsistent with its sample count and sample rate")
        demeaned = window.data.astype(float, copy=False) - np.mean(window.data, axis=1, keepdims=True)
        return signal.sosfiltfilt(self._sos, demeaned, axis=1)

    def _result(self, scores: dict[float, float]) -> DecodeResult:
        predicted_frequency_hz = max(scores, key=scores.__getitem__)
        return DecodeResult(
            predicted_frequency_hz=predicted_frequency_hz,
            command=self.config.commands[predicted_frequency_hz],
            scores=scores,
            confidence=_confidence(scores),
            timestamp_s=time.monotonic(),
        )


class FFTDecoder(_DecoderBase):
    """Welch-PSD decoder with channel-mean, harmonic-weighted band energy."""

    def __init__(
        self,
        config: DemoConfig,
        *,
        frequency_half_bandwidth_hz: float = 0.25,
        harmonic_weights: tuple[float, ...] = (1.0, 0.5, 0.25),
    ) -> None:
        if (
            isinstance(frequency_half_bandwidth_hz, bool)
            or not isinstance(frequency_half_bandwidth_hz, (int, float))
            or not math.isfinite(frequency_half_bandwidth_hz)
            or frequency_half_bandwidth_hz <= 0
        ):
            raise ValueError("frequency_half_bandwidth_hz must be greater than zero")
        if (
            len(harmonic_weights) < config.decoder.harmonics
            or any(not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0 for weight in harmonic_weights)
        ):
            raise ValueError("harmonic_weights must provide non-negative weights for all configured harmonics")
        self.frequency_half_bandwidth_hz = float(frequency_half_bandwidth_hz)
        self.harmonic_weights = tuple(float(weight) for weight in harmonic_weights)
        super().__init__(config)

    def decode(self, window: EEGWindow) -> DecodeResult:
        data = self._prepare_window(window)
        # nperseg=all samples gives 0.25 Hz resolution for the required minimum
        # four-second window. Integrating f ± bandwidth uses several bins, not a
        # single-bin FFT lookup.
        frequencies_hz, psd = signal.welch(data, fs=self._sample_rate_hz, axis=1, nperseg=data.shape[1])
        scores: dict[float, float] = {}
        for candidate in self._frequencies:
            channel_scores = np.zeros(data.shape[0], dtype=float)
            for harmonic in self._valid_harmonics(candidate):
                center_hz = harmonic * candidate
                mask = np.abs(frequencies_hz - center_hz) <= self.frequency_half_bandwidth_hz
                if not np.any(mask):
                    continue
                # PSD is integrated across the local band for each channel, then
                # channel energies are averaged to avoid channel-count scaling.
                band_energy = np.trapz(psd[:, mask], frequencies_hz[mask], axis=1)
                channel_scores += self.harmonic_weights[harmonic - 1] * band_energy
            scores[candidate] = float(np.mean(channel_scores))
        return self._result(scores)


class CCADecoder(_DecoderBase):
    """CCA decoder using [samples, features] EEG/reference matrices internally."""

    def __init__(self, config: DemoConfig, *, regularization: float = 1e-6) -> None:
        if (
            isinstance(regularization, bool)
            or not isinstance(regularization, (int, float))
            or not math.isfinite(regularization)
            or regularization <= 0
        ):
            raise ValueError("regularization must be greater than zero")
        self.regularization = float(regularization)
        super().__init__(config)

    def decode(self, window: EEGWindow) -> DecodeResult:
        data = self._prepare_window(window)
        time_s = np.arange(data.shape[1], dtype=float) / self._sample_rate_hz
        # EEG starts as [channels, samples], while CCA uses X=[samples, channels]
        # and a reference Y=[samples, 2 * valid_harmonics] of sin/cos columns.
        eeg_samples_by_channels = data.T
        scores = {
            candidate: self._cca_score(eeg_samples_by_channels, self._reference(time_s, candidate))
            for candidate in self._frequencies
        }
        return self._result(scores)

    def _reference(self, time_s: np.ndarray, frequency_hz: float) -> np.ndarray:
        columns = []
        for harmonic in self._valid_harmonics(frequency_hz):
            angle = 2.0 * np.pi * harmonic * frequency_hz * time_s
            columns.extend((np.sin(angle), np.cos(angle)))
        return np.column_stack(columns)

    def _cca_score(self, x: np.ndarray, y: np.ndarray) -> float:
        x = x - np.mean(x, axis=0, keepdims=True)
        y = y - np.mean(y, axis=0, keepdims=True)
        scale = x.shape[0] - 1
        cxx = x.T @ x / scale
        cyy = y.T @ y / scale
        cxy = x.T @ y / scale
        inverse_sqrt_x = self._inverse_sqrt(cxx)
        inverse_sqrt_y = self._inverse_sqrt(cyy)
        singular_values = np.linalg.svd(inverse_sqrt_x @ cxy @ inverse_sqrt_y, compute_uv=False)
        return float(np.clip(singular_values[0], 0.0, 1.0))

    def _inverse_sqrt(self, covariance: np.ndarray) -> np.ndarray:
        dimension = covariance.shape[0]
        scale = max(float(np.trace(covariance)) / dimension, np.finfo(float).eps)
        regularized = covariance + self.regularization * scale * np.eye(dimension)
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        return (eigenvectors / np.sqrt(np.maximum(eigenvalues, np.finfo(float).eps))) @ eigenvectors.T
