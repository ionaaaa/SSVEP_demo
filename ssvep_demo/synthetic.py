"""Deterministic, label-free synthetic EEG for decoder evaluation only."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np

from .config import DemoConfig
from .protocol import CANONICAL_CHANNELS, EEGWindow


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


class SyntheticEEGSource:
    """Generate multi-channel SSVEP windows with reproducible Gaussian noise.

    The target component is the sum of channel- and harmonic-specific sinusoids.
    Its discrete RMS is ``sqrt(mean(target_signal ** 2))`` over all channels and
    samples.  For the requested ``snr_db``, independent Gaussian noise has
    standard deviation ``target_rms / 10 ** (snr_db / 20)``; consequently
    ``snr_db = 20 * log10(target_rms / noise_rms)`` in expectation.

    Interference and line noise are added after that SNR-calibrated noise and
    therefore are intentionally *not* included in the SNR definition.  Their
    amplitudes use the same arbitrary signal unit as the target's fundamental.
    No target label is stored in the returned :class:`EEGWindow`.
    """

    def __init__(
        self,
        config: DemoConfig,
        *,
        harmonics: int = 3,
        seed: int | None = None,
        interference_frequencies_hz: Sequence[float] = (),
        interference_amplitude: float = 0.0,
        line_noise_hz: float | None = None,
        line_noise_amplitude: float = 0.0,
    ) -> None:
        if not isinstance(harmonics, int) or isinstance(harmonics, bool) or harmonics not in {2, 3}:
            raise ValueError("harmonics must be either 2 or 3 for SyntheticEEGSource")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer or None")
        interference_amplitude = _finite_number(interference_amplitude, "interference_amplitude")
        if interference_amplitude < 0:
            raise ValueError("interference_amplitude must not be negative")
        line_noise_amplitude = _finite_number(line_noise_amplitude, "line_noise_amplitude")
        if line_noise_amplitude < 0:
            raise ValueError("line_noise_amplitude must not be negative")
        if line_noise_hz is None and line_noise_amplitude != 0:
            line_noise_hz = 50.0
        if line_noise_hz is not None:
            line_noise_hz = _finite_number(line_noise_hz, "line_noise_hz")
            if line_noise_hz <= 0:
                raise ValueError("line_noise_hz must be greater than zero")

        candidate_frequencies = config.stimulus.frequencies_hz
        if not isinstance(interference_frequencies_hz, Sequence) or isinstance(interference_frequencies_hz, str):
            raise ValueError("interference_frequencies_hz must be a sequence of candidate frequencies")
        interference = tuple(
            _finite_number(frequency, "interference frequency") for frequency in interference_frequencies_hz
        )
        if len(set(interference)) != len(interference):
            raise ValueError("interference_frequencies_hz must not contain duplicates")
        if any(frequency not in candidate_frequencies for frequency in interference):
            raise ValueError("interference frequencies must be configured candidate frequencies")

        self._candidate_frequencies = candidate_frequencies
        self._harmonics = harmonics
        self._rng = np.random.default_rng(seed)
        self._interference_frequencies = interference
        self._interference_amplitude = interference_amplitude
        self._line_noise_hz = line_noise_hz
        self._line_noise_amplitude = line_noise_amplitude

    def generate(
        self,
        *,
        target_frequency_hz: float,
        duration_s: float,
        sample_rate_hz: float,
        channels: Sequence[str] | None = None,
        snr_db: float = 0.0,
    ) -> EEGWindow:
        """Generate one unlabeled ``[channels, samples]`` EEG window."""
        target_frequency_hz = _finite_number(target_frequency_hz, "target_frequency_hz")
        if target_frequency_hz not in self._candidate_frequencies:
            raise ValueError("target_frequency_hz must be one of the configured candidate frequencies")
        if target_frequency_hz in self._interference_frequencies:
            raise ValueError("interference_frequencies_hz must contain only non-target frequencies")
        duration_s = _finite_number(duration_s, "duration_s")
        sample_rate_hz = _finite_number(sample_rate_hz, "sample_rate_hz")
        snr_db = _finite_number(snr_db, "snr_db")
        if duration_s <= 0:
            raise ValueError("duration_s must be greater than zero")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be greater than zero")
        if channels is None:
            channels = CANONICAL_CHANNELS
        if not isinstance(channels, Sequence) or isinstance(channels, str) or tuple(channels) != CANONICAL_CHANNELS:
            raise ValueError("channels must exactly match the configured canonical channel order")
        if len(set(channels)) != len(channels):
            raise ValueError("channels must not contain duplicates")
        nyquist_hz = sample_rate_hz / 2.0
        active_frequencies = (target_frequency_hz, *self._interference_frequencies)
        for frequency in active_frequencies:
            if frequency <= 0 or self._harmonics * frequency >= nyquist_hz:
                raise ValueError("target/interference frequency or harmonic reaches the Nyquist frequency")
        if self._line_noise_hz is not None and self._line_noise_amplitude > 0 and self._line_noise_hz >= nyquist_hz:
            raise ValueError("line_noise_hz must be below the Nyquist frequency when enabled")

        samples = int(round(duration_s * sample_rate_hz))
        if samples < 2:
            raise ValueError("duration_s and sample_rate_hz must produce at least two samples")
        time_s = np.arange(samples, dtype=float) / sample_rate_hz
        target_signal = self._ssvep_component(time_s, target_frequency_hz, amplitude=1.0)
        target_rms = float(np.sqrt(np.mean(target_signal**2)))
        noise_std = target_rms / (10.0 ** (snr_db / 20.0))
        data = target_signal + self._rng.normal(0.0, noise_std, size=target_signal.shape)

        for frequency in self._interference_frequencies:
            if frequency != target_frequency_hz and self._interference_amplitude > 0:
                data += self._ssvep_component(time_s, frequency, amplitude=self._interference_amplitude)
        if self._line_noise_hz is not None and self._line_noise_amplitude > 0:
            phases = self._rng.uniform(0.0, 2.0 * np.pi, size=(len(channels), 1))
            data += self._line_noise_amplitude * np.sin(2.0 * np.pi * self._line_noise_hz * time_s + phases)

        start_time_s = time.monotonic()
        return EEGWindow(
            data=data.astype(np.float64, copy=False),
            sample_rate_hz=sample_rate_hz,
            channel_names=list(channels),
            start_time_s=start_time_s,
            end_time_s=start_time_s + samples / sample_rate_hz,
        )

    def _ssvep_component(self, time_s: np.ndarray, frequency_hz: float, *, amplitude: float) -> np.ndarray:
        """Build channel-major sinusoid sums with per-channel/per-harmonic phases."""
        channels = len(CANONICAL_CHANNELS)
        signal = np.zeros((channels, time_s.size), dtype=float)
        channel_gains = self._rng.uniform(0.8, 1.2, size=(channels, 1))
        for harmonic in range(1, self._harmonics + 1):
            phases = self._rng.uniform(0.0, 2.0 * np.pi, size=(channels, 1))
            harmonic_gains = self._rng.uniform(0.85, 1.15, size=(channels, 1))
            harmonic_amplitude = amplitude * channel_gains * harmonic_gains / harmonic
            signal += harmonic_amplitude * np.sin(2.0 * np.pi * harmonic * frequency_hz * time_s + phases)
        return signal
