from pathlib import Path

import numpy as np
import pytest

from ssvep_demo.config import load_config
from ssvep_demo.decoders import CCADecoder, FFTDecoder
from ssvep_demo.protocol import CANONICAL_CHANNELS, EEGWindow
from ssvep_demo.synthetic import SyntheticEEGSource


CONFIG = load_config(Path(__file__).parents[1] / "config" / "ssvep_demo.yaml")


@pytest.fixture(params=[FFTDecoder, CCADecoder], ids=["fft", "cca"])
def decoder(request):
    return request.param(CONFIG)


@pytest.mark.parametrize("decoder_class", [FFTDecoder, CCADecoder])
def test_all_candidate_frequencies_decode_at_high_snr(decoder_class) -> None:
    source = SyntheticEEGSource(CONFIG, seed=42)
    decoder = decoder_class(CONFIG)
    for frequency in CONFIG.stimulus.frequencies_hz:
        window = source.generate(
            target_frequency_hz=frequency,
            duration_s=4.0,
            sample_rate_hz=250.0,
            snr_db=10.0,
        )
        result = decoder.decode(window)
        assert result.predicted_frequency_hz == frequency
        assert result.command == CONFIG.commands[frequency]
        assert set(result.scores) == set(CONFIG.stimulus.frequencies_hz)
        assert np.isfinite(list(result.scores.values())).all()
        assert np.isfinite(result.confidence)
        # EEGWindow intentionally has no label field; an incidental dynamic
        # attribute must not affect decoding either.
        window.target_frequency_hz = 15.0
        assert decoder.decode(window).predicted_frequency_hz == frequency


def test_source_is_reproducible_by_seed_and_has_required_layout() -> None:
    parameters = dict(target_frequency_hz=10.0, duration_s=4.0, sample_rate_hz=250.0, snr_db=0.0)
    first = SyntheticEEGSource(CONFIG, seed=7).generate(**parameters)
    second = SyntheticEEGSource(CONFIG, seed=7).generate(**parameters)
    different = SyntheticEEGSource(CONFIG, seed=8).generate(**parameters)

    assert first.data.shape == (8, 1000)
    assert first.channel_names == list(CANONICAL_CHANNELS)
    assert np.array_equal(first.data, second.data)
    assert not np.array_equal(first.data, different.data)
    assert not hasattr(first, "target_frequency_hz")


def test_harmonics_interference_and_line_noise_options_change_generated_data() -> None:
    parameters = dict(target_frequency_hz=10.0, duration_s=4.0, sample_rate_hz=250.0, snr_db=0.0)
    baseline = SyntheticEEGSource(CONFIG, harmonics=2, seed=9).generate(**parameters)
    with_third_harmonic = SyntheticEEGSource(CONFIG, harmonics=3, seed=9).generate(**parameters)
    with_interference = SyntheticEEGSource(
        CONFIG, seed=9, interference_frequencies_hz=(12.0,), interference_amplitude=0.5
    ).generate(**parameters)
    with_line_noise = SyntheticEEGSource(CONFIG, seed=9, line_noise_amplitude=0.5).generate(**parameters)

    assert not np.array_equal(baseline.data, with_third_harmonic.data)
    assert not np.array_equal(baseline.data, with_interference.data)
    assert not np.array_equal(baseline.data, with_line_noise.data)


def test_source_rejects_invalid_frequency_channels_and_nyquist() -> None:
    source = SyntheticEEGSource(CONFIG, seed=1)
    with pytest.raises(ValueError, match="configured candidate"):
        source.generate(target_frequency_hz=9.0, duration_s=4.0, sample_rate_hz=250.0)
    with pytest.raises(ValueError, match="canonical channel order"):
        source.generate(target_frequency_hz=8.0, duration_s=4.0, sample_rate_hz=250.0, channels=["PO7"])
    with pytest.raises(ValueError, match="Nyquist"):
        source.generate(target_frequency_hz=15.0, duration_s=4.0, sample_rate_hz=80.0)


def test_decoders_fail_fast_for_invalid_windows(decoder) -> None:
    valid = SyntheticEEGSource(CONFIG, seed=2).generate(
        target_frequency_hz=8.0, duration_s=4.0, sample_rate_hz=250.0
    )
    with pytest.raises(ValueError, match="finite"):
        decoder.decode(EEGWindow(np.where(np.ones_like(valid.data), np.nan, valid.data), 250.0, list(CANONICAL_CHANNELS), 0, 4))
    with pytest.raises(ValueError, match="finite"):
        decoder.decode(EEGWindow(np.full_like(valid.data, np.inf), 250.0, list(CANONICAL_CHANNELS), 0, 4))
    with pytest.raises(ValueError, match="sample_rate"):
        decoder.decode(EEGWindow(valid.data, 200.0, list(CANONICAL_CHANNELS), 0, 5))
    with pytest.raises(ValueError, match="channels"):
        decoder.decode(EEGWindow(valid.data, 250.0, list(reversed(CANONICAL_CHANNELS)), 0, 4))
    with pytest.raises(ValueError, match="channels"):
        decoder.decode(EEGWindow(valid.data[:-1], 250.0, list(CANONICAL_CHANNELS[:-1]), 0, 4))
    with pytest.raises(ValueError, match="too short"):
        decoder.decode(EEGWindow(np.zeros((8, 100)), 250.0, list(CANONICAL_CHANNELS), 0, 0.4))


@pytest.mark.parametrize("decoder_class", [FFTDecoder, CCADecoder])
def test_high_snr_accuracy_exceeds_extremely_low_snr_in_fixed_batch(decoder_class) -> None:
    decoder = decoder_class(CONFIG)

    def accuracy(snr_db: float) -> float:
        source = SyntheticEEGSource(CONFIG, seed=123)
        predictions = []
        for frequency in CONFIG.stimulus.frequencies_hz:
            for _ in range(6):
                window = source.generate(
                    target_frequency_hz=frequency, duration_s=4.0, sample_rate_hz=250.0, snr_db=snr_db
                )
                predictions.append(decoder.decode(window).predicted_frequency_hz == frequency)
        return sum(predictions) / len(predictions)

    assert accuracy(10.0) >= 0.9
    assert accuracy(10.0) > accuracy(-35.0) + 0.25
