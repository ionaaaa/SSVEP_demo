import numpy as np
import pytest

from ssvep_demo.protocol import DecodeResult, EEGWindow, TrialMarker


def test_trial_marker_uses_valid_monotonic_boundaries() -> None:
    marker = TrialMarker(1, 10.0, 100.0, 104.0)

    assert marker.target_frequency_hz == 10.0
    assert marker.stimulus_end_monotonic_s == 104.0


def test_eeg_window_is_channel_major_and_validates_channel_count() -> None:
    window = EEGWindow(np.zeros((2, 250)), 250.0, ["PO7", "PO3"], 10.0, 11.0)
    assert window.data.shape == (2, 250)

    with pytest.raises(ValueError, match="channel dimension"):
        EEGWindow(np.zeros((2, 250)), 250.0, ["PO7"], 10.0, 11.0)


def test_decode_result_carries_scores_command_and_monotonic_timestamp() -> None:
    result = DecodeResult(12.0, "FORWARD", {8.0: 0.2, 12.0: 0.9}, 0.9, 105.0)

    assert result.command == "FORWARD"
    assert result.scores[12.0] == 0.9
    assert result.timestamp_s == 105.0
