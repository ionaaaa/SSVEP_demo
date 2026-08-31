import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ssvep_demo.config import load_config
from ssvep_demo.eeg_archive import EEGWindowArchiveWriter, ReplayEEGSource
from ssvep_demo.replay import ReplayDemoRunner
from ssvep_demo.synthetic_demo import SSVEPSyntheticDemoRunner


CONFIG_PATH = Path(__file__).parents[1] / "config" / "ssvep_demo.yaml"
CONFIG = load_config(CONFIG_PATH)


def _synthetic_session(tmp_path: Path, *, seed: int = 7) -> Path:
    return SSVEPSyntheticDemoRunner(
        CONFIG, CONFIG_PATH, decoder_type="cca", seed=seed, output_dir=tmp_path, max_trials=2
    ).run_offline()


def test_synthetic_offline_session_has_resolved_config_logs_and_pickle_free_archive(tmp_path: Path) -> None:
    session = _synthetic_session(tmp_path)
    expected = {"config.yaml", "trials.csv", "events.jsonl", "eeg_windows.npz", "summary.json", "frame_intervals.csv"}
    assert expected.issubset({path.name for path in session.iterdir()})
    config_text = (session / "config.yaml").read_text(encoding="utf-8")
    assert "type: cca" in config_text and "seed: 7" in config_text
    with np.load(session / "eeg_windows.npz", allow_pickle=False) as archive:
        assert archive["data"].shape == (2, 8, 1000)
        assert archive["data"].dtype == np.float32
        assert archive["channel_names"].tolist() == list(CONFIG.acquisition.channels)
    with (session / "trials.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {"attempt_id", "eeg_window_index", "decoder_scores_json", "dropped_frames", "decoder_latency_ms"}.issubset(rows[0])
    assert [row["eeg_window_index"] for row in rows] == ["0", "1"]
    summary = json.loads((session / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_trials"] == len(rows) == summary["archived_eeg_windows"]
    assert summary["total_dropped_frames"] == sum(int(row["dropped_frames"] or 0) for row in rows)
    assert summary["mode"] == "synthetic"


def test_archived_eeg_is_reproducible_and_replay_returns_original_decoder_input(tmp_path: Path) -> None:
    first = _synthetic_session(tmp_path / "a", seed=9)
    second = _synthetic_session(tmp_path / "b", seed=9)
    with np.load(first / "eeg_windows.npz", allow_pickle=False) as left, np.load(second / "eeg_windows.npz", allow_pickle=False) as right:
        assert np.array_equal(left["data"], right["data"])
    replay = ReplayEEGSource(first / "eeg_windows.npz")
    with np.load(first / "eeg_windows.npz", allow_pickle=False) as archive:
        assert np.array_equal(replay.window(0).data, archive["data"][0])
        assert replay.window(0).sample_rate_hz == archive["sample_rate_hz"][0]
    assert replay.indices_for_trial(0) == [0]


@pytest.mark.parametrize("decoder", ["fft", "cca"])
def test_replay_creates_independent_sessions_without_touching_source_or_synthetic_source(tmp_path: Path, decoder: str, monkeypatch) -> None:
    source_session = _synthetic_session(tmp_path / "source")
    source_archive = source_session / "eeg_windows.npz"
    original_bytes = source_archive.read_bytes()

    import ssvep_demo.replay as replay_module

    def forbidden_synthetic(*args, **kwargs):
        raise AssertionError("replay must not construct SyntheticEEGSource")

    monkeypatch.setattr(replay_module, "SyntheticEEGSource", forbidden_synthetic, raising=False)
    replay_session = ReplayDemoRunner(
        CONFIG, source_archive, decoder_type=decoder, output_dir=tmp_path / decoder, no_gui=True
    ).run()
    assert source_archive.read_bytes() == original_bytes
    assert replay_session != source_session
    summary = json.loads((replay_session / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "replay"
    assert summary["source_replay_file"] == "eeg_windows.npz"
    assert (replay_session / "eeg_windows.npz").is_file()


def test_archive_rejects_missing_schema_object_nan_and_channel_mismatch(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npz"
    np.savez_compressed(missing, schema_version=np.asarray(1))
    with pytest.raises(ValueError, match="missing required"):
        ReplayEEGSource(missing)

    wrong_schema = tmp_path / "wrong_schema.npz"
    np.savez_compressed(wrong_schema, schema_version=np.asarray(99))
    with pytest.raises(ValueError, match="missing required|Unsupported"):
        ReplayEEGSource(wrong_schema)

    object_data = tmp_path / "object.npz"
    np.savez_compressed(
        object_data,
        schema_version=np.asarray(1), data=np.asarray([[[object()]]], dtype=object), trial_id=np.asarray([0]),
        attempt_id=np.asarray([0]), sample_rate_hz=np.asarray([250.0]), channel_names=np.asarray(CONFIG.acquisition.channels),
        window_start_monotonic_s=np.asarray([0.0]), window_end_monotonic_s=np.asarray([1.0]),
        target_frequency_hz=np.asarray([8.0]), source_mode=np.asarray(["synthetic"]), data_unit=np.asarray("u"),
    )
    with pytest.raises(ValueError):
        ReplayEEGSource(object_data)

    writer = EEGWindowArchiveWriter(tmp_path / "valid.npz", CONFIG.acquisition.channels, 1000)
    from ssvep_demo.synthetic import SyntheticEEGSource

    window = SyntheticEEGSource(CONFIG, seed=1).generate(
        target_frequency_hz=8.0, duration_s=4.0, sample_rate_hz=250.0, snr_db=0.0
    )
    writer.add(window, trial_id=0, attempt_id=0, target_frequency_hz=8.0, source_mode="synthetic")
    with pytest.raises(ValueError, match="channel_names"):
        ReplayEEGSource(tmp_path / "valid.npz", expected_channels=tuple(reversed(CONFIG.acquisition.channels)))

    with np.load(tmp_path / "valid.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["data"] = arrays["data"].copy()
    arrays["data"][0, 0, 0] = np.nan
    nan_archive = tmp_path / "nan.npz"
    np.savez_compressed(nan_archive, **arrays)
    with pytest.raises(ValueError, match="finite"):
        ReplayEEGSource(nan_archive)


def test_empty_archive_is_valid_and_replay_is_headless_on_import(tmp_path: Path) -> None:
    writer = EEGWindowArchiveWriter(tmp_path / "empty.npz", CONFIG.acquisition.channels, 1000)
    source = ReplayEEGSource(writer.path)
    assert len(source) == 0
    import ssvep_demo.replay as module

    assert "psychopy" not in module.__dict__
