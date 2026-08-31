from pathlib import Path

import pytest
import yaml

from ssvep_demo.config import load_config
from ssvep_demo.exceptions import ConfigurationError
from ssvep_demo.protocol import CANONICAL_CHANNELS


CONFIG_PATH = Path(__file__).parents[1] / "config" / "ssvep_demo.yaml"


def _base_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_load_demo_config() -> None:
    config = load_config(CONFIG_PATH)

    assert config.stimulus.frequencies_hz == (8.0, 10.0, 12.0, 15.0)
    assert config.acquisition.channels == CANONICAL_CHANNELS
    assert config.decoder.type == "cca"
    assert config.commands == {8.0: "LEFT", 10.0: "RIGHT", 12.0: "FORWARD", 15.0: "STOP"}
    assert config.stimulus.cue_duration_s == 1.0
    assert config.stimulus.window_size == (1200, 800)
    assert config.ui.cjk_font_file is None


def test_old_config_without_visual_fields_uses_stimulus_defaults(tmp_path: Path) -> None:
    config = _base_config()
    config.pop("ui")
    for field in (
        "cue_duration_s", "trial_order", "random_seed", "fullscreen", "window_size", "screen_index",
        "background_color", "stimulus_on_color", "stimulus_off_color", "dropped_frame_threshold_ratio",
    ):
        del config["stimulus"][field]
    path = tmp_path / "stage_one.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(path)
    assert loaded.stimulus.trial_order == "fixed"
    assert loaded.stimulus.dropped_frame_threshold_ratio == 1.5
    assert loaded.ui.cjk_font_file is None


def test_load_config_accepts_optional_cjk_font_override(tmp_path: Path) -> None:
    config = _base_config()
    config["ui"] = {"cjk_font_file": "fonts/custom.otf", "cjk_font_name": "Custom CJK"}
    path = tmp_path / "font_override.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(path)
    # The raw path is preserved; GUI-independent resolution later interprets it
    # relative to the YAML file's directory.
    assert loaded.ui.cjk_font_file == "fonts/custom.otf"
    assert loaded.ui.cjk_font_name == "Custom CJK"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["stimulus"].update(frequencies=[8, 8]), "must not contain duplicates"),
        (lambda config: config["acquisition"].update(channels=list(reversed(CANONICAL_CHANNELS))), "must exactly be"),
        (lambda config: config["decoder"].update(type="trca"), "must be either"),
        (lambda config: config["decoder"].update(bandpass_hz=[6, 125]), "Nyquist"),
        (lambda config: config.update(commands={"8": "LEFT"}), "every and only"),
        (lambda config: config["stimulus"].update(trial_order="shuffle"), "trial_order"),
        (lambda config: config["stimulus"].update(window_size=[0, 800]), "window_size"),
        (lambda config: config["stimulus"].update(background_color=[2, 0, 0]), "between -1 and 1"),
    ],
)
def test_load_config_rejects_invalid_protocol(tmp_path: Path, mutate, message: str) -> None:
    config = _base_config()
    mutate(config)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_config(path)
