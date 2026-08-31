"""Loading and validation for the SSVEP demo's shared YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .exceptions import ConfigurationError
from .control import ControlCommand, ControlConfig
from .protocol import CANONICAL_CHANNELS


@dataclass(frozen=True)
class StimulusConfig:
    frequencies_hz: tuple[float, ...]
    trial_duration_s: float
    rest_duration_s: float
    repetitions: int
    cue_duration_s: float = 1.0
    trial_order: str = "fixed"
    random_seed: int = 42
    fullscreen: bool = False
    window_size: tuple[int, int] = (1200, 800)
    screen_index: int = 0
    background_color: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    stimulus_on_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    stimulus_off_color: tuple[float, float, float] = (-0.6, -0.6, -0.6)
    dropped_frame_threshold_ratio: float = 1.5


@dataclass(frozen=True)
class AcquisitionConfig:
    sample_rate_hz: float
    channels: tuple[str, ...]


@dataclass(frozen=True)
class DecoderConfig:
    type: str
    harmonics: int
    bandpass_hz: tuple[float, float]


@dataclass(frozen=True)
class UIConfig:
    """Optional visual UI settings, kept separate from signal configuration."""

    cjk_font_file: str | None = None
    cjk_font_name: str | None = None


@dataclass(frozen=True)
class SyntheticDemoConfig:
    """Deterministic settings for the explicitly simulated closed-loop demo."""

    snr_db: float = 0.0
    seed: int = 42
    confirmations_required: int = 1


@dataclass(frozen=True)
class DemoConfig:
    stimulus: StimulusConfig
    acquisition: AcquisitionConfig
    decoder: DecoderConfig
    commands: dict[float, ControlCommand]
    ui: UIConfig = field(default_factory=UIConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    synthetic_demo: SyntheticDemoConfig = field(default_factory=SyntheticDemoConfig)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field_name} must be a mapping")
    return value


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field_name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigurationError(f"{field_name} must be finite")
    return value


def _required(section: Mapping[str, Any], name: str, section_name: str) -> Any:
    try:
        return section[name]
    except KeyError as exc:
        raise ConfigurationError(f"Missing required field: {section_name}.{name}") from exc


def _optional(section: Mapping[str, Any], name: str, default: Any) -> Any:
    """Read a backwards-compatible optional configuration field."""
    return section.get(name, default)


def _color(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigurationError(f"{field_name} must be a three-item RGB list")
    color = tuple(_number(component, field_name) for component in value)
    if any(component < -1.0 or component > 1.0 for component in color):
        raise ConfigurationError(f"{field_name} values must be between -1 and 1")
    return color


def _window_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigurationError("stimulus.window_size must be a [width, height] list")
    if any(isinstance(component, bool) or not isinstance(component, int) or component <= 0 for component in value):
        raise ConfigurationError("stimulus.window_size values must be positive integers")
    return (value[0], value[1])


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field_name} must be a non-empty string or null")
    return value


def _load_mapping(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration: {config_path}") from exc
    return _mapping(loaded, "configuration root")


def load_config(path: str | Path) -> DemoConfig:
    """Load an SSVEP demo YAML file and enforce the stage-one protocol."""
    root = _load_mapping(path)
    stimulus_data = _mapping(_required(root, "stimulus", "root"), "stimulus")
    acquisition_data = _mapping(_required(root, "acquisition", "root"), "acquisition")
    decoder_data = _mapping(_required(root, "decoder", "root"), "decoder")
    commands_data = _mapping(_required(root, "commands", "root"), "commands")
    ui_data = _mapping(_optional(root, "ui", {}), "ui")
    control_data = _mapping(_optional(root, "control", {}), "control")
    synthetic_data = _mapping(_optional(root, "synthetic_demo", {}), "synthetic_demo")

    frequencies_value = _required(stimulus_data, "frequencies", "stimulus")
    if not isinstance(frequencies_value, list) or not frequencies_value:
        raise ConfigurationError("stimulus.frequencies must be a non-empty list")
    frequencies = tuple(_number(value, "stimulus.frequencies item") for value in frequencies_value)
    if any(frequency <= 0 for frequency in frequencies):
        raise ConfigurationError("stimulus.frequencies must contain only values greater than zero")
    if len(set(frequencies)) != len(frequencies):
        raise ConfigurationError("stimulus.frequencies must not contain duplicates")

    trial_duration_s = _number(_required(stimulus_data, "trial_duration_s", "stimulus"), "stimulus.trial_duration_s")
    rest_duration_s = _number(_required(stimulus_data, "rest_duration_s", "stimulus"), "stimulus.rest_duration_s")
    if trial_duration_s <= 0:
        raise ConfigurationError("stimulus.trial_duration_s must be greater than zero")
    if rest_duration_s < 0:
        raise ConfigurationError("stimulus.rest_duration_s must not be negative")
    repetitions = _required(stimulus_data, "repetitions", "stimulus")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ConfigurationError("stimulus.repetitions must be a positive integer")
    cue_duration_s = _number(_optional(stimulus_data, "cue_duration_s", 1.0), "stimulus.cue_duration_s")
    if cue_duration_s < 0:
        raise ConfigurationError("stimulus.cue_duration_s must not be negative")
    trial_order = _optional(stimulus_data, "trial_order", "fixed")
    if trial_order not in {"fixed", "random"}:
        raise ConfigurationError("stimulus.trial_order must be either 'fixed' or 'random'")
    random_seed = _optional(stimulus_data, "random_seed", 42)
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ConfigurationError("stimulus.random_seed must be an integer")
    fullscreen = _optional(stimulus_data, "fullscreen", False)
    if not isinstance(fullscreen, bool):
        raise ConfigurationError("stimulus.fullscreen must be a boolean")
    window_size = _window_size(_optional(stimulus_data, "window_size", [1200, 800]))
    screen_index = _optional(stimulus_data, "screen_index", 0)
    if isinstance(screen_index, bool) or not isinstance(screen_index, int) or screen_index < 0:
        raise ConfigurationError("stimulus.screen_index must be a non-negative integer")
    background_color = _color(_optional(stimulus_data, "background_color", [-1, -1, -1]), "stimulus.background_color")
    stimulus_on_color = _color(
        _optional(stimulus_data, "stimulus_on_color", [1, 1, 1]), "stimulus.stimulus_on_color"
    )
    stimulus_off_color = _color(
        _optional(stimulus_data, "stimulus_off_color", [-0.6, -0.6, -0.6]), "stimulus.stimulus_off_color"
    )
    dropped_frame_threshold_ratio = _number(
        _optional(stimulus_data, "dropped_frame_threshold_ratio", 1.5),
        "stimulus.dropped_frame_threshold_ratio",
    )
    if dropped_frame_threshold_ratio <= 1.0:
        raise ConfigurationError("stimulus.dropped_frame_threshold_ratio must be greater than 1")

    sample_rate_hz = _number(
        _required(acquisition_data, "sample_rate_hz", "acquisition"), "acquisition.sample_rate_hz"
    )
    if sample_rate_hz <= 0:
        raise ConfigurationError("acquisition.sample_rate_hz must be greater than zero")
    channels = _required(acquisition_data, "channels", "acquisition")
    if not isinstance(channels, list) or tuple(channels) != CANONICAL_CHANNELS:
        expected = ", ".join(CANONICAL_CHANNELS)
        raise ConfigurationError(f"acquisition.channels must exactly be: {expected}")
    if len(set(channels)) != len(channels):
        raise ConfigurationError("acquisition.channels must not contain duplicates")

    decoder_type = _required(decoder_data, "type", "decoder")
    if decoder_type not in {"fft", "cca"}:
        raise ConfigurationError("decoder.type must be either 'fft' or 'cca'")
    harmonics = _required(decoder_data, "harmonics", "decoder")
    if isinstance(harmonics, bool) or not isinstance(harmonics, int) or harmonics <= 0:
        raise ConfigurationError("decoder.harmonics must be a positive integer")
    bandpass = _required(decoder_data, "bandpass_hz", "decoder")
    if not isinstance(bandpass, list) or len(bandpass) != 2:
        raise ConfigurationError("decoder.bandpass_hz must contain exactly [low, high]")
    bandpass_hz = tuple(_number(value, "decoder.bandpass_hz item") for value in bandpass)
    if bandpass_hz[0] >= bandpass_hz[1]:
        raise ConfigurationError("decoder.bandpass_hz lower bound must be less than upper bound")
    if bandpass_hz[1] >= sample_rate_hz / 2:
        raise ConfigurationError("decoder.bandpass_hz upper bound must be below the Nyquist frequency")

    commands: dict[float, ControlCommand] = {}
    for raw_frequency, command in commands_data.items():
        try:
            frequency = float(raw_frequency)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("commands keys must be numeric stimulus frequencies") from exc
        if not math.isfinite(frequency):
            raise ConfigurationError("commands keys must be finite stimulus frequencies")
        try:
            command = ControlCommand.parse(command)
        except ValueError as exc:
            raise ConfigurationError(f"commands values must be known control commands: {exc}") from exc
        if command is ControlCommand.UNKNOWN:
            raise ConfigurationError("commands values must not be UNKNOWN")
        if command is ControlCommand.STOP:
            raise ConfigurationError("commands values must not map a stimulus frequency to STOP")
        if frequency in commands:
            raise ConfigurationError("commands must not contain duplicate frequency mappings")
        commands[frequency] = command
    if set(commands) != set(frequencies):
        raise ConfigurationError("commands must map every and only stimulus frequency")
    if len(set(commands.values())) != len(commands):
        raise ConfigurationError("each stimulus frequency must have a unique command mapping")

    ui = UIConfig(
        cjk_font_file=_optional_string(_optional(ui_data, "cjk_font_file", None), "ui.cjk_font_file"),
        cjk_font_name=_optional_string(_optional(ui_data, "cjk_font_name", None), "ui.cjk_font_name"),
    )
    try:
        control = ControlConfig(
            confidence_threshold=_number(
                _optional(control_data, "confidence_threshold", 0.60), "control.confidence_threshold"
            ),
            confirmations_required=_optional(control_data, "confirmations_required", 2),
            max_confirmation_gap_s=_number(
                _optional(control_data, "max_confirmation_gap_s", 1.0), "control.max_confirmation_gap_s"
            ),
            command_duration_s=_number(
                _optional(control_data, "command_duration_s", 0.5), "control.command_duration_s"
            ),
            input_timeout_s=_number(_optional(control_data, "input_timeout_s", 1.0), "control.input_timeout_s"),
            max_prediction_age_s=_number(
                _optional(control_data, "max_prediction_age_s", 0.5), "control.max_prediction_age_s"
            ),
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    synthetic_seed = _optional(synthetic_data, "seed", 42)
    synthetic_confirmations = _optional(synthetic_data, "confirmations_required", 1)
    if isinstance(synthetic_seed, bool) or not isinstance(synthetic_seed, int) or synthetic_seed < 0:
        raise ConfigurationError("synthetic_demo.seed must be a non-negative integer")
    if (
        isinstance(synthetic_confirmations, bool)
        or not isinstance(synthetic_confirmations, int)
        or synthetic_confirmations <= 0
    ):
        raise ConfigurationError("synthetic_demo.confirmations_required must be a positive integer")
    synthetic_demo = SyntheticDemoConfig(
        snr_db=_number(_optional(synthetic_data, "snr_db", 0.0), "synthetic_demo.snr_db"),
        seed=synthetic_seed,
        confirmations_required=synthetic_confirmations,
    )

    return DemoConfig(
        stimulus=StimulusConfig(
            frequencies,
            trial_duration_s,
            rest_duration_s,
            repetitions,
            cue_duration_s,
            trial_order,
            random_seed,
            fullscreen,
            window_size,
            screen_index,
            background_color,
            stimulus_on_color,
            stimulus_off_color,
            dropped_frame_threshold_ratio,
        ),
        acquisition=AcquisitionConfig(sample_rate_hz, tuple(channels)),
        decoder=DecoderConfig(decoder_type, harmonics, bandpass_hz),
        commands=commands,
        ui=ui,
        control=control,
        synthetic_demo=synthetic_demo,
    )
