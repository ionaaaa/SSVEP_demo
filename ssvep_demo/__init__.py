"""Shared configuration and data protocol for the SSVEP demonstration."""

from .config import DemoConfig, UIConfig, load_config
from .decoders import CCADecoder, FFTDecoder, SSVEPDecoder
from .protocol import CANONICAL_CHANNELS, DecodeResult, EEGWindow, TrialMarker
from .synthetic import SyntheticEEGSource
from .stimulus import FlickerScheduler, PsychoPyStimulusRunner, TrialState, TrialStateMachine

__all__ = [
    "CCADecoder",
    "CANONICAL_CHANNELS",
    "DecodeResult",
    "DemoConfig",
    "EEGWindow",
    "FFTDecoder",
    "FlickerScheduler",
    "PsychoPyStimulusRunner",
    "SSVEPDecoder",
    "SyntheticEEGSource",
    "TrialMarker",
    "TrialState",
    "TrialStateMachine",
    "UIConfig",
    "load_config",
]
