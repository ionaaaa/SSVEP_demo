"""Shared configuration and data protocol for the SSVEP demonstration."""

from .config import DemoConfig, UIConfig, load_config
from .control import ControlCommand, ControlConfig, Controller, DispatchDecision, SafeCommandDispatcher
from .decoders import CCADecoder, FFTDecoder, SSVEPDecoder
from .protocol import CANONICAL_CHANNELS, DecodeResult, EEGWindow, TrialMarker
from .synthetic import SyntheticEEGSource
from .stimulus import FlickerScheduler, PsychoPyStimulusRunner, TrialState, TrialStateMachine
from .virtual_car import VirtualCarController

__all__ = [
    "CCADecoder",
    "CANONICAL_CHANNELS",
    "ControlCommand",
    "ControlConfig",
    "Controller",
    "DecodeResult",
    "DispatchDecision",
    "DemoConfig",
    "EEGWindow",
    "FFTDecoder",
    "FlickerScheduler",
    "PsychoPyStimulusRunner",
    "SSVEPDecoder",
    "SafeCommandDispatcher",
    "SyntheticEEGSource",
    "TrialMarker",
    "TrialState",
    "TrialStateMachine",
    "UIConfig",
    "VirtualCarController",
    "load_config",
]
