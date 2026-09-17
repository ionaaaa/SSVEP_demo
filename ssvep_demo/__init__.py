"""Shared configuration and data protocol for the SSVEP demonstration."""

from .config import DemoConfig, LiveConfig, SyntheticDemoConfig, UIConfig, load_config
from .control import ControlCommand, ControlConfig, Controller, DispatchDecision, SafeCommandDispatcher
from .decoders import CCADecoder, FFTDecoder, SSVEPDecoder
from .protocol import CANONICAL_CHANNELS, DecodeResult, EEGWindow, TrialMarker
from .synthetic import SyntheticEEGSource
from .stimulus import FlickerScheduler, PsychoPyStimulusRunner, TrialState, TrialStateMachine
from .synthetic_demo import SSVEPSyntheticDemoRunner, SyntheticDemoState, SyntheticDemoStateMachine
from .eeg_archive import EEGWindowArchiveWriter, ReplayEEGSource
from .replay import ReplayDemoRunner
from .live_source import (
    LiveEEGSource,
    LiveSampleFrame,
    LiveSourceStatus,
    LiveStreamMetadata,
    OmniBCIWebSocketSource,
)
from .live_demo import LiveDemoState, LiveDemoStateMachine, SSVEPLiveDemoRunner
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
    "EEGWindowArchiveWriter",
    "FFTDecoder",
    "FlickerScheduler",
    "LiveConfig",
    "LiveDemoState",
    "LiveDemoStateMachine",
    "LiveEEGSource",
    "LiveSampleFrame",
    "LiveSourceStatus",
    "LiveStreamMetadata",
    "OmniBCIWebSocketSource",
    "PsychoPyStimulusRunner",
    "SSVEPDecoder",
    "SafeCommandDispatcher",
    "ReplayDemoRunner",
    "ReplayEEGSource",
    "SyntheticEEGSource",
    "SyntheticDemoConfig",
    "SyntheticDemoState",
    "SyntheticDemoStateMachine",
    "SSVEPSyntheticDemoRunner",
    "SSVEPLiveDemoRunner",
    "TrialMarker",
    "TrialState",
    "TrialStateMachine",
    "UIConfig",
    "VirtualCarController",
    "load_config",
]
