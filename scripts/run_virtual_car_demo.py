#!/usr/bin/env python3
"""Run the keyboard-driven virtual-car safety-control demonstration only."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssvep_demo.config import load_config
from ssvep_demo.control import SafeCommandDispatcher
from ssvep_demo.protocol import DecodeResult
from ssvep_demo.virtual_car import ControlDecisionLogger, PsychoPyVirtualCarView, VirtualCarController


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "ssvep_demo.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "virtual_car")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fullscreen", dest="fullscreen", action="store_true")
    mode.add_argument("--windowed", dest="fullscreen", action="store_false")
    parser.set_defaults(fullscreen=None)
    return parser.parse_args()


def _simulated_result(config, key: str, now_s: float) -> DecodeResult | None:
    key_map = {"1": 8.0, "2": 10.0, "3": 12.0, "4": 15.0}
    if key in key_map:
        frequency = key_map[key]
        confidence = 0.95
    elif key == "u":
        frequency = 8.0
        confidence = max(0.0, config.control.confidence_threshold - 0.10)
    else:
        return None
    return DecodeResult(
        predicted_frequency_hz=frequency,
        command=config.commands[frequency],
        scores={candidate: 0.0 for candidate in config.stimulus.frequencies_hz},
        confidence=confidence,
        timestamp_s=now_s,
    )


def main() -> None:
    args = arguments()
    config = load_config(args.config)
    try:
        import psychopy
        from psychopy import event, visual
    except ImportError as exc:
        raise RuntimeError("PsychoPy is required for the virtual-car GUI. Install the 'stimulus' extra.") from exc

    fullscreen = config.stimulus.fullscreen if args.fullscreen is None else args.fullscreen
    print(f"PsychoPy {psychopy.__version__}")
    print(f"Safety config: {config.control}")
    print("Keys: 1=LEFT (8 Hz), 2=RIGHT (10 Hz), 3=FORWARD (12 Hz), 4=BACKWARD (15 Hz), U=low confidence, Esc=exit")

    win = visual.Window(
        size=config.stimulus.window_size,
        fullscr=fullscreen,
        screen=config.stimulus.screen_index,
        color=config.stimulus.background_color,
        units="norm",
    )
    car = VirtualCarController()
    dispatcher = SafeCommandDispatcher(car, config.control, config.commands)
    logger = ControlDecisionLogger(
        args.output_dir,
        args.config,
        {
            "mode": "keyboard_virtual_car",
            "started_monotonic_s": time.monotonic(),
            "control": {
                "confidence_threshold": config.control.confidence_threshold,
                "confirmations_required": config.control.confirmations_required,
                "max_confirmation_gap_s": config.control.max_confirmation_gap_s,
                "command_duration_s": config.control.command_duration_s,
                "input_timeout_s": config.control.input_timeout_s,
                "max_prediction_age_s": config.control.max_prediction_age_s,
            },
        },
    )
    print(f"Output session: {logger.directory}")
    view = PsychoPyVirtualCarView(win, visual, config.control.confirmations_required)
    last_decision = None
    last_confidence = None
    previous_frame_s = time.monotonic()
    try:
        running = True
        while running:
            now_s = time.monotonic()
            frame_start_s = previous_frame_s
            deadline = dispatcher.motion_deadline_s
            elapsed_s = now_s - frame_start_s
            if deadline is not None:
                elapsed_s = max(0.0, min(now_s, deadline) - frame_start_s)
            car.update(max(0.0, elapsed_s))
            previous_frame_s = now_s
            for key in event.getKeys():
                if key == "escape":
                    running = False
                    break
                result = _simulated_result(config, key, now_s)
                if result is None:
                    continue
                last_confidence = result.confidence
                last_decision = dispatcher.submit(result)
                logger.write(
                    last_decision,
                    car,
                    prediction_timestamp_s=result.timestamp_s,
                    confidence=result.confidence,
                )
            if not running:
                break
            timeout_decision = dispatcher.tick(now_s)
            if timeout_decision is not None:
                last_decision = timeout_decision
                logger.write(timeout_decision, car, prediction_timestamp_s=None, confidence=None)
            view.draw(car, last_decision, last_confidence)
            win.flip()
    finally:
        try:
            dispatcher.close()
        finally:
            logger.close()
            win.close()


if __name__ == "__main__":
    main()
