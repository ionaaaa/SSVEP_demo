#!/usr/bin/env python3
"""Run synthetic, replay, or omniBCI WebSocket live SSVEP trials."""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssvep_demo.config import load_config
from ssvep_demo.controller_factory import (
    ControllerFaultError,
    ControllerStartupError,
    controller_config_with_overrides,
)
from ssvep_demo.replay import ReplayDemoRunner
from ssvep_demo.live_demo import SSVEPLiveDemoRunner
from ssvep_demo.synthetic_demo import SSVEPSyntheticDemoRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Bluetooth car control requires Windows 10/11, a car already paired in Windows, "
            "and an installed car_sdk package. Real hardware may move: raise the wheels before testing."
        ),
    )
    parser.add_argument("--mode", choices=["synthetic", "replay", "live"], default="synthetic")
    parser.add_argument("--source", help="Live input source; currently only omnibci-websocket is supported.")
    parser.add_argument("--server-url", help="Live WebSocket endpoint, e.g. ws://127.0.0.1:8766/v1/stream")
    parser.add_argument("--alignment-mode", choices=["buffer-sequence", "trigger"], help="Live stimulus/EEG alignment strategy.")
    parser.add_argument("--trigger-url", help="Existing omniBCI Trigger endpoint, e.g. http://127.0.0.1:8766/v1/trigger")
    parser.add_argument("--decoder", choices=["fft", "cca"])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "ssvep_demo.yaml")
    parser.add_argument("--snr-db", type=float)
    parser.add_argument("--seed", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fullscreen", dest="fullscreen", action="store_true")
    mode.add_argument("--windowed", dest="fullscreen", action="store_false")
    parser.set_defaults(fullscreen=None)
    parser.add_argument("--refresh-rate", type=float, help="Measured-rate fallback or explicit debugging override.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ssvep_demo")
    parser.add_argument("--max-trials", type=int, help="Limit the prebuilt schedule; useful for smoke tests.")
    parser.add_argument("--confirmations-required", type=int, help="Override the mode's command confirmation count.")
    parser.add_argument("--replay-file", type=Path, help="Required EEG archive for --mode replay.")
    parser.add_argument("--trial-ids", help="Comma-separated archived trial IDs to replay.")
    parser.add_argument("--no-gui", action="store_true", help="Run synthetic or replay acceptance offline without creating a PsychoPy window.")
    parser.add_argument("--controller", choices=["virtual", "bluetooth"], help="Controller backend; overrides controller.type in YAML.")
    parser.add_argument("--car-device-name", help="Exact paired Bluetooth device name; device ID takes priority when both are set.")
    parser.add_argument("--car-device-id", help="Exact Bluetooth address/device ID; preferred over device name.")
    parser.add_argument("--car-connect-timeout-s", type=float, help="Finite Bluetooth startup timeout in seconds.")
    parser.add_argument("--car-io-timeout-s", type=float, help="Finite Bluetooth I/O timeout in seconds.")
    parser.add_argument("--car-close-timeout-s", type=float, help="Finite Bluetooth shutdown timeout in seconds.")
    return parser


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def apply_controller_cli_overrides(config, args: argparse.Namespace):
    bluetooth_overrides = {
        "device_name": args.car_device_name,
        "device_id": args.car_device_id,
        "connect_timeout_s": args.car_connect_timeout_s,
        "io_timeout_s": args.car_io_timeout_s,
        "close_timeout_s": args.car_close_timeout_s,
    }
    if (args.controller or config.controller.type) == "virtual" and any(
        value is not None for value in bluetooth_overrides.values()
    ):
        raise ValueError(
            "Bluetooth-specific --car-* options cannot be used when the final controller is virtual."
        )
    controller = controller_config_with_overrides(
        config.controller,
        controller_type=args.controller,
        **bluetooth_overrides,
    )
    return replace(config, controller=controller)


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    try:
        config = apply_controller_cli_overrides(load_config(args.config), args)
    except ValueError as exc:
        raise SystemExit(f"Controller configuration error: {exc}") from exc
    if args.mode == "replay" and config.controller.type == "bluetooth":
        raise SystemExit("Bluetooth controller is disabled in replay mode for safety.")
    if args.mode == "live":
        if not args.source:
            raise SystemExit("--source is required when --mode live")
        if not args.server_url:
            raise SystemExit("--server-url is required when --mode live")
        if args.source != "omnibci-websocket":
            raise SystemExit(
                f"unsupported live source {args.source!r}; currently only 'omnibci-websocket' is supported"
            )
        if args.replay_file is not None:
            raise SystemExit("--replay-file cannot be used when --mode live")
        if args.no_gui:
            raise SystemExit("--no-gui is not supported in live mode because collection is stimulus-trial aligned")
        alignment_mode = (
            args.alignment_mode.replace("-", "_") if args.alignment_mode is not None else config.live.alignment_mode
        )
        trigger_url = args.trigger_url if args.trigger_url is not None else config.live.trigger_url
        if alignment_mode == "trigger" and trigger_url is None:
            raise SystemExit("--trigger-url is required when --alignment-mode trigger")
        config = replace(config, live=replace(config.live, alignment_mode=alignment_mode, trigger_url=trigger_url))
        print("LIVE EEG MODE: omniBCI WebSocket input; no synthetic or replay EEG is used.")
        runner = SSVEPLiveDemoRunner(
            config,
            args.config,
            server_url=args.server_url,
            decoder_type=args.decoder,
            fullscreen=args.fullscreen,
            refresh_rate_hz=args.refresh_rate,
            output_dir=args.output_dir,
            max_trials=args.max_trials,
            confirmations_required=args.confirmations_required,
        )
        try:
            session_directory = runner.run()
        except (ControllerStartupError, ControllerFaultError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Live session completed: {session_directory}")
        print(f"EEG archive: {session_directory / 'eeg_windows.npz'}")
        print(f"Summary: {session_directory / 'summary.json'}")
        print(
            "已完成基于 omniBCI WebSocket 的 SSVEP live 数据接收、8 通道映射、连续 4 秒 EEG trial "
            "收集及单次解码的软件链路；真实人体诱发与设备级同步仍需现场验证。"
        )
        return
    if args.mode == "synthetic":
        print("SIMULATION MODE / 模拟模式: EEG is synthesized from the current trial target.")
        print("This does not validate real human SSVEP, real EEG synchronization, or a real car.")
        runner = SSVEPSyntheticDemoRunner(
            config,
            args.config,
            decoder_type=args.decoder,
            snr_db=args.snr_db,
            seed=args.seed,
            fullscreen=args.fullscreen,
            refresh_rate_hz=args.refresh_rate,
            output_dir=args.output_dir,
            max_trials=args.max_trials,
            confirmations_required=args.confirmations_required,
        )
        try:
            session_directory = (runner.run_offline if args.no_gui else runner.run)()
        except (ControllerStartupError, ControllerFaultError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Synthetic session completed: {session_directory}")
        print(f"EEG archive: {session_directory / 'eeg_windows.npz'}")
        print(f"Summary: {session_directory / 'summary.json'}")
        return
    if args.replay_file is None:
        raise SystemExit("--replay-file is required when --mode replay")
    trial_ids = None
    if args.trial_ids:
        try:
            trial_ids = [int(value) for value in args.trial_ids.split(",") if value.strip()]
        except ValueError as exc:
            raise SystemExit("--trial-ids must be a comma-separated list of integers") from exc
    session_directory = ReplayDemoRunner(
        config,
        args.replay_file,
        decoder_type=args.decoder,
        output_dir=args.output_dir,
        trial_ids=trial_ids,
        max_trials=args.max_trials,
        no_gui=args.no_gui,
        confirmations_required=args.confirmations_required,
    ).run()
    print("Replay is an offline decoder-acceptance run; it does not re-present visual stimuli.")
    print(f"Replay session completed: {session_directory}")
    print(f"Summary: {session_directory / 'summary.json'}")


if __name__ == "__main__":
    main()
