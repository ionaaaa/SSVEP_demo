#!/usr/bin/env python3
"""Run the explicitly simulated SSVEP closed-loop demonstration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssvep_demo.config import load_config
from ssvep_demo.replay import ReplayDemoRunner
from ssvep_demo.synthetic_demo import SSVEPSyntheticDemoRunner


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "replay"], default="synthetic")
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
    parser.add_argument("--confirmations-required", type=int, help="Override synthetic_demo.confirmations_required.")
    parser.add_argument("--replay-file", type=Path, help="Required EEG archive for --mode replay.")
    parser.add_argument("--trial-ids", help="Comma-separated archived trial IDs to replay.")
    parser.add_argument("--no-gui", action="store_true", help="Run synthetic or replay acceptance offline without creating a PsychoPy window.")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = load_config(args.config)
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
        session_directory = (runner.run_offline if args.no_gui else runner.run)()
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
