#!/usr/bin/env python3
"""Run deterministic synthetic SSVEP decoder evaluations and save JSON metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssvep_demo.config import load_config
from ssvep_demo.decoders import CCADecoder, FFTDecoder
from ssvep_demo.synthetic import SyntheticEEGSource


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", choices=("fft", "cca"), required=True)
    parser.add_argument("--trials-per-frequency", type=int, default=100)
    parser.add_argument("--snr-db", type=float, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "ssvep_demo.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    if args.trials_per_frequency <= 0:
        raise ValueError("--trials-per-frequency must be positive")
    config = load_config(args.config)
    decoder = FFTDecoder(config) if args.decoder == "fft" else CCADecoder(config)
    labels = list(config.stimulus.frequencies_hz)
    reports = []
    for snr_db in args.snr_db:
        started = time.monotonic()
        source = SyntheticEEGSource(config, harmonics=min(3, config.decoder.harmonics), seed=args.seed)
        matrix = [[0 for _ in labels] for _ in labels]
        confidences = []
        per_frequency = {frequency: [0, 0] for frequency in labels}
        for actual_index, actual_frequency in enumerate(labels):
            for _ in range(args.trials_per_frequency):
                window = source.generate(
                    target_frequency_hz=actual_frequency,
                    duration_s=config.stimulus.trial_duration_s,
                    sample_rate_hz=config.acquisition.sample_rate_hz,
                    channels=config.acquisition.channels,
                    snr_db=snr_db,
                )
                result = decoder.decode(window)
                predicted_index = labels.index(result.predicted_frequency_hz)
                matrix[actual_index][predicted_index] += 1
                per_frequency[actual_frequency][0] += int(result.predicted_frequency_hz == actual_frequency)
                per_frequency[actual_frequency][1] += 1
                confidences.append(result.confidence)
        total = len(labels) * args.trials_per_frequency
        correct = sum(matrix[index][index] for index in range(len(labels)))
        reports.append(
            {
                "snr_db": snr_db,
                "total_trials": total,
                "overall_accuracy": correct / total,
                "per_frequency_accuracy": {
                    str(frequency): correct_count / count
                    for frequency, (correct_count, count) in per_frequency.items()
                },
                "confusion_matrix": {"labels_hz": labels, "rows_actual_columns_predicted": matrix},
                "mean_confidence": sum(confidences) / len(confidences),
                "elapsed_s": time.monotonic() - started,
            }
        )
    return {
        "decoder": args.decoder,
        "seed": args.seed,
        "trials_per_frequency": args.trials_per_frequency,
        "config_path": str(args.config),
        "candidate_frequencies_hz": labels,
        "reports": reports,
    }


def main() -> None:
    args = _arguments()
    report = evaluate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"synthetic_{args.decoder}_evaluation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for item in report["reports"]:
        print(
            f"SNR {item['snr_db']:g} dB: accuracy={item['overall_accuracy']:.3f}, "
            f"confidence={item['mean_confidence']:.3f}, elapsed={item['elapsed_s']:.2f}s"
        )
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
