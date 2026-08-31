#!/usr/bin/env python3
"""Run the PsychoPy frame-synchronised, visual-only SSVEP stimulus program."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssvep_demo.config import load_config
from ssvep_demo.stimulus import PsychoPyStimulusRunner


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "ssvep_demo.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fullscreen", dest="fullscreen", action="store_true")
    mode.add_argument("--windowed", dest="fullscreen", action="store_false")
    parser.set_defaults(fullscreen=None)
    parser.add_argument("--screen-index", type=int)
    parser.add_argument("--refresh-rate", type=float, help="Measured-rate fallback or explicit debugging override.")
    parser.add_argument("--cjk-font-file", type=Path, help="External static .otf/.ttf font; overrides YAML and bundled font.")
    parser.add_argument("--cjk-font-name", help="Internal font family name for --cjk-font-file or YAML font file.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ssvep_stimulus")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = load_config(args.config)
    runner = PsychoPyStimulusRunner(
        config,
        args.config,
        fullscreen=args.fullscreen,
        screen_index=args.screen_index,
        refresh_rate_hz=args.refresh_rate,
        output_dir=args.output_dir,
        cjk_font_file=args.cjk_font_file,
        cjk_font_name=args.cjk_font_name,
    )
    runner.run()


if __name__ == "__main__":
    main()
