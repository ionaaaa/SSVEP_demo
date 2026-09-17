#!/usr/bin/env python3
"""Local fake for the actual omniBCI ``/v1/stream`` batch JSON protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--scenario", choices=["continuous", "gap", "invalid", "reconnect"], default="continuous")
    parser.add_argument("--batch-size", type=int, default=5)
    return parser.parse_args()


def _message(sequence: int, count: int, scenario: str) -> tuple[str, int]:
    frames = []
    for offset in range(count):
        current = sequence
        sequence = (sequence + 1) % (1 << 32)
        if scenario == "gap" and current == 500:
            current = 501
            sequence = 502
        valid = not (scenario == "invalid" and current == 500)
        frames.append(
            {
                "sequence": current,
                "values_uv": [float(current % 100) + channel / 10.0 for channel in range(8)],
                "raw_counts": None,
                "valid": valid,
                "mode": 1,
                "status": [0, 0, 0],
                "source_timestamp_ms": None,
                "triggers": [],
            }
        )
    return json.dumps(
        {
            "type": "eeg",
            "sample_rate_hz": 250,
            "channel_names": [f"CH{index}" for index in range(1, 9)],
            "frames": frames,
        }
    ), sequence


async def serve(host: str, port: int, scenario: str, batch_size: int) -> None:
    from websockets.asyncio.server import serve as websocket_serve

    connection_count = 0
    next_sequence = 0

    async def handler(websocket) -> None:
        nonlocal connection_count, next_sequence
        connection_count += 1
        sent = 0
        try:
            while True:
                message, next_sequence = _message(next_sequence, batch_size, scenario)
                await websocket.send(message)
                sent += batch_size
                if scenario == "reconnect" and connection_count == 1 and sent >= 250:
                    await websocket.close()
                    return
                await asyncio.sleep(batch_size / 250.0)
        except Exception:
            return

    async with websocket_serve(handler, host, port):
        print(f"Fake omniBCI WebSocket: ws://{host}:{port}/v1/stream ({scenario})", flush=True)
        await asyncio.Future()


def main() -> None:
    args = arguments()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in 1..65535")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    try:
        asyncio.run(serve(args.host, args.port, args.scenario, args.batch_size))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

