import json
import asyncio
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pytest
import yaml

from ssvep_demo.config import load_config
from ssvep_demo.eeg_archive import EEGWindowArchiveWriter, ReplayEEGSource
from ssvep_demo.live_demo import LiveDemoState, LiveTrialCoordinator, SSVEPLiveDemoRunner
from ssvep_demo.live_source import (
    LiveProtocolError,
    LiveSampleFrame,
    LiveStreamMetadata,
    OmniBCIWebSocketSource,
    UINT32_MAX,
    _WorkerEvent,
    parse_omnibci_message,
)
from ssvep_demo.protocol import CANONICAL_CHANNELS, DecodeResult


CONFIG_PATH = Path(__file__).parents[1] / "config" / "ssvep_demo.yaml"
CONFIG = load_config(CONFIG_PATH)
SOURCE_NAMES = tuple(f"CH{index}" for index in range(1, 9))


def _payload(
    sequences,
    *,
    valid=True,
    sample_rate=250,
    channel_names=SOURCE_NAMES,
    value_factory=lambda sequence, channel: sequence * 10 + channel,
):
    frames = []
    for sequence in sequences:
        frame_valid = valid(sequence) if callable(valid) else valid
        frames.append(
            {
                "sequence": sequence,
                "valid": frame_valid,
                "values_uv": [value_factory(sequence, channel) for channel in range(8)],
                "mode": 1,
                "status": [0, 0, 0],
            }
        )
    return json.dumps(
        {
            "type": "eeg",
            "sample_rate_hz": sample_rate,
            "channel_names": list(channel_names),
            "frames": frames,
        }
    )


def _ready_source(*, queue_capacity=None):
    live_config = CONFIG.live if queue_capacity is None else replace(CONFIG.live, queue_capacity=queue_capacity)
    source = OmniBCIWebSocketSource("ws://127.0.0.1:1/v1/stream", live_config)
    source._started = True
    source._connection_state = "connected"
    metadata = LiveStreamMetadata(250.0, SOURCE_NAMES, CANONICAL_CHANNELS)
    source._handle_event(_WorkerEvent("metadata", 0.0, metadata=metadata))
    return source


def _feed(source, message, *, clock_start=1.0):
    ticks = iter(clock_start + index / 250.0 for index in range(10000))
    parsed = parse_omnibci_message(message, source.config, clock=lambda: next(ticks))
    for frame in parsed.frames:
        source._handle_event(_WorkerEvent("frame", frame.received_monotonic_s, frame=frame))


def test_actual_batch_message_parses_frames_and_maps_ch1_through_ch8() -> None:
    parsed = parse_omnibci_message(
        _payload([7], value_factory=lambda _sequence, channel: channel + 0.25),
        CONFIG.live,
        clock=lambda: 12.5,
    )
    assert parsed.metadata == LiveStreamMetadata(250.0, SOURCE_NAMES, CANONICAL_CHANNELS)
    frame = parsed.frames[0]
    assert frame.sequence == 7 and frame.valid and frame.received_monotonic_s == 12.5
    assert frame.values_uv.tolist() == pytest.approx([0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25])
    assert frame.values_uv.shape == (8,) and not frame.values_uv.flags.writeable


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (_payload([1], sample_rate=500), "sample_rate_hz"),
        (_payload([1], channel_names=SOURCE_NAMES[:-1]), "8 channels"),
        (_payload([1], channel_names=tuple(reversed(SOURCE_NAMES))), "physical source order"),
    ],
)
def test_parser_rejects_rate_channel_count_and_order(message, error) -> None:
    with pytest.raises(LiveProtocolError, match=error):
        parse_omnibci_message(message, CONFIG.live)


def test_parser_rejects_conflicting_explicit_unit() -> None:
    payload = json.loads(_payload([1]))
    payload["unit"] = "V"
    with pytest.raises(LiveProtocolError, match="unit must be exactly 'uV'"):
        parse_omnibci_message(json.dumps(payload), CONFIG.live)


def test_config_rejects_invalid_live_mapping_and_contract(tmp_path: Path) -> None:
    base = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cases = [
        (lambda value: value["live"].update(expected_sample_rate_hz=500), "exactly 250"),
        (lambda value: value["live"].update(trial_samples=999), "exactly 1000"),
        (lambda value: value["live"]["channel_map"].update(CH8="Oz"), "map once each"),
    ]
    for index, (mutate, expected) in enumerate(cases):
        value = json.loads(json.dumps(base))
        mutate(value)
        path = tmp_path / f"bad_{index}.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            load_config(path)


def test_exactly_1000_consecutive_frames_form_channel_major_window() -> None:
    source = _ready_source()
    source.begin_trial(0.5)
    # The 1001st and later samples are deliberately ignored after completion.
    _feed(source, _payload(range(100, 1101)))
    status = source.status()
    window = source.try_take_completed_window()
    assert window is not None and window.data.shape == (8, 1000)
    assert window.channel_names == list(CANONICAL_CHANNELS)
    assert window.data[0, 0] == 1000 and window.data[7, 999] == 10997
    assert status.start_sequence == 100 and status.end_sequence == 1099
    assert status.received_samples == 1000


def test_999_frames_never_produce_decoder_input() -> None:
    source = _ready_source()
    source.begin_trial(0.5)
    _feed(source, _payload(range(10, 1009)))
    assert source.status().collected_samples == 999
    assert source.try_take_completed_window() is None


@pytest.mark.parametrize(
    ("sequences", "valid", "expected_reason"),
    [
        ([10, 12], True, "sequence_gap"),
        ([10, 10], True, "duplicate_sequence"),
        ([10, 9], True, "sequence_backwards"),
        ([10, 11], lambda sequence: sequence != 11, "invalid_frame"),
    ],
)
def test_discontinuities_and_invalid_frame_abort_current_trial(sequences, valid, expected_reason) -> None:
    source = _ready_source()
    source.begin_trial(0.5)
    _feed(source, _payload(sequences, valid=valid))
    assert source.status().trial_failure_reason == expected_reason
    assert source.try_take_completed_window() is None


def test_disconnect_reconnect_and_queue_overflow_abort_current_trial() -> None:
    source = _ready_source(queue_capacity=1)
    source.begin_trial(0.0)
    source._handle_event(_WorkerEvent("disconnected", 1.0, reason="stream_disconnected"))
    assert source.status().trial_failure_reason == "stream_disconnected"

    source._connection_state = "connected"
    source.begin_trial(1.1)
    source._handle_event(_WorkerEvent("reconnected", 1.2))
    assert source.status().trial_failure_reason == "stream_reconnected"
    assert source.status().reconnect_count == 1

    source._connection_state = "connected"
    source.begin_trial(1.3)
    source._put_event(_WorkerEvent("connecting", 1.4))
    source._put_event(_WorkerEvent("frame", 1.5, frame=LiveSampleFrame(1, True, np.zeros(8), 1.5)))
    source.poll()
    assert source.status().trial_failure_reason == "queue_overflow"
    assert source.status().queue_overflow_count == 1


def test_protocol_error_and_metadata_change_abort_current_trial() -> None:
    source = _ready_source()
    source.begin_trial(0.0)
    source._handle_event(_WorkerEvent("protocol_error", 1.0, reason="protocol_error", detail="bad JSON"))
    assert source.status().trial_failure_reason == "protocol_error"

    source._connection_state = "connected"
    source.begin_trial(1.1)
    changed = LiveStreamMetadata(250.0, SOURCE_NAMES, tuple(reversed(CANONICAL_CHANNELS)))
    source._handle_event(_WorkerEvent("metadata_changed", 1.2, metadata=changed, reason="metadata_mismatch"))
    assert source.status().trial_failure_reason == "metadata_mismatch"


def test_pre_flip_frames_are_ignored_first_post_flip_sequence_starts_trial() -> None:
    source = _ready_source()
    old = parse_omnibci_message(_payload([20]), CONFIG.live, clock=lambda: 0.9).frames[0]
    source.begin_trial(1.0)
    source._handle_event(_WorkerEvent("frame", 0.9, frame=old))
    new = parse_omnibci_message(_payload([21]), CONFIG.live, clock=lambda: 1.01).frames[0]
    source._handle_event(_WorkerEvent("frame", 1.01, frame=new))
    assert source.status().start_sequence == 21
    assert source.status().collected_samples == 1


def test_gap_samples_never_mix_and_u32_rollover_is_explicitly_continuous() -> None:
    source = _ready_source()
    source.begin_trial(0.0)
    _feed(source, _payload([40, 42]))
    assert source.status().received_samples == 1
    source.begin_trial(2.0)
    _feed(source, _payload(range(43, 1043)), clock_start=2.1)
    window = source.try_take_completed_window()
    assert window is not None and source.status().start_sequence == 43

    rollover = _ready_source()
    rollover.begin_trial(0.0)
    _feed(rollover, _payload([UINT32_MAX, 0]))
    assert rollover.status().trial_failure_reason is None
    assert rollover.status().collected_samples == 2


class _CountingDecoder:
    def __init__(self):
        self.calls = 0

    def decode(self, window):
        self.calls += 1
        assert not hasattr(window, "target_frequency_hz")
        return DecodeResult(8.0, CONFIG.commands[8.0], {8.0: 1.0, 10.0: 0.0}, 0.9, time.monotonic())


class _CountingDispatcher:
    def __init__(self):
        self.calls = 0

    def submit(self, result):
        self.calls += 1
        from ssvep_demo.control import DispatchDecision, ControlCommand

        return DispatchDecision(result.command, ControlCommand.STOP, "stopped", "test", 0, time.monotonic())


def test_decoder_receives_only_complete_unlabeled_window_once() -> None:
    source = _ready_source()
    source.begin_trial(0.0)
    _feed(source, _payload(range(1000)))
    decoder, dispatcher = _CountingDecoder(), _CountingDispatcher()
    coordinator = LiveTrialCoordinator(decoder, dispatcher)
    window = source.try_take_completed_window()
    assert window is not None
    coordinator.process(window)
    assert decoder.calls == dispatcher.calls == 1

    incomplete = _ready_source()
    incomplete.begin_trial(0.0)
    _feed(incomplete, _payload(range(999)))
    assert incomplete.try_take_completed_window() is None
    assert decoder.calls == dispatcher.calls == 1


def test_live_window_archive_preserves_sequences_and_replays(tmp_path: Path) -> None:
    source = _ready_source()
    source.begin_trial(0.0)
    _feed(source, _payload(range(500, 1500)))
    window = source.try_take_completed_window()
    status = source.status()
    assert window is not None
    writer = EEGWindowArchiveWriter(tmp_path / "live.npz", CANONICAL_CHANNELS, 1000)
    writer.add(
        window, trial_id=2, attempt_id=0, target_frequency_hz=10.0, source_mode="live",
        start_sequence=status.start_sequence, end_sequence=status.end_sequence, data_unit="uV",
    )
    replay = ReplayEEGSource(writer.path)
    assert np.array_equal(replay.window(0).data, window.data)
    with np.load(writer.path, allow_pickle=False) as archive:
        assert archive["source_mode"].tolist() == ["live"]
        assert archive["data_unit"].tolist() == ["uV"]
        assert archive["start_sequence"].tolist() == [500]
        assert archive["end_sequence"].tolist() == [1499]


def test_poll_is_nonblocking_close_is_idempotent_and_import_is_headless() -> None:
    source = _ready_source()
    started = time.perf_counter()
    for _ in range(100):
        source.poll()
    assert time.perf_counter() - started < 0.1
    source.close()
    source.close()
    assert source.status().connection_state == "closed"

    import ssvep_demo.live_demo as live_demo
    import ssvep_demo.live_source as live_source

    assert "psychopy" not in live_demo.__dict__
    assert not any(
        isinstance(value, OmniBCIWebSocketSource) for value in live_source.__dict__.values()
    )


def test_background_async_worker_delivers_typed_frames_without_blocking_main_thread() -> None:
    class FakeWebSocket:
        def __init__(self):
            self.calls = 0

        async def recv(self):
            self.calls += 1
            if self.calls == 1:
                return _payload([0])
            if self.calls == 2:
                await asyncio.sleep(0.05)
                return _payload(range(1, 1001))
            await asyncio.sleep(10)

    class FakeConnection:
        def __init__(self):
            self.websocket = FakeWebSocket()

        async def __aenter__(self):
            return self.websocket

        async def __aexit__(self, *_args):
            return False

    source = OmniBCIWebSocketSource(
        "ws://fake/v1/stream", CONFIG.live, websocket_connect=lambda *_args, **_kwargs: FakeConnection()
    )
    source.start()
    deadline = time.monotonic() + 2.0
    begun = False
    window = None
    while time.monotonic() < deadline and window is None:
        source.poll()
        status = source.status()
        if not begun and status.connection_state == "connected" and status.metadata is not None:
            source.begin_trial(time.monotonic())
            begun = True
        window = source.try_take_completed_window()
        time.sleep(0.001)
    source.close()
    assert begun and window is not None and window.data.shape == (8, 1000)
    assert not source.status().worker_alive


def test_live_runner_constructs_no_synthetic_or_replay_source(monkeypatch) -> None:
    import ssvep_demo.live_demo as module

    monkeypatch.setattr(module, "SyntheticEEGSource", lambda *a, **k: (_ for _ in ()).throw(AssertionError()), raising=False)
    monkeypatch.setattr(module, "ReplayEEGSource", lambda *a, **k: (_ for _ in ()).throw(AssertionError()), raising=False)
    source = _ready_source()
    runner = SSVEPLiveDemoRunner(CONFIG, CONFIG_PATH, server_url="ws://127.0.0.1:8766/v1/stream", max_trials=1, live_source=source)
    assert runner.live_source is source


def test_escape_aborts_live_trial_and_stops_controller() -> None:
    from ssvep_demo.control import SafeCommandDispatcher
    from ssvep_demo.virtual_car import VirtualCarController

    source = _ready_source()
    source.begin_trial(0.0)
    runner = SSVEPLiveDemoRunner(
        CONFIG, CONFIG_PATH, server_url="ws://127.0.0.1:8766/v1/stream", max_trials=1, live_source=source
    )

    class EscapeEvent:
        @staticmethod
        def getKeys(**_kwargs):
            return ["escape"]

    class Logger:
        def event(self, *_args, **_kwargs):
            pass

    car = VirtualCarController()
    car.execute(CONFIG.commands[12.0])
    dispatcher = SafeCommandDispatcher(car, CONFIG.control, CONFIG.commands)
    runner._run_loop(
        None, EscapeEvent(), None, Logger(), None, None, dispatcher, car, 60.0, 1 / 60.0
    )
    assert runner.state_machine.state is LiveDemoState.STOPPED
    assert source.status().trial_failure_reason == "escape"
    assert not car.is_moving
